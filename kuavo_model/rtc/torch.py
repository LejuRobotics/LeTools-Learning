"""PyTorch RTC primitives; flow time may run noise-to-data or data-to-noise."""

import math
import torch


def prefix_weights(delay, end, total, schedule="exp", *, device=None, dtype=None):
    end = min(max(0, int(end)), total)
    start = min(max(0, int(delay)), end)
    weights = torch.zeros(total, device=device, dtype=dtype or torch.float32)
    weights[:start] = 1
    if schedule == "ones":
        weights[:end] = 1
    elif schedule in ("linear", "exp"):
        ramp = torch.linspace(1, 0, end - start + 2, device=device, dtype=weights.dtype)[1:-1]
        if schedule == "exp":
            ramp = ramp * torch.expm1(ramp) / math.expm1(1.0)
        weights[start:end] = ramp
    elif schedule != "zeros":
        raise ValueError(f"Unsupported RTC prefix schedule: {schedule}")
    return weights


def prepare_prefix(prefix, sample):
    if prefix is None:
        return None
    prefix = torch.as_tensor(prefix, device=sample.device, dtype=sample.dtype)
    if prefix.ndim == 2:
        prefix = prefix.unsqueeze(0)
    if prefix.ndim != 3 or prefix.shape[0] != sample.shape[0] or prefix.shape[2] != sample.shape[2]:
        raise ValueError(f"RTC prefix {tuple(prefix.shape)} must match sample batch/action dimensions {tuple(sample.shape)}")
    if prefix.shape[1] == 0 or not torch.isfinite(prefix).all():
        raise ValueError("RTC prefix must be nonempty and finite")
    return prefix[:, :sample.shape[1]].detach()


def guided_velocity(denoise, sample, time, prefix, *, delay=0, horizon=10,
                    schedule="exp", max_guidance_weight=10.0, noise_to_data=False):
    """Full VJP through endpoint prediction, matching Kinetix realtime_action.

    Do not detach the denoiser before differentiating: the velocity Jacobian is
    part of the official correction. Disable inference_mode in every outer caller.
    """
    if prefix is None:
        return denoise(sample)
    with torch.inference_mode(False), torch.enable_grad():
        x = sample.detach().clone().requires_grad_(True)
        target = prepare_prefix(prefix, x)
        length = target.shape[1]
        target = torch.nn.functional.pad(target, (0, 0, 0, x.shape[1] - length))
        time_tensor = torch.as_tensor(time, dtype=x.dtype, device=x.device)
        remaining = 1 - time_tensor if noise_to_data else time_tensor
        sign = 1 if noise_to_data else -1
        velocity = denoise(x)
        endpoint = x + sign * remaining * velocity
        weights = prefix_weights(delay, min(horizon, length), x.shape[1], schedule, device=x.device, dtype=x.dtype)[None, :, None]
        error = ((target - endpoint) * weights).detach()
        correction = torch.autograd.grad(endpoint, x, grad_outputs=error)[0]
        progress = 1 - remaining
        eps = torch.finfo(x.dtype).eps
        strength = (remaining.square() + progress.square()) / (remaining * progress).clamp_min(eps)
        strength = strength.clamp(max=max_guidance_weight)
        return (velocity + sign * strength * correction).detach()


def training_condition(actions, noise, time, max_delay, *, noise_to_data=False, delays=None):
    """Random clean prefix, token-wise flow time, and suffix-only loss mask."""
    if type(max_delay) is not int or not 0 < max_delay < actions.shape[1]:
        raise ValueError("rtc_training_max_delay must be positive and smaller than action horizon")
    batch, horizon, _ = actions.shape
    if delays is None:
        probabilities = torch.softmax(-torch.arange(max_delay + 1, device=actions.device, dtype=torch.float32), dim=0)
        delays = torch.multinomial(probabilities, batch, replacement=True)
    delays = torch.as_tensor(delays, device=actions.device)
    if delays.shape != (batch,) or torch.any(delays < 0) or torch.any(delays > max_delay):
        raise ValueError("Training RTC delays exceed the checkpoint's supported range")
    frozen = torch.arange(horizon, device=actions.device)[None] < delays[:, None]
    token_time = time[:, None].expand(batch, horizon)
    clean_time = 1.0 if noise_to_data else 0.0
    token_time = torch.where(frozen, clean_time, token_time)
    fraction = token_time[..., None]
    noisy = (1 - fraction) * noise + fraction * actions if noise_to_data else fraction * noise + (1 - fraction) * actions
    return noisy, token_time, (~frozen).unsqueeze(-1)


def trained_step(sample, time, prefix, delay, training_max_delay, *, noise_to_data=False):
    """Clamp the committed prefix and expose a clean timestep per prefix token."""
    if training_max_delay <= 0:
        raise ValueError("RTC trained mode requires a checkpoint trained with rtc_training_max_delay > 0")
    delay = int(delay)
    if delay < 0 or delay > training_max_delay:
        raise ValueError(f"RTC inference delay {delay} exceeds training maximum {training_max_delay}")
    times = torch.as_tensor(time, dtype=sample.dtype, device=sample.device)
    if times.ndim == 1:
        times = times[:, None]
    times = times.expand(sample.shape[0], sample.shape[1])
    if prefix is None:
        if delay:
            raise ValueError("RTC cannot condition a nonzero delay without previous actions")
        return sample, times, None
    prefix = prepare_prefix(prefix, sample)
    if delay > prefix.shape[1]:
        raise ValueError("RTC delay exceeds available prefix length")
    padded = torch.nn.functional.pad(prefix, (0, 0, 0, sample.shape[1] - prefix.shape[1]))
    frozen = torch.arange(sample.shape[1], device=sample.device)[None, :] < delay
    clamped = torch.where(frozen[..., None], padded, sample)
    times = torch.where(frozen, 1.0 if noise_to_data else 0.0, times)
    return clamped, times, (frozen[..., None], padded)


def finish_trained_step(sample, frozen_prefix):
    if frozen_prefix is None:
        return sample
    mask, prefix = frozen_prefix
    return torch.where(mask, prefix, sample)


def suffix_loss(losses, suffix_mask):
    """Preserve an unreduced loss API while averaging only supervised tokens."""
    scale = (losses.shape[0] * losses.shape[1]) / suffix_mask.sum().clamp_min(1)
    return losses * suffix_mask * scale
