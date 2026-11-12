"""JAX RTC equations matching Kinetix's flow-policy reference."""

import math
import jax
import jax.numpy as jnp


def prefix_weights(delay, end, total, schedule="exp"):
    index = jnp.arange(total)
    start = jnp.minimum(delay, end)
    ramp = jnp.clip((end - index) / jnp.maximum(end - start + 1, 1), 0, 1)
    if schedule == "exp":
        ramp = ramp * jnp.expm1(ramp) / math.expm1(1.0)
    elif schedule == "ones":
        ramp = jnp.ones(total)
    elif schedule == "zeros":
        ramp = jnp.zeros(total)
    elif schedule != "linear":
        raise ValueError(f"Unsupported RTC prefix schedule: {schedule}")
    return jnp.where(index < start, 1.0, jnp.where(index < end, ramp, 0.0))


def guided_velocity(denoise, sample, time, prefix, *, delay=0, horizon=10,
                    schedule="exp", max_guidance_weight=10.0):
    """OpenPI convention: t=1 noise, t=0 data; integrate negative dt."""
    if prefix is None:
        return denoise(sample)
    length = min(prefix.shape[1], sample.shape[1])
    target = jnp.pad(prefix[:, :length], ((0, 0), (0, sample.shape[1] - length), (0, 0)))

    def endpoint(x):
        velocity = denoise(x)
        return x - time * velocity, velocity

    predicted, vjp, velocity = jax.vjp(endpoint, sample, has_aux=True)
    weights = prefix_weights(delay, jnp.minimum(horizon, length), sample.shape[1], schedule)
    correction = vjp((target - predicted) * weights[None, :, None])[0]
    strength = (time**2 + (1 - time)**2) / jnp.maximum(time * (1 - time), 1e-6)
    return velocity - jnp.minimum(strength, max_guidance_weight) * correction


def training_condition(actions, noise, time, max_delay, rng):
    if not 0 < max_delay < actions.shape[1]:
        raise ValueError("rtc_training_max_delay must be positive and smaller than action horizon")
    logits = -jnp.arange(max_delay + 1, dtype=jnp.float32)
    delays = jax.random.categorical(rng, logits, shape=(actions.shape[0],))
    frozen = jnp.arange(actions.shape[1])[None, :] < delays[:, None]
    times = jnp.where(frozen, 0.0, time[:, None])
    sample = times[..., None] * noise + (1 - times[..., None]) * actions
    return sample, times, ~frozen


def trained_step(sample, time, prefix, delay):
    frozen = jnp.arange(sample.shape[1])[None, :] < delay
    times = jnp.where(frozen, 0.0, jnp.broadcast_to(time, sample.shape[:2]))
    if prefix is None:
        return sample, times, None
    length = min(prefix.shape[1], sample.shape[1])
    target = jnp.pad(prefix[:, :length], ((0, 0), (0, sample.shape[1] - length), (0, 0)))
    return jnp.where(frozen[..., None], target, sample), times, (frozen[..., None], target)
