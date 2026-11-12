"""RTC endpoint validation and reversible robot/model action packing."""

import math
import numpy as np


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def parse_request(payload, training_max_delay):
    observation = payload["observation"]
    rtc = dict(payload["rtc"])
    if rtc.get("mode") not in ("guided", "trained"):
        raise ValueError("RTC mode must be guided or trained")
    if rtc.get("prefix_attention_schedule", "exp") not in ("zeros", "ones", "linear", "exp"):
        raise ValueError("Unsupported RTC prefix_attention_schedule")
    for key in ("inference_delay", "execution_horizon"):
        if type(rtc.get(key)) is not int or rtc[key] < (1 if key == "execution_horizon" else 0):
            raise ValueError(f"RTC {key} is invalid")
    strength = rtc.get("max_guidance_weight", 10.0)
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("RTC max_guidance_weight must be finite and positive")
    if rtc["mode"] == "trained" and (training_max_delay <= 0 or rtc["inference_delay"] > training_max_delay):
        raise ValueError(f"RTC trained mode requires a trained checkpoint; supported maximum delay={training_max_delay}")
    original = payload.get("previous_original")
    absolute = payload.get("previous_absolute")
    if (original is None) != (absolute is None):
        raise ValueError("Both RTC action spaces must be provided together")
    if original is not None:
        if payload.get("previous_context") is None:
            raise ValueError("RTC prefix requires the observation context that produced it")
        original, absolute = as_numpy(original), as_numpy(absolute)
        if original.ndim != 2 or absolute.ndim != 2 or len(original) != len(absolute) or not len(original):
            raise ValueError("RTC original and absolute prefixes must be nonempty, aligned [T,D] arrays")
        if not np.isfinite(original).all() or not np.isfinite(absolute).all():
            raise ValueError("RTC prefixes must be finite")
        if rtc["inference_delay"] > len(original):
            raise ValueError("RTC inference_delay exceeds available previous actions")
        target = rtc["execution_horizon"]
        if rtc["inference_delay"] > target:
            raise ValueError("RTC inference_delay exceeds execution_horizon")
        if len(original) < target:
            original = np.concatenate([original, np.repeat(original[-1:], target - len(original), axis=0)])
            absolute = np.concatenate([absolute, np.repeat(absolute[-1:], target - len(absolute), axis=0)])
        original, absolute = original[:target], absolute[:target]
    elif rtc["inference_delay"]:
        raise ValueError("RTC inference_delay must be zero without a previous chunk")
    return observation, rtc, original, absolute


def reconcile_absolute_prefix(native_absolute, executed_absolute, pack_action):
    """Invert a model's packing permutation while preserving unexecuted dimensions.

    Native actions have already been decoded using their OLD anchor. Overwrite
    controlled coordinates with the execution queue, then the model's own input
    processor re-anchors/normalizes them against the NEW observation. Refuse a
    lossy or nonlinear packing instead of guessing joint/gripper/pose layouts.
    """
    native = as_numpy(native_absolute).copy()
    executed = as_numpy(executed_absolute)
    if native.ndim != 2 or executed.ndim != 2 or len(native) != len(executed):
        raise ValueError("RTC native/execution prefixes must have matching time dimensions")
    zero = np.zeros(native.shape[1], dtype=np.float32)
    baseline = as_numpy(pack_action(zero))
    matrix = np.stack([as_numpy(pack_action(basis)) - baseline for basis in np.eye(native.shape[1], dtype=np.float32)], axis=1)
    if matrix.shape[0] != executed.shape[1]:
        raise ValueError("RTC execution prefix dimension does not match adapter packing")
    for row in range(matrix.shape[0]):
        columns = np.flatnonzero(np.abs(matrix[row]) > 1e-6)
        if len(columns) != 1 or not np.isclose(matrix[row, columns[0]], 1.0) or not np.isclose(baseline[row], 0.0):
            raise ValueError("RTC requires an invertible coordinate selection in the adapter action packing")
        native[:, columns[0]] = executed[:, row]
    repacked = np.stack([pack_action(step) for step in native])
    if not np.allclose(repacked, executed, atol=1e-5):
        raise ValueError("RTC action packing round trip failed")
    return native


def flatten_groups(groups, keys):
    values = [as_numpy(groups[key]) for key in keys]
    dims = [value.shape[-1] for value in values]
    flat = np.concatenate(values, axis=-1)

    def unflatten(array):
        splits = np.cumsum(dims)[:-1]
        return dict(zip(keys, np.split(array, splits, axis=-1)))

    return flat, unflatten


def response(original, processed, training_max_delay, action_context=None):
    original, processed = as_numpy(original), as_numpy(processed)
    if (original.ndim != 2 or processed.ndim != 2 or len(original) != len(processed)
            or not len(original) or not np.isfinite(original).all() or not np.isfinite(processed).all()):
        raise ValueError("RTC response must preserve paired action spaces with matching time dimensions")
    return {"original_actions": original, "processed_actions": processed,
            "rtc_training_max_delay": int(training_max_delay), "action_context": action_context}
