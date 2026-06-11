#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuavo_deploy.config import KuavoConfig, load_kuavo_config

try:
    from kuavo_deploy.utils.logging_utils import setup_logger
except ModuleNotFoundError:
    import logging

    def setup_logger(name: str, level: str = "INFO"):
        logging.basicConfig(level=getattr(logging, level, logging.INFO))
        return logging.getLogger(name)

log_model = setup_logger("model")
log_robot = setup_logger("robot")


def _resolve_pretrained_path(cfg) -> Path:
    if cfg.pretrained_path:
        return Path(cfg.pretrained_path)
    return Path(f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}")


def _setup_policy(
    pretrained_path: Path,
    policy_type: str,
    device,
    task_prompt: str,
):
    from kuavo_deploy.kuavo_service.client import PolicyClient
    from kuavo_deploy.utils.policy_loader import load_native_policy_bundle

    if device.type == "cpu":
        log_model.warning("Using CPU for offline inference; this may be slow.")

    if policy_type == "client":
        preprocessor, postprocessor = lambda obs: obs, lambda action: action
        return PolicyClient(task_prompt=task_prompt), preprocessor, postprocessor, None

    policy, preprocessor, postprocessor, pretrained_model_dir = load_native_policy_bundle(
        pretrained_path=pretrained_path,
        device=device,
        strict=True,
    )
    log_model.info(f"Model loaded from {pretrained_model_dir}")
    log_model.info(f"Model type: {policy.config.type}")
    log_model.info(f"Model n_obs_steps: {getattr(policy.config, 'n_obs_steps', 'unknown')}")
    return policy, preprocessor, postprocessor, pretrained_model_dir


def _patch_data_config_from_deploy(data_cfg, config: KuavoConfig):
    """Make rosbag parsing follow the active deploy config where possible."""

    env = config.env
    inf = config.inference

    data_cfg.dataset.platform_type = env.platform_type
    data_cfg.dataset.eef_type = env.eef_type
    data_cfg.dataset.which_arm = env.which_arm
    data_cfg.dataset.depth_range = list(env.depth_range)
    data_cfg.dataset.dex_dof_needed = env.qiangnao_dof_needed
    data_cfg.dataset.is_binary = env.is_binary
    data_cfg.dataset.delta_action = env.use_delta
    data_cfg.dataset.train_hz = env.ros_rate
    data_cfg.dataset.task_description = inf.task_prompt or "robot manipulation"

    if env.image_size and len(env.image_size) == 2:
        # Deploy config stores [width, height] in the comments and current usage.
        data_cfg.dataset.resize.width = int(env.image_size[0])
        data_cfg.dataset.resize.height = int(env.image_size[1])

    obs_keys = set(env.obs_key_map.keys())
    data_cfg.dataset.use_depth = any("depth" in key for key in obs_keys)
    if "head_cam_h" in obs_keys:
        data_cfg.dataset.main_timeline = "head_cam_h"
    elif "wrist_cam_l" in obs_keys:
        data_cfg.dataset.main_timeline = "wrist_cam_l"
    elif "wrist_cam_r" in obs_keys:
        data_cfg.dataset.main_timeline = "wrist_cam_r"

    return data_cfg


def _load_bag_frames(
    *,
    bag_path: Path,
    data_config_path: Path,
    config: KuavoConfig,
    chunk_size: int,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    # These imports require the ROS/rosbag Python environment, so keep them
    # local to allow --help and config parsing without ROS.
    import torch
    from omegaconf import OmegaConf

    from kuavo_data.common import kuavo_dataset as kuavo
    from kuavo_data.common.config_platform import get_arm_joint_slice

    data_cfg = OmegaConf.load(data_config_path)
    data_cfg = _patch_data_config_from_deploy(data_cfg, config)
    kuavo.init_parameters(data_cfg)

    frames: list[dict[str, Any]] = []
    bag_reader = kuavo.KuavoRosbagReader()
    arm_start, arm_end = get_arm_joint_slice(kuavo.PLATFORM_TYPE)
    first_raw_state: np.ndarray | None = None
    first_raw_action: np.ndarray | None = None

    def _array(aligned_frame: dict[str, Any], key: str, dtype=np.float32) -> np.ndarray:
        item = aligned_frame.get(key)
        if item is None:
            return np.array([], dtype=dtype)
        return np.asarray(item.get("data", []), dtype=dtype)

    def _normalize_binary_or_range(values: np.ndarray, *, binary_threshold: float, scale: float) -> np.ndarray:
        if values.size == 0:
            return values
        if kuavo.IS_BINARY:
            return np.where(values > binary_threshold, 1.0, 0.0).astype(np.float32)
        return (values / scale).astype(np.float32)

    def _hand_pair(
        *,
        state: np.ndarray,
        action: np.ndarray,
        claw_state: np.ndarray,
        claw_action: np.ndarray,
        qiangnao_state: np.ndarray,
        qiangnao_action: np.ndarray,
        hand_side: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        robot_slice = kuavo.SLICE_ROBOT[hand_side]
        arm_state = state[robot_slice[0] : robot_slice[-1]]
        arm_action = action[robot_slice[0] : robot_slice[-1]]

        if kuavo.USE_LEJU_CLAW:
            claw_slice = kuavo.SLICE_CLAW[hand_side]
            eef_state = claw_state[claw_slice[0] : claw_slice[-1]]
            eef_action = claw_action[claw_slice[0] : claw_slice[-1]]
        elif kuavo.USE_QIANGNAO:
            dex_slice = kuavo.SLICE_DEX[hand_side]
            eef_state = qiangnao_state[dex_slice[0] : dex_slice[-1]]
            eef_action = qiangnao_action[dex_slice[0] : dex_slice[-1]]
        else:
            raise ValueError("Only leju_claw, rq2f85 and qiangnao end effectors are supported.")

        return (
            np.concatenate((arm_state, eef_state)).astype(np.float32),
            np.concatenate((arm_action, eef_action)).astype(np.float32),
        )

    def on_frame(aligned_frame: dict[str, Any], frame_idx: int) -> None:
        nonlocal first_raw_state, first_raw_action

        if max_frames is not None and len(frames) >= max_frames:
            return

        state = _array(aligned_frame, "observation.state")
        action = _array(aligned_frame, "action")
        arm_traj = _array(aligned_frame, "action.kuavo_arm_traj")
        if state.size == 0 or action.size == 0 or arm_traj.size == 0:
            return

        action = action.copy()
        action[arm_start:arm_end] = arm_traj

        if first_raw_state is None:
            first_raw_state = state.copy()
        if first_raw_action is None:
            first_raw_action = action.copy()

        if kuavo.RELATIVE_START:
            state = state - first_raw_state
            action = action - first_raw_action
        if kuavo.DELTA_ACTION:
            action = action - state

        claw_state = _array(aligned_frame, "observation.claw")
        claw_action = _array(aligned_frame, "action.claw")
        qiangnao_state = _array(aligned_frame, "observation.qiangnao")
        qiangnao_action = _array(aligned_frame, "action.qiangnao")
        rq2f85_state = _array(aligned_frame, "observation.rq2f85")
        rq2f85_action = _array(aligned_frame, "action.rq2f85")

        if claw_state.size == 0 and qiangnao_state.size == 0 and rq2f85_state.size == 0:
            return
        if claw_action.size == 0 and qiangnao_action.size == 0 and rq2f85_action.size == 0:
            return

        claw_state = _normalize_binary_or_range(claw_state, binary_threshold=50, scale=100)
        claw_action = _normalize_binary_or_range(claw_action, binary_threshold=50, scale=100)
        qiangnao_state = _normalize_binary_or_range(qiangnao_state, binary_threshold=50, scale=100)
        qiangnao_action = _normalize_binary_or_range(qiangnao_action, binary_threshold=50, scale=100)
        rq2f85_state = _normalize_binary_or_range(rq2f85_state, binary_threshold=0.4, scale=0.8)
        rq2f85_action = _normalize_binary_or_range(rq2f85_action, binary_threshold=70, scale=255)

        if claw_state.size == 0 and qiangnao_state.size == 0:
            claw_state = rq2f85_state
            claw_action = rq2f85_action

        state_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        if kuavo.CONTROL_HAND_SIDE in ("left", "both"):
            s, a = _hand_pair(
                state=state,
                action=action,
                claw_state=claw_state,
                claw_action=claw_action,
                qiangnao_state=qiangnao_state,
                qiangnao_action=qiangnao_action,
                hand_side=0,
            )
            state_parts.append(s)
            action_parts.append(a)
        if kuavo.CONTROL_HAND_SIDE in ("right", "both"):
            s, a = _hand_pair(
                state=state,
                action=action,
                claw_state=claw_state,
                claw_action=claw_action,
                qiangnao_state=qiangnao_state,
                qiangnao_action=qiangnao_action,
                hand_side=1,
            )
            state_parts.append(s)
            action_parts.append(a)

        frame = {
            "observation.state": torch.from_numpy(np.concatenate(state_parts).astype(np.float32)),
            "action": torch.from_numpy(np.concatenate(action_parts).astype(np.float32)),
            "frame_index": frame_idx,
            "timestamp": aligned_frame.get("timestamp"),
        }

        for cam_key in kuavo.DEFAULT_CAMERA_NAMES:
            cam_data = aligned_frame.get(cam_key)
            if not cam_data or "data" not in cam_data:
                return
            img = cam_data["data"]
            if "depth" in cam_key:
                min_d, max_d = kuavo.DEPTH_RANGE
                depth = np.clip(img, min_d, max_d)
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                frame[f"observation.{cam_key}"] = depth_uint8[..., None].repeat(3, -1)
            else:
                frame[f"observation.images.{cam_key}"] = img

        frames.append(frame)

    bag_reader.process_rosbag_chunked(
        bag_file=str(bag_path),
        frame_callback=on_frame,
        chunk_size=chunk_size,
        save_callback=lambda: None,
    )
    return frames


def _frame_to_policy_observation(frame: dict[str, Any], device) -> dict[str, Any]:
    import torch

    obs: dict[str, Any] = {}
    for key, value in frame.items():
        if key in {"action", "frame_index", "timestamp"}:
            continue
        if key == "observation.state":
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
            obs[key] = tensor.float().unsqueeze(0).to(device, non_blocking=True)
        else:
            obs[key] = value
    return obs


def _action_to_numpy(action: Any) -> np.ndarray:
    import torch

    if isinstance(action, torch.Tensor):
        tensor = action
    elif isinstance(action, np.ndarray):
        return np.asarray(action).reshape(-1).astype(np.float64)
    else:
        tensor = torch.as_tensor(action)

    if tensor.ndim > 1:
        tensor = tensor.squeeze(0)
    return tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _infer_one_action(policy, preprocessor, postprocessor, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
    import torch

    start = time.perf_counter()
    observation = preprocessor(observation)
    with torch.inference_mode():
        action = policy.select_action(observation)
    action = postprocessor(action)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return _action_to_numpy(action), elapsed_ms


def _to_chunk_tensor(actions: Any):
    import torch

    if isinstance(actions, torch.Tensor):
        tensor = actions
    elif isinstance(actions, np.ndarray):
        tensor = torch.from_numpy(actions)
    else:
        tensor = torch.as_tensor(actions)

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected action chunk shape [T, D], got {tuple(tensor.shape)}")
    return tensor


def _select_action_chunk(policy, observation: dict[str, Any]) -> Any:
    if hasattr(policy, "select_action_chunk"):
        return policy.select_action_chunk(observation)
    if hasattr(policy, "predict_action_chunk"):
        return policy.predict_action_chunk(observation)
    return policy.select_action(observation)


def _postprocess_chunk(actions: Any, postprocessor) -> list[np.ndarray]:
    chunk = _to_chunk_tensor(actions)
    out: list[np.ndarray] = []
    for idx in range(chunk.shape[0]):
        processed = postprocessor(chunk[idx : idx + 1])
        processed = _to_chunk_tensor(processed)
        out.append(processed[0].detach().cpu().numpy().reshape(-1).astype(np.float64))
    return out


def _infer_action_chunk(
    policy,
    preprocessor,
    postprocessor,
    observation: dict[str, Any],
) -> tuple[list[np.ndarray], float]:
    import torch

    start = time.perf_counter()
    observation = preprocessor(observation)
    with torch.inference_mode():
        actions = _select_action_chunk(policy, observation)
    actions_np = _postprocess_chunk(actions, postprocessor)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return actions_np, elapsed_ms


def _infer_action_chunk_rtc(
    policy,
    preprocessor,
    observation: dict[str, Any],
    *,
    prev_chunk_leftover: np.ndarray | None,
    inference_delay: int,
    execution_horizon: int,
    rtc_options: dict[str, Any],
) -> tuple[list[np.ndarray], np.ndarray, float]:
    start = time.perf_counter()
    observation = preprocessor(observation)
    if not hasattr(policy, "select_action_chunk_rtc"):
        raise RuntimeError(
            f"rtc_full_enabled=true but policy {type(policy).__name__} "
            "does not implement select_action_chunk_rtc()"
        )
    result = policy.select_action_chunk_rtc(
        observation,
        prev_chunk_leftover=prev_chunk_leftover,
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
        rtc_options=rtc_options,
    )
    if not isinstance(result, dict) or not {"processed_actions", "original_actions"} <= result.keys():
        raise ValueError("RTC policy result must contain processed_actions and original_actions")
    processed = _to_chunk_tensor(result["processed_actions"]).detach().cpu().numpy()
    original = _to_chunk_tensor(result["original_actions"]).detach().cpu().numpy()
    if processed.shape[0] != original.shape[0]:
        raise ValueError("RTC processed/original chunks must contain the same number of steps")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return [np.asarray(action, dtype=np.float64) for action in processed], original, elapsed_ms


def _ramp_weights(blend_count: int, ramp: str) -> np.ndarray:
    if blend_count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if blend_count == 1:
        return np.ones((1,), dtype=np.float64)
    idx = np.arange(1, blend_count + 1, dtype=np.float64)
    if ramp == "linear":
        return idx / blend_count
    if ramp == "cosine":
        return 0.5 - 0.5 * np.cos(math.pi * idx / blend_count)
    raise ValueError(f"Unsupported ramp '{ramp}'. Valid: linear, cosine")


class OfflineActionBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = max(1, int(maxlen))
        self._actions: list[np.ndarray] = []
        self._total_popped = 0

    def qsize(self) -> int:
        return len(self._actions)

    def get_action_index(self) -> int:
        return self._total_popped

    def put_chunk(self, actions: list[np.ndarray]) -> dict[str, Any]:
        free = self.maxlen - len(self._actions)
        inserted_actions = actions[: max(0, free)]
        self._actions.extend(np.asarray(a).copy() for a in inserted_actions)
        return {
            "mode": "append",
            "produced": len(actions),
            "inserted": len(inserted_actions),
            "queued": len(self._actions),
        }

    def merge_chunk(
        self,
        actions: list[np.ndarray],
        *,
        delay_steps: int,
        overlap_steps: int,
        freeze_steps: int,
        keep_min_actions: int,
        ramp: str,
    ) -> dict[str, Any]:
        overlap_steps = max(0, int(overlap_steps))
        freeze_steps = max(0, int(freeze_steps))
        keep_min_actions = max(0, int(keep_min_actions))
        delay_steps = max(0, int(delay_steps))

        old = list(self._actions)
        old_size = len(old)
        new_len = len(actions)
        delay = min(delay_steps, new_len)
        new_valid = list(actions[delay:])

        if not new_valid:
            return {
                "mode": "blend_replace",
                "old_size": old_size,
                "produced": new_len,
                "delay_steps": delay,
                "kept": old_size,
                "blended": 0,
                "queued": old_size,
                "dropped_new_prefix": delay,
            }

        old_keep_count = min(old_size, max(freeze_steps, keep_min_actions))
        kept = old[:old_keep_count]
        old_blend = old[old_keep_count : old_keep_count + overlap_steps]
        new_blend = new_valid[:overlap_steps]
        blend_count = min(len(old_blend), len(new_blend))

        blended: list[np.ndarray] = []
        if blend_count > 0:
            weights = _ramp_weights(blend_count, ramp)
            for i, w in enumerate(weights):
                a_old = np.asarray(old_blend[i], dtype=np.float64)
                a_new = np.asarray(new_blend[i], dtype=np.float64)
                blended.append((1.0 - float(w)) * a_old + float(w) * a_new)

        tail = new_valid[blend_count:]
        self._actions = (kept + blended + tail)[: self.maxlen]
        return {
            "mode": "blend_replace",
            "old_size": old_size,
            "produced": new_len,
            "delay_steps": delay,
            "kept": len(kept),
            "blended": len(blended),
            "queued": len(self._actions),
            "dropped_new_prefix": delay,
        }

    def get(self) -> np.ndarray | None:
        if not self._actions:
            return None
        action = self._actions.pop(0)
        self._total_popped += 1
        return action


class OfflineRtcActionBuffer(OfflineActionBuffer):
    def __init__(self, maxlen: int):
        super().__init__(maxlen)
        self._original_actions: list[np.ndarray] = []

    def snapshot_original_actions(self) -> np.ndarray | None:
        if not self._original_actions:
            return None
        return np.stack([np.asarray(action).copy() for action in self._original_actions], axis=0)

    def replace_after_delay(
        self,
        actions: list[np.ndarray],
        original_actions: np.ndarray,
        *,
        delay_steps: int,
    ) -> dict[str, Any]:
        original = np.asarray(original_actions)
        if len(actions) != original.shape[0]:
            raise ValueError("RTC processed/original chunks must contain the same number of steps")
        old_size = len(self._actions)
        delay = min(max(0, int(delay_steps)), len(actions))
        remaining = actions[delay:]
        if len(remaining) > self.maxlen:
            raise ValueError(
                "RTC Full must preserve the model's remaining native chunk; "
                f"async_buffer_size={self.maxlen} is smaller than remaining chunk={len(remaining)}"
            )
        self._actions = [np.asarray(action).copy() for action in remaining]
        self._original_actions = [np.asarray(action).copy() for action in original[delay:]]
        return {
            "mode": "rtc_full_replace",
            "old_size": old_size,
            "produced": len(actions),
            "delay_steps": delay,
            "queued": len(self._actions),
        }

    def get(self) -> np.ndarray | None:
        action = super().get()
        if action is not None:
            if not self._original_actions:
                raise RuntimeError("RTC full offline queues are no longer aligned")
            self._original_actions.pop(0)
        return action


def _resolve_offline_control_hz(config: KuavoConfig) -> float:
    cfg = config.inference
    if cfg.async_control_hz and cfg.async_control_hz > 0:
        return float(cfg.async_control_hz)
    if config.env.ros_rate and config.env.ros_rate > 0:
        return float(config.env.ros_rate)
    return 10.0


def _chunk_delay_steps(infer_ms: float, control_hz: float, max_delay_steps: int | None = None) -> int:
    delay = max(0, int(math.ceil((infer_ms / 1000.0) * control_hz)))
    if max_delay_steps is not None:
        delay = min(delay, max(0, int(max_delay_steps)))
    return delay


def _write_csv(path: Path, predictions: np.ndarray, targets: np.ndarray, errors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = predictions.shape[1]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step"]
            + [f"pred_{i}" for i in range(dim)]
            + [f"target_{i}" for i in range(dim)]
            + [f"error_{i}" for i in range(dim)]
            + ["mse", "mae", "rmse"]
        )
        for step in range(predictions.shape[0]):
            err = errors[step]
            mse = float(np.mean(err**2))
            mae = float(np.mean(np.abs(err)))
            rmse = math.sqrt(mse)
            writer.writerow(
                [step]
                + predictions[step].tolist()
                + targets[step].tolist()
                + err.tolist()
                + [mse, mae, rmse]
            )


def _plot_actions(path: Path, predictions: np.ndarray, targets: np.ndarray, dims: list[int] | None) -> None:
    import matplotlib.pyplot as plt

    action_dim = predictions.shape[1]
    if dims is None:
        dims = list(range(min(action_dim, 16)))
    dims = [d for d in dims if 0 <= d < action_dim]
    if not dims:
        return

    cols = 2
    rows = math.ceil(len(dims) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3, rows * 2.4)), squeeze=False)
    steps = np.arange(predictions.shape[0])
    for ax, dim in zip(axes.ravel(), dims):
        ax.plot(steps, targets[:, dim], label="bag action", linewidth=1.2)
        ax.plot(steps, predictions[:, dim], label="pred action", linewidth=1.0, alpha=0.85)
        ax.set_title(f"action[{dim}]")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(dims) :]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _parse_dims(value: str | None) -> list[int] | None:
    if not value:
        return None
    dims: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        dims.append(int(item))
    return dims


def _prepare_observation_for_policy(
    *,
    frame: dict[str, Any],
    device,
    policy_type: str,
    task_prompt: str,
) -> dict[str, Any]:
    from kuavo_deploy.utils.policy_loader import inject_task_prompt

    observation = _frame_to_policy_observation(frame, device=device)
    if policy_type != "client":
        observation = inject_task_prompt(observation, task_prompt)
    return observation


def _run_sync_offline(
    *,
    frames: list[dict[str, Any]],
    total: int,
    policy,
    preprocessor,
    postprocessor,
    device,
    policy_type: str,
    task_prompt: str,
    log_step_times: bool,
) -> tuple[np.ndarray, np.ndarray, list[float], list[dict[str, Any]], dict[str, Any]]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    infer_times_ms: list[float] = []

    for step, frame in enumerate(tqdm(frames[:total], desc="Offline sync inference", unit="step")):
        observation = _prepare_observation_for_policy(
            frame=frame,
            device=device,
            policy_type=policy_type,
            task_prompt=task_prompt,
        )
        pred, infer_ms = _infer_one_action(policy, preprocessor, postprocessor, observation)
        target = _action_to_numpy(frame["action"])
        if pred.shape != target.shape:
            raise ValueError(
                f"Predicted action shape {pred.shape} does not match bag action shape {target.shape}. "
                "Check deploy env.which_arm/eef_type and data conversion config."
            )
        predictions.append(pred)
        targets.append(target)
        infer_times_ms.append(infer_ms)
        if log_step_times:
            print(f"[offline-sync] step={step} select_action_ms={infer_ms:.3f}", flush=True)

    return np.stack(predictions, axis=0), np.stack(targets, axis=0), infer_times_ms, [], {}


def _run_async_offline(
    *,
    config: KuavoConfig,
    frames: list[dict[str, Any]],
    total: int,
    policy,
    preprocessor,
    postprocessor,
    device,
    policy_type: str,
    task_prompt: str,
    log_step_times: bool,
) -> tuple[np.ndarray, np.ndarray, list[float], list[dict[str, Any]], dict[str, Any]]:
    cfg = config.inference
    rtc_full_enabled = bool(getattr(cfg, "rtc_full_enabled", False))
    buffer = (
        OfflineRtcActionBuffer(maxlen=cfg.async_buffer_size)
        if rtc_full_enabled
        else OfflineActionBuffer(maxlen=cfg.async_buffer_size)
    )
    control_hz = _resolve_offline_control_hz(config)
    low_watermark = max(0, int(cfg.async_low_watermark))
    rtc_lite_enabled = bool(getattr(cfg, "rtc_lite_enabled", False))
    rtc_options = {
        "enabled": True,
        "mode": str(cfg.rtc_full_mode),
        "prefix_attention_schedule": str(cfg.rtc_full_prefix_attention_schedule),
        "max_guidance_weight": float(cfg.rtc_full_max_guidance_weight),
        # RTC continuity target length plus inpainting ramp controls.
        "overlap_steps": int(cfg.rtc_full_overlap_steps),
        "frozen_steps": int(cfg.rtc_full_frozen_steps),
        "ramp_rate": float(cfg.rtc_full_ramp_rate),
        "debug": bool(cfg.rtc_full_debug),
    }

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    infer_times_ms: list[float] = []
    events: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    last_action: np.ndarray | None = None
    estimated_delay_steps = 0

    def start_inference(step: int, *, warmup: bool = False) -> dict[str, Any]:
        observation = _prepare_observation_for_policy(
            frame=frames[step],
            device=device,
            policy_type=policy_type,
            task_prompt=task_prompt,
        )
        action_index_before = buffer.get_action_index()
        if rtc_full_enabled:
            actions, original_actions, infer_ms = _infer_action_chunk_rtc(
                policy,
                preprocessor,
                observation,
                prev_chunk_leftover=buffer.snapshot_original_actions(),
                inference_delay=estimated_delay_steps,
                execution_horizon=int(cfg.rtc_full_overlap_steps),
                rtc_options=rtc_options,
            )
        else:
            actions, infer_ms = _infer_action_chunk(policy, preprocessor, postprocessor, observation)
            original_actions = None
        if log_step_times:
            print(
                f"[offline-async] request_step={step} chunk_infer_ms={infer_ms:.3f} "
                f"produced={len(actions)} warmup={warmup}",
                flush=True,
            )
        max_delay = (
            int(cfg.rtc_full_max_delay_steps)
            if rtc_full_enabled
            else int(cfg.rtc_lite_max_delay_steps) if rtc_lite_enabled else None
        )
        measured_delay = _chunk_delay_steps(infer_ms, control_hz, max_delay)
        # During warmup the real control loop has not started yet, so the
        # completed chunk is available before step 0 consumes anything.
        ready_delay = 0 if warmup else measured_delay
        return {
            "request_step": step,
            "ready_step": step + ready_delay,
            "actions": actions,
            "original_actions": original_actions,
            "infer_ms": infer_ms,
            "measured_delay": measured_delay,
            "action_index_before": action_index_before,
            "warmup": warmup,
        }

    def apply_pending(done: dict[str, Any], step: int) -> None:
        nonlocal estimated_delay_steps
        actual_consumed = max(0, buffer.get_action_index() - int(done["action_index_before"]))
        if rtc_full_enabled:
            delay_steps = 0 if done["warmup"] else max(actual_consumed, int(done["measured_delay"]))
            delay_steps = min(delay_steps, int(cfg.rtc_full_max_delay_steps))
            stats = buffer.replace_after_delay(
                done["actions"],
                done["original_actions"],
                delay_steps=delay_steps,
            )
            estimated_delay_steps = delay_steps
        elif rtc_lite_enabled:
            delay_steps = 0 if done["warmup"] else max(actual_consumed, int(done["measured_delay"]))
            delay_steps = min(delay_steps, int(cfg.rtc_lite_max_delay_steps))
            stats = buffer.merge_chunk(
                done["actions"],
                delay_steps=delay_steps,
                overlap_steps=int(cfg.rtc_lite_overlap_steps),
                freeze_steps=int(cfg.rtc_lite_freeze_steps),
                keep_min_actions=int(cfg.rtc_lite_keep_min_actions),
                ramp=str(cfg.rtc_lite_ramp),
            )
        else:
            stats = buffer.put_chunk(done["actions"])
            delay_steps = int(done["measured_delay"])

        events.append(
            {
                "request_step": done["request_step"],
                "ready_step": done["ready_step"],
                "apply_step": step,
                "infer_ms": done["infer_ms"],
                "measured_delay": done["measured_delay"],
                "actual_consumed": actual_consumed,
                "delay_steps": delay_steps,
                "rtc_lite_enabled": rtc_lite_enabled,
                "rtc_full_enabled": rtc_full_enabled,
                **stats,
            }
        )
        infer_times_ms.append(float(done["infer_ms"]))

    # Real async rollout waits for warmup actions before control starts.
    pending = start_inference(0, warmup=True)
    apply_pending(pending, 0)
    pending = None

    for step in tqdm(range(total), desc="Offline async inference", unit="step"):
        if pending is not None and pending["ready_step"] <= step:
            apply_pending(pending, step)
            pending = None

        if pending is None and buffer.qsize() <= low_watermark:
            pending = start_inference(step, warmup=False)
            if pending["ready_step"] <= step:
                apply_pending(pending, step)
                pending = None

        action_get_start = time.perf_counter()
        action = buffer.get()
        action_get_ms = (time.perf_counter() - action_get_start) * 1000.0
        if log_step_times:
            print(
                f"[offline-async] step={step} action_get_ms={action_get_ms:.6f} "
                f"buffer_after_get={buffer.qsize()}",
                flush=True,
            )
        if action is None:
            if last_action is None:
                raise RuntimeError(
                    f"Offline async buffer is empty at step {step} before any action was available. "
                    "Increase async_warmup_actions/async_low_watermark or inspect model chunk output."
                )
            action = last_action.copy()
            events.append({"step": step, "event": "hold_last_action", "action_get_ms": action_get_ms})

        target = _action_to_numpy(frames[step]["action"])
        if action.shape != target.shape:
            raise ValueError(
                f"Predicted action shape {action.shape} does not match bag action shape {target.shape}. "
                "Check deploy env.which_arm/eef_type and data conversion config."
            )
        predictions.append(np.asarray(action, dtype=np.float64))
        targets.append(target)
        last_action = np.asarray(action, dtype=np.float64)

    return np.stack(predictions, axis=0), np.stack(targets, axis=0), infer_times_ms, events, {}


def run_offline_bag_eval(
    *,
    config: KuavoConfig,
    bag_path: Path,
    data_config_path: Path,
    output_dir: Path | None,
    max_frames: int | None,
    chunk_size: int,
    mode: str,
    plot: bool,
    plot_dims: list[int] | None,
    log_step_times: bool,
) -> Path:
    import torch
    from lerobot.utils.random_utils import set_seed
    from lerobot_patches import custom_patches  # noqa: F401  Ensure LeRobot patches are installed.
    from kuavo_deploy.utils.policy_loader import resolve_eval_output_dir

    cfg = config.inference
    pretrained_path = _resolve_pretrained_path(cfg)
    eval_root = output_dir or resolve_eval_output_dir(pretrained_path, Path("outputs/offline_eval"))
    run_dir = eval_root / f"{bag_path.stem}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed=cfg.seed)
    device = torch.device(cfg.device)
    task_prompt = getattr(cfg, "task_prompt", "robot manipulation")
    policy, preprocessor, postprocessor, pretrained_model_dir = _setup_policy(
        pretrained_path=pretrained_path,
        policy_type=cfg.policy_type,
        device=device,
        task_prompt=task_prompt,
    )

    frames = _load_bag_frames(
        bag_path=bag_path,
        data_config_path=data_config_path,
        config=config,
        chunk_size=chunk_size,
        max_frames=max_frames,
    )
    if not frames:
        raise RuntimeError(f"No valid frames loaded from bag: {bag_path}")

    policy.reset()
    total = min(len(frames), cfg.max_episode_steps)
    if max_frames is not None:
        total = min(total, max_frames)

    resolved_mode = "async" if mode == "auto" and cfg.async_inference else "sync" if mode == "auto" else mode
    if resolved_mode == "async":
        pred_arr, target_arr, infer_times_ms, events, timing = _run_async_offline(
            config=config,
            frames=frames,
            total=total,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            policy_type=cfg.policy_type,
            task_prompt=task_prompt,
            log_step_times=log_step_times,
        )
    else:
        pred_arr, target_arr, infer_times_ms, events, timing = _run_sync_offline(
            frames=frames,
            total=total,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            policy_type=cfg.policy_type,
            task_prompt=task_prompt,
            log_step_times=log_step_times,
        )

    err_arr = pred_arr - target_arr

    mse_per_step = np.mean(err_arr**2, axis=1)
    mae_per_step = np.mean(np.abs(err_arr), axis=1)
    metrics = {
        "bag_path": str(bag_path),
        "config_path": None,
        "data_config_path": str(data_config_path),
        "pretrained_model_dir": str(pretrained_model_dir or pretrained_path),
        "mode": resolved_mode,
        "async_inference": bool(cfg.async_inference),
        "rtc_lite_enabled": bool(getattr(cfg, "rtc_lite_enabled", False)),
        "rtc_full_enabled": bool(getattr(cfg, "rtc_full_enabled", False)),
        "num_steps": int(pred_arr.shape[0]),
        "action_dim": int(pred_arr.shape[1]),
        "mse": float(np.mean(err_arr**2)),
        "mae": float(np.mean(np.abs(err_arr))),
        "rmse": float(math.sqrt(np.mean(err_arr**2))),
        "max_abs_error": float(np.max(np.abs(err_arr))),
        "mean_infer_ms": float(np.mean(infer_times_ms)),
        "p95_infer_ms": float(np.percentile(infer_times_ms, 95)),
        "per_step_mse_mean": float(np.mean(mse_per_step)),
        "per_step_mse_max": float(np.max(mse_per_step)),
        "per_step_mae_mean": float(np.mean(mae_per_step)),
    }

    np.savez_compressed(
        run_dir / "offline_predictions.npz",
        pred_action=pred_arr,
        bag_action=target_arr,
        error=err_arr,
        infer_times_ms=np.asarray(infer_times_ms, dtype=np.float64),
    )
    _write_csv(run_dir / "offline_predictions.csv", pred_arr, target_arr, err_arr)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if events:
        with (run_dir / "async_events.jsonl").open("w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    if plot:
        _plot_actions(run_dir / "action_compare.png", pred_arr, target_arr, plot_dims)

    log_model.info(
        "Offline bag eval complete: mode=%s steps=%d action_dim=%d mse=%.6f mae=%.6f rmse=%.6f "
        "mean_infer=%.2fms",
        metrics["mode"],
        metrics["num_steps"],
        metrics["action_dim"],
        metrics["mse"],
        metrics["mae"],
        metrics["rmse"],
        metrics["mean_infer_ms"],
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kuavo policy inference offline on one rosbag.")
    parser.add_argument("--config", type=Path, default=Path("configs/deploy/deploy.yaml"))
    parser.add_argument("--bag", type=Path, default=None, help="Rosbag path. Defaults to inference.go_bag_path.")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/KuavoRosbag2Lerobot.yaml"),
        help="Rosbag conversion config used for topic alignment and frame construction.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["auto", "sync", "async"],
        default="auto",
        help="auto follows inference.async_inference; sync uses select_action; async simulates chunk buffer flow.",
    )
    parser.add_argument("--plot", action="store_true", help="Save action comparison plot.")
    parser.add_argument("--plot-dims", type=str, default=None, help="Comma-separated action dims to plot.")
    parser.add_argument("--log-step-times", action="store_true", help="Print per-step select_action/buffer timing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_kuavo_config(str(args.config))
    bag_path = args.bag or Path(config.inference.go_bag_path or "")
    if not str(bag_path):
        raise ValueError("No bag path provided. Use --bag or set inference.go_bag_path in deploy config.")
    bag_path = bag_path.expanduser().resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag does not exist: {bag_path}")

    run_dir = run_offline_bag_eval(
        config=config,
        bag_path=bag_path,
        data_config_path=args.data_config.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        max_frames=args.max_frames,
        chunk_size=args.chunk_size,
        mode=args.mode,
        plot=args.plot,
        plot_dims=_parse_dims(args.plot_dims),
        log_step_times=args.log_step_times,
    )
    print(f"Offline eval saved to: {run_dir}")


if __name__ == "__main__":
    main()
