#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import random
import sys
import time
from contextlib import ExitStack
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
    async_inference: bool = False,
):
    if device.type == "cpu":
        log_model.warning("Using CPU for offline inference; this may be slow.")

    if policy_type == "client":
        preprocessor, postprocessor = lambda obs: obs, lambda action: action
        if async_inference:
            # Construct the unchanged client in the inference worker.
            return None, preprocessor, postprocessor, None
        from kuavo_deploy.kuavo_service.client import PolicyClient

        return PolicyClient(task_prompt=task_prompt), preprocessor, postprocessor, None

    from kuavo_deploy.utils.policy_loader import load_native_policy_bundle

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
    data_cfg.dataset["5w_wholebody"] = env.use_5w_wholebody
    data_cfg.dataset["5w_base_move"] = env.use_5w_base_move
    data_cfg.dataset.eef_type = env.eef_type
    data_cfg.dataset.which_arm = env.which_arm
    data_cfg.dataset.depth_range = list(env.depth_range)
    data_cfg.dataset.dex_dof_needed = (
        env.sg100_dof_needed if env.eef_type == "sg100" else env.qiangnao_dof_needed
    )
    data_cfg.dataset.dex_dof_offset = env.sg100_dof_offset if env.eef_type == "sg100" else 0
    data_cfg.dataset.is_binary = env.is_binary
    if env.eef_type == "sg100":
        # Offline parsing must use the same calibration as live observations.
        data_cfg.dataset.sg100_binary_open = list(env.sg100_open_pose)
        data_cfg.dataset.sg100_binary_close = list(env.sg100_close_pose)
        data_cfg.dataset.sg100_binary_ratio = env.sg100_binary_ratio
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
    from kuavo_data.common.config_platform import get_arm_joint_slice, get_lower_body_joint_slice

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
        arm_action: np.ndarray,
        claw_state: np.ndarray,
        claw_action: np.ndarray,
        qiangnao_state: np.ndarray,
        qiangnao_action: np.ndarray,
        sg100_state: np.ndarray,
        sg100_action: np.ndarray,
        hand_side: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        robot_slice = kuavo.SLICE_ROBOT[hand_side]
        arm_state = state[robot_slice[0] : robot_slice[-1]]
        arm_dof_per_side = arm_action.size // 2
        arm_start = hand_side * arm_dof_per_side
        selected_arm_action = arm_action[arm_start : arm_start + arm_dof_per_side]

        if kuavo.USE_LEJU_CLAW:
            claw_slice = kuavo.SLICE_CLAW[hand_side]
            eef_state = claw_state[claw_slice[0] : claw_slice[-1]]
            eef_action = claw_action[claw_slice[0] : claw_slice[-1]]
        elif kuavo.USE_QIANGNAO:
            dex_slice = kuavo.SLICE_DEX[hand_side]
            eef_state = qiangnao_state[dex_slice[0] : dex_slice[-1]]
            eef_action = qiangnao_action[dex_slice[0] : dex_slice[-1]]
        elif kuavo.USE_SG100:
            dex_slice = kuavo.SLICE_DEX[hand_side]
            eef_state = sg100_state[dex_slice[0] : dex_slice[-1]]
            eef_action = sg100_action[dex_slice[0] : dex_slice[-1]]
        else:
            raise ValueError("Only leju_claw, rq2f85, qiangnao and sg100 end effectors are supported.")

        return (
            np.concatenate((arm_state, eef_state)).astype(np.float32),
            np.concatenate((selected_arm_action, eef_action)).astype(np.float32),
        )

    def on_frame(aligned_frame: dict[str, Any], frame_idx: int) -> None:
        nonlocal first_raw_state, first_raw_action

        if max_frames is not None and len(frames) >= max_frames:
            return

        state = _array(aligned_frame, "observation.state")
        arm_traj = _array(aligned_frame, "action.kuavo_arm_traj")
        arm_dof = arm_end - arm_start
        if state.size == 0 or arm_traj.size != arm_dof:
            return

        if first_raw_state is None:
            first_raw_state = state.copy()
        if first_raw_action is None:
            first_raw_action = arm_traj.copy()

        if kuavo.RELATIVE_START:
            state = state - first_raw_state
            arm_traj = arm_traj - first_raw_action

        claw_state = _array(aligned_frame, "observation.claw")
        claw_action = _array(aligned_frame, "action.claw")
        qiangnao_state = _array(aligned_frame, "observation.qiangnao")
        qiangnao_action = _array(aligned_frame, "action.qiangnao")
        sg100_state = _array(aligned_frame, "observation.sg100")
        sg100_action = _array(aligned_frame, "action.sg100")
        rq2f85_state = _array(aligned_frame, "observation.rq2f85")
        rq2f85_action = _array(aligned_frame, "action.rq2f85")

        if claw_state.size == 0 and qiangnao_state.size == 0 and sg100_state.size == 0 and rq2f85_state.size == 0:
            return
        if claw_action.size == 0 and qiangnao_action.size == 0 and sg100_action.size == 0 and rq2f85_action.size == 0:
            return

        claw_state = _normalize_binary_or_range(claw_state, binary_threshold=50, scale=100)
        claw_action = _normalize_binary_or_range(claw_action, binary_threshold=50, scale=100)
        qiangnao_state = _normalize_binary_or_range(qiangnao_state, binary_threshold=50, scale=100)
        qiangnao_action = _normalize_binary_or_range(qiangnao_action, binary_threshold=50, scale=100)
        if kuavo.USE_SG100 and kuavo.IS_BINARY:
            from kuavo_data.common.sg100 import sg100_binarize

            def binary_sg100(values):
                return sg100_binarize(
                    values, kuavo.SG100_BINARY_OPEN, kuavo.SG100_BINARY_CLOSE,
                    kuavo.SG100_BINARY_JOINTS, kuavo.SG100_BINARY_RATIO,
                ).astype(np.float32)

            sg100_state, sg100_action = binary_sg100(sg100_state), binary_sg100(sg100_action)
        # Non-binary SG100 features are physical joint angles, as in conversion/live feedback.
        rq2f85_state = _normalize_binary_or_range(rq2f85_state, binary_threshold=0.4, scale=0.8)
        rq2f85_action = _normalize_binary_or_range(rq2f85_action, binary_threshold=70, scale=255)

        if claw_state.size == 0 and qiangnao_state.size == 0 and sg100_state.size == 0:
            claw_state = rq2f85_state
            claw_action = rq2f85_action

        state_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        if kuavo.CONTROL_HAND_SIDE in ("left", "both"):
            s, a = _hand_pair(
                state=state,
                arm_action=arm_traj,
                claw_state=claw_state,
                claw_action=claw_action,
                qiangnao_state=qiangnao_state,
                qiangnao_action=qiangnao_action,
                sg100_state=sg100_state,
                sg100_action=sg100_action,
                hand_side=0,
            )
            state_parts.append(s)
            action_parts.append(a)
        if kuavo.CONTROL_HAND_SIDE in ("right", "both"):
            s, a = _hand_pair(
                state=state,
                arm_action=arm_traj,
                claw_state=claw_state,
                claw_action=claw_action,
                qiangnao_state=qiangnao_state,
                qiangnao_action=qiangnao_action,
                sg100_state=sg100_state,
                sg100_action=sg100_action,
                hand_side=1,
            )
            state_parts.append(s)
            action_parts.append(a)

        if kuavo.USE_5W_WHOLEBODY:
            lower_start, lower_end = get_lower_body_joint_slice(kuavo.PLATFORM_TYPE)
            lower_action = _array(aligned_frame, "action.lb_leg_traj")
            lower_state = state[lower_start:lower_end]
            lower_dof = lower_end - lower_start
            if lower_state.size != lower_dof or lower_action.size != lower_dof:
                return
            state_parts.append(lower_state)
            action_parts.append(lower_action)

        if kuavo.USE_5W_BASE_MOVE:
            base_velocity = _array(aligned_frame, "action.base_velocity")
            if base_velocity.size != 3:
                return
            action_parts.append(base_velocity)

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

    with ExitStack() as stack:
        frame_callback = on_frame
        if config.env.camera_encoding == "h265":
            import rosbag
            from kuavo_deploy.utils.h265_decoder import RealtimeH265Decoder

            # Use the same topics and decoder as live camera observations.
            topic_to_camera = {}
            decoders = {}
            for cam in kuavo.DEFAULT_CAMERA_NAMES:
                topic = config.env._H265_TOPIC_MAP.get(cam)
                if topic is not None:
                    bag_reader._topic_process_map[cam]["topic"] = topic
                    bag_reader._h265_cameras.add(cam)
                    topic_to_camera[topic] = cam
                    decoders[cam] = RealtimeH265Decoder(kuavo.RESIZE_W, kuavo.RESIZE_H)

            # Read continuously from the start so chunking and sample_drop do
            # not skip reference frames or decode overlapping ranges twice.
            camera_bag = stack.enter_context(rosbag.Bag(str(bag_path)))
            messages = iter(camera_bag.read_messages(topics=list(topic_to_camera)))
            pending = next(messages, None)
            latest_images = {}

            def on_h265_frame(aligned_frame: dict[str, Any], frame_idx: int) -> None:
                nonlocal pending
                if max_frames is not None and len(frames) >= max_frames:
                    return
                while pending is not None and pending[2].to_sec() <= aligned_frame["timestamp"]:
                    topic, msg, stamp = pending
                    cam = topic_to_camera[topic]
                    rgb = decoders[cam].decode(bytes(msg.data))
                    if rgb is not None:
                        latest_images[cam] = {"data": rgb, "timestamp": stamp.to_sec()}
                    pending = next(messages, None)
                aligned_frame.update(latest_images)
                on_frame(aligned_frame, frame_idx)

            frame_callback = on_h265_frame

        bag_reader.process_rosbag_chunked(
            bag_file=str(bag_path),
            frame_callback=frame_callback,
            chunk_size=chunk_size,
            save_callback=lambda: None,
        )
    return frames


def _frame_to_policy_observation(
    frame: dict[str, Any],
    device,
    *,
    native_policy: bool,
) -> dict[str, Any]:
    import torch

    obs: dict[str, Any] = {}
    for key, value in frame.items():
        if key in {"action", "frame_index", "timestamp"}:
            continue
        if key == "observation.state":
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
            obs[key] = tensor.float().unsqueeze(0).to(device, non_blocking=True)
        elif not native_policy:
            # Kuavo Server adapters own their model-specific image conversion and
            # expect the transport representation (normally HWC uint8).
            obs[key] = value
        else:
            # Native LeRobot policies share the dataset image convention:
            # BCHW float32 in [0, 1]. Policy-specific preprocessing is then
            # handled by the processor pipeline stored with the checkpoint.
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float().div(255.0)
            else:
                tensor = tensor.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            obs[key] = tensor.contiguous().to(device, non_blocking=True)
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
        dims = list(range(action_dim))
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
    observation = _frame_to_policy_observation(
        frame,
        device=device,
        native_policy=policy_type != "client",
    )
    if policy_type != "client":
        from kuavo_deploy.utils.policy_loader import inject_task_prompt

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
) -> tuple[np.ndarray, np.ndarray, list[float]]:
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

    return np.stack(predictions, axis=0), np.stack(targets, axis=0), infer_times_ms


def _replay_offsets(frames, control_hz, replay_speed):
    """Use bag frame spacing, falling back to the configured action frequency."""
    if not math.isfinite(replay_speed) or replay_speed <= 0:
        raise ValueError("replay_speed must be finite and positive")
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("env.ros_rate must be finite and positive for offline async inference")
    try:
        timestamps = np.asarray([float(frame["timestamp"]) for frame in frames], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        timestamps = np.full(len(frames), np.nan)
    if np.all(np.isfinite(timestamps)) and np.all(np.diff(timestamps) > 0):
        return (timestamps - timestamps[0]) / replay_speed, "bag_timestamps"
    log_model.warning("Bag timestamps missing or invalid; pacing offline async at env.ros_rate")
    return np.arange(len(frames), dtype=np.float64) / (control_hz * replay_speed), "env.ros_rate"


def _run_async_offline(
    *, frames, total, policy, preprocessor, postprocessor, device, config,
    log_step_times, replay_speed=1.0,
):
    from kuavo_deploy.utils.async_inference import AsyncInference

    selected = frames[:total]
    offsets, pacing_source = _replay_offsets(selected, config.env.ros_rate, replay_speed)
    # A faster replay consumes more model actions per wall-clock second. RTC's
    # latency estimate must use that cadence rather than the nominal ROS rate.
    effective_hz = 1.0 / float(np.median(np.diff(offsets))) if len(offsets) > 1 else config.env.ros_rate * replay_speed
    predictions, targets, wait_times_ms, timeline = [], [], [], []
    next_deadline = None
    first_action_time = None
    wall_start = time.monotonic()
    with AsyncInference(policy, preprocessor, postprocessor, config.inference, record_chunks=True, control_hz=effective_hz) as runner:
        for step, frame in enumerate(tqdm(selected, desc="Offline async inference", unit="step")):
            if next_deadline is not None:
                remaining = next_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            observation = _prepare_observation_for_policy(
                frame=frame, device=device, policy_type=config.inference.policy_type,
                task_prompt=config.inference.task_prompt,
            )
            start = time.perf_counter()
            action = runner.select_action(observation)
            wait_ms = (time.perf_counter() - start) * 1000.0
            action_time = time.monotonic()
            if first_action_time is None:
                first_action_time = action_time
            pred, target = _action_to_numpy(action), _action_to_numpy(frame["action"])
            if pred.shape != target.shape:
                raise ValueError(
                    f"Predicted action shape {pred.shape} does not match bag action shape {target.shape}. "
                    "Check deploy env.which_arm/eef_type and data conversion config."
                )
            predictions.append(pred)
            targets.append(target)
            wait_times_ms.append(wait_ms)
            elapsed = action_time - first_action_time
            row = {
                "step": step,
                "frame_index": int(frame.get("frame_index", step)),
                "replay_time_s": float(offsets[step]),
                "elapsed_s": elapsed,
                "drift_ms": (elapsed - float(offsets[step])) * 1000.0,
                "select_action_ms": wait_ms,
                "queue_after_pop": runner.queue_size,
            }
            timeline.append(row)
            if log_step_times:
                print(
                    f"[offline-async] step={step} select_action_ms={wait_ms:.3f} "
                    f"queue={row['queue_after_pop']} drift_ms={row['drift_ms']:.3f}", flush=True,
                )
            if step + 1 < len(selected):
                # Hold each consumed action for one frame interval. If inference
                # stalls, delay the replay instead of skipping frames or catching
                # up with a burst of actions; this matches the logical step queue.
                next_deadline = action_time + float(offsets[step + 1] - offsets[step])
    return np.stack(predictions), np.stack(targets), wait_times_ms, {
        "timeline": timeline,
        "chunks": runner.chunk_history,
        "pacing_source": pacing_source,
        "effective_control_hz": effective_hz,
        "wall_duration_s": time.monotonic() - wall_start,
    }


def run_offline_bag_eval(
    *,
    config: KuavoConfig,
    bag_path: Path,
    data_config_path: Path,
    output_dir: Path | None,
    max_frames: int | None,
    chunk_size: int,
    plot: bool,
    plot_dims: list[int] | None,
    log_step_times: bool,
    replay_speed: float = 1.0,
) -> Path:
    import torch

    cfg = config.inference
    cfg.validate()
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if cfg.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if not math.isfinite(replay_speed) or replay_speed <= 0:
        raise ValueError("replay_speed must be finite and positive")
    pretrained_path = _resolve_pretrained_path(cfg)
    if cfg.policy_type == "client":
        eval_root = output_dir or Path("outputs/offline_eval/client")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    else:
        from lerobot.utils.random_utils import set_seed
        from lerobot_patches import custom_patches  # noqa: F401
        from kuavo_deploy.utils.policy_loader import resolve_eval_output_dir

        eval_root = output_dir or resolve_eval_output_dir(pretrained_path, Path("outputs/offline_eval"))
        set_seed(seed=cfg.seed)
    run_dir = eval_root / f"{bag_path.stem}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    task_prompt = getattr(cfg, "task_prompt", "robot manipulation")
    policy, preprocessor, postprocessor, pretrained_model_dir = _setup_policy(
        pretrained_path=pretrained_path,
        policy_type=cfg.policy_type,
        device=device,
        task_prompt=task_prompt,
        async_inference=cfg.async_inference,
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

    total = min(len(frames), cfg.max_episode_steps)
    if max_frames is not None:
        total = min(total, max_frames)

    replay = None
    if cfg.async_inference:
        pred_arr, target_arr, infer_times_ms, replay = _run_async_offline(
            frames=frames, total=total, policy=policy, preprocessor=preprocessor,
            postprocessor=postprocessor, device=device, config=config,
            log_step_times=log_step_times, replay_speed=replay_speed,
        )
    else:
        policy.reset()
        pred_arr, target_arr, infer_times_ms = _run_sync_offline(
            frames=frames, total=total, policy=policy, preprocessor=preprocessor,
            postprocessor=postprocessor, device=device, policy_type=cfg.policy_type,
            task_prompt=task_prompt, log_step_times=log_step_times,
        )

    err_arr = pred_arr - target_arr

    mse_per_step = np.mean(err_arr**2, axis=1)
    mae_per_step = np.mean(np.abs(err_arr), axis=1)
    metrics = {
        "bag_path": str(bag_path),
        "config_path": None,
        "data_config_path": str(data_config_path),
        "pretrained_model_dir": str(pretrained_model_dir or pretrained_path),
        "mode": "async" if cfg.async_inference else "sync",
        "rtc_enabled": cfg.rtc_enabled,
        "rtc_mode": cfg.rtc_mode if cfg.rtc_enabled else None,
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

    extra_arrays = {}
    if replay is not None:
        chunks = replay["chunks"]
        chunk_times = np.asarray([chunk["infer_ms"] for chunk in chunks], dtype=np.float64)
        metrics.update({
            "aggregate_fn_name": None if cfg.rtc_enabled else cfg.aggregate_fn_name,
            "actions_per_chunk": cfg.actions_per_chunk,
            "chunk_size_threshold": cfg.chunk_size_threshold,
            "replay_speed": replay_speed,
            "pacing_source": replay["pacing_source"],
            "effective_control_hz": replay["effective_control_hz"],
            "wall_duration_s": replay["wall_duration_s"],
            "num_chunks": len(chunks),
            "dropped_actions": sum(chunk["dropped"] for chunk in chunks),
            "overlap_actions": sum(chunk["overlap_steps"] for chunk in chunks),
            "mean_chunk_infer_ms": float(np.mean(chunk_times)),
            "p95_chunk_infer_ms": float(np.percentile(chunk_times, 95)),
            "initial_wait_ms": infer_times_ms[0],
            "mean_steady_state_wait_ms": float(np.mean(infer_times_ms[1:])) if total > 1 else 0.0,
            "final_drift_ms": replay["timeline"][-1]["drift_ms"],
            "infer_times_semantics": "main-thread select_action wait; chunk inference reported separately",
        })
        with (run_dir / "async_timeline.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(replay["timeline"][0]))
            writer.writeheader()
            writer.writerows(replay["timeline"])
        with (run_dir / "async_chunks.json").open("w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        extra_arrays = {
            "chunk_infer_times_ms": chunk_times,
            "elapsed_s": np.asarray([row["elapsed_s"] for row in replay["timeline"]]),
            "replay_time_s": np.asarray([row["replay_time_s"] for row in replay["timeline"]]),
            "queue_sizes": np.asarray([row["queue_after_pop"] for row in replay["timeline"]]),
        }

    np.savez_compressed(
        run_dir / "offline_predictions.npz",
        pred_action=pred_arr,
        bag_action=target_arr,
        error=err_arr,
        infer_times_ms=np.asarray(infer_times_ms, dtype=np.float64),
        **extra_arrays,
    )
    _write_csv(run_dir / "offline_predictions.csv", pred_arr, target_arr, err_arr)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
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
    parser.add_argument("--chunk-size", type=int, default=100, help="Bag parsing batch size; unrelated to action chunk length.")
    parser.add_argument("--mode", choices=("sync", "async"), default=None, help="Override inference.async_inference for this offline run.")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Async replay speed multiplier; 1 follows bag frame intervals. Sync mode remains unpaced.")
    parser.add_argument("--plot", action="store_true", help="Save action comparison plot.")
    parser.add_argument("--plot-dims", type=str, default=None, help="Comma-separated action dims to plot.")
    parser.add_argument("--log-step-times", action="store_true", help="Print per-step select_action timing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_kuavo_config(str(args.config))
    if args.mode is not None:
        config.inference.async_inference = args.mode == "async"
    configured_bag = config.inference.go_bag_path
    if args.bag is None and not configured_bag:
        raise ValueError("No bag path provided. Use --bag or set inference.go_bag_path in deploy config.")
    bag_path = args.bag or Path(configured_bag)
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
        plot=args.plot,
        plot_dims=_parse_dims(args.plot_dims),
        log_step_times=args.log_step_times,
        replay_speed=args.replay_speed,
    )
    print(f"Offline eval saved to: {run_dir}")


if __name__ == "__main__":
    main()
