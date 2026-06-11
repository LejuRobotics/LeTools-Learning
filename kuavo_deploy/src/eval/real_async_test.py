# Copyright (C) 2025-2026 LejuRobotics.

from __future__ import annotations

import datetime
import math
import time
import traceback
from collections import deque
from pathlib import Path
from threading import Condition, Event, Thread
from typing import Any

import numpy as np
import rospy
import torch
from std_msgs.msg import Bool
from tqdm import tqdm

from lerobot.utils.random_utils import set_seed
from lerobot_patches import custom_patches  # noqa: F401

from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.kuavo_service.client import PolicyClient
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.utils.policy_loader import (
    inject_task_prompt,
    load_native_policy_bundle,
    resolve_eval_output_dir,
)

log_model = setup_logger("model")
log_robot = setup_logger("robot")

pause_flag = Event()
stop_flag = Event()


def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()


def stop_callback(msg):
    if msg.data:
        stop_flag.set()


pause_sub = rospy.Subscriber("/kuavo/pause_state", Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber("/kuavo/stop_state", Bool, stop_callback, queue_size=10)


def _ramp_weights(blend_count: int, ramp: str) -> np.ndarray:
    """Return blend weights w_i for i in [0, blend_count).

    w=0 means keep the old action; w=1 means take the new action. blend_count=1
    short-circuits to a single w=1 (immediate handover) per the spec.
    """
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


class ActionChunkBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = max(1, int(maxlen))
        self._actions: deque[np.ndarray] = deque()
        self._cond = Condition()
        self._total_popped: int = 0

    def clear(self) -> None:
        with self._cond:
            self._actions.clear()
            self._cond.notify_all()

    def qsize(self) -> int:
        with self._cond:
            return len(self._actions)

    def get_action_index(self) -> int:
        """Total number of actions that have actually been popped via get()."""
        with self._cond:
            return self._total_popped

    def snapshot_actions(self) -> list[np.ndarray]:
        """Read a copy of the currently queued actions without popping anything."""
        with self._cond:
            return [np.asarray(a).copy() for a in self._actions]

    def put_chunk(self, actions: list[np.ndarray]) -> int:
        with self._cond:
            free = self.maxlen - len(self._actions)
            for action in actions[:free]:
                self._actions.append(action)
            self._cond.notify_all()
            return min(len(actions), max(0, free))

    def merge_chunk(
        self,
        actions: list[np.ndarray],
        *,
        delay_steps: int,
        overlap_steps: int,
        freeze_steps: int,
        keep_min_actions: int,
        ramp: str,
        maxlen: int | None = None,
    ) -> dict[str, Any]:
        """Replace the queue with kept-prefix + blended-overlap + tail-of-new.

        Returns a stats dict describing what happened (sizes pre/post, delay,
        blend lengths, dropped new prefix length). When delay consumes the whole
        new chunk the queue is left untouched.
        """
        cap = self.maxlen if maxlen is None else max(1, int(maxlen))
        overlap_steps = max(0, int(overlap_steps))
        freeze_steps = max(0, int(freeze_steps))
        keep_min_actions = max(0, int(keep_min_actions))
        delay_steps = max(0, int(delay_steps))

        new_len = len(actions)
        delay = min(delay_steps, new_len)
        new_valid = list(actions[delay:])

        with self._cond:
            old = list(self._actions)
            old_size = len(old)

            if not new_valid:
                stats = {
                    "mode": "blend_replace",
                    "old_size": old_size,
                    "produced": new_len,
                    "delay_steps": delay,
                    "kept": old_size,
                    "blended": 0,
                    "queued": old_size,
                    "dropped_new_prefix": delay,
                }
                # Nothing to merge in; keep existing queue unchanged.
                return stats

            old_keep_count = min(old_size, max(freeze_steps, keep_min_actions))
            kept = old[:old_keep_count]
            old_blend = old[old_keep_count : old_keep_count + overlap_steps]
            new_blend = new_valid[:overlap_steps]
            blend_count = min(len(old_blend), len(new_blend))

            blended: list[np.ndarray] = []
            if blend_count > 0:
                weights = _ramp_weights(blend_count, ramp)
                for i in range(blend_count):
                    w = float(weights[i])
                    a_old = np.asarray(old_blend[i], dtype=np.float64)
                    a_new = np.asarray(new_blend[i], dtype=np.float64)
                    blended_action = (1.0 - w) * a_old + w * a_new
                    # Preserve dtype of the new action if possible.
                    target_dtype = np.asarray(new_blend[i]).dtype
                    blended.append(blended_action.astype(target_dtype, copy=False))

            tail = new_valid[blend_count:]
            merged = kept + blended + tail
            merged = merged[:cap]

            self._actions.clear()
            for action in merged:
                self._actions.append(action)
            self._cond.notify_all()

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

    def get(self, timeout: float) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._actions:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            action = self._actions.popleft()
            self._total_popped += 1
            return action

    def wait_for_size(self, size: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self._actions) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            return True


class RtcActionQueue(ActionChunkBuffer):
    """Synchronized executable and model-space queues for full RTC."""

    def __init__(self, maxlen: int):
        super().__init__(maxlen)
        self._original_actions: deque[np.ndarray] = deque()

    def clear(self) -> None:
        with self._cond:
            self._actions.clear()
            self._original_actions.clear()
            self._cond.notify_all()

    def snapshot_original_actions(self) -> np.ndarray | None:
        with self._cond:
            if not self._original_actions:
                return None
            return np.stack([np.asarray(action).copy() for action in self._original_actions], axis=0)

    def replace_after_delay(
        self,
        *,
        processed_actions: Any,
        original_actions: Any,
        delay_steps: int,
    ) -> dict[str, int | str]:
        processed = _to_chunk_tensor(processed_actions).detach().cpu().numpy()
        original = _to_chunk_tensor(original_actions).detach().cpu().numpy()
        if processed.shape[0] != original.shape[0]:
            raise ValueError(
                "RTC full requires aligned processed/original chunks, got "
                f"{processed.shape[0]} and {original.shape[0]} steps"
            )
        delay = min(max(0, int(delay_steps)), processed.shape[0])
        produced = int(processed.shape[0])
        processed = processed[delay:]
        original = original[delay:]
        if processed.shape[0] > self.maxlen:
            raise ValueError(
                "RTC Full must preserve the model's remaining native chunk; "
                f"async_buffer_size={self.maxlen} is smaller than remaining chunk={processed.shape[0]}"
            )
        with self._cond:
            old_size = len(self._actions)
            self._actions.clear()
            self._original_actions.clear()
            self._actions.extend(np.asarray(step) for step in processed)
            self._original_actions.extend(np.asarray(step) for step in original)
            self._cond.notify_all()
            return {
                "mode": "rtc_full_replace",
                "old_size": old_size,
                "produced": produced,
                "delay_steps": delay,
                "queued": len(self._actions),
            }

    def get(self, timeout: float) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._actions:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            action = self._actions.popleft()
            if not self._original_actions:
                raise RuntimeError("RTC full queues are no longer aligned")
            self._original_actions.popleft()
            self._total_popped += 1
            return action


def setup_policy(pretrained_path, policy_type, device, task_prompt: str):
    if device.type == "cpu":
        log_model.warning("Using CPU for inference, this may be slow.")
        time.sleep(3)

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
    log_model.info(f"Model n_obs_steps: {policy.config.n_obs_steps}")
    return policy, preprocessor, postprocessor, pretrained_model_dir


def _to_chunk_tensor(actions: Any) -> torch.Tensor:
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


def _postprocess_chunk(actions: Any, postprocessor) -> list[np.ndarray]:
    chunk = _to_chunk_tensor(actions)
    out: list[np.ndarray] = []
    for idx in range(chunk.shape[0]):
        action = chunk[idx : idx + 1]
        processed = postprocessor(action)
        processed = _to_chunk_tensor(processed)
        out.append(processed[0].detach().cpu().numpy())
    return out


def _select_action_chunk(policy, observation: dict[str, Any]) -> Any:
    if hasattr(policy, "select_action_chunk"):
        return policy.select_action_chunk(observation)
    if hasattr(policy, "predict_action_chunk"):
        return policy.predict_action_chunk(observation)
    return policy.select_action(observation)


def _select_action_chunk_rtc(
    policy,
    observation: dict[str, Any],
    *,
    prev_chunk_leftover: np.ndarray | None,
    inference_delay: int,
    execution_horizon: int,
    rtc_options: dict[str, Any],
) -> dict[str, Any]:
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
    return result


def _rtc_full_options(cfg) -> dict[str, Any]:
    return {
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


def _resolve_control_hz(cfg, env) -> float | None:
    """Resolve a control-rate (Hz) for delay estimation.

    Order per plan: async_control_hz > env.control_rate > env rate. Returns
    None if nothing usable is available (the caller will fall back to
    actual_consumed and log a warning).
    """
    async_hz = getattr(cfg, "async_control_hz", 0.0) or 0.0
    if async_hz and async_hz > 0:
        return float(async_hz)
    deploy_rate = getattr(env, "control_rate_hz", 0) or getattr(env, "control_rate", 0) or 0
    if deploy_rate and deploy_rate > 0:
        return float(deploy_rate)
    rate = getattr(env, "rate", None)
    if rate is not None:
        sleep_dur = getattr(rate, "sleep_dur", None)
        to_sec = getattr(sleep_dur, "to_sec", None)
        try:
            seconds = to_sec() if callable(to_sec) else None
        except Exception:
            seconds = None
        if seconds and seconds > 0:
            return 1.0 / seconds
    return None


def _action_norm(action: np.ndarray | None) -> float | None:
    if action is None:
        return None
    try:
        return float(np.linalg.norm(np.asarray(action, dtype=np.float64)))
    except Exception:
        return None


def _delta_norm(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    try:
        return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))
    except Exception:
        return None


def inference_worker(
    *,
    config: KuavoConfig,
    env,
    policy,
    preprocessor,
    postprocessor,
    buffer: ActionChunkBuffer,
    stop_event: Event,
) -> None:
    cfg = config.inference
    policy_type = cfg.policy_type
    task_prompt = getattr(cfg, "task_prompt", "robot manipulation")
    low_watermark = max(0, int(cfg.async_low_watermark))

    rtc_lite_enabled = bool(getattr(cfg, "rtc_lite_enabled", False))
    rtc_full_enabled = bool(getattr(cfg, "rtc_full_enabled", False))
    if rtc_lite_enabled and not cfg.async_inference:
        log_model.warning(
            "rtc_lite_enabled=true but async_inference=false; RTC-Lite will not take effect."
        )
        rtc_lite_enabled = False

    control_hz: float | None = None
    if rtc_full_enabled and not cfg.async_inference:
        log_model.warning(
            "rtc_full_enabled=true but async_inference=false; Full RTC will not take effect."
        )
        rtc_full_enabled = False

    if rtc_lite_enabled or rtc_full_enabled:
        control_hz = _resolve_control_hz(cfg, env)
        if control_hz is None:
            log_model.warning(
                "RTC: no control rate available (async_control_hz / env.control_rate / env.rate "
                "all unset); falling back to actual_consumed for delay estimation."
            )

    log_deltas = bool(getattr(cfg, "rtc_lite_log_deltas", True))
    estimated_delay_steps = 0
    rtc_options = _rtc_full_options(cfg) if rtc_full_enabled else {}
    rtc_full_warmup = True
    if rtc_full_enabled and not isinstance(buffer, RtcActionQueue):
        raise RuntimeError("Full RTC requires RtcActionQueue")

    try:
        while not stop_event.is_set() and not rospy.is_shutdown():
            if stop_flag.is_set():
                stop_event.set()
                break
            if pause_flag.is_set():
                time.sleep(0.05)
                continue
            if buffer.qsize() > low_watermark:
                time.sleep(0.005)
                continue

            action_index_before = buffer.get_action_index()
            mono_start = time.monotonic()
            start = time.time()
            observation = env.get_obs()
            if policy_type != "client":
                observation = inject_task_prompt(observation, task_prompt)
            observation = preprocessor(observation)

            if rtc_full_enabled:
                result = _select_action_chunk_rtc(
                    policy,
                    observation,
                    prev_chunk_leftover=buffer.snapshot_original_actions(),
                    inference_delay=estimated_delay_steps,
                    execution_horizon=int(cfg.rtc_full_overlap_steps),
                    rtc_options=rtc_options,
                )
                actions_np = _to_chunk_tensor(result["processed_actions"]).detach().cpu().numpy()
                original_actions = result["original_actions"]
            else:
                with torch.inference_mode():
                    actions = _select_action_chunk(policy, observation)
                actions_np = _postprocess_chunk(actions, postprocessor)
            elapsed = time.time() - start

            if rtc_full_enabled:
                mono_elapsed = time.monotonic() - mono_start
                actual_consumed = max(0, buffer.get_action_index() - action_index_before)
                measured_delay = math.ceil(mono_elapsed * control_hz) if control_hz else actual_consumed
                delay_steps = (
                    0
                    if rtc_full_warmup
                    else min(
                        max(actual_consumed, measured_delay),
                        int(cfg.rtc_full_max_delay_steps),
                    )
                )
                stats = buffer.replace_after_delay(
                    processed_actions=actions_np,
                    original_actions=original_actions,
                    delay_steps=delay_steps,
                )
                if cfg.rtc_full_debug:
                    log_model.info(
                        "RTC Full chunk: produced={produced}, old={old_size}, "
                        "estimated_delay={estimated}, actual_delay={delay_steps}, queued={queued}, "
                        "infer={elapsed:.3f}s, consumed_during_infer={consumed}".format(
                            produced=stats["produced"],
                            old_size=stats["old_size"],
                            estimated=estimated_delay_steps,
                            delay_steps=stats["delay_steps"],
                            queued=stats["queued"],
                            elapsed=elapsed,
                            consumed=actual_consumed,
                        )
                    )
                estimated_delay_steps = delay_steps
                rtc_full_warmup = False
            elif rtc_lite_enabled:
                mono_elapsed = time.monotonic() - mono_start
                action_index_after = buffer.get_action_index()
                actual_consumed = max(0, action_index_after - action_index_before)
                if control_hz is not None and control_hz > 0:
                    measured_delay = math.ceil(mono_elapsed * control_hz)
                else:
                    measured_delay = actual_consumed
                delay_steps = max(actual_consumed, measured_delay)
                delay_steps = min(delay_steps, int(cfg.rtc_lite_max_delay_steps))

                # Optional boundary-delta logging: snapshot pre-state cheaply.
                pre_first = None
                if log_deltas:
                    pre_snapshot = buffer.snapshot_actions()
                    pre_first = pre_snapshot[0] if pre_snapshot else None

                stats = buffer.merge_chunk(
                    actions_np,
                    delay_steps=delay_steps,
                    overlap_steps=int(cfg.rtc_lite_overlap_steps),
                    freeze_steps=int(cfg.rtc_lite_freeze_steps),
                    keep_min_actions=int(cfg.rtc_lite_keep_min_actions),
                    ramp=str(cfg.rtc_lite_ramp),
                )
                log_model.info(
                    "Async chunk merged: produced={produced}, old={old_size}, kept={kept}, "
                    "blended={blended}, delay={delay_steps}, dropped_prefix={dropped_new_prefix}, "
                    "queued={queued}, infer={elapsed:.3f}s, consumed_during_infer={consumed}".format(
                        produced=stats["produced"],
                        old_size=stats["old_size"],
                        kept=stats["kept"],
                        blended=stats["blended"],
                        delay_steps=stats["delay_steps"],
                        dropped_new_prefix=stats["dropped_new_prefix"],
                        queued=stats["queued"],
                        elapsed=elapsed,
                        consumed=actual_consumed,
                    )
                )
                if log_deltas:
                    post_snapshot = buffer.snapshot_actions()
                    post_first = post_snapshot[0] if post_snapshot else None
                    new_first_raw = (
                        actions_np[stats["delay_steps"]]
                        if stats["delay_steps"] < len(actions_np)
                        else None
                    )
                    blend_delta_max: float | None = None
                    blend_window = post_snapshot[: stats["kept"] + stats["blended"] + 1]
                    for i in range(1, len(blend_window)):
                        d = _delta_norm(blend_window[i - 1], blend_window[i])
                        if d is None:
                            continue
                        if blend_delta_max is None or d > blend_delta_max:
                            blend_delta_max = d
                    log_model.info(
                        "RTC-Lite deltas: pre_first_norm={pre}, new_first_raw_norm={new_raw}, "
                        "post_first_norm={post}, pre_vs_post={pre_post}, "
                        "pre_vs_new_raw={pre_new}, blend_delta_max={blend_max}".format(
                            pre=_action_norm(pre_first),
                            new_raw=_action_norm(new_first_raw),
                            post=_action_norm(post_first),
                            pre_post=_delta_norm(pre_first, post_first),
                            pre_new=_delta_norm(pre_first, new_first_raw),
                            blend_max=blend_delta_max,
                        )
                    )
            else:
                inserted = buffer.put_chunk(actions_np)
                log_model.info(
                    f"Async chunk ready: produced={len(actions_np)}, inserted={inserted}, "
                    f"buffer={buffer.qsize()}, time={elapsed:.3f}s"
                )
    except Exception:
        log_model.error("Async inference worker failed:\n" + traceback.format_exc())
        stop_event.set()


def control_worker(
    *,
    env,
    buffer: ActionChunkBuffer,
    stop_event: Event,
    max_steps: int,
    action_timeout: float,
) -> int:
    step = 0
    last_action = None
    try:
        with tqdm(total=max_steps, desc="Async episode", unit="step", leave=False) as pbar:
            while step < max_steps and not stop_event.is_set() and not rospy.is_shutdown():
                while pause_flag.is_set() and not stop_flag.is_set():
                    log_model.info("Paused. Waiting for resume signal...")
                    time.sleep(0.5)
                if stop_flag.is_set():
                    stop_event.set()
                    break

                action = buffer.get(timeout=action_timeout)
                if action is None:
                    if last_action is None:
                        log_model.error("No action available before timeout; stopping async rollout.")
                        stop_event.set()
                        break
                    log_model.warning("No fresh action available; holding last action for one step.")
                    action = last_action

                env.step(action)
                last_action = action
                step += 1
                pbar.update(1)
    except Exception:
        log_robot.error("Async control worker failed:\n" + traceback.format_exc())
        stop_event.set()
    return step


def kuavo_eval_async(config: KuavoConfig, env) -> None:
    cfg = config.inference
    eval_episodes = cfg.eval_episodes
    policy_type = cfg.policy_type
    task_prompt = getattr(cfg, "task_prompt", "robot manipulation")

    pretrained_path = (
        Path(cfg.pretrained_path)
        if cfg.pretrained_path
        else Path(f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}")
    )
    output_directory = resolve_eval_output_dir(pretrained_path, Path("outputs/eval"))
    output_directory.mkdir(parents=True, exist_ok=True)

    set_seed(seed=cfg.seed)
    device = torch.device(cfg.device)
    policy, preprocessor, postprocessor, _ = setup_policy(
        pretrained_path,
        policy_type,
        device,
        task_prompt=task_prompt,
    )

    if cfg.async_control_hz and cfg.async_control_hz > 0:
        env.rate = rospy.Rate(cfg.async_control_hz)
        log_robot.info(f"Async control rate set to {cfg.async_control_hz} Hz")

    log_file_path = output_directory / "evaluation_async.log"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")
        log_file.write(f"Policy Type: {policy_type}\n")

    for episode in tqdm(range(eval_episodes), desc="Async evaluating", unit="episode"):
        if stop_flag.is_set() or rospy.is_shutdown():
            break

        policy.reset()
        env.reset(seed=episode + cfg.start_seed)

        buffer = (
            RtcActionQueue(maxlen=cfg.async_buffer_size)
            if getattr(cfg, "rtc_full_enabled", False)
            else ActionChunkBuffer(maxlen=cfg.async_buffer_size)
        )
        stop_event = Event()
        infer_thread = Thread(
            target=inference_worker,
            kwargs={
                "config": config,
                "env": env,
                "policy": policy,
                "preprocessor": preprocessor,
                "postprocessor": postprocessor,
                "buffer": buffer,
                "stop_event": stop_event,
            },
            daemon=True,
            name="kuavo-async-inference",
        )
        infer_thread.start()

        warmup_actions = max(1, int(cfg.async_warmup_actions))
        if not buffer.wait_for_size(warmup_actions, timeout=cfg.async_action_timeout):
            log_model.warning("Warmup action wait timed out; control loop will wait on the buffer.")

        steps = control_worker(
            env=env,
            buffer=buffer,
            stop_event=stop_event,
            max_steps=cfg.max_episode_steps,
            action_timeout=cfg.async_action_timeout,
        )
        stop_event.set()
        infer_thread.join(timeout=2.0)

        with log_file_path.open("a") as log_file:
            log_file.write(f"Episode {episode + 1}: steps={steps}\n")

        if stop_flag.is_set():
            break
