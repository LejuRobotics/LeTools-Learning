# Copyright (C) 2025-2026 LejuRobotics.

from __future__ import annotations

import datetime
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


class ActionChunkBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = max(1, int(maxlen))
        self._actions: deque[np.ndarray] = deque()
        self._cond = Condition()

    def clear(self) -> None:
        with self._cond:
            self._actions.clear()
            self._cond.notify_all()

    def qsize(self) -> int:
        with self._cond:
            return len(self._actions)

    def put_chunk(self, actions: list[np.ndarray]) -> int:
        with self._cond:
            free = self.maxlen - len(self._actions)
            for action in actions[:free]:
                self._actions.append(action)
            self._cond.notify_all()
            return min(len(actions), max(0, free))

    def get(self, timeout: float) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._actions:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            return self._actions.popleft()

    def wait_for_size(self, size: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self._actions) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            return True


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

            start = time.time()
            observation = env.get_obs()
            if policy_type != "client":
                observation = inject_task_prompt(observation, task_prompt)
            observation = preprocessor(observation)

            with torch.inference_mode():
                actions = _select_action_chunk(policy, observation)
            actions_np = _postprocess_chunk(actions, postprocessor)
            inserted = buffer.put_chunk(actions_np)
            elapsed = time.time() - start
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

        buffer = ActionChunkBuffer(maxlen=cfg.async_buffer_size)
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
