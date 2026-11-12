# Copyright (C) 2025-2026 LejuRobotics.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ---
#
# This project includes code from LeRobot (https://github.com/huggingface/lerobot),
# which is licensed under the Apache License, Version 2.0.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run kuavo_train/train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from lerobot_patches import custom_patches

from pathlib import Path
from contextlib import nullcontext
from kuavo_deploy.utils.async_inference import AsyncInference
from kuavo_deploy.utils.dagger_session import PolicyRefreshRequested

from sympy import im
from dataclasses import dataclass, field
import hydra
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm
from lerobot.utils.random_utils import set_seed
import datetime
import time
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading

from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.kuavo_service.client import PolicyClient
from kuavo_deploy.utils.policy_loader import (
    inject_task_prompt,
    load_native_policy_bundle,
    resolve_eval_output_dir,
)
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

log_model = setup_logger("model")
log_robot = setup_logger("robot")

def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()

def stop_callback(msg):
    if msg.data:
        stop_flag.set()

stop_flag = threading.Event()
pause_flag = threading.Event()
pause_sub = rospy.Subscriber('/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber('/kuavo/stop_state', Bool, stop_callback, queue_size=10)


def save_rollout_video(output_path: Path, frames: list[np.ndarray], fps: int | float) -> None:
    if not frames:
        return
    try:
        imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264")
    except Exception as exc:
        log_robot.warning(f"Failed to write mp4 '{output_path.name}': {exc}. Falling back to gif.")
        gif_path = output_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), frames, fps=fps)


def setup_policy(pretrained_path, policy_type, device=torch.device("cuda"), task_prompt="robot manipulation", inference_config=None):
    """
    Set up and load the policy model.

    Args:
        pretrained_path: Path to the checkpoint
        policy_type: Type of policy ('diffusion' or 'act')

    Returns:
        Loaded policy model and device
    """

    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)

    if policy_type == 'client':
        preprocessor, postprocessor = lambda obs: obs, lambda action: action
        if inference_config is not None and inference_config.async_inference:
            # The async worker creates the existing client in its own thread.
            return None, preprocessor, postprocessor, None
        return PolicyClient(task_prompt=task_prompt), preprocessor, postprocessor, None

    policy, preprocessor, postprocessor, pretrained_model_dir = load_native_policy_bundle(
        pretrained_path=pretrained_path,
        device=device,
        strict=True,
    )
    log_model.info(f"Model loaded from {pretrained_model_dir}")
    log_model.info(f"Model type: {policy.config.type}")
    log_model.info(f"Model n_obs_steps: {policy.config.n_obs_steps}")
    log_model.info(f"Model device: {device}")
    return policy, preprocessor, postprocessor, pretrained_model_dir

def _rollout(config: KuavoConfig, env: gym.Env, dagger=None):
    # load config
    cfg = config.inference
    stop_signal = env.unwrapped.control_signal_manager.stop_flag if dagger is not None else stop_flag
    pause_signal = env.unwrapped.control_signal_manager.pause_flag if dagger is not None else pause_flag

    eval_episodes = cfg.eval_episodes
    seed = cfg.seed
    start_seed = cfg.start_seed
    policy_type = cfg.policy_type
    if cfg.pretrained_path:
        pretrained_path = Path(cfg.pretrained_path)
    else:
        pretrained_path = Path(f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}")
    output_directory = resolve_eval_output_dir(pretrained_path, Path("outputs/eval"))
    # Create a directory to store the video of the evaluation
    output_directory.mkdir(parents=True, exist_ok=True)

    # set seed
    set_seed(seed=seed)

    # Select your device
    device = torch.device(cfg.device)

    task_prompt = getattr(cfg, 'task_prompt', "robot manipulation")

    policy, preprocessor, postprocessor, pretrained_model_dir = setup_policy(pretrained_path, policy_type, device, task_prompt=task_prompt, inference_config=cfg)

    if dagger is not None and not cfg.async_inference:
        prepared = dagger.prepare_policy(policy, preprocessor, postprocessor)
        if prepared is not None:
            policy = prepared
        set_seed(seed=seed)

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent.
    max_episode_steps = cfg.max_episode_steps
    env = env

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    if policy_type != 'client':
        log_model.info(f"policy.config.input_features: {policy.config.input_features}")
        log_robot.info(f"env.observation_space: {env.observation_space}")

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    if policy_type != 'client':
        log_model.info(f"policy.config.output_features: {policy.config.output_features}")
        log_robot.info(f"env.action_space: {env.action_space}")

    # Log evaluation results
    log_file_path = output_directory / "evaluation.log"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")

    success_count = 0
    for episode in tqdm(range(eval_episodes), desc="Evaluating model", unit="episode"):
        # Reset the policy and environments to prepare for rollout
        if not cfg.async_inference:
            policy.reset()
        observation, info = env.reset(seed=episode+start_seed)
        if dagger is not None:
            dagger.begin_episode(observation)
        if policy_type != 'client':
            observation = inject_task_prompt(observation, task_prompt)
        if not cfg.async_inference:
            observation = preprocessor(observation)
        # log_file.write(f"~~~~~~~~~~~~~~~~~~preprocess observation ok!~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
        rewards = []

        cam_keys = [k for k in observation.keys() if "images" in k or "depth" in k]
        frame_map = {k: [] for k in cam_keys}

        average_exec_time = 0
        average_action_infer_time = 0
        average_step_time = 0

        step = 0
        done = False
        runner_options = {"policy_ready": dagger.prepare_policy} if dagger is not None else {}
        runner_context = AsyncInference(policy, preprocessor, postprocessor, cfg, control_hz=config.env.ros_rate, **runner_options) if cfg.async_inference else nullcontext(None)
        with runner_context as async_runner:
            try:
                with tqdm(total=max_episode_steps, desc=f"Episode {episode+1}", unit="step", leave=False) as pbar:
                    while not done:
                        # --- Pause support: block here if pause_signal is set ---
                        while not cfg.async_inference and pause_signal.is_set() and not stop_signal.is_set():
                            log_model.info("Paused. Waiting for resume signal...")
                            time.sleep(0.5)
                        if stop_signal.is_set():
                            log_model.info("Stop flag detected during pause. Exiting loop.")
                            return

                        start_time = time.time()

                        try:
                            if dagger is not None:
                                token = dagger.inference_token()
                            if async_runner is not None:
                                action = async_runner.select_action(
                                    observation, stop_event=stop_signal, pause_event=pause_signal,
                                    observation_fn=env.unwrapped.get_obs,
                                    on_pause=None if dagger is not None else env.unwrapped.stop_base,
                                    **({"check_interrupt": dagger.check_interrupt} if dagger is not None else {}),
                                )
                                if action is None:
                                    return
                            else:
                                with torch.inference_mode():
                                    action = policy.select_action(observation)
                                action = postprocessor(action)
                            action_infer_time = time.time()
                            log_model.info(f"action infer time: {action_infer_time - start_time:.3f}s")
                            average_action_infer_time += action_infer_time - start_time

                            numpy_action = action.squeeze(0).cpu().numpy()
                            log_model.debug(f"numpy_action: {numpy_action}")

                            if dagger is not None:
                                dagger.accept_policy(token)
                            # 执行动作
                            observation, reward, terminated, truncated, info = env.step(numpy_action)
                        except PolicyRefreshRequested:
                            if dagger is None:
                                raise
                            dagger.check_error()
                            if stop_signal.is_set() or dagger.env.control_signal_manager.stop_flag.is_set():
                                return
                            dagger.start_refresh()
                            if async_runner is not None:
                                async_runner.invalidate(preserve_error=True)
                            else:
                                policy.reset()
                                for processor in (preprocessor, postprocessor):
                                    if callable(getattr(processor, "reset", None)):
                                        processor.reset()
                            observation = env.unwrapped.get_obs()
                            if policy_type != 'client':
                                observation = inject_task_prompt(observation, task_prompt)
                            if not cfg.async_inference:
                                observation = preprocessor(observation)
                            continue
                        if policy_type != 'client':
                            observation = inject_task_prompt(observation, task_prompt)
                        if not cfg.async_inference:
                            observation = preprocessor(observation)
                        exec_time = time.time()
                        log_model.debug(f"exec time: {exec_time - action_infer_time:.3f}s")
                        average_exec_time += exec_time - action_infer_time

                        rewards.append(reward)

                        # 相机帧记录，真机请取消，否则会一直堆叠卡死

                        # for k in cam_keys:
                        #     frame_map[k].append(observation[k].squeeze(0).cpu().numpy().transpose(1, 2, 0))

                        # The rollout is considered done when the success state is reached (i.e. terminated is True),
                        # or the maximum number of iterations is reached (i.e. truncated is True)
                        done = terminated | truncated | done
                        step += 1
                        if dagger is not None and step >= max_episode_steps:
                            truncated = done = True

                        end_time = time.time()
                        log_model.info(f"Step {step} time: {end_time - start_time:.3f}s")

                        # Update progress bar
                        status = "Success" if terminated else "Running"
                        pbar.set_postfix({
                            "Reward": f"{reward:.3f}",
                            "Status": status,
                            "Total Reward": f"{sum(rewards):.3f}"
                        })
                        pbar.update(1)
            finally:
                if dagger is not None:
                    dagger.end_episode()
                if async_runner is not None:
                    env.unwrapped.stop_base()


        if terminated:
            success_count += 1
            log_model.info(f"✅ Episode {episode+1}: Success! Total reward: {sum(rewards):.3f}")
        else:
            log_model.info(f"❌ Episode {episode+1}: Failed! Total reward: {sum(rewards):.3f}")

        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.ros_rate

        log_model.info(f"average exec time: {average_exec_time / step:.3f}s")
        log_model.info(f"average action infer time: {average_action_infer_time / step:.3f}s")
        log_model.info(f"average step time: {average_step_time / step:.3f}s")
        log_model.info(f"average sleep time: {env.average_sleep_time / step:.3f}s")


        # Encode all frames into a mp4 video.
        if len(frame_map.keys()) == 0:
            for cam in cam_keys:
                frames = frame_map[cam]
                output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
                save_rollout_video(output_path, frames, fps)

        # print(f"Video of the evaluation is available in '{video_path}'.")

        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Rewards per Episode: {numpy.array(rewards).sum()}")

    with log_file_path.open("a") as log_file:
        log_file.write("\n")
        log_file.write(f"Success Count: {success_count}\n")
        log_file.write(f"Success Rate: {success_count / eval_episodes:.2f}\n")

    # Display final statistics
    log_model.info("\n" + "="*50)
    log_model.info(f"🎯 Evaluation completed!")
    log_model.info(f"📊 Success count: {success_count}/{eval_episodes}")
    log_model.info(f"📈 Success rate: {success_count / eval_episodes:.2%}")
    print(f"📁 Videos and logs saved to: {output_directory}")
    print("="*50)

def main(config: KuavoConfig, env: gym.Env):
    if config.inference.dagger_on:
        from kuavo_deploy.utils.dagger_session import DaggerSession
        with DaggerSession(config, env.unwrapped) as dagger:
            return _rollout(config, env, dagger=dagger)
    return _rollout(config, env)


def kuavo_eval(config: KuavoConfig, env: gym.Env):
    return main(config, env)

if __name__ == "__main__":
    config_path = Path("test.yaml")
    env = gym.make(
        "Kuavo-Real",
        max_episode_steps=150,
        config_path=config_path,
    )
