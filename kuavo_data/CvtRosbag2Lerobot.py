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
分块流式rosbag转换器 - 低内存版本

核心优化（参考Diffusion Policy的按需读取方式）：
1. 第一遍扫描：只读取时间戳（内存占用几MB）
2. 第二遍扫描：按时间窗口分块读取+对齐+写入dataset

与原始CvtRosbag2Lerobot.py的区别：
- 原始：一次性加载整个rosbag到内存 → 对齐 → 写入（内存峰值巨大）
- 本版：分块读取 → 即时对齐 → 即时写入 → 释放内存（内存可控）

使用方法：
    python CvtRosbag2Lerobot_chunked.py --config-name=KuavoRosbag2Lerobot \
        rosbag.rosbag_dir=/path/to/rosbag \
        rosbag.target_dir=/path/to/output \
        rosbag.chunk_size=100
"""
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
import os
import gc
import shutil
from pathlib import Path
import numpy as np
import torch
import tqdm
import hydra
from omegaconf import DictConfig
from typing import Literal

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
import dataclasses
from kuavo_data.common import kuavo_dataset as kuavo
from kuavo_data.common.config_platform import get_arm_joint_slice, DEFAULT_PLATFORM
from rich.logging import RichHandler
import logging

log_print = logging.getLogger(__name__)


def setup_logging():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    from rich.logging import RichHandler
    root.addHandler(
        RichHandler(
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
        )
    )


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset_chunked(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
) -> LeRobotDataset:
    
    # 根据config的参数决定是否为半身和末端的关节类型
    motors = DEFAULT_JOINT_NAMES_LIST
    # TODO: auto detect cameras
    cameras = kuavo.DEFAULT_CAMERA_NAMES


    action_dim = (len(motors),)

    # set action name/dim, state name/dim,
    action_name =  motors

    state_dim = (len(motors),)

    # state_name = kuavo.DEFAULT_ARM_JOINT_NAMES[:len(kuavo.DEFAULT_ARM_JOINT_NAMES)//2] + ["gripper_l"] + kuavo.DEFAULT_ARM_JOINT_NAMES[len(kuavo.DEFAULT_ARM_JOINT_NAMES)//2:] + ["gripper_r"]
    state_name = motors

    # create corresponding features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_dim,
            "names": {
                "state_names": state_name
            }
        },
        "action": {
            "dtype": "float32",
            "shape": action_dim,
            "names": {
                "action_names": action_name
            }
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        if 'depth' in cam:
            features[f"observation.{cam}"] = {
                "dtype": mode, 
                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),  # Attention: for datasets.features "image" and "video", it must be c,h,w style! 
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }
        else:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=kuavo.TRAIN_HZ,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )


def resume_dataset_chunked(
    repo_id: str,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
) -> LeRobotDataset:
    return LeRobotDataset.resume(
        repo_id=repo_id,
        root=root,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def infer_local_repo_id(dataset_root: Path) -> str:
    """Generate a stable local repo id for datasets used from disk only."""
    return f"lerobot/{dataset_root.resolve().name}"


def resolve_lerobot_output_dir(target_dir: Path) -> Path:
    """Use target_dir as a parent directory and store the dataset in target_dir/lerobot."""
    return target_dir.expanduser().resolve() / "lerobot"


def ensure_clean_lerobot_output_dir(target_dir: Path) -> Path:
    output_dir = resolve_lerobot_output_dir(target_dir)
    target_dir = output_dir.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    return output_dir


def prepare_resume_target(source_dir: Path, output_dir: Path) -> Path:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"resume source dataset does not exist: {source_dir}")

    if source_dir == output_dir:
        return output_dir

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    return output_dir


def load_merge_sources(roots: list[Path]) -> list[LeRobotDataset]:
    datasets = []
    for root in roots:
        ds_root = root.expanduser().resolve()
        if not ds_root.exists():
            raise FileNotFoundError(f"merge source dataset does not exist: {ds_root}")
        datasets.append(
            LeRobotDataset(
                repo_id=infer_local_repo_id(ds_root),
                root=ds_root,
            )
        )
    return datasets


def populate_dataset_chunked(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    chunk_size: int = 100,
    platform_type: str = DEFAULT_PLATFORM,
) -> LeRobotDataset:
    """
    使用分块流式处理填充数据集
    
    核心优化：
    1. 第一遍扫描只读取时间戳（内存几MB）
    2. 第二遍扫描按时间窗口分块读取+对齐+写入
    3. 每个chunk处理完立即保存并释放内存
    
    Args:
        dataset: LeRobotDataset实例
        bag_files: rosbag文件路径列表
        task: 任务描述
        episodes: 要处理的episode索引列表
        chunk_size: 每个chunk包含的帧数（默认100帧）
    """
    if episodes is None:
        episodes = range(len(bag_files))
    
    failed_bags = []
    log_print.info(f"Total episodes to process: {len(episodes)}")
    bag_reader = kuavo.KuavoRosbagReader()
    
    # 内存监控
    process = None
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except ImportError:
        pass
    
    def log_memory(prefix: str):
        if process:
            mem_mb = process.memory_info().rss / 1024 / 1024
            log_print.info(f"{prefix} Memory: {mem_mb:.2f} MB")
    
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = bag_files[ep_idx]
        log_print.warning(f"Processing {ep_path}")
        log_memory("Before processing")
        
        try:
            # 收集当前episode的所有帧
            frames_buffer = []
            frame_count = [0]
            
            def on_frame(aligned_frame: dict, frame_idx: int):
                """处理单帧对齐数据"""

                def get_array(key, dtype, default_empty=True):
                    item = aligned_frame.get(key)
                    if item is None:
                        return np.array([], dtype=dtype) if default_empty else None
                    return np.array(item.get("data", []), dtype=dtype)

                # =========================
                # 1. state / action
                # =========================
                state = get_array('observation.state', np.float32)
                action = get_array('action', np.float32)
                

                if state.size == 0 or action.size == 0:
                    log_print.warning(f"Episode {ep_idx} Frame {frame_idx}: Missing state/action data")
                    return

                # =========================
                # 2. arm trajectory
                # =========================
                arm_traj     = get_array("action.kuavo_arm_traj", np.float32)
                if arm_traj.size == 0:
                    log_print.warning(f"Episode {ep_idx} Frame {frame_idx}: Missing arm trajectory data")
                    return
                arm_start, arm_end = get_arm_joint_slice(platform_type)
                action[arm_start:arm_end] = arm_traj

                # 接口留用
                velocity = None
                effort = None

                # =========================
                # 3. 手部数据读取
                # =========================
                claw_state     = get_array("observation.claw", np.float32)
                claw_action    = get_array("action.claw", np.float32)
                qiangnao_state = get_array("observation.qiangnao", np.float32)
                qiangnao_action= get_array("action.qiangnao", np.float32)
                rq2f85_state   = get_array("observation.rq2f85", np.float32)
                rq2f85_action  = get_array("action.rq2f85", np.float32)

                if claw_state.size == 0 and qiangnao_state.size == 0 and rq2f85_state.size==0:
                    # log_print.warning(f"Episode {ep_idx} Frame {frame_idx}: Missing eef state data")
                    return
                if claw_action.size == 0 and qiangnao_action.size==0 and rq2f85_action.size ==0:
                    # log_print.warning(f"Episode {ep_idx} Frame {frame_idx}: Missing eef action data")
                    return
                # =========================
                # 4. 手部归一化（保持原逻辑）
                # =========================
                if kuavo.IS_BINARY:
                    qiangnao_state  = np.where(qiangnao_state > 50, 1, 0)
                    qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
                    claw_state      = np.where(claw_state > 50, 1, 0)
                    claw_action     = np.where(claw_action > 50, 1, 0)
                    rq2f85_state    = np.where(rq2f85_state > 0.4, 1, 0)
                    rq2f85_action   = np.where(rq2f85_action > 70, 1, 0)
                    # rq2f85_state = np.where(rq2f85_state > 0.1, 1, 0)
                    # rq2f85_action = np.where(rq2f85_action > 128, 1, 0)
                else:
                    if claw_state.size:      claw_state /= 100
                    if claw_action.size:     claw_action /= 100
                    if qiangnao_state.size:  qiangnao_state /= 100
                    if qiangnao_action.size: qiangnao_action /= 100
                    if rq2f85_state.size:    rq2f85_state /= 0.8
                    if rq2f85_action.size:   rq2f85_action /= 255
                    # rq2f85_state = rq2f85_state / 0.8
                    # rq2f85_action = rq2f85_action / 255

                if claw_action.size == 0 and qiangnao_action.size == 0:
                    claw_action = rq2f85_action
                    claw_state  = rq2f85_state

                # =========================
                # 5. 构建最终 state / action
                # =========================
                if kuavo.USE_LEJU_CLAW or kuavo.USE_QIANGNAO:
                    hand_type = "LEJU" if kuavo.USE_LEJU_CLAW else "QIANGNAO"
                    s_list, a_list = [], []

                    def get_hand_slice(hand_side):
                        s_slice = kuavo.SLICE_ROBOT[hand_side]

                        if hand_type == "LEJU":
                            c_slice = kuavo.SLICE_CLAW[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                claw_state[c_slice[0]:c_slice[-1]]))
                            a = np.concatenate((action[s_slice[0]:s_slice[-1]],
                                                claw_action[c_slice[0]:c_slice[-1]]))
                        else:
                            d_slice = kuavo.SLICE_DEX[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                qiangnao_state[d_slice[0]:d_slice[-1]]))
                            a = np.concatenate((action[s_slice[0]:s_slice[-1]],
                                                qiangnao_action[d_slice[0]:d_slice[-1]]))
                        return s, a

                    if kuavo.CONTROL_HAND_SIDE in ("left", "both"):
                        s, a = get_hand_slice(0)
                        s_list.append(s)
                        a_list.append(a)

                    if kuavo.CONTROL_HAND_SIDE in ("right", "both"):
                        s, a = get_hand_slice(1)
                        s_list.append(s)
                        a_list.append(a)

                    final_state  = np.concatenate(s_list).astype(np.float32)
                    final_action = np.concatenate(a_list).astype(np.float32)
                else:
                    raise ValueError(f"eef type are not supported! ")

                # =========================
                # 6. 构建 frame
                # =========================
                frame = {
                    "observation.state": torch.from_numpy(final_state).type(torch.float32),
                    "action": torch.from_numpy(final_action).type(torch.float32),
                }

                for cam_key in kuavo.DEFAULT_CAMERA_NAMES:
                    cam_data = aligned_frame.get(cam_key)
                    if cam_data and "data" in cam_data:
                        img = cam_data["data"]
                        if "depth" in cam_key:
                            min_d, max_d = kuavo.DEPTH_RANGE
                            depth = np.clip(img, min_d, max_d)
                            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
                            depth_uint8 = (depth_norm * 255).astype(np.uint8)
                            frame[f"observation.{cam_key}"] = depth_uint8[..., None].repeat(3, -1)
                        else:
                            frame[f"observation.images.{cam_key}"] = img
                    else:
                        return
                
                if velocity is not None:
                    frame["observation.velocity"] = velocity
                if effort is not None:
                    frame["observation.effort"] = effort
                
                frames_buffer.append(frame)
                frame_count[0] += 1

            
            def on_chunk_done():
                """每个chunk处理完后的回调：保存并释放内存"""
                if len(frames_buffer) == 0:
                    log_memory(f"After saving chunk (total frames: {frame_count[0]})")
                    return
                
                # 将所有缓存的帧添加到dataset
                for frame in frames_buffer:
                    frame["task"] = task
                    dataset.add_frame(frame)
                
                # 清空buffer并释放内存
                frames_buffer.clear()
                gc.collect()
                
                log_memory(f"After saving chunk (total frames: {frame_count[0]})")
            
            # 使用分块流式处理
            bag_reader.process_rosbag_chunked(
                bag_file=str(ep_path),
                frame_callback=on_frame,
                chunk_size=chunk_size,
                save_callback=on_chunk_done
            )
             
            # 处理剩余的帧
            if len(frames_buffer) > 0:
                for frame in frames_buffer:
                    dataset.add_frame(frame, task=task)
            dataset.save_episode()
            frames_buffer.clear()
            gc.collect()
            
            log_print.info(f"Episode {ep_idx} completed: {frame_count[0]} frames")
            
        except Exception as e:
            log_print.error(f"Error processing {ep_path}: {e}")
            import traceback
            traceback.print_exc()
            failed_bags.append(str(ep_path))
            continue
        
        log_memory("After episode")
        gc.collect()
    
    if failed_bags:
        with open("error.txt", "w") as f:
            for bag in failed_bags:
                f.write(bag + "\n")
        log_print.error(f"{len(failed_bags)} failed bags written to error.txt")
    
    return dataset


def port_kuavo_rosbag_chunked(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    n: int | None = None,
    chunk_size: int = 100,
    platform_type: str = "4pro",
):
    """
    分块流式转换rosbag到LeRobot格式
    
    Args:
        raw_dir: rosbag目录
        repo_id: 输出数据集ID
        task: 任务描述
        chunk_size: 每个chunk的帧数（默认100）
    """
    bag_reader = kuavo.KuavoRosbagReader()
    bag_files = bag_reader.list_bag_files(raw_dir)
    if not bag_files:
        log_print.error(
            f"No .bag files found in rosbag_dir(s). Check that paths exist and contain *.bag. "
            f"Looked in: {raw_dir}"
        )
        raise FileNotFoundError(f"No .bag files in: {raw_dir}")

    log_print.info(f"Found {len(bag_files)} bag file(s) in rosbag dir: {raw_dir}")
    
    if isinstance(n, int) and n > 0:
        num_available_bags = len(bag_files)
        if n > num_available_bags:
            log_print.warning(f"Requested {n} bags, but only {num_available_bags} available. Using all available bags.")
            n = num_available_bags
        select_idx = np.random.choice(num_available_bags, n, replace=False)
        bag_files = [bag_files[i] for i in select_idx]
    
    dataset = create_empty_dataset_chunked(
        repo_id,
        robot_type=f"kuavo-{platform_type}",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
        root=root,
    )
    
    dataset = populate_dataset_chunked(
        dataset,
        bag_files,
        task=task,
        episodes=episodes,
        chunk_size=chunk_size,
        platform_type=platform_type,
    )
    dataset.finalize()
    
    return dataset


def resume_kuavo_rosbag_chunked(
    raw_dir: Path,
    repo_id: str,
    *,
    resume_root: str | Path,
    task: str = "DEBUG",
    episodes: list[int] | None = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    n: int | None = None,
    chunk_size: int = 100,
    platform_type: str = "4pro",
) -> LeRobotDataset:
    bag_reader = kuavo.KuavoRosbagReader()
    bag_files = bag_reader.list_bag_files(raw_dir)
    if not bag_files:
        log_print.error(
            f"No .bag files found in rosbag_dir(s). Check that paths exist and contain *.bag. "
            f"Looked in: {raw_dir}"
        )
        raise FileNotFoundError(f"No .bag files in: {raw_dir}")

    log_print.info(f"Found {len(bag_files)} bag file(s) in rosbag dir: {raw_dir}")

    if isinstance(n, int) and n > 0:
        num_available_bags = len(bag_files)
        if n > num_available_bags:
            log_print.warning(f"Requested {n} bags, but only {num_available_bags} available. Using all available bags.")
            n = num_available_bags
        select_idx = np.random.choice(num_available_bags, n, replace=False)
        bag_files = [bag_files[i] for i in select_idx]

    dataset = resume_dataset_chunked(
        repo_id=repo_id,
        dataset_config=dataset_config,
        root=str(resume_root),
    )

    dataset = populate_dataset_chunked(
        dataset,
        bag_files,
        task=task,
        episodes=episodes,
        chunk_size=chunk_size,
        platform_type=platform_type,
    )
    dataset.finalize()
    return dataset


def merge_lerobot_datasets(
    source_dirs: list[Path],
    output_dir: Path,
) -> LeRobotDataset:
    if len(source_dirs) < 2:
        raise ValueError("merge mode requires at least two dataset directories in `rosbag.lerobot_dir_merge`.")

    output_dir = output_dir.expanduser().resolve()
    if any(source.expanduser().resolve() == output_dir for source in source_dirs):
        raise ValueError("merge output target_dir/lerobot cannot be one of rosbag.lerobot_dir_merge sources.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    datasets = load_merge_sources(source_dirs)
    output_repo_id = infer_local_repo_id(output_dir)
    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
    )
    return merged


@hydra.main(
    config_path="../configs/data/",
    config_name="KuavoRosbag2Lerobot",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """
    分块流式转换入口
    
    使用方法：
        python CvtRosbag2Lerobot.py \
            rosbag.mode=normal \
            rosbag.rosbag_dir=/path/to/rosbag \
            rosbag.target_dir=/path/to/output \
            rosbag.chunk_size=100
    """
    setup_logging()  # set logger 

    global DEFAULT_JOINT_NAMES_LIST
    kuavo.init_parameters(cfg)

    mode = str(cfg.rosbag.get("mode", "normal")).strip().lower()
    n = cfg.rosbag.get("num_used", None)
    raw_dir_cfg = cfg.rosbag.get("rosbag_dir")
    raw_dir = Path(raw_dir_cfg).expanduser().resolve() if raw_dir_cfg else None
    target_dir = Path(cfg.rosbag.target_dir).expanduser().resolve()
    output_dir = resolve_lerobot_output_dir(target_dir)
    resume_source_cfg = cfg.rosbag.get("lerobot_dir_resume")
    resume_source = Path(resume_source_cfg).expanduser().resolve() if resume_source_cfg else None
    merge_source_cfg = cfg.rosbag.get("lerobot_dir_merge", [])
    merge_sources = [Path(p).expanduser().resolve() for p in merge_source_cfg] if merge_source_cfg else []

    chunk_size = cfg.rosbag.get("chunk_size", 100)
    
    log_print.info(f"=== Chunked Streaming Rosbag Converter ===")
    log_print.info(f"Mode: {mode}")
    if raw_dir is not None:
        log_print.info(f"Rosbag dir: {raw_dir}")
    if resume_source is not None:
        log_print.info(f"Resume source dir: {resume_source}")
    if merge_sources:
        log_print.info(f"Merge source dirs: {merge_sources}")
    log_print.info(f"Target dir: {target_dir}")
    log_print.info(f"LeRobot output dir: {output_dir}")
    log_print.info(f"Chunk size: {chunk_size}")

    half_arm = len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(kuavo.DEFAULT_DEXHAND_JOINT_NAMES) // 2
    arm_base_start, _ = get_arm_joint_slice(kuavo.PLATFORM_TYPE)
    if kuavo.USE_LEJU_CLAW:
        DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
        arm_slice = [
            (kuavo.SLICE_ROBOT[0][0] - arm_base_start, kuavo.SLICE_ROBOT[0][-1] - arm_base_start),(kuavo.SLICE_CLAW[0][0] + half_arm, kuavo.SLICE_CLAW[0][-1] + half_arm),
            (kuavo.SLICE_ROBOT[1][0] - arm_base_start + half_claw, kuavo.SLICE_ROBOT[1][-1] - arm_base_start + half_claw), (kuavo.SLICE_CLAW[1][0] + half_arm * 2, kuavo.SLICE_CLAW[1][-1] + half_arm * 2)
            ]
    elif kuavo.USE_QIANGNAO:  
        DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]               
        arm_slice = [
            (kuavo.SLICE_ROBOT[0][0] - arm_base_start, kuavo.SLICE_ROBOT[0][-1] - arm_base_start),(kuavo.SLICE_DEX[0][0] + half_arm, kuavo.SLICE_DEX[0][-1] + half_arm),
            (kuavo.SLICE_ROBOT[1][0] - arm_base_start + half_dexhand, kuavo.SLICE_ROBOT[1][-1] - arm_base_start + half_dexhand), (kuavo.SLICE_DEX[1][0] + half_arm * 2, kuavo.SLICE_DEX[1][-1] + half_arm * 2)
            ]
    DEFAULT_JOINT_NAMES_LIST = [DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)]

    if mode == "normal":
        if raw_dir is None:
            raise ValueError("normal mode requires `rosbag.rosbag_dir`.")
        if raw_dir.resolve() == output_dir.resolve():
            raise ValueError("rosbag_dir and target_dir/lerobot cannot be the same.")
        task_name = raw_dir.name
        repo_id = f"lerobot/{task_name}"
        output_dir = ensure_clean_lerobot_output_dir(target_dir)
        port_kuavo_rosbag_chunked(
            raw_dir=raw_dir,
            repo_id=repo_id,
            task=kuavo.TASK_DESCRIPTION,
            mode="video",
            root=str(output_dir),
            n=n,
            chunk_size=chunk_size,
            platform_type=kuavo.PLATFORM_TYPE,
        )
        log_print.info("Normal conversion completed!")
        return

    if mode == "resume":
        if raw_dir is None:
            raise ValueError("resume mode requires `rosbag.rosbag_dir`.")
        if resume_source is None:
            raise ValueError("resume mode requires `rosbag.lerobot_dir_resume`.")
        resume_target = prepare_resume_target(resume_source, output_dir)
        repo_id = infer_local_repo_id(output_dir)
        resume_kuavo_rosbag_chunked(
            raw_dir=raw_dir,
            repo_id=repo_id,
            resume_root=resume_target,
            task=kuavo.TASK_DESCRIPTION,
            n=n,
            chunk_size=chunk_size,
            platform_type=kuavo.PLATFORM_TYPE,
        )
        log_print.info("Resume conversion completed!")
        return

    if mode == "merge":
        if not merge_sources:
            raise ValueError("merge mode requires `rosbag.lerobot_dir_merge` with at least two dataset directories.")
        merge_lerobot_datasets(merge_sources, output_dir)
        log_print.info("Merge completed!")
        return

    raise ValueError(f"Unsupported rosbag.mode: {mode}. Expected one of: normal, resume, merge")


if __name__ == "__main__":
    np.random.seed(42)
    main()
