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
import tempfile
import time
import concurrent.futures
from pathlib import Path
import numpy as np
import torch
import tqdm
import hydra
from omegaconf import DictConfig
from typing import Literal

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets import dataset_writer as lerobot_dataset_writer
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.feature_utils import validate_episode_buffer
import dataclasses
from kuavo_data.common import kuavo_dataset as kuavo
from kuavo_data.common.config_platform import (
    get_arm_joint_slice,
    get_lower_body_joint_slice,
    DEFAULT_PLATFORM,
)
from kuavo_data.common.sg100 import sg100_amounts, sg100_binarize
from kuavo_data.common.h265_utils import (
    H265CameraContext,
    build_camera_timestamp_sources,
    preload_h265_cameras,
    save_h265_videos_direct,
    trim_main_timestamps_strict,
)
from kuavo_data.common.parallel_cvt import parallel_options, run_parallel_conversion
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


# ============================================================================
# SG100 1-DOF 二值化
# 只在 eef_type='sg100' 且 is_binary=True 时生效，其他末端走原有分支。
# ============================================================================

# 样本太少时不做健全性判定，避免把"某条 episode 夹爪没动"误判成配置错误。
SG100_CHECK_MIN_SAMPLES = 30
# 闭合占比超过此值即告警（疑似 open/close 标反、选错关节，或该机器人零位不同）。
SG100_MAX_CLOSED_RATIO = 0.80
# 累积桶的键 -> (日志显示名, 来源话题)。action 即数据集的 action 字段。
SG100_KIND_SOURCE = {"state": ("state", "/sg100_hand_state"),
                     "action": ("command", "/sg100_hand_command")}


def _sg100_amounts(values: np.ndarray, hand: int) -> list:
    """该手判定关节的开合进度 amount = (q - open) / (close - open)。

    分母带符号，因此"越大越闭合"和"越小越闭合"两种关节都适用。
    真正的计算在 kuavo_data.common.sg100 里，这里只补上本次转换的参考位姿。
    """
    return sg100_amounts(
        values,
        hand,
        kuavo.SG100_BINARY_OPEN,
        kuavo.SG100_BINARY_CLOSE,
        kuavo.SG100_BINARY_JOINTS,
    )


def _sg100_binarize(values: np.ndarray) -> np.ndarray:
    """把 sg100 原始值二值化为 0/1，结果写在每手的代表维上（其余维保持 0）。

    每手使用 dex_dof_offset 指定的关节及手动配置的 open/close 参考值，
    amount 超过 ratio 即判为闭合(1)。
    真正的计算在 kuavo_data.common.sg100 里，与在线推理共用同一份实现。
    """
    return sg100_binarize(
        values,
        kuavo.SG100_BINARY_OPEN,
        kuavo.SG100_BINARY_CLOSE,
        kuavo.SG100_BINARY_JOINTS,
        kuavo.SG100_BINARY_RATIO,
    )


def _sg100_collect(bucket: dict, kind: str, values: np.ndarray) -> None:
    """累积 sg100 原始值（22 维），供 episode 结束时做健全性检查。

    逐帧调用时 values 是 1 维，atleast_2d 保证拼出来始终是 (N, 22)。
    """
    if values.size == 0 or values.shape[-1] != 22:
        return
    bucket[kind].append(np.atleast_2d(values.astype(np.float64)))


def _sg100_check(bucket: dict, episode_index: int) -> None:
    """按 episode 打印每只手的 state / command 闭合占比，异常时告警。

    state  <- /sg100_hand_state    （数据集的 observation.state）
    command <- /sg100_hand_command （数据集的 action）
    判定关节/开合参考位姿可能不适用于其他机器人，所以只提示、不中断转换：
    闭合占比过高 -> 疑似 open/close 标反，或判定关节选错；
    全程未闭合   -> 该通道本 episode 没动，属正常数据。
    """
    ratios: dict = {}
    used = {"left": (0,), "right": (1,), "both": (0, 1)}[kuavo.CONTROL_HAND_SIDE]
    for kind in ("state", "action"):
        if not bucket[kind]:
            continue
        column = np.concatenate(bucket[kind])          # (N, 22)
        if column.shape[0] < SG100_CHECK_MIN_SAMPLES:
            continue
        for hand in used:
            amount = np.max(_sg100_amounts(column, hand), axis=0)
            ratios[(kind, ("left", "right")[hand])] = float(
                np.mean(amount > kuavo.SG100_BINARY_RATIO))

    if not ratios:
        return

    def pct(kind: str, hand: str) -> str:
        value = ratios.get((kind, hand))
        return f"{value:5.1%}" if value is not None else "    -"

    for hand in ("left", "right"):
        if ("state", hand) in ratios or ("action", hand) in ratios:
            log_print.info(
                f"Episode {episode_index} sg100 {hand:<5} 闭合占比（占本 episode 时长）: "
                f"state={pct('state', hand)}  command={pct('action', hand)}"
            )

    for (kind, hand), closed in sorted(ratios.items()):
        label, topic = SG100_KIND_SOURCE[kind]
        if closed > SG100_MAX_CLOSED_RATIO:
            log_print.warning(
                f"Episode {episode_index} sg100 {label}/{hand} ({topic}): 闭合占比 {closed:.1%} 过高，"
                f"疑似开合参考位姿标反，或判定关节选错了"
            )
        elif closed == 0.0:
            log_print.warning(
                f"Episode {episode_index} sg100 {label}/{hand} ({topic}): 本 episode 全程未闭合"
                f"（该手没动，或判定关节选错了）"
            )


@dataclasses.dataclass
class BagConversionTiming:
    """Per-rosbag conversion timing record."""

    episode_index: int
    bag_path: str
    codec: str  # "h265" | "jpeg" | "mixed"
    frames: int
    wall_time_sec: float
    train_hz: int

    @property
    def content_duration_sec(self) -> float:
        return self.frames / self.train_hz if self.train_hz > 0 else 0.0

    @property
    def content_duration_min(self) -> float:
        return self.content_duration_sec / 60.0

    @property
    def sec_per_content_min(self) -> float:
        """Wall seconds to convert one minute of episode data."""
        if self.content_duration_min <= 0:
            return float("inf")
        return self.wall_time_sec / self.content_duration_min


def _bag_codec_label(h265_cameras: set[str], camera_encodings: dict[str, str]) -> str:
    h265 = {cam for cam, enc in camera_encodings.items() if enc == "h265"}
    jpeg = {cam for cam, enc in camera_encodings.items() if enc == "jpeg"}
    if h265 and jpeg:
        return "mixed"
    if h265 or h265_cameras:
        return "h265"
    return "jpeg"


def _log_bag_conversion_timing(record: BagConversionTiming) -> None:
    log_print.info(
        "Conversion timing ep=%d codec=%s frames=%d bag=%s",
        record.episode_index,
        record.codec,
        record.frames,
        Path(record.bag_path).name,
    )
    log_print.info(
        "  wall=%.1fs (%.2f min) content=%.2f min @ %dHz sec_per_content_min=%.1f",
        record.wall_time_sec,
        record.wall_time_sec / 60.0,
        record.content_duration_min,
        record.train_hz,
        record.sec_per_content_min,
    )


def _log_conversion_timing_summary(records: list[BagConversionTiming]) -> None:
    if not records:
        return

    log_print.info("=== Conversion timing summary ===")
    by_codec: dict[str, list[BagConversionTiming]] = {}
    for rec in records:
        by_codec.setdefault(rec.codec, []).append(rec)

    summary_rows: dict[str, dict[str, float]] = {}
    for codec, items in sorted(by_codec.items()):
        n = len(items)
        total_frames = sum(r.frames for r in items)
        total_wall = sum(r.wall_time_sec for r in items)
        total_content_min = sum(r.content_duration_min for r in items)
        avg_sec_per_min = sum(r.sec_per_content_min for r in items) / n
        summary_rows[codec] = {
            "bags": n,
            "frames": total_frames,
            "wall_sec": total_wall,
            "content_min": total_content_min,
            "avg_sec_per_content_min": avg_sec_per_min,
        }
        log_print.info(
            "  [%s] bags=%d frames=%d total_wall=%.1fs total_content=%.2f min "
            "avg_sec_per_content_min=%.1f",
            codec,
            n,
            total_frames,
            total_wall,
            total_content_min,
            avg_sec_per_min,
        )

    if "h265" in summary_rows and "jpeg" in summary_rows:
        h265_avg = summary_rows["h265"]["avg_sec_per_content_min"]
        jpeg_avg = summary_rows["jpeg"]["avg_sec_per_content_min"]
        if jpeg_avg > 0:
            delta_pct = (h265_avg - jpeg_avg) / jpeg_avg * 100.0
            log_print.info(
                "  [h265 vs jpeg] avg_sec_per_content_min: h265=%.1f jpeg=%.1f delta=%+.1f%%",
                h265_avg,
                jpeg_avg,
                delta_pct,
            )
        req_max_pct = 30.0
        if jpeg_avg > 0 and h265_avg <= jpeg_avg * (1.0 + req_max_pct / 100.0):
            log_print.info(
                "  [h265 vs jpeg] within requirement: H.265 overhead <= %.0f%% vs JPEG",
                req_max_pct,
            )
        elif jpeg_avg > 0:
            log_print.warning(
                "  [h265 vs jpeg] exceeds requirement: H.265 overhead=%+.1f%% (limit %.0f%%)",
                delta_pct,
                req_max_pct,
            )

    total_wall = sum(r.wall_time_sec for r in records)
    log_print.info(
        "  [all] bags=%d total_wall=%.1fs (%.2f min)",
        len(records),
        total_wall,
        total_wall / 60.0,
    )


_H265_PLACEHOLDER: np.ndarray | None = None


def _get_h265_placeholder() -> np.ndarray:
    global _H265_PLACEHOLDER
    if _H265_PLACEHOLDER is None:
        _H265_PLACEHOLDER = np.zeros((kuavo.RESIZE_H, kuavo.RESIZE_W, 3), dtype=np.uint8)
    return _H265_PLACEHOLDER


def _encode_h265_videos(
    h265_contexts: dict[str, H265CameraContext],
    main_timestamps: list[float],
) -> tuple[dict[str, Path], dict[str, dict[str, np.ndarray]]]:
    """Encode all H.265 cameras to MP4 (libsvtav1, preset 12, CRF 30, GOP 2)."""
    mp4_paths: dict[str, Path] = {}
    video_stats: dict[str, dict[str, np.ndarray]] = {}
    for cam, ctx in h265_contexts.items():
        temp_mp4 = Path(tempfile.mkdtemp(dir=ctx.temp_dir.parent)) / f"{ctx.video_key}.mp4"
        _, cam_stats = save_h265_videos_direct(
            ctx.nal_path,
            ctx.source_timestamps,
            main_timestamps,
            temp_mp4,
            kuavo.RESIZE_W,
            kuavo.RESIZE_H,
            kuavo.TRAIN_HZ,
        )
        mp4_paths[ctx.video_key] = temp_mp4
        if cam_stats is None:
            raise RuntimeError(f"Failed to compute video stats for {ctx.video_key}")
        video_stats[ctx.video_key] = cam_stats
    return mp4_paths, video_stats


def save_episode_with_h265_videos(
    dataset: LeRobotDataset,
    h265_mp4_paths: dict[str, Path],
    h265_video_stats: dict[str, dict[str, np.ndarray]],
    *,
    parallel_encoding: bool = True,
) -> None:
    """Save episode parquet + inject pre-encoded H.265 MP4s into the dataset."""
    writer = dataset.writer
    episode_buffer = writer.episode_buffer

    validate_episode_buffer(episode_buffer, writer._meta.total_episodes, writer._meta.features)

    episode_length = episode_buffer.pop("size")
    tasks = episode_buffer.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = episode_buffer["episode_index"]

    episode_buffer["index"] = np.arange(
        writer._meta.total_frames, writer._meta.total_frames + episode_length
    )
    episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

    writer._meta.save_episode_tasks(episode_tasks)
    episode_buffer["task_index"] = np.array([writer._meta.get_task_index(task) for task in tasks])

    for key, ft in writer._meta.features.items():
        if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
            continue
        episode_buffer[key] = np.stack(episode_buffer[key])

    writer._wait_image_writer()

    # H.265 videos are pre-encoded — compute stats from non-video features only.
    non_video_buffer = {
        k: v
        for k, v in episode_buffer.items()
        if writer._meta.features.get(k, {}).get("dtype") not in ("video", "image")
    }
    non_video_features = {k: v for k, v in writer._meta.features.items() if v["dtype"] not in ("video", "image")}
    ep_stats = compute_episode_stats(non_video_buffer, non_video_features)
    ep_stats.update(h265_video_stats)

    ep_metadata = writer._save_episode_data(episode_buffer)

    for video_key in h265_mp4_paths:
        img_dir = writer._get_image_file_dir(episode_index, video_key)
        if img_dir.is_dir():
            shutil.rmtree(img_dir)
        ep_metadata.update(writer._save_episode_video(video_key, episode_index, temp_path=h265_mp4_paths[video_key]))

    jpeg_video_keys = [k for k in writer._meta.video_keys if k not in h265_mp4_paths]
    if jpeg_video_keys:
        original_worker = lerobot_dataset_writer._encode_video_worker

        def patched_worker(video_key, ep_index, root, fps, vcodec, encoder_threads):
            if video_key in h265_mp4_paths:
                return h265_mp4_paths[video_key]
            return original_worker(video_key, ep_index, root, fps, vcodec, encoder_threads)

        lerobot_dataset_writer._encode_video_worker = patched_worker
        try:
            if parallel_encoding and len(jpeg_video_keys) > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=len(jpeg_video_keys)) as executor:
                    future_to_key = {
                        executor.submit(
                            patched_worker,
                            video_key,
                            episode_index,
                            writer._root,
                            writer._meta.fps,
                            writer._vcodec,
                            writer._encoder_threads,
                        ): video_key
                        for video_key in jpeg_video_keys
                    }
                    for future in concurrent.futures.as_completed(future_to_key):
                        video_key = future_to_key[future]
                        temp_path = future.result()
                        ep_metadata.update(
                            writer._save_episode_video(video_key, episode_index, temp_path=temp_path)
                        )
            else:
                for video_key in jpeg_video_keys:
                    ep_metadata.update(writer._save_episode_video(video_key, episode_index))
        finally:
            lerobot_dataset_writer._encode_video_worker = original_worker

    writer._meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)
    writer.clear_episode_buffer(delete_images=len(writer._meta.image_keys) > 0)

    for path in h265_mp4_paths.values():
        if path.parent.exists():
            shutil.rmtree(path.parent, ignore_errors=True)

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


    # set action name/dim, state name/dim,
    action_name = list(motors)
    if kuavo.USE_5W_BASE_MOVE:
        action_name.extend(kuavo.DEFAULT_BASE_VELOCITY_NAMES)
    action_dim = (len(action_name),)

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
        vcodec="libsvtav1",
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
    conversion_timings: list[BagConversionTiming] = []
    log_print.info(f"Total episodes to process: {len(episodes)}")

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
        bag_t0 = time.perf_counter()
        
        try:
            # Fresh reader per bag so H.265/JPEG topic maps do not leak across episodes.
            bag_reader = kuavo.KuavoRosbagReader()
            bag_reader.configure_for_bag(str(ep_path))
            h265_cameras = bag_reader.h265_cameras
            h265_contexts: dict[str, H265CameraContext] = {}
            h265_temp_root: Path | None = None
            if h265_cameras:
                h265_temp_root = Path(tempfile.mkdtemp(prefix=f"h265_ep{ep_idx}_"))
                h265_contexts = preload_h265_cameras(
                    str(ep_path),
                    {cam: "h265" for cam in h265_cameras},
                    h265_temp_root,
                )

            # 收集当前episode的所有帧
            frames_buffer = []
            frame_count = [0]
            # sg100 二值化健全性检查用的原始值累积（非 sg100 时为空）
            sg100_check = {"state": [], "action": []}
            
            def on_frame(aligned_frame: dict, frame_idx: int):
                """处理单帧对齐数据"""

                def get_array(key, dtype, default_empty=True):
                    item = aligned_frame.get(key)
                    if item is None:
                        return np.array([], dtype=dtype) if default_empty else None
                    return np.array(item.get("data", []), dtype=dtype)

                # =========================
                # 1. state
                # =========================
                state = get_array('observation.state', np.float32)

                if state.size == 0:
                    log_print.warning(f"Episode {ep_idx} Frame {frame_idx}: Missing state data")
                    return

                # =========================
                # 2. arm trajectory
                # =========================
                arm_traj     = get_array("action.kuavo_arm_traj", np.float32)
                arm_start, arm_end = get_arm_joint_slice(platform_type)
                arm_dof = arm_end - arm_start
                if arm_traj.size != arm_dof:
                    log_print.warning(
                        f"Episode {ep_idx} Frame {frame_idx}: expected {arm_dof} arm trajectory "
                        f"values, got {arm_traj.size}"
                    )
                    return
                arm_dof_per_side = arm_dof // 2

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
                sg100_state    = get_array("observation.sg100", np.float32)
                sg100_action   = get_array("action.sg100", np.float32)
                rq2f85_state   = get_array("observation.rq2f85", np.float32)
                rq2f85_action  = get_array("action.rq2f85", np.float32)

                if claw_state.size == 0 and qiangnao_state.size == 0 and sg100_state.size == 0 and rq2f85_state.size==0:
                    # log_print.warning(f"Episode {ep_idx} Frame {frame_idx}: Missing eef state data")
                    return
                if claw_action.size == 0 and qiangnao_action.size==0 and sg100_action.size == 0 and rq2f85_action.size ==0:
                    # log_print.warning(f"Episode {ep_idx} Frame {frame_idx}: Missing eef action data")
                    return
                # =========================
                # 4. 手部归一化（保持原逻辑）
                # =========================
                if kuavo.IS_BINARY:
                    qiangnao_state  = np.where(qiangnao_state > 50, 1, 0)
                    qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
                    _sg100_collect(sg100_check, "state", sg100_state)
                    _sg100_collect(sg100_check, "action", sg100_action)
                    sg100_state     = _sg100_binarize(sg100_state)
                    sg100_action    = _sg100_binarize(sg100_action)
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

                if claw_action.size == 0 and qiangnao_action.size == 0 and sg100_action.size == 0:
                    claw_action = rq2f85_action
                    claw_state  = rq2f85_state

                # =========================
                # 5. 构建最终 state / action
                # =========================
                if kuavo.USE_LEJU_CLAW or kuavo.USE_QIANGNAO or kuavo.USE_SG100:
                    hand_type = "LEJU" if kuavo.USE_LEJU_CLAW else "SG100" if kuavo.USE_SG100 else "QIANGNAO"
                    s_list, a_list = [], []

                    def get_hand_slice(hand_side):
                        s_slice = kuavo.SLICE_ROBOT[hand_side]
                        arm_action_start = hand_side * arm_dof_per_side
                        arm_action_end = arm_action_start + arm_dof_per_side
                        arm_action = arm_traj[arm_action_start:arm_action_end]

                        if hand_type == "LEJU":
                            c_slice = kuavo.SLICE_CLAW[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                claw_state[c_slice[0]:c_slice[-1]]))
                            a = np.concatenate((arm_action,
                                                claw_action[c_slice[0]:c_slice[-1]]))
                        elif hand_type == "QIANGNAO":
                            d_slice = kuavo.SLICE_DEX[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                qiangnao_state[d_slice[0]:d_slice[-1]]))
                            a = np.concatenate((arm_action,
                                                qiangnao_action[d_slice[0]:d_slice[-1]]))
                        else:
                            d_slice = kuavo.SLICE_DEX[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                sg100_state[d_slice[0]:d_slice[-1]]))
                            a = np.concatenate((arm_action,
                                                sg100_action[d_slice[0]:d_slice[-1]]))
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

                # 5W whole-body data is always appended after the selected
                # arm/end-effector layout.  Both sources use radians in the
                # converted dataset.
                if kuavo.USE_5W_WHOLEBODY:
                    lower_action = get_array("action.lb_leg_traj", np.float32)
                    lower_start, lower_end = get_lower_body_joint_slice(platform_type)
                    lower_state = state[lower_start:lower_end]
                    lower_dof = lower_end - lower_start
                    if lower_state.size != lower_dof or lower_action.size != lower_dof:
                        log_print.warning(
                            f"Episode {ep_idx} Frame {frame_idx}: expected {lower_dof} lower-body "
                            f"values, got state={lower_state.size}, action={lower_action.size}"
                        )
                        return
                    final_state = np.concatenate((final_state, lower_state)).astype(np.float32)
                    final_action = np.concatenate((final_action, lower_action)).astype(np.float32)

                # Base velocity is the final action segment regardless of whether
                # the optional 4-DOF lower-body segment is enabled.
                if kuavo.USE_5W_BASE_MOVE:
                    base_velocity = get_array("action.base_velocity", np.float32)
                    if base_velocity.size != len(kuavo.DEFAULT_BASE_VELOCITY_NAMES):
                        log_print.warning(
                            f"Episode {ep_idx} Frame {frame_idx}: expected 3 base velocity "
                            f"values, got {base_velocity.size}"
                        )
                        return
                    final_action = np.concatenate((final_action, base_velocity)).astype(np.float32)

                # =========================
                # 6. 构建 frame
                # =========================
                frame = {
                    "observation.state": torch.from_numpy(final_state).type(torch.float32),
                    "action": torch.from_numpy(final_action).type(torch.float32),
                }

                for cam_key in kuavo.DEFAULT_CAMERA_NAMES:
                    if cam_key in h265_cameras:
                        frame[f"observation.images.{cam_key}"] = _get_h265_placeholder()
                        continue
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

            from kuavo_data.common.chunk_process import ChunkedRosbagProcessor

            processor = ChunkedRosbagProcessor(
                msg_processer=bag_reader._msg_processer,
                topic_process_map=bag_reader._topic_process_map,
                camera_names=kuavo.DEFAULT_CAMERA_NAMES,
                train_hz=kuavo.TRAIN_HZ,
                main_timeline=kuavo.MAIN_TIMELINE,
                main_timeline_fps=kuavo.MAIN_TIMELINE_FPS,
                sample_drop=kuavo.SAMPLE_DROP,
                h265_cameras=h265_cameras,
            )

            h265_encode_future = None
            h265_executor = None
            if h265_contexts:
                _, main_timestamps, all_timestamps = processor.scan_timestamps_only(str(ep_path))
                camera_ts = build_camera_timestamp_sources(
                    kuavo.DEFAULT_CAMERA_NAMES,
                    h265_cameras,
                    h265_contexts,
                    all_timestamps,
                )
                main_timestamps, valid_start, valid_end, trim_dropped = trim_main_timestamps_strict(
                    main_timestamps, camera_ts
                )
                if trim_dropped > 0:
                    log_print.info(
                        f"Strict main timeline trim: dropped {trim_dropped} frames, "
                        f"valid window [{valid_start:.6f}, {valid_end:.6f}], "
                        f"{len(main_timestamps)} frames remain"
                    )
                h265_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                h265_encode_future = h265_executor.submit(
                    _encode_h265_videos, h265_contexts, main_timestamps
                )
                processor.process_in_chunks(
                    bag_file=str(ep_path),
                    main_timestamps=main_timestamps,
                    all_timestamps=all_timestamps,
                    frame_callback=on_frame,
                    chunk_size=chunk_size,
                    save_callback=on_chunk_done,
                )
            else:
                bag_reader.process_rosbag_chunked(
                    bag_file=str(ep_path),
                    frame_callback=on_frame,
                    chunk_size=chunk_size,
                    save_callback=on_chunk_done,
                )

            # 处理剩余的帧
            if len(frames_buffer) > 0:
                for frame in frames_buffer:
                    frame["task"] = task
                    dataset.add_frame(frame)

            if h265_contexts and h265_encode_future is not None:
                h265_mp4_paths, h265_video_stats = h265_encode_future.result()
                if h265_executor is not None:
                    h265_executor.shutdown(wait=False)
                save_episode_with_h265_videos(dataset, h265_mp4_paths, h265_video_stats)
            else:
                dataset.save_episode()

            for ctx in h265_contexts.values():
                ctx.cleanup()
            if h265_temp_root and h265_temp_root.exists():
                shutil.rmtree(h265_temp_root, ignore_errors=True)
            frames_buffer.clear()
            gc.collect()
            
            log_print.info(f"Episode {ep_idx} completed: {frame_count[0]} frames")

            if kuavo.USE_SG100 and kuavo.IS_BINARY:
                _sg100_check(sg100_check, ep_idx)

            bag_elapsed = time.perf_counter() - bag_t0
            timing = BagConversionTiming(
                episode_index=ep_idx,
                bag_path=str(ep_path),
                codec=_bag_codec_label(h265_cameras, bag_reader.camera_encodings),
                frames=frame_count[0],
                wall_time_sec=bag_elapsed,
                train_hz=kuavo.TRAIN_HZ,
            )
            conversion_timings.append(timing)
            _log_bag_conversion_timing(timing)
            
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

    _log_conversion_timing_summary(conversion_timings)
    
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

    arm_start, arm_end = get_arm_joint_slice(kuavo.PLATFORM_TYPE)
    arm_dof = arm_end - arm_start
    arm_dof_per_side = arm_dof // 2
    if len(kuavo.DEFAULT_ARM_JOINT_NAMES) == arm_dof:
        arm_names = kuavo.DEFAULT_ARM_JOINT_NAMES
    else:
        arm_names = [
            f"{side}_arm_joint_{joint_index}"
            for side in ("left", "right")
            for joint_index in range(arm_dof_per_side)
        ]

    eef_names = (kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES if kuavo.USE_LEJU_CLAW
                 else kuavo.DEFAULT_SG100_JOINT_NAMES if kuavo.USE_SG100
                 else kuavo.DEFAULT_DEXHAND_JOINT_NAMES)
    eef_slices = kuavo.SLICE_CLAW if kuavo.USE_LEJU_CLAW else kuavo.SLICE_DEX
    # SG100 1dof 二值化使用统一的开合标量名称
    binary_grip = kuavo.USE_SG100 and kuavo.IS_BINARY and kuavo.DEX_DOF_NEEDED == 1
    DEFAULT_JOINT_NAMES_LIST = []
    selected_sides = (0, 1) if kuavo.CONTROL_HAND_SIDE == "both" else ((0,) if kuavo.CONTROL_HAND_SIDE == "left" else (1,))
    for side_index in selected_sides:
        arm_name_start = side_index * arm_dof_per_side
        DEFAULT_JOINT_NAMES_LIST.extend(arm_names[arm_name_start:arm_name_start + arm_dof_per_side])
        if binary_grip:
            DEFAULT_JOINT_NAMES_LIST.append(("left", "right")[side_index] + "_grip")
        else:
            eef_start, eef_end = eef_slices[side_index]
            DEFAULT_JOINT_NAMES_LIST.extend(eef_names[eef_start:eef_end])
    if kuavo.USE_5W_WHOLEBODY:
        lower_start, lower_end = get_lower_body_joint_slice(kuavo.PLATFORM_TYPE)
        lower_names = [
            f"{kuavo.DEFAULT_LOWER_BODY_JOINT_NAME_PREFIX}_{index}"
            for index in range(lower_start, lower_end)
        ]
        DEFAULT_JOINT_NAMES_LIST.extend(lower_names)

    use_parallel, parallel_workers, keep_shards, writer_threads = parallel_options(cfg)

    if mode == "normal":
        if raw_dir is None:
            raise ValueError("normal mode requires `rosbag.rosbag_dir`.")
        if raw_dir.resolve() == output_dir.resolve():
            raise ValueError("rosbag_dir and target_dir/lerobot cannot be the same.")
        task_name = raw_dir.name
        repo_id = f"lerobot/{task_name}"
        output_dir = ensure_clean_lerobot_output_dir(target_dir)
        ran_parallel = False
        if use_parallel and parallel_workers > 1:
            bag_files = _select_kuavo_bag_files(raw_dir, n)
            if len(bag_files) > 1:
                log_print.info("Parallel conversion: workers=%s", min(parallel_workers, len(bag_files)))
                run_parallel_conversion(
                    kind="kuavo",
                    cfg=cfg,
                    bag_files=bag_files,
                    target_dir=target_dir,
                    output_dir=output_dir,
                    chunk_size=chunk_size,
                    workers=parallel_workers,
                    keep_shards=keep_shards,
                    image_writer_threads=writer_threads,
                )
                ran_parallel = True
            else:
                log_print.info("Falling back to sequential conversion (need >1 bag).")
        elif use_parallel:
            log_print.info("Falling back to sequential conversion (need parallel_workers>1).")
        if not ran_parallel:
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


def _select_kuavo_bag_files(raw_dir: Path, n: int | None) -> list[Path]:
    bag_reader = kuavo.KuavoRosbagReader()
    bag_files = [Path(path) for path in bag_reader.list_bag_files(raw_dir)]
    if not bag_files:
        log_print.error(
            "No .bag files found in rosbag_dir(s). Check that paths exist and contain *.bag. "
            f"Looked in: {raw_dir}"
        )
        raise FileNotFoundError(f"No .bag files in: {raw_dir}")
    if isinstance(n, int) and n > 0:
        num_available_bags = len(bag_files)
        if n > num_available_bags:
            log_print.warning(
                "Requested %s bags, but only %s available. Using all available bags.",
                n,
                num_available_bags,
            )
            n = num_available_bags
        select_idx = np.random.choice(num_available_bags, n, replace=False)
        bag_files = [bag_files[index] for index in select_idx]
    return bag_files


if __name__ == "__main__":
    np.random.seed(42)
    main()
