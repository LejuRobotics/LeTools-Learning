import argparse
import contextlib
import json
import os
import shutil
import traceback

import numpy as np
import pandas as pd
from termcolor import colored


def load_jsonl(file_path):
    """
    Load data from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: A list containing JSON objects parsed from each line.
    """
    data = []

    # Special handling for episodes_stats.jsonl
    if "episodes_stats.jsonl" in file_path:
        try:
            # Try to load the entire file as a JSON array
            with open(file_path) as f:
                content = f.read()
                if content.strip().startswith("[") and content.strip().endswith("]"):
                    return json.loads(content)
                else:
                    try:
                        return json.loads("[" + content + "]")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error loading {file_path} as JSON array: {e}")

        # Fall back to line-by-line parsing
        try:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        with contextlib.suppress(json.JSONDecodeError):
                            data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file_path} line by line: {e}")
    else:
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    with contextlib.suppress(json.JSONDecodeError):
                        data.append(json.loads(line))

    return data


def save_jsonl(data, file_path):
    """
    Save data in JSONL format.

    Args:
        data (list): List of JSON objects to save.
        file_path (str): Output file path.
    """
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def merge_stats(stats_list):
    """
    Merge stats only when all feature/stat shapes are consistent.
    If shapes are inconsistent, raise an error instead of padding.
    """
    merged_stats = {}

    if not stats_list:
        return merged_stats

    common_features = set(stats_list[0].keys())
    for stats in stats_list[1:]:
        common_features = common_features.intersection(set(stats.keys()))

    for feature in stats_list[0]:
        if feature not in common_features:
            continue

        merged_stats[feature] = {}

        common_stat_types = []
        for stat_type in ["mean", "std", "max", "min"]:
            if all(stat_type in stats[feature] for stats in stats_list):
                common_stat_types.append(stat_type)

        # Check shape consistency
        for stat_type in common_stat_types:
            shapes = {
                tuple(np.array(stats[feature][stat_type]).shape)
                for stats in stats_list
            }
            if len(shapes) > 1:
                raise ValueError(
                    f"Inconsistent shapes for feature '{feature}', stat '{stat_type}': {shapes}"
                )

        # Merge values directly with numpy, without padding
        for stat_type in common_stat_types:
            values = [np.array(stats[feature][stat_type]) for stats in stats_list]

            try:
                if stat_type == "mean":
                    if all("count" in stats[feature] for stats in stats_list):
                        counts = [stats[feature]["count"][0] for stats in stats_list]
                        total_count = sum(counts)
                        weighted_values = [
                            val * count / total_count
                            for val, count in zip(values, counts, strict=False)
                        ]
                        merged = np.sum(weighted_values, axis=0)
                    else:
                        merged = np.mean(values, axis=0)

                elif stat_type == "std":
                    if all("count" in stats[feature] for stats in stats_list):
                        counts = [stats[feature]["count"][0] for stats in stats_list]
                        total_count = sum(counts)
                        variances = [val**2 for val in values]
                        weighted_variances = [
                            var * count / total_count
                            for var, count in zip(variances, counts, strict=False)
                        ]
                        merged = np.sqrt(np.sum(weighted_variances, axis=0))
                    else:
                        merged = np.mean(values, axis=0)

                elif stat_type == "max":
                    merged = np.maximum.reduce(values)

                elif stat_type == "min":
                    merged = np.minimum.reduce(values)

                merged_stats[feature][stat_type] = merged.tolist()

            except Exception as e:
                print(f"Warning: Error processing {feature}.{stat_type}: {e}")
                raise

        if all("count" in stats[feature] for stats in stats_list):
            try:
                merged_stats[feature]["count"] = [
                    sum(stats[feature]["count"][0] for stats in stats_list)
                ]
            except Exception as e:
                print(f"Warning: Error processing {feature}.count: {e}")
                raise

    return merged_stats


def validate_info_features(source_folders):
    """
    Validate that all source datasets have the same feature keys and shapes
    in meta/info.json["features"].

    Returns:
        dict: The reference features dictionary from the first dataset.
    """
    reference_folder = source_folders[0]
    reference_info_path = os.path.join(reference_folder, "meta", "info.json")

    if not os.path.exists(reference_info_path):
        raise FileNotFoundError(f"Missing info.json in {reference_folder}")

    with open(reference_info_path) as f:
        reference_info = json.load(f)

    if "features" not in reference_info:
        raise KeyError(f'"features" not found in {reference_info_path}')

    reference_features = reference_info["features"]
    reference_keys = set(reference_features.keys())

    for folder in source_folders[1:]:
        info_path = os.path.join(folder, "meta", "info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Missing info.json in {folder}")

        with open(info_path) as f:
            info = json.load(f)

        if "features" not in info:
            raise KeyError(f'"features" not found in {info_path}')

        features = info["features"]
        keys = set(features.keys())

        if keys != reference_keys:
            missing_in_current = sorted(reference_keys - keys)
            extra_in_current = sorted(keys - reference_keys)
            raise ValueError(
                f"Feature keys mismatch in {folder}. "
                f"Missing keys: {missing_in_current}. "
                f"Extra keys: {extra_in_current}."
            )

        for key in sorted(reference_keys):
            ref_shape = reference_features[key].get("shape", None)
            cur_shape = features[key].get("shape", None)
            if ref_shape != cur_shape:
                raise ValueError(
                    f'Feature shape mismatch for key "{key}" in {folder}: '
                    f"expected {ref_shape}, got {cur_shape}"
                )

    return reference_features


def copy_videos(source_folders, output_folder, episode_mapping):
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    video_path_template = info["video_path"]

    video_keys = []
    for feature_name, feature_info in info["features"].items():
        if feature_info.get("dtype") == "video":
            video_keys.append(feature_name)

    print(f"Found video keys: {video_keys}")

    for old_folder, old_index, new_index in episode_mapping:
        episode_chunk = old_index // info["chunks_size"]
        new_episode_chunk = new_index // info["chunks_size"]

        for video_key in video_keys:
            source_patterns = [
                os.path.join(
                    old_folder,
                    video_path_template.format(
                        episode_chunk=episode_chunk,
                        video_key=video_key,
                        episode_index=old_index,
                    ),
                ),
                os.path.join(
                    old_folder,
                    video_path_template.format(
                        episode_chunk=0,
                        video_key=video_key,
                        episode_index=0,
                    ),
                ),
                os.path.join(
                    old_folder,
                    f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{old_index}.mp4",
                ),
                os.path.join(
                    old_folder,
                    f"videos/chunk-000/{video_key}/episode_000000.mp4",
                ),
            ]

            source_video_path = None
            for pattern in source_patterns:
                if os.path.exists(pattern):
                    source_video_path = pattern
                    break

            if source_video_path:
                dest_video_path = os.path.join(
                    output_folder,
                    video_path_template.format(
                        episode_chunk=new_episode_chunk,
                        video_key=video_key,
                        episode_index=new_index,
                    ),
                )
                os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                print(f"Copying video: {source_video_path} -> {dest_video_path}")
                shutil.copy2(source_video_path, dest_video_path)
            else:
                found = False
                for root, _, files in os.walk(os.path.join(old_folder, "videos")):
                    for file in files:
                        if file.endswith(".mp4") and video_key in root:
                            source_video_path = os.path.join(root, file)

                            dest_video_path = os.path.join(
                                output_folder,
                                video_path_template.format(
                                    episode_chunk=new_episode_chunk,
                                    video_key=video_key,
                                    episode_index=new_index,
                                ),
                            )

                            os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                            print(
                                f"Copying video (found by search): "
                                f"{source_video_path} -> {dest_video_path}"
                            )
                            shutil.copy2(source_video_path, dest_video_path)
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print(
                        f"Warning: Video file not found for {video_key}, "
                        f"episode {old_index} in {old_folder}"
                    )


def validate_timestamps(source_folders, tolerance_s=1e-4):
    issues = []
    fps_values = []

    for folder in source_folders:
        try:
            info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                    if "fps" in info:
                        fps = info["fps"]
                        fps_values.append(fps)
                        print(f"Dataset {folder} FPS={fps}")

            parquet_path = None
            for root, _, files in os.walk(os.path.join(folder, "parquet")):
                for file in files:
                    if file.endswith(".parquet"):
                        parquet_path = os.path.join(root, file)
                        break
                if parquet_path:
                    break

            if not parquet_path:
                for root, _, files in os.walk(os.path.join(folder, "data")):
                    for file in files:
                        if file.endswith(".parquet"):
                            parquet_path = os.path.join(root, file)
                            break
                    if parquet_path:
                        break

            if parquet_path:
                df = pd.read_parquet(parquet_path)
                timestamp_cols = [col for col in df.columns if "timestamp" in col or "time" in col]
                if timestamp_cols:
                    print(f"Dataset {folder} contains timestamp columns: {timestamp_cols}")
                else:
                    issues.append(
                        f"Warning: Dataset {folder} has no timestamp columns"
                    )
            else:
                issues.append(
                    f"Warning: No parquet files found in dataset {folder}"
                )

        except Exception as e:
            issues.append(
                f"Error: Failed to validate dataset {folder}: {e}"
            )
            print(f"Validation error: {e}")
            traceback.print_exc()

    if len(set(fps_values)) > 1:
        issues.append(
            f"Warning: Inconsistent FPS across datasets: {fps_values}"
        )

    return issues, fps_values


def process_single_parquet(
    source_path,
    old_folder,
    old_index,
    new_index,
    output_folder,
    episode_to_frame_index=None,
    folder_task_mapping=None,
    chunks_size=1000,
):
    df = pd.read_parquet(source_path)

    if "episode_index" in df.columns:
        print(
            f"Updating episode_index from {df['episode_index'].iloc[0]} to {new_index}"
        )
        df["episode_index"] = new_index

    if "index" in df.columns:
        if episode_to_frame_index and new_index in episode_to_frame_index:
            first_index = episode_to_frame_index[new_index]
            print(
                f"Updating index column, start value: {first_index} "
                f"(using global cumulative frame count)"
            )
        else:
            first_index = new_index * len(df)
            print(
                f"Updating index column, start value: {first_index} "
                f"(using episode index multiplied by length)"
            )

        df["index"] = [first_index + i for i in range(len(df))]

    if "task_index" in df.columns and folder_task_mapping and old_folder in folder_task_mapping:
        current_task_index = df["task_index"].iloc[0]

        if current_task_index in folder_task_mapping[old_folder]:
            new_task_index = folder_task_mapping[old_folder][current_task_index]
            print(
                f"Updating task_index from {current_task_index} to {new_task_index}"
            )
            df["task_index"] = new_task_index
        else:
            print(
                f"Warning: No mapping found for task_index {current_task_index}"
            )

    chunk_index = new_index // chunks_size
    chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
    os.makedirs(chunk_dir, exist_ok=True)

    dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")
    df.to_parquet(dest_path, index=False)

    print(f"Processed and saved: {dest_path}")
    return True


def copy_data_files(
    source_folders,
    output_folder,
    episode_mapping,
    fps=None,
    episode_to_frame_index=None,
    folder_task_mapping=None,
    chunks_size=1000,
    default_fps=20,
):
    if fps is None:
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                fps = info.get("fps", default_fps)
        else:
            fps = default_fps

    print(f"Using FPS={fps}")

    total_copied = 0
    total_failed = 0
    failed_files = []

    for old_folder, old_index, new_index in episode_mapping:
        episode_str = f"episode_{old_index:06d}.parquet"
        source_paths = [
            os.path.join(old_folder, "parquet", episode_str),
            os.path.join(old_folder, "data", episode_str),
        ]

        source_path = None
        for path in source_paths:
            if os.path.exists(path):
                source_path = path
                break

        if source_path:
            try:
                process_single_parquet(
                    source_path=source_path,
                    old_folder=old_folder,
                    old_index=old_index,
                    new_index=new_index,
                    output_folder=output_folder,
                    episode_to_frame_index=episode_to_frame_index,
                    folder_task_mapping=folder_task_mapping,
                    chunks_size=chunks_size,
                )
                total_copied += 1

            except Exception as e:
                error_msg = f"Processing {source_path} failed: {e}"
                print(error_msg)
                traceback.print_exc()
                failed_files.append({"file": source_path, "reason": str(e), "episode": old_index})
                total_failed += 1
        else:
            found = False
            for root, _, files in os.walk(old_folder):
                for file in files:
                    if file.endswith(".parquet") and f"episode_{old_index:06d}" in file:
                        try:
                            source_path = os.path.join(root, file)

                            process_single_parquet(
                                source_path=source_path,
                                old_folder=old_folder,
                                old_index=old_index,
                                new_index=new_index,
                                output_folder=output_folder,
                                episode_to_frame_index=episode_to_frame_index,
                                folder_task_mapping=folder_task_mapping,
                                chunks_size=chunks_size,
                            )

                            total_copied += 1
                            found = True
                            break

                        except Exception as e:
                            error_msg = f"Processing {source_path} failed: {e}"
                            print(error_msg)
                            traceback.print_exc()
                            failed_files.append({"file": source_path, "reason": str(e), "episode": old_index})
                            total_failed += 1
                if found:
                    break

            if not found:
                error_msg = f"Could not find parquet file for episode {old_index}, source folder: {old_folder}"
                print(error_msg)
                failed_files.append(
                    {
                        "file": f"episode_{old_index:06d}.parquet",
                        "reason": "File not found",
                        "folder": old_folder,
                    }
                )
                total_failed += 1

    print(f"Copied {total_copied} data files, {total_failed} failed")

    if failed_files:
        print("\nFailed file details:")
        for i, failed in enumerate(failed_files):
            print(f"{i + 1}. File: {failed['file']}")
            if "folder" in failed:
                print(f"   Folder: {failed['folder']}")
            if "episode" in failed:
                print(f"   Episode index: {failed['episode']}")
            print(f"   Reason: {failed['reason']}")
            print("---")

    return total_copied > 0


def count_video_frames_torchvision(video_path):
    """
    Count the number of frames in a video file using torchvision.

    Args:
        video_path (str): Path to the video file.

    Returns:
        int: Frame count.
    """
    try:
        import torchvision

        reader = torchvision.io.VideoReader(video_path, "video")
        metadata = reader.get_metadata()
        frame_count = 0

        if "video" in metadata and "num_frames" in metadata["video"] and len(metadata["video"]["num_frames"]) > 0:
            frame_count = int(metadata["video"]["num_frames"][0])
            if frame_count > 0:
                return frame_count

        count_manually = 0
        for _ in reader:
            count_manually += 1

        if count_manually > 0:
            return count_manually
        elif frame_count > 0:
            print(
                f"Warning: Manual count is 0, but metadata indicates {frame_count} frames. "
                f"Video might be empty or there was a read issue. Returning metadata count."
            )
            return frame_count
        else:
            print(f"Video appears to have no frames: {video_path}")
            return 0

    except ImportError:
        print("Warning: torchvision or its dependencies (like ffmpeg) not installed, cannot count video frames")
        return 0
    except RuntimeError as e:
        if "No video stream found" in str(e):
            print(f"Error: No video stream found in video file: {video_path}")
        elif "Could not open" in str(e) or "Demuxing video" in str(e):
            print(
                f"Error: Could not open or demux video file "
                f"(possibly unsupported format or corrupted file): {video_path} - {e}"
            )
        else:
            print(f"Runtime error counting video frames: {e}")
        return 0
    except Exception as e:
        print(f"Error counting video frames: {e}")
        return 0


def early_validation(source_folders, episode_mapping, default_fps=20, fps=None):
    """
    Validate images and videos in source folders before copying,
    to ensure dataset consistency.
    """
    if fps is None:
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                fps = info.get("fps", default_fps)
        else:
            fps = default_fps

    print(f"Using FPS={fps}")

    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    video_path_template = info["video_path"]
    image_keys = []

    for feature_name, feature_info in info["features"].items():
        if feature_info.get("dtype") == "video":
            image_keys.append(feature_name)

    print(f"Found video/image keys: {image_keys}")

    print("Starting validation of images and videos...")
    validation_results = {}
    validation_failed = False

    episode_file_mapping = {}
    for old_folder, old_index, new_index in episode_mapping:
        episode_file = os.path.join(old_folder, "meta", "episodes.jsonl")
        expected_frames = 0
        if os.path.exists(episode_file):
            if episode_file not in episode_file_mapping:
                episodes = load_jsonl(episode_file)
                episodes = {ep["episode_index"]: ep for ep in episodes}
                episode_file_mapping[episode_file] = episodes
            episode_data = episode_file_mapping[episode_file].get(old_index, None)
            if episode_data and "length" in episode_data:
                expected_frames = episode_data["length"]

        validation_key = f"{old_folder}_{old_index}"
        validation_results[validation_key] = {
            "expected_frames": expected_frames,
            "image_counts": {},
            "video_frames": {},
            "old_index": old_index,
            "new_index": new_index,
            "is_valid": True,
        }

        episode_chunk = old_index // info["chunks_size"]
        for image_dir in image_keys:
            source_video_path = os.path.join(
                old_folder,
                video_path_template.format(
                    episode_chunk=episode_chunk,
                    video_key=image_dir,
                    episode_index=old_index,
                ),
            )
            source_image_dir = os.path.join(old_folder, "images", image_dir, f"episode_{old_index:06d}")
            image_dir_exists = os.path.exists(source_image_dir)
            video_file_exists = os.path.exists(source_video_path)

            if not video_file_exists:
                print(
                    f"{colored('WARNING', 'yellow', attrs=['bold'])}: "
                    f"Video file not found for {image_dir}, episode {old_index} in {old_folder}"
                )
                if image_dir_exists:
                    print("  Image directory exists, encoding video from images.")
                    from lerobot.datasets.video_utils import encode_video_frames
                    encode_video_frames(source_image_dir, source_video_path, fps, overwrite=True)
                    print("  Encoded video frames successfully.")
                else:
                    print(
                        f"{colored('ERROR', 'red', attrs=['bold'])}: "
                        f"No video or image directory found for {image_dir}, "
                        f"episode {old_index} in {old_folder}"
                    )
                    validation_results[validation_key]["is_valid"] = False
                    validation_failed = True
                    continue

            video_frame_count = count_video_frames_torchvision(source_video_path)
            validation_results[validation_key]["video_frames"][image_dir] = video_frame_count

            if image_dir_exists:
                image_files = sorted([f for f in os.listdir(source_image_dir) if f.endswith(".png")])
                images_count = len(image_files)
                validation_results[validation_key]["image_counts"][image_dir] = images_count

                error_msg = (
                    f"expected_frames: {expected_frames}, "
                    f"images_count: {images_count}, "
                    f"video_frame_count: {video_frame_count}"
                )
                assert expected_frames > 0 and expected_frames == images_count, (
                    f"{colored('ERROR', 'red', attrs=['bold'])}: "
                    f"Image count should match expected frames for {source_image_dir}.\n  {error_msg}"
                )
                assert expected_frames >= video_frame_count, (
                    f"{colored('ERROR', 'red', attrs=['bold'])}: "
                    f"Video frame count should be less or equal than expected frames for {source_video_path}.\n  {error_msg}"
                )

                if video_frame_count != expected_frames:
                    print(
                        f"{colored('WARNING', 'yellow', attrs=['bold'])}: "
                        f"Video frame count mismatch for {source_video_path}"
                    )
                    print(f"  Expected: {expected_frames}, Found: {video_frame_count}")
                    print(f"  Re-encoded video frames from {source_image_dir} to {source_video_path}")

                    from lerobot.datasets.video_utils import encode_video_frames
                    encode_video_frames(source_image_dir, source_video_path, fps, overwrite=True)
                    print("  Re-encoded video frames successfully.")
            else:
                print(
                    f"{colored('WARNING', 'yellow', attrs=['bold'])}: "
                    f"No image directory {image_dir} found for episode {old_index} in {old_folder}"
                )
                print("  You can ignore this if you are not using images and your video frame count is equal to expected frames.")

                if expected_frames > 0 and video_frame_count != expected_frames:
                    print(
                        f"{colored('ERROR', 'red', attrs=['bold'])}: "
                        f"Video frame count mismatch for {source_video_path}"
                    )
                    print(f"  Expected: {expected_frames}, Found: {video_frame_count}")

                    validation_results[validation_key]["is_valid"] = False
                    validation_failed = True

    print("\nValidation Results:")
    valid_count = sum(1 for result in validation_results.values() if result["is_valid"])
    print(f"{valid_count} of {len(validation_results)} episodes are valid")

    if validation_failed:
        print(colored("Validation failed. Please fix the issues before continuing.", "red", attrs=["bold"]))


def copy_images(source_folders, output_folder, episode_mapping, default_fps=20, fps=None):
    """
    Copy image files from source folders to the output folder.
    This function assumes validation has already been performed by early_validation().
    """
    if fps is None:
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                fps = info.get("fps", default_fps)
        else:
            fps = default_fps

    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    image_keys = []
    for feature_name, feature_info in info["features"].items():
        if feature_info.get("dtype") == "video":
            image_keys.append(feature_name)

    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)

    print(f"Starting to copy images for {len(image_keys)} video keys...")
    total_copied = 0
    skipped_episodes = 0

    for old_folder, old_index, new_index in episode_mapping:
        episode_copied = False

        for image_dir in image_keys:
            os.makedirs(os.path.join(output_folder, "images", image_dir), exist_ok=True)

            source_image_dir = os.path.join(old_folder, "images", image_dir, f"episode_{old_index:06d}")

            if os.path.exists(source_image_dir):
                target_image_dir = os.path.join(output_folder, "images", image_dir, f"episode_{new_index:06d}")
                os.makedirs(target_image_dir, exist_ok=True)

                image_files = sorted([f for f in os.listdir(source_image_dir) if f.endswith(".png")])
                num_images = len(image_files)

                if num_images > 0:
                    print(f"Copying {num_images} images from {source_image_dir} to {target_image_dir}")

                    for image_file in image_files:
                        try:
                            frame_part = image_file.split("_")[1] if "_" in image_file else image_file
                            frame_num = int(frame_part.split(".")[0])

                            dest_file = os.path.join(target_image_dir, f"frame_{frame_num:06d}.png")
                            shutil.copy2(
                                os.path.join(source_image_dir, image_file),
                                dest_file,
                            )
                            total_copied += 1
                            episode_copied = True
                        except Exception as e:
                            print(f"Error copying image {image_file}: {e}")

        if not episode_copied:
            skipped_episodes += 1

    print(f"\nCopied {total_copied} images for {len(episode_mapping) - skipped_episodes} episodes")
    if skipped_episodes > 0:
        print(f"{colored('WARNING', 'yellow', attrs=['bold'])}: Skipped {skipped_episodes} episodes with no images")


def merge_datasets(
    source_folders,
    output_folder,
    validate_ts=False,
    tolerance_s=1e-4,
    default_fps=20,
    copy_images_flag=False,
):
    """
    Merge multiple dataset folders into one, handling indices and metadata.

    No state/action padding is performed.
    Feature consistency is validated using meta/info.json["features"] keys and shapes.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    fps = default_fps
    print(f"Using default FPS value: {fps}")

    all_episodes = []
    all_episodes_stats = []
    all_tasks = []

    total_frames = 0
    total_episodes = 0
    episode_mapping = []
    total_videos = 0

    cumulative_frame_count = 0
    episode_to_frame_index = {}

    task_desc_to_new_index = {}
    folder_task_mapping = {}
    all_unique_tasks = []

    info_path = os.path.join(source_folders[0], "meta", "info.json")
    images_dir_exists = all(os.path.exists(os.path.join(folder, "images")) for folder in source_folders)

    chunks_size = 1000
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
            chunks_size = info.get("chunks_size", 1000)

    reference_features = validate_info_features(source_folders)
    print("Validated feature keys and shapes from info.json across all source datasets")

    if validate_ts:
        issues, fps_values = validate_timestamps(source_folders, tolerance_s=tolerance_s)
        if issues:
            print("\nTimestamp validation issues:")
            for issue in issues:
                print(f"- {issue}")
        if fps_values:
            unique_fps = set(fps_values)
            if len(unique_fps) == 1:
                fps = fps_values[0]

    for folder in source_folders:
        try:
            folder_info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(folder_info_path):
                with open(folder_info_path) as f:
                    folder_info = json.load(f)
                    if "total_videos" in folder_info:
                        folder_videos = folder_info["total_videos"]
                        total_videos += folder_videos
                        print(
                            f"Read video count from {folder}'s info.json: {folder_videos}"
                        )

            episodes_path = os.path.join(folder, "meta", "episodes.jsonl")
            if not os.path.exists(episodes_path):
                print(f"Warning: Episodes file not found in {folder}, skipping")
                continue

            episodes = load_jsonl(episodes_path)

            episodes_stats_path = os.path.join(folder, "meta", "episodes_stats.jsonl")
            episodes_stats = []
            if os.path.exists(episodes_stats_path):
                episodes_stats = load_jsonl(episodes_stats_path)

            stats_map = {}
            for stat in episodes_stats:
                if "episode_index" in stat:
                    stats_map[stat["episode_index"]] = stat

            tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
            folder_tasks = []
            if os.path.exists(tasks_path):
                folder_tasks = load_jsonl(tasks_path)

            folder_task_mapping[folder] = {}

            for task in folder_tasks:
                task_desc = task["task"]
                old_index = task["task_index"]

                if task_desc not in task_desc_to_new_index:
                    new_task_index = len(all_unique_tasks)
                    task_desc_to_new_index[task_desc] = new_task_index
                    all_unique_tasks.append({"task_index": new_task_index, "task": task_desc})

                folder_task_mapping[folder][old_index] = task_desc_to_new_index[task_desc]

            for episode in episodes:
                old_index = episode["episode_index"]
                new_index = total_episodes

                episode["episode_index"] = new_index
                all_episodes.append(episode)

                if old_index in stats_map:
                    stats = stats_map[old_index]
                    stats["episode_index"] = new_index
                    all_episodes_stats.append(stats)

                episode_mapping.append((folder, old_index, new_index))

                total_episodes += 1
                total_frames += episode["length"]

                episode_to_frame_index[new_index] = cumulative_frame_count
                cumulative_frame_count += episode["length"]

            all_tasks = all_unique_tasks

        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

    print(f"Processed {total_episodes} episodes from {len(source_folders)} folders")

    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(all_episodes_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))
    save_jsonl(all_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))

    stats_list = []
    for folder in source_folders:
        stats_path = os.path.join(folder, "meta", "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
                stats_list.append(stats)

    if stats_list:
        merged_stats = merge_stats(stats_list)
        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)

    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(all_tasks)
    info["total_chunks"] = (total_episodes + info["chunks_size"] - 1) // info["chunks_size"]
    info["splits"] = {"train": f"0:{total_episodes}"}

    if "features" in info:
        info["features"] = reference_features
        print("Updated info.json features from the validated reference dataset")

    info["total_videos"] = total_videos
    print(f"Updated total videos to: {total_videos}")

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    if images_dir_exists:
        early_validation(
            source_folders,
            episode_mapping,
            fps=fps,
        )

    copy_videos(source_folders, output_folder, episode_mapping)
    copy_data_files(
        source_folders,
        output_folder,
        episode_mapping,
        fps=fps,
        episode_to_frame_index=episode_to_frame_index,
        folder_task_mapping=folder_task_mapping,
        chunks_size=chunks_size,
    )

    if copy_images_flag:
        print("Starting to copy images and validate video frame counts")
        copy_images(source_folders, output_folder, episode_mapping, fps=fps)

    print(f"Merged {total_episodes} episodes with {total_frames} frames into {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge datasets from multiple sources.")

    parser.add_argument(
        "--sources",
        type=str,
        required=True,
        help="Comma-separated list of source folder paths",
    )
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--fps", type=int, default=20, help="Your datasets FPS (default: 20)")
    parser.add_argument("--copy_images", action="store_true", help="Whether to copy images (default: False)")
    parser.add_argument("--validate_ts", action="store_true", help="Whether to validate timestamps")
    parser.add_argument("--tolerance_s", type=float, default=1e-4, help="Timestamp tolerance in seconds")

    args = parser.parse_args()

    source_folders = [s.strip() for s in args.sources.split(",") if s.strip()]

    merge_datasets(
        source_folders,
        args.output,
        validate_ts=args.validate_ts,
        tolerance_s=args.tolerance_s,
        default_fps=args.fps,
        copy_images_flag=args.copy_images,
    )