# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Callable, Dict, List, Literal, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer
import json
import datasets
from collections.abc import Callable
from pathlib import Path

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from .utils import FeatureTransform


class LeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        load_image: bool = True,
        **kwargs,
    ):
        super().__init__(repo_id, **kwargs)
        self.load_image = load_image

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        Query dataset for indices across keys, skipping video keys.

        Tries column-first [key][indices] for speed, falls back to row-first.

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            # Map absolute indices to relative indices if needed
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset.select(relative_indices)[key])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def __getitem__(self, idx) -> dict:
        # Ensure dataset is loaded when we actually need to read from it
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
            
        if len(self.meta.video_keys) > 0 and self.load_image:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None and self.load_image:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name

        return item
        

class VLADataset(Dataset):
    def __init__(
        self,
        repo_id,
        data_name,
        data_config,
        robot_config_root,
        config=PI0Config,
        tokenizer=AutoTokenizer,
        image_processor=None,
        video_backend = 'torchcodec',
        chunk_size = 50,
        image_size = (224, 224),
        do_nomalize = True,
        use_depth_align = False,
    ):

        self.image_processor = image_processor
        self.config = config
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.data_name = data_name

        if do_nomalize:
            data_config.max_state_dim = self.config.max_state_dim
            data_config.max_action_dim = self.config.max_action_dim
            data_config.resize_imgs_with_padding = self.config.resize_imgs_with_padding
            data_config.tokenizer_max_length = self.config.tokenizer_max_length
        
        load_image = True if do_nomalize else False
        robot_config = os.path.join(robot_config_root, f'{data_name}.yaml')
        self.feature_transform = FeatureTransform(robot_config, data_config, \
                    tokenizer, image_processor, do_nomalize, \
                    chunk_size=chunk_size,use_depth_align=use_depth_align,
                    norm_stats_path=data_config.norm_stats_file,
                    load_image=load_image)

        self.action_features = self.feature_transform.actions
        self.state_features = self.feature_transform.states
        self.image_features = self.feature_transform.images
        
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=Resize(image_size),
            delta_timestamps=self.get_delta_timestamps(),
            load_image=load_image
        )
        
        self.task_mapping = dict(zip(self.dataset_meta.tasks['task_index'], self.dataset_meta.tasks.index))

    def __len__(self):
        return len(self.dataset)

    def get_features(self):
        features = set()
        for feature_category, _features in self.feature_transform.org_features.items():
            features.update(_features)
        features.update(self.feature_transform.feature_to_keep)
        features = [x for x in list(features) if x not in ['action_is_pad', 'task']]
        return features

    def get_delta_timestamps(self):
        delta_timestamps = {}
        for action_feature in self.feature_transform.org_features['actions']:
                delta_timestamps[action_feature] = [t / self.dataset_meta.fps for t in range(self.chunk_size)]
        return delta_timestamps

    def getdata(self, idx):
        item = self.dataset[idx]
        item = self.feature_transform.apply(item)
        return item

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                print(f'Error occurred while getting data {cur}: {str(e)}')
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )