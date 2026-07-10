# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import math
import random
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Union

from torch.utils.data import IterableDataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .batching_strategy import TextBatchingStrategy
from .data_collator import (
    CollatePipeline,
    DataCollatorWithPacking,
    DataCollatorWithPadding,
    DataCollatorWithPositionIDs,
    MakeMicroBatchCollator,
    TextSequenceShardCollator,
    UnpackDataCollator,
)
from .dynamic_batching import DynamicBatchSizeDataLoader


if TYPE_CHECKING:
    from torch.utils.data import Dataset
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


logger = logging.get_logger(__name__)


class LazyStatefulDistributedSampler(Sampler[int]):
    """Distributed sampler that does not materialize len(dataset) indices."""

    _YIELDED = "yielded"

    def __init__(
        self,
        dataset: "Dataset",
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas should be a positive integer")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank should be in the interval [0, num_replicas)")

        self.dataset = dataset
        self.dataset_len = len(dataset)
        if self.dataset_len <= 0:
            raise ValueError("dataset should contain at least one sample")

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.yielded = 0
        self.next_yielded = None

        if self.drop_last and self.dataset_len % self.num_replicas != 0:
            self.num_samples = math.ceil((self.dataset_len - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.dataset_len / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _shuffle_params(self) -> tuple[int, int]:
        if self.total_size <= 1:
            return 0, 1
        rng = random.Random(self.seed + self.epoch)
        offset = rng.randrange(self.total_size)
        stride = rng.randrange(1, self.total_size)
        while math.gcd(stride, self.total_size) != 1:
            stride += 1
            if stride >= self.total_size:
                stride = 1
        return offset, stride

    def __iter__(self) -> Iterator[int]:
        self.yielded = 0
        if self.next_yielded is not None:
            self.yielded = self.next_yielded
            self.next_yielded = None

        offset, stride = self._shuffle_params()
        for sample_idx in range(self.yielded, self.num_samples):
            global_idx = self.rank + sample_idx * self.num_replicas
            if self.shuffle:
                idx = (offset + stride * global_idx) % self.total_size
            else:
                idx = global_idx
            if idx >= self.dataset_len:
                idx %= self.dataset_len
            self.yielded += 1
            yield idx

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {self._YIELDED: self.yielded}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        if self._YIELDED not in state_dict:
            raise ValueError("Invalid state_dict")
        if state_dict[self._YIELDED] < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")
        self.next_yielded = state_dict[self._YIELDED]


class DistributedDataloader(StatefulDataLoader):
    dataset: "Dataset"
    sampler: "StatefulDistributedSampler"

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


def build_dataloader(
    dataset: "Dataset",
    micro_batch_size: int,
    global_batch_size: int,
    dataloader_batch_size: int,
    max_seq_len: int,
    train_steps: int,
    rmpad: bool = True,
    rmpad_with_pos_ids: bool = False,
    bsz_warmup_ratio: float = 0.02,
    bsz_warmup_init_mbtoken: int = 200,
    dyn_bsz_buffer_size: int = 500,
    dyn_bsz_margin: int = 0,
    collate_fn: Optional[Union[Callable, List[Callable]]] = None,
    num_workers: int = 8,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
    seed: int = 0,
) -> "DistributedDataloader":
    parallel_state = get_parallel_state()
    token_micro_bsz = micro_batch_size * max_seq_len
    num_micro_batch = global_batch_size // (
        micro_batch_size * parallel_state.dp_size
    )  # num_micro_batch = num accumulation steps
    bsz_warmup_steps = int(train_steps * bsz_warmup_ratio)
    use_rmpad = rmpad or rmpad_with_pos_ids
    logger.info_rank0(
        f"train_steps: {train_steps}, max_seq_len: {max_seq_len}, use_rmpad: {use_rmpad}, "
        f"bsz_warmup_steps: {bsz_warmup_steps}, bsz_warmup_init_mbtoken: {bsz_warmup_init_mbtoken}, "
        f"token_micro_bsz: {token_micro_bsz}, num_micro_batch: {num_micro_batch}, "
        f"micro_batch_size: {micro_batch_size}, global_batch_size: {global_batch_size}, "
        f"dp_size: {parallel_state.dp_size}, sp_size: {parallel_state.sp_size}."
    )

    if collate_fn is None:
        collate_fn_list = []
        if rmpad_with_pos_ids:
            collate_fn_list.append(DataCollatorWithPositionIDs())
        elif rmpad:
            collate_fn_list.append(DataCollatorWithPacking())
        else:
            collate_fn_list.append(DataCollatorWithPadding())

        if parallel_state.sp_enabled:
            collate_fn_list.append(TextSequenceShardCollator(rmpad=rmpad, rmpad_with_pos_ids=rmpad_with_pos_ids))

        collate_fn = CollatePipeline(collate_fn_list)

    if isinstance(collate_fn, list):
        collate_fn = CollatePipeline(collate_fn)

    if use_rmpad:
        batching_strategy = TextBatchingStrategy(
            token_micro_bsz=token_micro_bsz - dyn_bsz_margin * max_seq_len,
            buffer_size=dyn_bsz_buffer_size,
            bsz_warmup_steps=bsz_warmup_steps if bsz_warmup_steps else -1,
            bsz_warmup_init_mbtoken=bsz_warmup_init_mbtoken,
        )
        dyn_bsz_collate_fn = collate_fn
        collate_fn = UnpackDataCollator()
    else:
        collate_fn = MakeMicroBatchCollator(num_micro_batch=num_micro_batch, internal_data_collator=collate_fn)

    sampler = None
    if not isinstance(dataset, IterableDataset):
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=parallel_state.dp_size,
            rank=parallel_state.dp_rank,
            shuffle=True,
            seed=seed,
        )

    dataloader = DistributedDataloader(
        dataset,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )
    if use_rmpad:
        dataloader = DynamicBatchSizeDataLoader(
            dataloader,
            batching_strategy=batching_strategy,
            collate_fn=dyn_bsz_collate_fn,
            num_micro_batch=num_micro_batch,
            length=train_steps,
            drop_last=drop_last,
        )

    return dataloader
