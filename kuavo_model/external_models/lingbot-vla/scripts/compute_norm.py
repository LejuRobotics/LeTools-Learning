import json
import numpy as np
import os
import re
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from lingbotvla.models import build_processor
from lingbotvla.utils import helper
from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
import lingbotvla.utils.normalize as normalize
from lingbotvla.data.vla_data.base_dataset import VLADataset


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from lingbotvla.data.chat_template import ChatTemplate

logger = helper.create_logger(__name__)

@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)

def compute_norm(dataset, repo_id, batch_size, num_workers, stats, state_norm_keys, acton_norm_keys, delta_norm, shuffle=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=False)
    
    for batch in tqdm(data_loader, desc=f"Computing stats of {repo_id}"):
        for key in state_norm_keys:
            values = np.asarray(batch[key])
            stats[key].update(values.reshape(-1, values.shape[-1]))
        for key in acton_norm_keys:
            values = np.asarray(batch[key][:,0]) if not delta_norm[key] else np.asarray(batch[key].reshape(batch[key].shape[0], -1))
            stats[key].update(values.reshape(-1, values.shape[-1]))


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    
    logger.info_rank0("Prepare data")
    stats = None
    
    assert args.data.datasets_type == 'vla'
    dataset = VLADataset(repo_id=args.data.train_path, data_name =args.data.data_name, robot_config_root=args.data.robot_config_root, config=None, data_config=args.data,  do_nomalize=False)

    state_norm_keys = dataset.feature_transform.states
    acton_norm_keys = dataset.feature_transform.actions
    delta_norm = dataset.feature_transform.action_subtract_state 

    stats = {key: normalize.RunningStats() for key in acton_norm_keys+state_norm_keys}
    
    chunk_size = args.train.chunk_size

    compute_norm(dataset, args.data.train_path, args.train.micro_batch_size, args.data.num_workers, stats, state_norm_keys, acton_norm_keys, delta_norm)
        
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    norm_stats = {}
    for key, stats in stats.items():
        if key in delta_norm and delta_norm[key]==True:
            norm_stats[key] = stats.get_statistics(chunk_size=chunk_size)
        else:
            norm_stats[key] = stats.get_statistics()

    output_path = Path(args.data.norm_stats_file)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats, stats._count)
    


if __name__ == "__main__":
    main()