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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py

import importlib
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Type, Union
from transformers import PretrainedConfig

import torch.nn as nn

from ..utils import logging


logger = logging.get_logger(__name__)

MODELING_PATH = ["lingbotvla.models.vla"]

@dataclass
class _ConfigRegistry:

    modeling_path: List[str] = field(default_factory=list)
    config_key_to_cls: Dict[str, Type[PretrainedConfig]] = field(default_factory=dict)

    def __post_init__(self):
        for path in self.modeling_path:
            self._mapping_config_key_to_cls(path)

    @property
    def supported_configs(self) -> List[str]:
        return list(self.config_key_to_cls.keys())

    def get_config_cls_from_config_key(self, config_key: str) -> Type[PretrainedConfig]:
        if config_key not in self.config_key_to_cls:
            raise KeyError(f"Config key '{config_key}' not found in registry. "
                          f"Supported keys: {self.supported_configs}")
        return self.config_key_to_cls[config_key]

    def _mapping_config_key_to_cls(self, modeling_path: str):
        try:
            package = importlib.import_module(modeling_path)
        except ImportError as e:
            logger.warning(f"Ignore import error when loading base path {modeling_path}. {e}")
            return

        for _, name, ispkg in pkgutil.walk_packages(package.__path__, modeling_path + "."):
            if not ispkg:
                try:
                    module = importlib.import_module(name)
                except Exception as e:
                    logger.warning(f"Ignore import error when loading config from {name}. {e}")
                    continue

                if hasattr(module, "ConfigClass"):
                    entry = module.ConfigClass
                    entries = entry if isinstance(entry, list) else [entry]
                    
                    for config_cls in entries:
                        cls_name = config_cls.__name__
                        if cls_name not in self.config_key_to_cls:
                            self.config_key_to_cls[cls_name] = config_cls

@lru_cache
def get_config_registry():
    return _ConfigRegistry(modeling_path=MODELING_PATH)

