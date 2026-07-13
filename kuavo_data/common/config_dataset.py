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

from dataclasses import dataclass
from typing import List, Tuple
from omegaconf import OmegaConf
from kuavo_data.common.config_platform import get_arm_joint_slice, DEFAULT_PLATFORM

@dataclass
class ResizeConfig:
    width: int
    height: int

@dataclass
class Config:
    # Basic settings
    eef_type: str  # 'qiangnao' or 'leju_claw'
    which_arm: str  # 'left', 'right', or 'both'
    use_depth: bool # 是否使用深度数据
    depth_range: tuple[int, int]
    dex_dof_needed: int  # 通常为1，表示只需要第一个关节作为开合依据
    platform_type: str # 机器人类型：'4pro' 或 '5w'
    include_waist: bool
    waist_state_index: int
    waist_command_topic: str
    
    # Timeline settings
    train_hz: int
    main_timeline: str
    main_timeline_fps: int
    sample_drop: int
    
    # Processing flags
    is_binary: bool
    delta_action: bool
    relative_start: bool

    # Image resize settings
    resize: ResizeConfig

    # Task description
    task_description: str = "Pick and Place Task"
    @property
    def use_leju_claw(self) -> bool:
        """Determine if using leju claw based on eef_type."""
        # return self.eef_type == 'leju_claw'
        return "claw" in self.eef_type or self.eef_type=="rq2f85"
    
    @property
    def use_qiangnao(self) -> bool:
        """Determine if using qiangnao based on eef_type."""
        return self.eef_type == 'qiangnao'
    
    @property
    def default_camera_names(self) -> List[str]:
        """Get camera names based on which arm is being used."""
        cameras = ['head_cam_h',"depth_h"]
        cameras = [{"left":['head_cam_h','wrist_cam_l'],
                    "right":['head_cam_h','wrist_cam_r'],
                    "both":['head_cam_h','wrist_cam_l','wrist_cam_r']
                    },
                    {"left":['head_cam_h','depth_h','wrist_cam_l','depth_l'],
                    "right":['head_cam_h','depth_h','wrist_cam_r','depth_r'],
                    "both":['head_cam_h','depth_h','wrist_cam_l','depth_l','wrist_cam_r','depth_r']
                    }][int(self.use_depth)][self.which_arm]
        return cameras
    
    @property
    def slice_robot(self) -> List[Tuple[int, int]]:
        """Get robot slice based on which arm is being used."""
        arm_start, arm_end = get_arm_joint_slice(self.platform_type)
        left_end = arm_start + 7
        right_start = left_end
        
        if self.which_arm == 'left':
            return [(arm_start, left_end), (left_end, left_end)]
        elif self.which_arm == 'right':
            return [(arm_start, arm_start), (right_start, arm_end)]
        elif self.which_arm == 'both':
            return [(arm_start, left_end), (right_start, arm_end)]
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
    
    
    @property
    def dex_slice(self) -> List[List[int]]:
        """Get dex slice based on which arm and dex_dof_needed."""
        if self.which_arm == 'left':
            return [[0, self.dex_dof_needed], [6, 6]]  # 左手使用指定自由度，右手不使用
        elif self.which_arm == 'right':
            return [[0, 0], [6, 6 + self.dex_dof_needed]]  # 左手不使用，右手使用指定自由度
        elif self.which_arm == 'both':
            return [[0, self.dex_dof_needed], [6, 6 + self.dex_dof_needed]]  # 双手都使用指定自由度
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
    
    @property
    def claw_slice(self) -> List[List[int]]:
        """Get claw slice based on which arm."""
        if self.which_arm == 'left':
            return [[0, 1], [1, 1]]  # 左手使用夹爪，右手不使用
        elif self.which_arm == 'right':
            return [[0, 0], [1, 2]]  # 左手不使用，右手使用夹爪
        elif self.which_arm == 'both':
            return [[0, 1], [1, 2]]  # 双手都使用夹爪
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")

def load_config(cfg) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file. If None, uses default path.
        
    Returns:
        Config object containing all settings
    """
    
    # Validate eef_type
    eef_type = OmegaConf.select(cfg, "dataset.eef_type")

    if eef_type not in ['qiangnao', 'leju_claw', 'rq2f85']:
        raise ValueError(f"Invalid eef_type: {eef_type}, must be 'qiangnao' or 'leju_claw','rq2f85' .")
    
    # Validate which_arm
    which_arm = OmegaConf.select(cfg, 'dataset.which_arm')
    if which_arm not in ['left', 'right', 'both']:
        raise ValueError(f"Invalid which_arm: {which_arm}, must be 'left', 'right', or 'both'")

    platform_type = OmegaConf.select(cfg, 'dataset.platform_type', default=DEFAULT_PLATFORM)
    include_waist = OmegaConf.select(cfg, 'dataset.include_waist', default=False)
    waist_state_index = OmegaConf.select(cfg, 'dataset.waist_state_index', default=12)
    waist_command_topic = OmegaConf.select(
        cfg,
        'dataset.waist_command_topic',
        default='/robot_waist_motion_data',
    )
    if include_waist and (platform_type != '5' or which_arm != 'right'):
        raise ValueError(
            "dataset.include_waist currently requires platform_type='5' and which_arm='right'"
        )
    if waist_state_index < 0:
        raise ValueError("dataset.waist_state_index must be >= 0")
    if include_waist and not waist_command_topic:
        raise ValueError("dataset.waist_command_topic must not be empty")
    
    # Create ResizeConfig object
    resize_config = ResizeConfig(
        width=cfg.dataset.resize.width,
        height=cfg.dataset.resize.height
    )
    
    # Create main Config object
    return Config(
        eef_type=eef_type,
        which_arm=which_arm,
        use_depth=OmegaConf.select(cfg, 'dataset.use_depth', default=False),
        depth_range=OmegaConf.select(cfg, 'dataset.depth_range', default=(0,1000)),
        dex_dof_needed=OmegaConf.select(cfg, 'dataset.dex_dof_needed', default=1),
        train_hz=OmegaConf.select(cfg, 'dataset.train_hz', default=10),
        main_timeline=OmegaConf.select(cfg, 'dataset.main_timeline', default='head_cam_h'),
        main_timeline_fps=OmegaConf.select(cfg, 'dataset.main_timeline_fps', default=30),
        sample_drop=OmegaConf.select(cfg, 'dataset.sample_drop', default=0),
        is_binary=OmegaConf.select(cfg, 'dataset.is_binary', default=False),
        delta_action=OmegaConf.select(cfg, 'dataset.delta_action', default=False),
        relative_start=OmegaConf.select(cfg, 'dataset.relative_start', default=False),
        resize=resize_config,
        task_description=OmegaConf.select(cfg, 'dataset.task_description', default="Pick and Place Task"),
        platform_type=platform_type,
        include_waist=include_waist,
        waist_state_index=waist_state_index,
        waist_command_topic=waist_command_topic,
    )
