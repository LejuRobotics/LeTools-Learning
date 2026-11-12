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

# ================ SG100 1-DOF 二值化 ================
SG100_JOINTS_PER_HAND = 11
SG100_DEFAULT_CLOSE_RATIO = 0.2   # 超过开合行程的这个比例即判为闭合


def _parse_sg100_limits(value, name: str, dof_needed: int, offsets: Tuple[int, int],
                        fill: float) -> Tuple[float, ...]:
    """把开合参考位姿展开成 22 维（左手 11 + 右手 11）。

    接受两种写法：
      - 2 * dex_dof_needed 维，每手一段（dex_dof_needed=1 时即 [左手, 右手]）
      - 22 维，逐关节指定
    未选中的位置用 fill 填充，保证 open/close 之间不为零（避免除零）。
    """
    if value is None:
        raise ValueError(
            f"eef_type=sg100 且 is_binary=true 时必须提供 {name}"
            f"（每手 {dof_needed} 维，或 22 维逐关节）"
        )
    values = tuple(float(v) for v in value)
    if len(values) == 2 * SG100_JOINTS_PER_HAND:
        return values
    if len(values) == 2 * dof_needed:
        pad = lambda part, off: ((fill,) * off + part
                                 + (fill,) * (SG100_JOINTS_PER_HAND - off - dof_needed))
        return pad(values[:dof_needed], offsets[0]) + pad(values[dof_needed:], offsets[1])
    raise ValueError(
        f"{name} 长度应为 {2 * dof_needed} 或 {2 * SG100_JOINTS_PER_HAND}，实际 {len(values)}"
    )


@dataclass
class ResizeConfig:
    width: int
    height: int

@dataclass
class Config:
    # Basic settings
    eef_type: str  # 'qiangnao', 'sg100', 'leju_claw', or 'rq2f85'
    which_arm: str  # 'left', 'right', or 'both'
    use_depth: bool # 是否使用深度数据
    depth_range: tuple[int, int]
    dex_dof_needed: int  # 通常为1，表示只需要第一个关节作为开合依据
    dex_dof_offset: Tuple[int, int]  # 单手内起始偏移；可写成 [左手, 右手]，标量则两手相同
    platform_type: str # 机器人类型：'4pro' 或 '5w'
    enable_5w_wholebody: bool
    
    # Timeline settings
    train_hz: int
    main_timeline: str
    main_timeline_fps: int
    sample_drop: int
    
    # Processing flags
    is_binary: bool
    relative_start: bool

    # Image resize settings
    resize: ResizeConfig

    # Optional 5W mobile-base action extension
    enable_5w_base_move: bool = False

    # Task description
    task_description: str = "Pick and Place Task"

    # ---- SG100 1-DOF 二值化专用配置（仅 eef_type='sg100' 生效，其他末端不受影响）----
    # 手动配置的开合参考位姿，已展开为 22 维（左手 11 + 右手 11）。
    sg100_binary_open: Tuple[float, ...] = ()
    sg100_binary_close: Tuple[float, ...] = ()
    # 超过开合行程的该比例即判为闭合。
    sg100_binary_ratio: float = SG100_DEFAULT_CLOSE_RATIO

    @property
    def use_5w_wholebody(self) -> bool:
        """The lower-body extension is only meaningful on the 5W platform."""
        return self.platform_type.lower() == "5w" and self.enable_5w_wholebody

    @property
    def use_5w_base_move(self) -> bool:
        """The mobile-base velocity action is only meaningful on the 5W platform."""
        return self.platform_type.lower() == "5w" and self.enable_5w_base_move

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
    def use_sg100(self) -> bool:
        """Determine if using the 11-DOF-per-hand SG100."""
        return self.eef_type == 'sg100'
    
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
        arm_dof = arm_end - arm_start
        if arm_dof % 2:
            raise ValueError("The configured arm joint range must split evenly between both arms")
        left_end = arm_start + arm_dof // 2
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
        """Get dex slice based on hand type, selected arm, DOF count and offset."""
        half_hand_dof = 11 if self.use_sg100 else 6
        left, right = self.dex_dof_offset
        if self.which_arm == 'left':
            return [[left, left + self.dex_dof_needed], [half_hand_dof, half_hand_dof]]
        elif self.which_arm == 'right':
            return [[0, 0], [half_hand_dof + right, half_hand_dof + right + self.dex_dof_needed]]
        elif self.which_arm == 'both':
            return [[left, left + self.dex_dof_needed],
                    [half_hand_dof + right, half_hand_dof + right + self.dex_dof_needed]]
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

    if eef_type not in ['qiangnao', 'sg100', 'leju_claw', 'rq2f85']:
        raise ValueError(
            f"Invalid eef_type: {eef_type}, must be 'qiangnao', 'sg100', 'leju_claw', or 'rq2f85'."
        )
    
    # Validate which_arm
    which_arm = OmegaConf.select(cfg, 'dataset.which_arm')
    if which_arm not in ['left', 'right', 'both']:
        raise ValueError(f"Invalid which_arm: {which_arm}, must be 'left', 'right', or 'both'")
    
    # Create ResizeConfig object
    resize_config = ResizeConfig(
        width=cfg.dataset.resize.width,
        height=cfg.dataset.resize.height
    )
    
    dex_dof_needed = int(OmegaConf.select(cfg, 'dataset.dex_dof_needed', default=1))
    raw_offset = OmegaConf.select(cfg, 'dataset.dex_dof_offset', default=0)
    if OmegaConf.is_list(raw_offset):
        if len(raw_offset) != 2:
            raise ValueError("dex_dof_offset 写列表时必须是 [左手, 右手] 两个值")
        dex_dof_offset = tuple(int(v) for v in raw_offset)
    else:
        dex_dof_offset = (int(raw_offset),) * 2
    for offset in dex_dof_offset:
        if offset < 0:
            raise ValueError("dex_dof_offset must be >= 0")
    if eef_type == 'sg100':
        if dex_dof_needed not in (1, 11):
            raise ValueError("SG100 dex_dof_needed must be 1 or 11")
        for offset in dex_dof_offset:
            if offset + dex_dof_needed > 11:
                raise ValueError("SG100 dex_dof_offset + dex_dof_needed must be <= 11")

    # SG100 二值化选项（其他末端不读取，保持原状）
    sg100_binary_open: Tuple[float, ...] = ()
    sg100_binary_close: Tuple[float, ...] = ()
    sg100_binary_ratio = float(OmegaConf.select(
        cfg, 'dataset.sg100_binary_ratio', default=SG100_DEFAULT_CLOSE_RATIO))
    if eef_type == 'sg100' and OmegaConf.select(cfg, 'dataset.is_binary', default=False):
        if not 0.0 < sg100_binary_ratio < 1.0:
            raise ValueError("sg100_binary_ratio must be in (0, 1)")
        sg100_binary_open = _parse_sg100_limits(
            OmegaConf.select(cfg, 'dataset.sg100_binary_open', default=None),
            "sg100_binary_open", dex_dof_needed, dex_dof_offset, fill=0.0)
        sg100_binary_close = _parse_sg100_limits(
            OmegaConf.select(cfg, 'dataset.sg100_binary_close', default=None),
            "sg100_binary_close", dex_dof_needed, dex_dof_offset, fill=1.0)
        # 开合参考值不能相同，避免计算开合比例时除零
        for side in {"left": (0,), "right": (1,), "both": (0, 1)}[which_arm]:
            index = side * 11 + dex_dof_offset[side]
            if sg100_binary_close[index] == sg100_binary_open[index]:
                raise ValueError(
                    f"sg100 {'left' if side == 0 else 'right'} 的 "
                    f"sg100_binary_open/close 不能相同"
                )

    # Create main Config object
    return Config(
        eef_type=eef_type,
        which_arm=which_arm,
        use_depth=OmegaConf.select(cfg, 'dataset.use_depth', default=False),
        depth_range=OmegaConf.select(cfg, 'dataset.depth_range', default=(0,1000)),
        dex_dof_needed=dex_dof_needed,
        dex_dof_offset=dex_dof_offset,
        train_hz=OmegaConf.select(cfg, 'dataset.train_hz', default=10),
        main_timeline=OmegaConf.select(cfg, 'dataset.main_timeline', default='head_cam_h'),
        main_timeline_fps=OmegaConf.select(cfg, 'dataset.main_timeline_fps', default=30),
        sample_drop=OmegaConf.select(cfg, 'dataset.sample_drop', default=0),
        is_binary=OmegaConf.select(cfg, 'dataset.is_binary', default=False),
        relative_start=OmegaConf.select(cfg, 'dataset.relative_start', default=False),
        sg100_binary_open=sg100_binary_open,
        sg100_binary_close=sg100_binary_close,
        sg100_binary_ratio=sg100_binary_ratio,
        resize=resize_config,
        task_description=OmegaConf.select(cfg, 'dataset.task_description', default="Pick and Place Task"),
        platform_type=OmegaConf.select(cfg, 'dataset.platform_type', default=DEFAULT_PLATFORM),
        enable_5w_wholebody=OmegaConf.select(cfg, 'dataset.5w_wholebody', default=False),
        enable_5w_base_move=OmegaConf.select(cfg, 'dataset.5w_base_move', default=False),
    )
