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

"""SG100 开合二值化（数据转换与在线推理共用）。

放在这里是为了让两侧走同一份实现：数据转换
（``kuavo_data/CvtRosbag2Lerobot.py``）和在线推理
（``kuavo_deploy``）都从这里取，避免两套公式各自漂移。

本模块**只依赖 numpy**：既不能被 ``kuavo_data`` 的重依赖（lerobot / hydra /
rosbag）污染，也不能被 ``kuavo_deploy`` 的 ROS 依赖污染。
"""

from typing import Sequence

import numpy as np

# 每只手（左手 / 右手）的关节数，与 SG100HandState.msg 一致。
SG100_JOINTS_PER_HAND = 11
# 双手拼接后的维度：左手 11 + 右手 11。
SG100_TOTAL_DOF = 2 * SG100_JOINTS_PER_HAND


def sg100_amounts(
    values: np.ndarray,
    hand: int,
    open_v: Sequence[float],
    close_v: Sequence[float],
    joints: Sequence[Sequence[int]],
) -> list:
    """该手判定关节的开合进度 amount = (q - open) / (close - open)。

    分母带符号，因此"越大越闭合"和"越小越闭合"两种关节都适用。

    Args:
        values: 末维为 22 的手部数值，(..., 22) 或 (22,)。
        hand: 0 左手 / 1 右手。
        open_v: 22 维完全张开参考位姿。
        close_v: 22 维完全闭合参考位姿。
        joints: 每只手的判定关节（手内下标），(左手下标, 右手下标)。
    """
    open_arr = np.asarray(open_v, dtype=np.float64)
    close_arr = np.asarray(close_v, dtype=np.float64)
    base = SG100_JOINTS_PER_HAND * hand
    return [(values[..., base + j] - open_arr[base + j])
            / (close_arr[base + j] - open_arr[base + j])
            for j in joints[hand]]


def sg100_binarize(
    values: np.ndarray,
    open_v: Sequence[float],
    close_v: Sequence[float],
    joints: Sequence[Sequence[int]],
    ratio: float,
) -> np.ndarray:
    """把 sg100 原始值二值化为 0/1，结果写在每手的代表维上（其余维保持 0）。

    每手按 ``joints`` 判定：任一关节的 amount 超过 ratio 即判为闭合(1)（偏激进）。
    只用单关节时等价于逐维判定。``joints[hand]`` 为空表示该手全程张开，输出全 0。

    Args:
        values: 末维为 22 的手部数值。
        open_v: 22 维完全张开参考位姿。
        close_v: 22 维完全闭合参考位姿。
        joints: 每只手的判定关节（手内下标），(左手下标, 右手下标)。
        ratio: 开合行程中判为闭合的比例。
    """
    if values.size == 0:
        return values

    open_arr = np.asarray(open_v, dtype=np.float64)
    if open_arr.size == 0 or values.shape[-1] != open_arr.shape[0]:
        raise ValueError(
            f"sg100 二值化需要 {open_arr.size} 维开合参考位姿，实际收到 {values.shape[-1]} 维；"
            f"请检查开合参考位姿配置"
        )

    out = np.zeros(values.shape, dtype=values.dtype)
    for hand, hand_joints in enumerate(joints):
        if not hand_joints:
            continue
        amount = np.max(sg100_amounts(values, hand, open_v, close_v, joints), axis=0)
        out[..., SG100_JOINTS_PER_HAND * hand + hand_joints[0]] = (
            amount > ratio
        ).astype(values.dtype)
    return out
