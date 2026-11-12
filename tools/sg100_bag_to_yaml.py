#!/usr/bin/env python3
"""从 rosbag 提取 SG100 灵巧手参数，生成 kuavo_deploy 推理配置 YAML。

只做一件事：读 bag 里的 SG100 三个话题，把能实测到的参数覆盖到
`configs/deploy/total/deploy_total.yaml` 的 SG100 字段上；**bag 里没有的
（话题不存在 / 数组为空 / 该手未启用），一律沿用 deploy_total.yaml**。

话题与字段映射
--------------
/sg100_hand_command            控制指令（推荐作为姿态来源）
    left_hand_positions / right_hand_positions     11 维弧度
    left_enable_mask   / right_enable_mask         0 = 该手未启用
    left_hand_kp/kd/torque_ff/output_limit/control_mode
                                 可选 per-joint 力控数组，位置控制时为空
/sg100_hand_filtered_command   滤波后的指令，字段同上但名字带 filtered
    left_hand_filtered_positions / right_hand_filtered_positions
    （话题不存在时自动跳过，仅告警）
/sg100_hand_state              实际状态
    left_hand_positions / right_hand_positions     11 维弧度
    left_hand_modes / right_hand_modes             11 维控制模式（7=位置）
    left_hand_currents / right_hand_currents       11 维电流（仅统计，不写 YAML）

写入 YAML 的规则
----------------
bag 实测  which_arm（enable mask）、sg100_init_pose（command 首帧）、
          sg100_joint_min/max（command + state 全 bag 极值，只扩不缩）、
          sg100_open_pose/close_pose（开合量投影极值帧）、
          sg100_force_kp/kd/torque_ff/output_limit/velocities
          （仅当 command 的 per-joint 数组非空，即阻抗手势）
照抄 total sg100_dof_needed / sg100_dof_offset / sg100_force_threshold /
          sg100_action_threshold / is_binary_sg100_action / arm_state_keys /
          limits.gripper（dof=11 时改为 sg100_joint_min/max）

注意：`/sg100_hand_command` 的 per-joint 力控数组是**可选**的，语义是
"空数组 -> 用驱动侧全局 ROS 参数"（sg100_hand_driver.cpp: kp=10.0, kd=0.02,
output_limit=0.5）。驱动仅在阻抗手势（mode=9）时才填充它们，位置控制（mode=7）
下为空。此时脚本沿用 total；本 YAML 里非空的 per-joint 数组会在部署时随
`/sg100_hand_command` 下发，**覆盖驱动全局参数**（这是预期行为，因此脚本只
记录来源、不告警）。

用法
----
    # 默认输出 configs/deploy/sg100_<which_arm>.yaml
    python tools/sg100_bag_to_yaml.py --bag /path/to/go.bag

    # 只打印不写文件 / 写后校验
    python tools/sg100_bag_to_yaml.py --bag /path/to/go.bag --print-only
    python tools/sg100_bag_to_yaml.py --bag /path/to/go.bag --validate

    # 模型按每手 11 关节训练（动作维度 18/36）
    python tools/sg100_bag_to_yaml.py --bag /path/to/go.bag --sg100-dof 11

    # 显式指定 per-joint 力控参数（11 个=左右手广播，22 个=左+右）
    python tools/sg100_bag_to_yaml.py --bag /path/to/go.bag \
        --sg100-kp 10,10,10,10,10,10,10,10,10,10,10 --sg100-kd 0.02
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOTAL = REPO_ROOT / "configs" / "deploy" / "total" / "deploy_total.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "configs" / "deploy"

HANDS: Tuple[str, str] = ("left", "right")
NJ = 11                      # 每手关节数
POSE_LEN = NJ * 2            # 22 = 左手 11 + 右手 11
MAX_FRAMES = 200_000         # 单话题最多读多少帧
MIN_GRIP_SPAN = 0.5          # 标定 open/close 所需的最小开合量跨度

# 话题
TOPIC_CMD = "/sg100_hand_command"
TOPIC_FILTERED = "/sg100_hand_filtered_command"
TOPIC_STATE = "/sg100_hand_state"
SG100_TOPICS = (TOPIC_CMD, TOPIC_FILTERED, TOPIC_STATE)

# 消息字段名 -> YAML 键（每手 11 维，合并成 22 维）
FORCE_FIELDS: Dict[str, str] = {
    "kp": "sg100_force_kp",
    "kd": "sg100_force_kd",
    "torque_ff": "sg100_force_torque_ff",
    "output_limit": "sg100_force_output_limit",
    "velocities": "sg100_force_velocities",
}

# 驱动侧全局 ROS 参数默认值（sg100_hand_driver.cpp）。消息里数组为空时驱动用它；
# 本仓库恒发 22 维 per-joint 数组，部署时以 YAML 值为准（覆盖驱动全局参数）。
# 这里仅用于在报告里对比，方便确认"是否真的想改掉驱动默认"。
DRIVER_DEFAULTS: Dict[str, float] = {
    "sg100_force_kp": 10.0,
    "sg100_force_kd": 0.02,
    "sg100_force_output_limit": 0.5,
}

# 驱动量化范围（sg100_types.h CTRL_PID_RANGE / CTRL_OUTLIM_RANGE），超出被静默裁剪
ENCODE_RANGE: Dict[str, float] = {
    "sg100_force_kp": 10.0,
    "sg100_force_kd": 10.0,
    "sg100_force_output_limit": 10.0,
}

# 本脚本会写出的 SG100 字段（用于文件头来源清单）
SG100_KEYS: Tuple[str, ...] = (
    "eef_type", "which_arm", "platform_type", "arm_state_keys",
    "sg100_dof_needed", "sg100_dof_offset",
    "sg100_init_pose", "sg100_open_pose", "sg100_close_pose",
    "sg100_joint_min", "sg100_joint_max",
    "sg100_force_kp", "sg100_force_kd", "sg100_force_torque_ff",
    "sg100_force_output_limit", "sg100_force_velocities",
    "sg100_force_threshold", "is_binary_sg100_action", "sg100_action_threshold",
    "limits.gripper",
)

ENV_KEY_ORDER: Tuple[str, ...] = (
    "eef_type", "which_arm", "platform_type", "arm_state_keys",
    "sg100_dof_needed", "sg100_dof_offset",
    "sg100_open_pose", "sg100_close_pose", "sg100_init_pose",
    "sg100_joint_min", "sg100_joint_max",
    "sg100_force_threshold", "sg100_force_kp", "sg100_force_kd",
    "sg100_force_torque_ff", "sg100_force_output_limit", "sg100_force_velocities",
    "is_binary_sg100_action", "sg100_action_threshold",
    "limits",
)

# 平台 -> 手臂末端索引（来自 configs/platform/platform_config.yaml），用于推断型号
PLATFORM_ARM_END = {"4pro": 26, "5w": 18, "5": 27}


# ===========================================================================
# 通用工具
# ===========================================================================
def _field(msg: Any, name: str, default: Any = None) -> Any:
    """按名字从消息对象 / dict 取字段，取不到返回 default。"""
    if isinstance(msg, dict):
        return msg.get(name, default)
    return getattr(msg, name, default)


def _floats(values: Any) -> List[float]:
    """转成 float 列表；长度不对或类型不符返回空列表。"""
    if values is None:
        return []
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return []


def _get11(msg: Any, name: str) -> List[float]:
    """取一个 11 维数组字段，长度不为 11 时返回空列表（视为"未提供"）。"""
    values = _floats(_field(msg, name))
    return values if len(values) == NJ else []


def _round(values: Sequence[float], digits: int = 4) -> List[float]:
    return [round(float(v), digits) for v in values]


def _slice(side: str) -> Tuple[int, int]:
    """该手在 22 维数组里的 [start, end)。"""
    return (0, NJ) if side == "left" else (NJ, POSE_LEN)


def _short(values: Sequence[float], keep: int = 3) -> str:
    items = [str(round(float(v), 4)) for v in values]
    if len(items) <= keep * 2:
        return "[" + ", ".join(items) + "]"
    return "[" + ", ".join(items[:keep]) + ", ... , " + ", ".join(items[-keep:]) + "]"


def _distinct(values: Sequence[float]) -> List[float]:
    return sorted({round(float(v), 6) for v in values})


def _varies(frames: List[List[float]], tol: float = 1e-6) -> bool:
    """一批 11 维数组是否随时间变化。"""
    if len(frames) < 2:
        return False
    first = frames[0]
    return any(
        any(abs(a - b) > tol for a, b in zip(first, frame)) for frame in frames[1:]
    )


# ===========================================================================
# bag 读取（优先 ROS1 rosbag，回退纯 Python rosbags）
# ===========================================================================
class _Reader:
    def topics(self) -> Dict[str, str]:
        raise NotImplementedError

    def messages(self, topic: str, limit: int = 0) -> Iterator[Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _RosbagReader(_Reader):
    """ROS1 自带 rosbag 包。"""

    def __init__(self, path: str):
        import rosbag  # type: ignore

        self._bag = rosbag.Bag(path, "r")

    def topics(self) -> Dict[str, str]:
        info = self._bag.get_type_and_topic_info()
        return {name: meta.msg_type for name, meta in info.topics.items()}

    def messages(self, topic: str, limit: int = 0) -> Iterator[Any]:
        count = 0
        for _, msg, _ in self._bag.read_messages(topics=[topic]):
            yield msg
            count += 1
            if limit and count >= limit:
                return

    def close(self) -> None:
        try:
            self._bag.close()
        except Exception:
            pass


class _RosbagsReader(_Reader):
    """纯 Python rosbags 包，可从 bag 内嵌的 msg 定义反序列化。"""

    def __init__(self, path: str):
        from rosbags.rosbag1 import Reader  # type: ignore
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg  # type: ignore

        self._reader = Reader(path)
        self._reader.open()
        self._store = get_typestore(Stores.ROS1_NOETIC)
        for conn in self._reader.connections:
            msgdef = getattr(conn, "msgdef", None)
            if not msgdef:
                continue
            try:
                self._store.register(get_types_from_msg(msgdef, conn.msgtype))
            except Exception:
                continue  # 个别嵌套类型注册失败不影响其他话题

    def topics(self) -> Dict[str, str]:
        return {c.topic: c.msgtype for c in self._reader.connections}

    def messages(self, topic: str, limit: int = 0) -> Iterator[Any]:
        conns = [c for c in self._reader.connections if c.topic == topic]
        if not conns:
            return
        count = 0
        for conn, _ts, raw in self._reader.messages(connections=conns):
            try:
                yield self._store.deserialize_ros1(raw, conn.msgtype)
            except Exception:
                continue
            count += 1
            if limit and count >= limit:
                return

    def close(self) -> None:
        try:
            self._reader.close()
        except Exception:
            pass


def open_bag(path: str) -> _Reader:
    errors: List[str] = []
    for cls in (_RosbagReader, _RosbagsReader):
        try:
            return cls(path)
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise SystemExit(
        "无法打开 bag（需要 ROS1 `rosbag` 或纯 Python `rosbags`）：\n  "
        + "\n  ".join(errors)
    )


# ===========================================================================
# 日志 / 结果容器
# ===========================================================================
class Report:
    """收集探测依据、告警和每个字段的来源，便于人工复核。"""

    def __init__(self) -> None:
        self.env: Dict[str, Any] = {}
        self.inf: Dict[str, Any] = {}
        self.notes: List[str] = []
        self.warnings: List[str] = []
        self.todos: List[str] = []
        self.source: Dict[str, str] = {}

    def set(self, key: str, value: Any, source: str) -> None:
        self.env[key] = value
        self.source[key] = source

    def prov(self, key: str, source: str) -> None:
        """只登记来源，不写入 env（用于 limits.gripper 这类嵌套字段）。"""
        self.source[key] = source

    def note(self, text: str) -> None:
        self.notes.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def todo(self, text: str) -> None:
        self.todos.append(text)


# ===========================================================================
# 采集：把三个话题读成按手分组的数据
# ===========================================================================
def collect(bag: _Reader, topics: Dict[str, str], rep: Report) -> Dict[str, Dict[str, Any]]:
    """读 SG100 话题。

    返回 {side: {
        "cmd":        [[11], ...]   # /sg100_hand_command 位置帧
        "filtered":   [[11], ...]   # /sg100_hand_filtered_command 位置帧
        "state":      [[11], ...]   # /sg100_hand_state 位置帧
        "mask":       {int, ...}    # 出现过的 enable mask（cmd）
        "force":      {字段: [[11], ...]}   # 仅记录非空的 per-joint 数组
        "modes":      {int, ...}    # per-joint control_mode / hand_modes
    }}
    """
    data: Dict[str, Dict[str, Any]] = {
        side: {
            "cmd": [], "filtered": [], "state": [],
            "mask": set(),
            "force": {name: [] for name in FORCE_FIELDS},
            "modes": set(),
        }
        for side in HANDS
    }

    def read(topic: str, key: str, prefix: str) -> None:
        """prefix 为字段名前缀（"" 或 "filtered_"）。"""
        if topic not in topics:
            rep.note(f"{topic}: 话题不存在，跳过")
            return
        frames = 0
        for msg in bag.messages(topic):
            if frames >= MAX_FRAMES:
                rep.warn(f"{topic} 帧数超过 {MAX_FRAMES}，已截断，极值标定可能不完整。")
                break
            frames += 1
            for side in HANDS:
                pos = _get11(msg, f"{side}_hand_{prefix}positions")
                if pos:
                    data[side][key].append(pos)

                # ---- enable mask（只有 command 有）----------------------
                if key == "cmd":
                    mask = _field(msg, f"{side}_enable_mask")
                    if mask is not None:
                        data[side]["mask"].add(int(mask))

                # ---- per-joint 力控数组（可选，可能为空）-----------------
                if key == "cmd":
                    for name in FORCE_FIELDS:
                        values = _get11(msg, f"{side}_hand_{name}")
                        if values:
                            data[side]["force"][name].append(values)
                    mode_field = f"{side}_hand_control_mode"
                elif key == "state":
                    mode_field = f"{side}_hand_modes"
                else:
                    mode_field = None
                if mode_field:
                    for mode in _field(msg, mode_field) or []:
                        data[side]["modes"].add(int(mode))

        rep.note(f"{topic}: 读取 {frames} 帧")

    read(TOPIC_CMD, "cmd", "")
    read(TOPIC_FILTERED, "filtered", "filtered_")
    read(TOPIC_STATE, "state", "")
    return data


# ===========================================================================
# 逐项探测
# ===========================================================================
def _active(side_data: Dict[str, Any]) -> Optional[bool]:
    """该手是否被真正控制：True/False，无数据时 None。"""
    if any(m != 0 for m in side_data["mask"]):
        return True
    if side_data["mask"]:
        return False
    frames = side_data["cmd"] or side_data["filtered"] or side_data["state"]
    if not frames:
        return None
    return any(any(abs(v) > 1e-6 for v in frame) for frame in frames)


def detect_arm(data: Dict[str, Dict[str, Any]], rep: Report, override: Optional[str]) -> str:
    if override:
        rep.set("which_arm", override, "cli")
        rep.note(f"which_arm={override}（命令行指定）")
        return override

    active = {side: _active(data[side]) for side in HANDS}
    rep.note("每手活动性: " + ", ".join(f"{s}={active[s]}" for s in HANDS))
    if active["left"] is True and active["right"] is True:
        arm = "both"
    elif active["left"] is True:
        arm = "left"
    elif active["right"] is True:
        arm = "right"
    else:
        arm = "both"
        rep.warn(
            "两只手都没有 enable mask / 数值非零证据，which_arm 回退 both，"
            "请用 --which-arm 确认。"
        )
    rep.set("which_arm", arm, "bag")
    rep.note(f"which_arm={arm}（由 enable mask / 数值推断）")
    return arm


def detect_platform(
    bag: _Reader, topics: Dict[str, str], rep: Report, override: Optional[str]
) -> str:
    """用 /sensors_data_raw 的关节数推断型号；推断不出就沿用 total。"""
    if override:
        rep.set("platform_type", override, "cli")
        rep.note(f"platform_type={override}（命令行指定）")
        return override

    if "/lb_leg_traj" in topics:
        rep.set("platform_type", "5w", "bag")
        rep.note("platform_type=5w（bag 中存在 /lb_leg_traj）")
        return "5w"

    joint_len = 0
    for msg in bag.messages("/sensors_data_raw", limit=1):
        joint_len = len(_floats(_field(msg, "joint_data.joint_q") or _field(msg, "joint_q")))
        break

    candidates = [p for p, end in PLATFORM_ARM_END.items() if joint_len >= end]
    if candidates:
        platform = max(candidates, key=lambda p: PLATFORM_ARM_END[p])
        rep.set("platform_type", platform, "bag")
        rep.note(f"platform_type={platform}（/sensors_data_raw 关节数 {joint_len}）")
        return platform

    rep.warn(
        "无法从 /sensors_data_raw 推断 platform_type，沿用 deploy_total.yaml；"
        "该值决定手臂维度，配错会导致动作维度不匹配，请确认。"
    )
    return ""


def _grip_amount(
    frame: Sequence[float], open_ref: Sequence[float], close_ref: Sequence[float]
) -> float:
    """把一帧姿态投影到 total 的 open->close 轴上，得到归一化开合量。"""
    num = den = 0.0
    for value, open_v, close_v in zip(frame, open_ref, close_ref):
        delta = close_v - open_v
        num += (value - open_v) * delta
        den += delta * delta
    return num / den if den > 1e-9 else 0.0


def _resolve_binary(
    amounts: List[float], total: Dict[str, Any], rep: Report, override: Optional[bool]
) -> None:
    """is_binary_sg100_action 描述模型输出，bag 无法反推：默认沿用 total 只做告警。"""
    total_value = bool(total.get("is_binary_sg100_action", False))
    if amounts:
        n = len(amounts)
        low = sum(1 for a in amounts if a < 0.2) / n
        high = sum(1 for a in amounts if a > 0.8) / n
        mid = max(0.0, 1.0 - low - high)
        rep.note(f"开合量分布: <0.2 {low:.1%}, 0.2~0.8 {mid:.1%}, >0.8 {high:.1%}")
        if total_value and mid >= 0.5:
            rep.warn(
                f"is_binary_sg100_action=true 但开合量中间段占 {mid:.1%}（连续变化），"
                "若模型输出连续开合量请加 --sg100-binary false。"
            )
        elif not total_value and mid <= 0.2:
            rep.warn(
                f"is_binary_sg100_action=false 但开合量呈双峰（中间段 {mid:.1%}），"
                "若模型输出 0/1 开合可加 --sg100-binary true。"
            )

    if override is not None:
        rep.set("is_binary_sg100_action", bool(override), "cli")
        rep.note(f"is_binary_sg100_action={bool(override)}（命令行指定）")
    else:
        rep.set("is_binary_sg100_action", total_value, "total")
        rep.note(
            f"is_binary_sg100_action={total_value}（描述模型输出，bag 无法反推，沿用 total）"
        )


def resolve_force(data: Dict[str, Dict[str, Any]], total: Dict[str, Any], rep: Report) -> None:
    """per-joint 力控参数：bag 非空则用 bag，否则沿用 total（并对比驱动默认值）。"""
    modes = set()
    for side in HANDS:
        modes |= data[side]["modes"]
    if modes:
        rep.note(f"control_mode 取值 {sorted(modes)}（7=位置, 9=阻抗）")
        if 9 in modes:
            rep.warn(
                "bag 中出现阻抗控制（mode=9）；部署侧恒发 MODE_JOINT_POSITION(7)，"
                "这些阻抗增益不会按阻抗方式生效。"
            )
        elif modes == {7}:
            rep.note("bag 全程位置控制（mode=7）：torque_ff 无效，kp/kd/output_limit 生效。")
    else:
        rep.note("control_mode 数组为空 -> 全局 MODE_JOINT_POSITION(7)。")

    for name, key in FORCE_FIELDS.items():
        values = total.get(key)
        if not isinstance(values, list) or len(values) != POSE_LEN:
            raise SystemExit(f"deploy_total.yaml 的 {key} 必须是 {POSE_LEN} 个值")

        merged = [float(v) for v in values]
        used: List[str] = []
        for side in HANDS:
            frames = data[side]["force"][name]
            if not frames:
                continue
            lo, hi = _slice(side)
            merged[lo:hi] = _round(frames[-1])
            used.append(side)
            if _varies(frames):
                rep.warn(
                    f"{key}[{side}] 在 bag 中逐帧变化（{len(frames)} 帧），已取最后一帧，"
                    "请确认这是期望的增益。"
                )

        if used and _round(merged) != _round(values):
            rep.set(key, merged, "bag")
            rep.note(f"{key} 取自 {TOPIC_CMD} 的 per-joint 数组（{'+'.join(used)} 手）。")
            limit = ENCODE_RANGE.get(key)
            if limit is not None and max(abs(v) for v in merged) > limit:
                rep.warn(
                    f"{key} 超出驱动量化范围 ±{limit}，驱动侧会被静默裁剪"
                    "（sg100_types.h CTRL_PID_RANGE/CTRL_OUTLIM_RANGE）。"
                )
        else:
            rep.set(key, merged, "total")
            default = DRIVER_DEFAULTS.get(key)
            distinct = _distinct(merged)
            if used:
                rep.note(f"{key}：bag 的 per-joint 数组与 deploy_total.yaml 一致。")
            elif default is not None:
                if len(distinct) == 1 and abs(distinct[0] - default) < 1e-9:
                    rep.note(
                        f"{key}={distinct[0]}（bag 中该数组为空，沿用 total，"
                        "与驱动全局默认值一致）。"
                    )
                else:
                    rep.note(
                        f"{key}={distinct}（bag 中该数组为空，沿用 total；"
                        f"驱动全局默认是 {default}，部署时以本 YAML 下发的 per-joint "
                        "值为准，覆盖驱动全局参数）。"
                    )
            else:
                rep.note(f"{key}（bag 中该数组为空，沿用 deploy_total.yaml）。")

    rep.note(
        "sg100_force_torque_ff 仅在阻抗模式（mode=9）生效，位置控制下无效，建议保持 0。"
    )
    rep.note(
        "sg100_force_velocities 会下发到 velocities 字段，但驱动命令回调不读取它"
        "（仅状态回调上报速度），属透传不生效参数。"
    )


# ===========================================================================
# 构造 SG100 参数块
# ===========================================================================
def _need22(total: Dict[str, Any], key: str) -> List[float]:
    values = total.get(key)
    if not isinstance(values, list) or len(values) != POSE_LEN:
        raise SystemExit(f"deploy_total.yaml 的 {key} 必须是 {POSE_LEN} 个值")
    return [float(v) for v in values]


def build(
    data: Dict[str, Dict[str, Any]],
    total: Dict[str, Any],
    rep: Report,
    which_arm: str,
    platform: str,
    args: argparse.Namespace,
) -> None:
    rep.set("eef_type", "sg100", "fixed")
    rep.note("eef_type=sg100（本脚本固定输出 SG100 配置）")

    # ---- dof / offset：模型侧选择，bag 无法反推 ---------------------------
    if args.sg100_dof is None:
        dof = int(total.get("sg100_dof_needed", 1))
        rep.set("sg100_dof_needed", dof, "total")
        rep.note(f"sg100_dof_needed={dof}（bag 无法反推，沿用 total）")
        rep.todo(
            "sg100_dof_needed：若模型按每手 11 关节训练（动作维度 18/36），"
            "请用 --sg100-dof 11 重新生成。"
        )
    else:
        dof = int(args.sg100_dof)
        rep.set("sg100_dof_needed", dof, "cli")
        rep.note(f"sg100_dof_needed={dof}（命令行指定）")

    if dof == NJ:
        if args.sg100_dof_offset not in (None, 0):
            rep.warn("sg100_dof_needed=11 时 offset 必须为 0，已忽略 --sg100-dof-offset。")
        offset = 0
    elif args.sg100_dof_offset is None:
        offset = int(total.get("sg100_dof_offset", 0))
        rep.note(f"sg100_dof_offset={offset}（沿用 total）")
    else:
        offset = int(args.sg100_dof_offset)
        rep.note(f"sg100_dof_offset={offset}（命令行指定）")
    rep.set(
        "sg100_dof_offset",
        offset,
        "cli" if args.sg100_dof_offset is not None or dof == NJ else "total",
    )
    if offset < 0 or offset + dof > NJ:
        raise SystemExit(f"sg100_dof_offset({offset}) + sg100_dof_needed({dof}) 必须 <= {NJ}")

    # ---- 照抄 total 的字段 ------------------------------------------------
    for key in ("sg100_force_threshold", "sg100_action_threshold"):
        if key in total:
            rep.set(key, copy.deepcopy(total[key]), "total")
    if args.sg100_action_threshold is not None:
        threshold = float(args.sg100_action_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise SystemExit("sg100_action_threshold 必须在 [0, 1] 之间")
        rep.set("sg100_action_threshold", threshold, "cli")
        rep.note(f"sg100_action_threshold={threshold}（命令行指定）")
    rep.set("arm_state_keys", ["joint_q", "gripper"], "total")
    rep.note(
        'arm_state_keys=["joint_q", "gripper"]：build_obs_key_map() 会把 obs_key_map 的 '
        "sg100 键重命名为 gripper，写 sg100 会取不到状态。"
    )

    # ---- per-joint 力控参数 ----------------------------------------------
    resolve_force(data, total, rep)
    for name, key in FORCE_FIELDS.items():
        override = getattr(args, key)
        if override is not None:
            rep.set(key, list(override), "cli")
            rep.note(f"{key}={_short(override)}（命令行指定）")

    # ---- 姿态 / 限位 ------------------------------------------------------
    open_pose = _need22(total, "sg100_open_pose")
    close_pose = _need22(total, "sg100_close_pose")
    init_pose = _need22(total, "sg100_init_pose")
    joint_min = _need22(total, "sg100_joint_min")
    joint_max = _need22(total, "sg100_joint_max")
    ref_open = _need22(total, "sg100_open_pose")
    ref_close = _need22(total, "sg100_close_pose")

    from_bag = {"sg100_init_pose": False, "sg100_open_pose": False, "sg100_close_pose": False,
                "sg100_joint_min": False, "sg100_joint_max": False}
    amounts: List[float] = []

    for side in HANDS:
        side_data = data[side]
        active = _active(side_data)
        if active is not True:
            reason = "无数据" if active is None else "未启用（enable mask=0 或数值全 0）"
            rep.note(f"{side} 手{reason}，该侧姿态/限位沿用 deploy_total.yaml。")
            continue

        lo, hi = _slice(side)
        frames = side_data["cmd"] or side_data["filtered"] or side_data["state"]
        topic = (
            TOPIC_CMD if side_data["cmd"]
            else TOPIC_FILTERED if side_data["filtered"]
            else TOPIC_STATE
        )

        # init pose = 首帧
        init_pose[lo:hi] = _round(frames[0])
        from_bag["sg100_init_pose"] = True
        rep.note(f"sg100_init_pose[{side}] 取 {topic} 首帧。")

        # joint_min/max = command + state 全 bag 极值 ∪ total（只扩不缩）
        all_frames = side_data["cmd"] + side_data["state"] or frames
        expanded = False
        for i in range(NJ):
            idx = lo + i
            obs_min = min(f[i] for f in all_frames)
            obs_max = max(f[i] for f in all_frames)
            if obs_min < joint_min[idx]:
                rep.note(f"sg100_joint_min[{idx}] {joint_min[idx]} -> {round(obs_min, 4)}（bag 更小）")
                joint_min[idx] = round(obs_min, 4)
                expanded = True
            if obs_max > joint_max[idx]:
                rep.note(f"sg100_joint_max[{idx}] {joint_max[idx]} -> {round(obs_max, 4)}（bag 更大）")
                joint_max[idx] = round(obs_max, 4)
                expanded = True
        if expanded:
            from_bag["sg100_joint_min"] = from_bag["sg100_joint_max"] = True
        else:
            rep.note(f"{side} 手实测极值都在 total 的 sg100_joint_min/max 内，无需扩展。")

        # open/close = 开合量投影极值帧
        side_amounts = [_grip_amount(f, ref_open[lo:hi], ref_close[lo:hi]) for f in frames]
        amounts.extend(side_amounts)
        span = max(side_amounts) - min(side_amounts)
        if span >= MIN_GRIP_SPAN:
            i_open = side_amounts.index(min(side_amounts))
            i_close = side_amounts.index(max(side_amounts))
            open_pose[lo:hi] = _round(frames[i_open])
            close_pose[lo:hi] = _round(frames[i_close])
            from_bag["sg100_open_pose"] = from_bag["sg100_close_pose"] = True
            rep.note(
                f"sg100_open_pose/close_pose[{side}] 取 {topic} 开合量最小/最大帧"
                f"（帧 {i_open}/{i_close}，跨度 {span:.2f}）。"
            )
            rep.warn(
                f"{side} 手 open/close 来自 demo 实际极限而非物理极限，"
                "若该 bag 未做完整开合，请按实机标定值修正。"
            )
        else:
            rep.warn(
                f"{side} 手开合量跨度仅 {span:.2f}（< {MIN_GRIP_SPAN}），"
                "不足以标定 open/close，该侧沿用 total。"
            )

    for key, value in (
        ("sg100_open_pose", open_pose), ("sg100_close_pose", close_pose),
        ("sg100_init_pose", init_pose), ("sg100_joint_min", joint_min),
        ("sg100_joint_max", joint_max),
    ):
        rep.set(key, value, "bag+total" if from_bag[key] else "total")

    for idx, (low, high) in enumerate(zip(joint_min, joint_max)):
        if low >= high:
            raise SystemExit(
                f"sg100_joint_min[{idx}]={low} 必须小于 sg100_joint_max[{idx}]={high}"
            )

    # ---- 二值化开关 & limits.gripper -------------------------------------
    _resolve_binary(amounts, total, rep, args.sg100_binary)

    if dof == NJ:
        rep.env["limits"] = {"gripper": {"min": copy.deepcopy(joint_min),
                                         "max": copy.deepcopy(joint_max)}}
        rep.prov("limits.gripper", "bag+total")
        rep.note("limits.gripper = sg100_joint_min/max（dof=11 时动作是弧度，不能沿用 [0,1]）。")
    else:
        rep.env["limits"] = {"gripper": {"min": [0.0, 0.0], "max": [1.0, 1.0]}}
        rep.prov("limits.gripper", "total")
        rep.note("limits.gripper=[0,1]（dof=1 时动作是归一化开合量）。")

    if platform:
        rep.set("platform_type", platform, rep.source.get("platform_type", "bag"))
    else:
        rep.set("platform_type", total.get("platform_type", "4pro"), "total")
    rep.set("which_arm", which_arm, rep.source.get("which_arm", "bag"))


# ===========================================================================
# 渲染 / 校验
# ===========================================================================
def _dumper():
    class D(yaml.SafeDumper):
        pass

    def mapping(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)

    def sequence(dumper, data):
        flow = all(not isinstance(v, (dict, list)) for v in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

    D.add_representer(dict, mapping)
    D.add_representer(list, sequence)
    return D


def _ordered(env: Dict[str, Any]) -> Dict[str, Any]:
    ordered = {k: env[k] for k in ENV_KEY_ORDER if k in env}
    ordered.update({k: v for k, v in env.items() if k not in ordered})
    return ordered


def _header(bag_path: str, rep: Report) -> str:
    lines = [
        "# " + "=" * 74,
        "# 由 tools/sg100_bag_to_yaml.py 自动生成（SG100 专用）",
        f"# 来源 bag: {bag_path}",
        "# " + "-" * 74,
        "# 探测依据:",
    ]
    lines += [f"#   - {n}" for n in rep.notes]
    if rep.warnings:
        lines += ["# " + "-" * 74, "# 需要人工确认:"]
        lines += [f"#   ! {w}" for w in rep.warnings]
    if rep.todos:
        lines += ["# " + "-" * 74, "# 必须人工修改:"]
        lines += [f"#   TODO {t}" for t in rep.todos]
    lines += ["# " + "-" * 74,
              "# 来源（bag=bag 实测 / cli=命令行 / total=deploy_total.yaml）:"]
    lines += [f"#   {k}: {rep.source[k]}" for k in SG100_KEYS if k in rep.source]
    lines += ["# " + "-" * 74,
              "# 未写出的字段（相机/图像尺寸/推理参数等）由 "
              "configs/deploy/total/deploy_total.yaml 提供。",
              "# " + "=" * 74]
    return "\n".join(lines) + "\n"


def validate(config_path: str) -> int:
    """用仓库自己的 loader 校验生成结果。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from kuavo_deploy.config import load_kuavo_config
    except Exception as exc:
        print(f"[validate] 跳过：无法导入 kuavo_deploy.config（{exc}）")
        return 0
    try:
        cfg = load_kuavo_config(config_path)
    except Exception as exc:
        print(f"[validate] 失败：{exc}")
        return 1

    env = cfg.env
    action_dof = (env.arm_dof_per_side + env.eef_dof_per_side) * (
        2 if env.which_arm == "both" else 1
    )
    print("\n=== 配置校验通过 ===")
    print(f"  eef_type/which_arm : {env.eef_type} / {env.which_arm}")
    print(f"  platform_type      : {env.platform_type}")
    print(f"  arm_dof / eef_dof  : {env.arm_dof_per_side} / {env.eef_dof_per_side}"
          f"  (sg100_dof_needed={env.sg100_dof_needed}, offset={env.sg100_dof_offset})")
    print(f"  gripper_slice      : {env.gripper_slice}")
    print(f"  limits.gripper     : len={len(env.limits['gripper']['min'])}")
    print(f"  joint_min/max len  : {len(env.sg100_joint_min)} / {len(env.sg100_joint_max)}")
    print(f"  is_binary_sg100    : {env.is_binary_sg100_action}"
          f" (threshold={env.sg100_action_threshold})")
    print(f"  arm_state_keys     : {env.arm_state_keys}")
    print(f"  obs keys           : {list(env.obs_key_map.keys())}")
    print(f"  模型动作维度        : {action_dof}")
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def _parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("布尔值应为 true/false")


def _parse_floats(text: str) -> List[float]:
    """逗号/空格分隔的数组：11 个则左右手广播，22 个则左+右。"""
    items = [i for i in text.replace(" ", ",").replace("，", ",").split(",") if i]
    try:
        values = [float(i) for i in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析为数字: {text}") from exc
    if len(values) == NJ:
        return values * 2
    if len(values) == POSE_LEN:
        return values
    raise argparse.ArgumentTypeError(
        f"需要 {NJ} 个（单侧广播）或 {POSE_LEN} 个（左+右）数值，实际 {len(values)} 个"
    )


def _resolve_bag(path: str) -> str:
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.bag")))
        if not candidates:
            raise SystemExit(f"目录中没有 .bag 文件: {path}")
        print(f"[info] --bag 是目录，使用第一个 bag: {candidates[0]}")
        return candidates[0]
    if not os.path.exists(path):
        raise SystemExit(f"bag 文件不存在: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 rosbag 提取 SG100 灵巧手参数，生成推理配置 YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--bag", required=True, help="go bag 路径，或包含 .bag 的目录")
    parser.add_argument("--output", "-o", default=None,
                        help="输出路径；省略时写 configs/deploy/sg100_<which_arm>.yaml")
    parser.add_argument("--print-only", action="store_true", help="只打印，不写文件")
    parser.add_argument("--validate", action="store_true", help="生成后用 load_kuavo_config 校验")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")

    parser.add_argument("--which-arm", choices=["left", "right", "both"], default=None,
                        help="覆盖 which_arm（默认由 enable mask 推断）")
    parser.add_argument("--platform-type", choices=["4pro", "5w", "5"], default=None,
                        help="覆盖 platform_type（决定手臂维度，配错会维度不匹配）")
    parser.add_argument("--sg100-dof", type=int, choices=[1, NJ], default=None,
                        help="每手模型自由度：1（整手开合）或 11（逐关节）")
    parser.add_argument("--sg100-dof-offset", type=int, default=None,
                        help="dof=1 时选第几个关节；dof=11 时必须为 0")
    parser.add_argument("--sg100-binary", type=_parse_bool, default=None,
                        help="显式设置 is_binary_sg100_action（默认沿用 total）")
    parser.add_argument("--sg100-action-threshold", type=float, default=None,
                        help="显式设置 sg100_action_threshold（[0,1]）")

    for flag, key, desc in (
        ("--sg100-kp", "sg100_force_kp", "每手 11 个位置增益 P"),
        ("--sg100-kd", "sg100_force_kd", "每手 11 个阻尼增益 D"),
        ("--sg100-torque-ff", "sg100_force_torque_ff", "每手 11 个力矩前馈（仅 mode=9 生效）"),
        ("--sg100-output-limit", "sg100_force_output_limit", "每手 11 个输出限幅"),
        ("--sg100-velocities", "sg100_force_velocities", "每手 11 个前馈速度 rad/s"),
    ):
        parser.add_argument(flag, dest=key, type=_parse_floats, default=None, metavar="11值",
                            help=f"{desc}；逗号分隔，11 个=左右手广播，22 个=左+右")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bag_path = os.path.abspath(_resolve_bag(args.bag))
    total = yaml.safe_load(DEFAULT_TOTAL.read_text(encoding="utf-8"))["env"]

    rep = Report()
    bag = open_bag(bag_path)
    try:
        topics = bag.topics()
        missing = [t for t in SG100_TOPICS if t not in topics]
        rep.note(f"bag 话题: {', '.join(sorted(topics))}")
        if len(missing) == len(SG100_TOPICS):
            raise SystemExit(
                "bag 中没有任何 SG100 话题，本脚本只处理 SG100 灵巧手 bag。"
            )
        if missing:
            rep.warn(f"缺少话题 {missing}，对应参数沿用 deploy_total.yaml。")
        data = collect(bag, topics, rep)
        which_arm = detect_arm(data, rep, args.which_arm)
        platform = detect_platform(bag, topics, rep, args.platform_type)
    finally:
        bag.close()

    build(data, total, rep, which_arm, platform, args)

    rep.inf["go_bag_path"] = bag_path
    rep.note(f"go_bag_path={bag_path}（--bag 原样写入）")
    rep.todo(
        "policy_type / pretrained_path / 相机 / image_size 不在本脚本覆盖范围内，"
        "由 deploy_total.yaml 提供。"
    )

    out_path: Optional[str] = args.output and os.path.abspath(args.output)
    if out_path is None and not args.print_only:
        out_path = str(DEFAULT_OUTPUT_DIR / f"sg100_{which_arm}.yaml")

    text = _header(bag_path, rep) + yaml.dump(
        {"env": _ordered(rep.env), "inference": rep.inf},
        Dumper=_dumper(), allow_unicode=True, sort_keys=False,
        default_flow_style=False, width=100000,
    )

    if args.print_only:
        print(text)
    else:
        assert out_path is not None
        if os.path.exists(out_path) and not args.overwrite:
            raise SystemExit(f"输出文件已存在，加 --overwrite 覆盖: {out_path}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text, encoding="utf-8")

    print("\n=== 探测结果 ===")
    for note in rep.notes:
        print(f"  [探测] {note}")
    for warning in rep.warnings:
        print(f"  [确认] {warning}")
    for todo in rep.todos:
        print(f"  [TODO] {todo}")
    print("\n=== 参数来源 ===")
    for key in SG100_KEYS:
        if key in rep.source:
            print(f"  {key:28s} <- {rep.source[key]}")
    if out_path:
        print(f"\n已生成: {out_path}")
        print(f"运行: python kuavo_deploy/src/scripts/script.py --task go_run --config {out_path}")

    return validate(out_path) if args.validate and out_path else 0


if __name__ == "__main__":
    sys.exit(main())
