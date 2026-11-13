"""
Single merged configuration loader for Kuavo deploy.

Default behavior:
- load ./configs/deploy/deploy.yaml
- if a matching total config exists at ./configs/deploy/total/deploy_total.yaml,
  merge it first and let deploy.yaml override it

This mirrors the train-side `total + simple override` pattern.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any, Dict
import os
from pathlib import Path
from copy import deepcopy
import math
import yaml

DEFAULT_DEPLOY_CONFIG = "deploy.yaml"


def get_arm_joint_slice(platform_type: str) -> Tuple[int, int]:
    config_file = (
        Path(__file__).resolve().parent.parent / "configs" / "platform" / "platform_config.yaml"
    )
    with open(config_file, "r", encoding="utf-8") as f:
        platform_cfg = yaml.safe_load(f) or {}
    platforms = platform_cfg.get("platforms", {})
    platform = platforms.get(platform_type.lower())
    if not platform:
        raise ValueError(
            f"Unsupported platform type: {platform_type}. Supported: {list(platforms.keys())}"
        )
    return (platform["arm_joint_start"], platform["arm_joint_end"])


def get_lower_body_joint_slice(platform_type: str) -> Tuple[int, int]:
    config_file = (
        Path(__file__).resolve().parent.parent / "configs" / "platform" / "platform_config.yaml"
    )
    with open(config_file, "r", encoding="utf-8") as f:
        platform_cfg = yaml.safe_load(f) or {}
    platform = platform_cfg.get("platforms", {}).get(platform_type.lower())
    if not platform:
        raise ValueError(f"Unsupported platform type: {platform_type}")
    if "lower_body_joint_start" not in platform or "lower_body_joint_end" not in platform:
        raise ValueError(f"Platform {platform_type!r} does not define lower-body joints")
    return (platform["lower_body_joint_start"], platform["lower_body_joint_end"])

@dataclass
class Range:
    min: List[float]
    max: List[float]

@dataclass
class LimitsConfig:
    joint_q: Range = field(default_factory=lambda: Range([-3.14]*14, [3.14]*14))
    gripper: Range = field(default_factory=lambda: Range([0, 0], [1, 1]))
    lower_body: Range = field(default_factory=lambda: Range([-3.14]*4, [3.14]*4))

# -----------------------
# Environment Dataclass
# -----------------------
@dataclass
class ConfigEnv:
    inference_env: str = "real"  # "sim" or "real"
    env_name: str = "Kuavo-Sim"
    real: bool = False
    eef_type: str = "rq2f85"
    platform_type: str = "4pro"
    enable_5w_wholebody: bool = False
    enable_5w_base_move: bool = False
    lowerlock_5w: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    control_mode: str = "joint"
    which_arm: str = "both"
    head_init: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.0])
    ros_rate: int = 10
    control_rate: int = 100  # WBC/插值后的实际控制指令频率
    enable_action_interpolation: bool = True
    direct_to_wbc: bool = False  # 是否直接将动作发送到WBC
    # inline: 在推理线程内插值下发（原行为）；threaded: 独立线程按 control_rate 独占下发，
    # step() 只投递最新目标，避免推理阻塞导致的突发下发抖动。threaded 需要 direct_to_wbc=true。
    execution_mode: str = "inline"
    image_size: List[int] = field(default_factory=lambda: [640, 480])
    depth_range: List[int] = field(default_factory=lambda: [0, 1500])
    obs_key_map: Dict[str, List[Any]] = field(default_factory=dict)
    # Derived by the loader before selecting policy cameras; raw recording keeps all views.
    recording_obs_key_map: Dict[str, Any] = field(default_factory=dict)
    arm_state_keys: List[str]=field(default_factory=list)
    ratio: float = 0.5
    frame_alignment: bool = True
    qiangnao_dof_needed: int = 1
    sg100_dof_needed: int = 1
    sg100_dof_offset: int = 0
    # 姿态使用弧度，顺序为左手 11 + 右手 11（TH_CMC_ABD, TH_MCP_FLEX, TH_IP_FLEX,
    # IF_MCP_ABD, IF_MCP_FLEX, IF_PIP_FLEX, MF_MCP_FLEX, MF_PIP_FLEX,
    # LF_MCP_ABD, LF_MCP_FLEX, LF_PIP_FLEX）。
    # 注意 SG100 并非所有关节都在 [0, 1.57]：拇指 MCP 为负值区间，小指 MCP_ABD 约 ±3.1。
    sg100_open_pose: List[float] = field(default_factory=lambda: [
        1.57, -2.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.1, 0.0, 0.0,
        1.57, -2.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.1, 0.0, 0.0,
    ])
    sg100_close_pose: List[float] = field(default_factory=lambda: [
        1.57, -0.89, 1.57, 0.0, 1.57, 1.57, 1.57, 1.57, 3.1, 1.57, 1.57,
        1.57, -0.89, 1.57, 0.0, 1.57, 1.57, 1.57, 1.57, 3.1, 1.57, 1.57,
    ])
    sg100_init_pose: List[float] = field(default_factory=lambda: [
        1.57, -2.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.1, 0.0, 0.0,
        1.57, -2.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.1, 0.0, 0.0,
    ])
    # 逐关节合法区间（弧度），用于姿态校验与下发前限幅。
    # 默认值来自 heiman 5W 实机 bag 实测范围，可按需覆盖。
    sg100_joint_min: List[float] = field(default_factory=lambda: [
        0.0, -2.62, -1.05, -1.57, 0.0, -1.05, 0.0, -1.05, 0.0, 0.0, -1.05,
        0.0, -2.62, -1.05, -1.57, 0.0, -1.05, 0.0, -1.05, 0.0, 0.0, -1.05
    ])
    sg100_joint_max: List[float] = field(default_factory=lambda: [
        2.18, 0.0,  1.57,  0.0,  2.18, 1.57, 2.18, 1.57, 3.14, 2.18, 1.57,
        2.18, 0.0,  1.57,  0.0,  2.18, 1.57, 2.18, 1.57, 3.14, 2.18, 1.57,
    ])
    sg100_force_threshold: float = 0.5
    sg100_force_kp: List[float] = field(default_factory=lambda: [10.0] * 22)
    sg100_force_kd: List[float] = field(default_factory=lambda: [0.2] * 22)
    sg100_force_torque_ff: List[float] = field(default_factory=lambda: [0.8] * 22)
    sg100_force_output_limit: List[float] = field(default_factory=lambda: [0.6] * 22)
    sg100_force_velocities: List[float] = field(default_factory=lambda: [0.0] * 22)
    is_binary_sg100_action: bool = False
    sg100_action_threshold: float = 0.5
    # SG100 观测二值化的开合行程比例，与数转 dataset.sg100_binary_ratio 同名同义。
    # 打开 env.is_binary 后生效，判定关节由 sg100_dof_offset 指定。
    sg100_binary_ratio: float = 0.1

    limits: LimitsConfig = field(default_factory=LimitsConfig)
    is_binary: bool = False
    camera_encoding: str = "jpeg"  # "jpeg" or "h265"

    # -------- Validation ----------
    def validate(self):
        if (not isinstance(self.lowerlock_5w, list)
                or len(self.lowerlock_5w) != 4
                or any(type(value) is not int or value not in (0, 1)
                       for value in self.lowerlock_5w)):
            raise ValueError("env.5w_lowerlock must be a four-element array of 0 or 1")
        if self.inference_env not in ["sim", "real"]:
            raise ValueError("env.inference_env must be 'sim' or 'real'")
        if self.eef_type not in ["rq2f85", "leju_claw", "qiangnao", "sg100"]:
            raise ValueError(f"Invalid eef_type: {self.eef_type}. Valid: rq2f85, leju_claw, qiangnao, sg100")
        if self.platform_type not in ["4pro", "5w", "5"]:
            raise ValueError(f"Invalid platform_type: {self.platform_type}. Valid: 4pro, 5w, 5")
        if self.which_arm not in ["left", "right", "both"]:
            raise ValueError(f"Invalid which_arm: {self.which_arm}. Valid: left, right, both")
        if not isinstance(self.image_size, list) or len(self.image_size) != 2:
            raise ValueError("image_size must be a list [height, width]")
        if self.camera_encoding not in ["jpeg", "h265"]:
            raise ValueError(f"Invalid camera_encoding: {self.camera_encoding}. Valid: jpeg, h265")
        if self.execution_mode not in ("inline", "threaded"):
            raise ValueError(
                f"Invalid execution_mode: {self.execution_mode}. Valid: inline, threaded"
            )
        if self.execution_mode == "threaded" and not self.direct_to_wbc:
            raise ValueError("env.execution_mode=threaded requires env.direct_to_wbc=true")
        # Derive dimensions from the platform/config instead of a packed 16D assumption.
        arm_start, arm_end = get_arm_joint_slice(self.platform_type)
        arm_dof = arm_end - arm_start
        if len(self.limits["joint_q"]["max"]) != len(self.limits["joint_q"]["min"]):
            raise ValueError("joint_q min/max must have equal lengths")
        if len(self.limits["joint_q"]["min"]) != arm_dof:
            raise ValueError(f"Robot joint_q limits must contain {arm_dof} arm joints")
        if len(self.limits["gripper"]["max"]) != len(self.limits["gripper"]["min"]):
            raise ValueError("gripper min/max must have equal lengths")
        if len(self.limits["gripper"]["min"]) % 2:
            raise ValueError("gripper limits must contain equal left/right dimensions")
        if self.use_5w_wholebody:
            lower_start, lower_end = get_lower_body_joint_slice(self.platform_type)
            lower_dof = lower_end - lower_start
            lower_limits = self.limits["lower_body"]
            if not (len(lower_limits["min"]) == len(lower_limits["max"]) == lower_dof):
                raise ValueError(f"lower_body min/max must each contain {lower_dof} values")
        if self.qiangnao_dof_needed != 1: # not in [1, 7]:
            raise ValueError("qiangnao_dof_needed must be 1 now!")
            # raise ValueError("qiangnao_dof_needed must be either 1 or 7")
        if self.sg100_dof_needed not in (1, 11):
            raise ValueError("sg100_dof_needed must be 1 or 11")
        if not isinstance(self.sg100_dof_offset, int) or self.sg100_dof_offset < 0:
            raise ValueError("sg100_dof_offset must be a non-negative integer")
        if self.sg100_dof_offset + self.sg100_dof_needed > 11:
            raise ValueError("sg100_dof_offset + sg100_dof_needed must be <= 11")
        for name in (
            "sg100_open_pose", "sg100_close_pose", "sg100_init_pose",
            "sg100_joint_min", "sg100_joint_max",
            "sg100_force_kp", "sg100_force_kd", "sg100_force_torque_ff",
            "sg100_force_output_limit", "sg100_force_velocities",
        ):
            if len(getattr(self, name)) != 22:
                raise ValueError(f"{name} must contain 22 values (11 left + 11 right)")
        for idx, (low, high) in enumerate(zip(self.sg100_joint_min, self.sg100_joint_max)):
            if low >= high:
                raise ValueError(
                    f"sg100_joint_min[{idx}] ({low}) must be smaller than "
                    f"sg100_joint_max[{idx}] ({high})"
                )
        if not 0.0 <= self.sg100_force_threshold <= 1.0:
            raise ValueError("sg100_force_threshold must be in [0, 1]")
        if not 0.0 <= self.sg100_action_threshold <= 1.0:
            raise ValueError("sg100_action_threshold must be in [0, 1]")
        if not 0.0 < self.sg100_binary_ratio < 1.0:
            raise ValueError("sg100_binary_ratio must be in (0, 1)")

    # -------- Derived properties ----------
    @property
    def use_5w_wholebody(self) -> bool:
        return self.platform_type.lower() == "5w" and self.enable_5w_wholebody

    @property
    def use_5w_base_move(self) -> bool:
        return self.platform_type.lower() == "5w" and self.enable_5w_base_move

    @property
    def arm_dof_per_side(self) -> int:
        arm_start, arm_end = get_arm_joint_slice(self.platform_type)
        arm_dof = arm_end - arm_start
        if arm_dof % 2:
            raise ValueError("The configured arm joint range must split evenly between both arms")
        return arm_dof // 2

    @property
    def eef_dof_per_side(self) -> int:
        if self.eef_type == "sg100":
            return self.sg100_dof_needed
        return len(self.limits["gripper"]["min"]) // 2

    @property
    def joint_q_slice(self):
        arm_start, arm_end = get_arm_joint_slice(self.platform_type)
        left_end = arm_start + self.arm_dof_per_side
        right_start = left_end
        return {
            "left": [[arm_start, left_end]],
            "right": [[right_start, arm_end]],
            "both": [[arm_start, left_end], [right_start, arm_end]]
        }[self.which_arm]

    @property
    def gripper_slice(self):
        if self.eef_type == "rq2f85" or self.eef_type == "leju_claw":
            dof = self.eef_dof_per_side
            return {
                "left": [[0, dof]],
                "right": [[dof, dof * 2]],
                "both": [[0, dof], [dof, dof * 2]]
            }[self.which_arm]
        elif self.eef_type == "qiangnao" and self.qiangnao_dof_needed == 1:
            return {
                "left": [[0, 1]],
                "right": [[6, 7]],
                "both": [[0, 1], [6, 7]]
            }[self.which_arm]
        elif self.eef_type == "sg100":
            offset = self.sg100_dof_offset
            count = self.sg100_dof_needed
            return {
                "left": [[offset, offset + count]],
                "right": [[11 + offset, 11 + offset + count]],
                "both": [[offset, offset + count], [11 + offset, 11 + offset + count]],
            }[self.which_arm]
        else:
            raise ValueError("Unsupported eef_type or dof config")

    # ---------------- obs_key_map build ----------------
    _H265_TOPIC_MAP = {
        "head_cam_h": "/cam_h/color/h265_stream",
        "wrist_cam_l": "/cam_l/color/h265_stream",
        "wrist_cam_r": "/cam_r/color/h265_stream",
    }

    def build_obs_key_map(self) -> Dict[str, Any]:
        obs_map = {}
        for key, info in self.obs_key_map.items():
            base = {
                "topic": info[0],
                "msg_type": info[1],
                "frequency": info[2],
                "handle": {"params": {}}
            }
            # 统一规则化参数处理
            if len(info) == 4 and isinstance(info[3], list):
                base["handle"]["params"]["resize_wh"] = info[3]
            if len(info) == 5 and isinstance(info[3], list) and isinstance(info[4], list):
                base["handle"]["params"]["resize_wh"] = info[3]
                base["handle"]["params"]["depth_range"] = info[4]

            # 特殊键处理
            if key == "joint_q":
                base["handle"]["params"]["slice"] = self.joint_q_slice
            if key in ["rq2f85", "qiangnao", "leju_claw", "sg100"]:
                base["handle"]["params"]["slice"] = self.gripper_slice
                obs_map["gripper"] = base
                continue
            if key == "eef_pose" and len(info) >= 3 and info[0] == "computed":
                obs_map["eef_pose"] = {
                    "type": "computed",
                    "source": info[1],
                    "frequency": info[2]
                }
                continue
            obs_map[key] = base

        if self.camera_encoding == "h265":
            for key, obs_info in obs_map.items():
                if key in self._H265_TOPIC_MAP:
                    obs_info["topic"] = self._H265_TOPIC_MAP[key]
                    obs_info["h265"] = True

        if self.use_5w_wholebody:
            joint_info = obs_map.get("joint_q")
            if joint_info is None:
                raise ValueError("5w_wholebody requires joint_q in env.obs_key_map")
            lower_start, lower_end = get_lower_body_joint_slice(self.platform_type)
            lower_info = deepcopy(joint_info)
            lower_info["handle"]["params"]["slice"] = [[lower_start, lower_end]]
            obs_map["lower_body"] = lower_info
        return obs_map



# -----------------------
# Inference Dataclass
# -----------------------
@dataclass
class ConfigInference:
    go_bag_path: str = ""
    policy_type: str = "diffusion"  # 支持 diffusion, act 等
    pretrained_path: str = ""
    eval_episodes: int = 1
    seed: int = 42
    start_seed: int = 42
    device: str = "cuda"  # or "cpu"
    task: str = ""
    method: str = ""
    timestamp: str = ""
    epoch: int = 1
    max_episode_steps: int = 1000
    task_prompt: str = ""
    async_inference: bool = False
    actions_per_chunk: Optional[int] = None  # None: use policy n_action_steps / returned chunk length
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "weighted_average"
    async_timeout_s: float = 30.0
    rtc_enabled: bool = False
    rtc_mode: str = "guided"
    rtc_execution_horizon: int = 10
    rtc_prefix_attention_schedule: str = "exp"
    rtc_max_guidance_weight: float = 10.0
    rtc_initial_delay_steps: int = 1

    # Runtime state derived from --task, never a YAML switch.
    dagger_on: bool = field(default=False, init=False)
    rosbag_output_dir: str = "outputs/dagger"
    rosbag_min_free_gb: float = 10.0
    rosbag_buffer_mb: int = 256
    rosbag_extra_topics: List[str] = field(default_factory=list)
    dagger_vr_action_timeout: float = 0.2
    dagger_signal_timeout: float = 0.2
    dagger_handover_duration: float = 0.3  # Fade the initial arm position/velocity mismatch to VR.
    dagger_vr_candidate_topic: str = "/dagger/vr/arm_candidate"
    dagger_policy_handover_duration: float = 0.5
    dagger_vr_eef_topic: str = "/dagger/vr/eef_action"
    dagger_left_intervention_topic: str = "/dagger/vr/left_intervention"
    dagger_right_intervention_topic: str = "/dagger/vr/right_intervention"
    dagger_waist_on: bool = False  # Follow VR body pose while an active-side grip is held.
    dagger_waist_pose_max_speed: float = 1.0  # rad/s per joint; sent to KRC torso IK


    def validate(self):
        if type(self.rtc_enabled) is not bool:
            raise ValueError("rtc_enabled must be a boolean")
        if self.rtc_enabled and (not self.async_inference or self.policy_type != "client"):
            raise ValueError("RTC requires async_inference=true and policy_type=client for the external model adapters")
        if self.rtc_mode not in ("guided", "trained"):
            raise ValueError("rtc_mode must be guided or trained")
        if type(self.rtc_execution_horizon) is not int or self.rtc_execution_horizon <= 0:
            raise ValueError("rtc_execution_horizon must be a positive integer")
        if self.rtc_prefix_attention_schedule not in ("zeros", "ones", "linear", "exp"):
            raise ValueError("Unsupported rtc_prefix_attention_schedule")
        if not math.isfinite(self.rtc_max_guidance_weight) or self.rtc_max_guidance_weight <= 0:
            raise ValueError("rtc_max_guidance_weight must be finite and positive")
        if type(self.rtc_initial_delay_steps) is not int or self.rtc_initial_delay_steps < 0:
            raise ValueError("rtc_initial_delay_steps must be a non-negative integer")
        if type(self.async_inference) is not bool:
            raise ValueError("async_inference must be a boolean")
        if self.actions_per_chunk is not None and (
            type(self.actions_per_chunk) is not int or self.actions_per_chunk <= 0
        ):
            raise ValueError("actions_per_chunk must be a positive integer or null")
        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("chunk_size_threshold must be in [0, 1]")
        if self.aggregate_fn_name not in (
            "latest_only", "weighted_average", "average", "conservative", "exp", "cosine",
        ):
            raise ValueError("Unknown aggregate_fn_name")
        if not math.isfinite(self.async_timeout_s) or self.async_timeout_s <= 0:
            raise ValueError("async_timeout_s must be finite and positive")
        supported_policy_types = [
            "",
            "client",
            "act",
            "diffusion",
            "pi0",
            "pi0_fast",
            "pi05",
            "groot",
            "smolvla",
            "xvla",
            "wall_x",
            "multi_task_dit",
        ]
        if self.policy_type not in supported_policy_types:
            raise ValueError(f"Unsupported policy_type '{self.policy_type}'")
        if not isinstance(self.rosbag_output_dir, str) or not self.rosbag_output_dir.strip():
            raise ValueError("inference.rosbag_output_dir must be a nonempty path")
        if (isinstance(self.rosbag_min_free_gb, bool)
                or not isinstance(self.rosbag_min_free_gb, (int, float))
                or not math.isfinite(self.rosbag_min_free_gb) or self.rosbag_min_free_gb <= 0):
            raise ValueError("inference.rosbag_min_free_gb must be finite and > 0")
        if type(self.rosbag_buffer_mb) is not int or self.rosbag_buffer_mb <= 0:
            raise ValueError("inference.rosbag_buffer_mb must be a positive integer")
        if not isinstance(self.rosbag_extra_topics, list) or any(
                not isinstance(t, str) or not t.startswith("/") or any(c.isspace() for c in t)
                for t in self.rosbag_extra_topics):
            raise ValueError("inference.rosbag_extra_topics must be absolute ROS topic names")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")
        if self.dagger_vr_action_timeout <= 0 or self.dagger_signal_timeout <= 0:
            raise ValueError("DAgger action and signal timeouts must be > 0")
        if self.dagger_handover_duration < 0:
            raise ValueError("DAgger handover duration must be >= 0")
        if not self.dagger_vr_candidate_topic.startswith("/"):
            raise ValueError("dagger_vr_candidate_topic must be an absolute topic")
        if self.dagger_policy_handover_duration < 0:
            raise ValueError("DAgger policy handover duration must be >= 0")


# -----------------------
# Master config
# -----------------------
@dataclass
class KuavoConfig:
    env: ConfigEnv
    inference: ConfigInference

    def validate(self):
        self.env.validate()
        self.inference.validate()
        if self.inference.dagger_on:
            if not (self.env.inference_env == "real" and self.env.control_mode == "joint"
                    and self.env.direct_to_wbc):
                raise ValueError("DAgger requires real joint control and direct_to_wbc")
            for name, value in (("env.ros_rate", self.env.ros_rate),
                                ("env.control_rate", self.env.control_rate)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and > 0")
            if self.env.ros_rate > self.env.control_rate:
                raise ValueError("Policy rate must not exceed env.control_rate")
            for name in ("eval_episodes", "max_episode_steps"):
                value = getattr(self.inference, name)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"inference.{name} must be a positive integer")
            timeout = self.inference.async_timeout_s
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("DAgger async_timeout_s must be finite and > 0")
            if self.env.arm_dof_per_side != 7 or self.env.eef_dof_per_side != 1:
                raise ValueError("DAgger VR requires joint7 + eef1 per arm")
        if type(self.inference.dagger_waist_on) is not bool:
            raise ValueError("inference.dagger_waist_on must be a YAML boolean")
        if self.inference.dagger_waist_on:
            if not (self.inference.dagger_on and self.env.use_5w_wholebody
                    and self.env.inference_env == "real"):
                raise ValueError("Waist DAgger requires the dagger task, real 5w and 5w_wholebody=true")
            if all(self.env.lowerlock_5w):
                raise ValueError("Waist DAgger requires at least one unlocked lower-body joint")
            if (not math.isfinite(self.inference.dagger_waist_pose_max_speed)
                    or self.inference.dagger_waist_pose_max_speed <= 0):
                raise ValueError("dagger_waist_pose_max_speed must be finite and positive")


# -----------------------
# Loader
# -----------------------
def load_kuavo_config(config_path: Optional[str] = None, *, task: Optional[str] = None) -> KuavoConfig:
    """
    Load config from YAML.
    Default path: ./configs/deploy/deploy.yaml

    If the selected config is a simple profile and a matching total config exists
    in a sibling `total/` directory, the total config is loaded first and then the
    simple config overrides it.

    Only task="dagger" enables intervention before validation and environment
    creation. With no task, intervention remains disabled. The YAML on disk is
    never modified.
    """
    if task not in (None, "go", "run", "dagger", "go_run", "here_run", "back_to_zero",
                    "auto_test", "offline_bag"):
        raise ValueError(f"Unknown deployment task: {task!r}")
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "../configs", "deploy", DEFAULT_DEPLOY_CONFIG)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.abspath(config_path)

    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _resolve_template(value: Any, scope: Dict[str, Any]) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            expr = value[2:-1]
            parts = expr.split(".")
            cur: Any = scope
            for part in parts:
                if not isinstance(cur, dict) or part not in cur:
                    raise KeyError(f"Unknown config template reference: {value}")
                cur = cur[part]
            return deepcopy(cur)
        if isinstance(value, list):
            return [_resolve_template(v, scope) for v in value]
        if isinstance(value, dict):
            return {k: _resolve_template(v, scope) for k, v in value.items()}
        return value

    def _apply_inference_env_defaults(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        env_cfg = cfg_dict.setdefault("env", {})
        inference_env = env_cfg.get("inference_env", "real")

        if inference_env == "sim":
            env_cfg["env_name"] = "Kuavo-Sim"
            env_cfg["real"] = False

        else:
            env_cfg["env_name"] = "Kuavo-Real"
            env_cfg["real"] = True
            env_cfg.setdefault("platform_type", "4pro")
            env_cfg.setdefault("eef_type", "leju_claw")
            if env_cfg["eef_type"] not in {"leju_claw", "qiangnao", "sg100"}:
                raise ValueError("When inference_env=real, eef_type must be 'leju_claw', 'qiangnao', or 'sg100'")
            env_cfg["head_init"] = None
            env_cfg["image_size"] = [848, 480]

        return cfg_dict

    def _filter_obs_key_map_for_eef(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        env_cfg = cfg_dict.get("env", {})
        eef_type = env_cfg.get("eef_type")
        obs_key_map = env_cfg.get("obs_key_map", {})
        if not isinstance(obs_key_map, dict):
            return cfg_dict

        filtered = {}
        for key, value in obs_key_map.items():
            if key in {"rq2f85", "leju_claw", "qiangnao", "sg100"} and key != eef_type:
                continue
            filtered[key] = value
        env_cfg["obs_key_map"] = filtered
        return cfg_dict

    cfg = _load_yaml(config_path)

    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    total_path = os.path.join(config_dir, "total", config_name.replace(".yaml", "_total.yaml"))
    fallback_total_path = os.path.join(current_dir, "../configs", "deploy", "total", "deploy_total.yaml")
    is_total_config = (
        os.path.basename(os.path.dirname(config_path)) == "total"
        or config_name.endswith("_total.yaml")
    )
    if not is_total_config:
        base_total_path = total_path if os.path.exists(total_path) else fallback_total_path
        if os.path.exists(base_total_path):
            cfg = _deep_merge(_load_yaml(base_total_path), cfg)

    cfg = _apply_inference_env_defaults(cfg)
    cfg = _resolve_template(cfg, cfg)
    cfg = _filter_obs_key_map_for_eef(cfg)

    # YAML keeps the user-facing option name requested for 5W, while the
    # dataclass uses a valid Python identifier internally.
    for option_key, field_name in (
        ("5w_wholebody", "enable_5w_wholebody"),
        ("5w_base_move", "enable_5w_base_move"),
        ("5w_lowerlock", "lowerlock_5w"),
    ):
        if isinstance(cfg.get("env"), dict) and option_key in cfg["env"]:
            cfg["env"][field_name] = cfg["env"].pop(option_key)
        elif option_key in cfg:
            cfg[field_name] = cfg.pop(option_key)

    # The user's original YAML was mostly top-level keys (not nested under env/inference).
    # We'll support both styles:
    #  - top-level flat (as your original): keys like 'real', 'policy_type', ...
    #  - nested style: {env: {...}, inference: {...}}
    if 'env' in cfg and 'inference' in cfg:
        env_cfg: Dict[str, Any] = cfg.get('env', {})
        inf_cfg: Dict[str, Any] = cfg.get('inference', {})
    else:
        # 自动根据 dataclass 字段划分 env / inference
        env_fields = set(ConfigEnv.__dataclass_fields__.keys())
        inf_fields = set(ConfigInference.__dataclass_fields__.keys())

        env_cfg = {}
        inf_cfg = {}

        for k, v in cfg.items():
            if k in env_fields:
                env_cfg[k] = v
            elif k in inf_fields:
                inf_cfg[k] = v
            elif k == "limits" and isinstance(v, dict):
                def dict_to_range(d):
                    return Range(d.get("min", []), d.get("max", []))

                env_cfg["limits"] = LimitsConfig(
                    joint_q=dict_to_range(v.get("joint_q", {})),
                    gripper=dict_to_range(v.get("gripper", {})),
                    lower_body=dict_to_range(v.get("lower_body", {})),
                )
            else:
                env_cfg[k] = v
    # Merge defaults with provided config
    default_env = ConfigEnv()
    default_inf = ConfigInference()

    merged_env = {**asdict(default_env), **env_cfg}
    if isinstance(env_cfg.get("limits"), dict):
        merged_env["limits"] = _deep_merge(asdict(default_env.limits), env_cfg["limits"])
    if merged_env["eef_type"] == "sg100" and merged_env["sg100_dof_needed"] == 11:
        for bound in ("min", "max"):
            values = merged_env["limits"]["gripper"][bound]
            if len(values) == 2:
                merged_env["limits"]["gripper"][bound] = [values[0]] * 11 + [values[1]] * 11
    merged_inf = {**{k: v for k, v in asdict(default_inf).items()
                    if ConfigInference.__dataclass_fields__[k].init}, **inf_cfg}

    env = ConfigEnv(**merged_env)
    inference = ConfigInference(**merged_inf)

    config = KuavoConfig(env=env, inference=inference)
    config.inference.dagger_on = task == "dagger"
    if not config.inference.dagger_on:
        config.inference.dagger_waist_on = False
    config.env.obs_key_map = config.env.build_obs_key_map()
    config.env.recording_obs_key_map = deepcopy(config.env.obs_key_map)
    # Model observations use dev's arm-specific cameras for every task.
    # Recording keeps the complete physical topic map independently.
    excluded = {"left": {"wrist_cam_r", "depth_r"},
                "right": {"wrist_cam_l", "depth_l"}}.get(config.env.which_arm, set())
    config.env.obs_key_map = {key: spec for key, spec in config.env.obs_key_map.items()
                              if key not in excluded}
    config.validate()
    return config


# -----------------------
# Quick test when run as script
# -----------------------
if __name__ == "__main__":
    cfg = load_kuavo_config()
    print(isinstance(cfg, KuavoConfig))
    print("=== Env basic ===")
    print("eef_type:", cfg.env.eef_type)
    print("eef_name:", cfg.env.env_name)
    print("which_arm:", cfg.env.which_arm)
    print("cam keys:", cfg.env.obs_key_map)
    print("slice_robot:", cfg.env.gripper_slice)
    print("qiangnao_slice:", cfg.env.joint_q_slice)
    print("=== Inference basic ===")
    print("policy_type:", cfg.inference.policy_type)
    print("device:", cfg.inference.device)
    print("arm_state_keys",cfg.env.arm_state_keys)
