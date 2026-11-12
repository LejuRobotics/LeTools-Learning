#!/usr/bin/env python3
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import Twist
import cv2
import gymnasium as gym
import threading
import time
import sys
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot, KuavoRobotState, DexterousHand
from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import lejuClawCommand
from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import (changeArmCtrlMode, changeArmCtrlModeRequest, changeLbQuickModeSrv, changeLbQuickModeSrvRequest)
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.ros_manager import ROSManager
import traceback
import torch
from torchvision.transforms.functional import to_tensor
from kuavo_deploy.utils.obs_buffer import ObsBuffer
from kuavo_deploy.utils.dagger_layout import DaggerLayout
from kuavo_deploy.utils.signal_controller import ControlSignalManager
from kuavo_deploy.utils.arm_velocity_transport import SDKArmVelocityTransport, validate_arm_vector
from kuavo_deploy.utils.interp_thread import ControlLoop
from kuavo_deploy.utils.sg100_msgs import SG100HandCommand
from std_srvs.srv import SetBool

log_robot = setup_logger("robot")
BASE_VELOCITY_DOF = 3


class KuavoBaseRosEnv(gym.Env):
    """Kuavo机器人ROS环境基类"""

    WHEEL_QUICK_MODE_DISABLED = 0
    WHEEL_QUICK_MODE_ARM = 2
    WHEEL_ARM_TRANSPORT_NONE = 0
    WHEEL_ARM_TRANSPORT_ROS = 2

    def __init__(self, config: KuavoConfig):
        self.dagger_on = config.inference.dagger_on
        self.dagger_session = None
        self._arm_velocity_transport = None
        self._wheel_arm_control_configured = False
        self._wheel_arm_cleanup_required = False
        self.single_arm_dagger = self.dagger_on and config.env.which_arm != "both"
        self._held_arm = None
        self._held_eef = None
        if self.dagger_on:
            from kuavo_deploy.utils.dagger_action_arbiter import validate_dagger_vr_mode
            validate_dagger_vr_mode(config)
        # 保护 _last_lower_cmd / _last_base_cmd：threaded 模式下由下发线程写入，
        # reset() 在推理线程读取。
        self._last_cmd_lock = threading.Lock()
        self._last_lower_cmd = None
        self._last_base_cmd = None
        self._init_lower_body_lock(config)
        self._set_config(config.env)
        
        # 初始化ROS管理器
        self.ros_manager = ROSManager()
        self.control_signal_manager = ControlSignalManager()
        
        # 初始化其他组件
        self.bridge = CvBridge()
        self._set_observation_space()
        self._set_action_space()
        if self.dagger_on:
            self.action_layout = DaggerLayout(self)
        self._init_kuavo_sdk()
        self._set_ros_topics()
        try:
            self.prepare_inactive_hold()
        except Exception:
            self.close()
            raise
        
        # 等待ROS话题初始化
        log_robot.info(f"Inializing done!")
        print(f"Inializing done!")

    def _init_lower_body_lock(self, config: KuavoConfig):
        """Cache the configured go bag's first lower-body target in radians."""
        self.lower_body_lock_mask = np.asarray(config.env.lowerlock_5w, dtype=bool)
        self.lower_body_lock_target = None
        if not config.env.use_5w_wholebody or not self.lower_body_lock_mask.any():
            return
        bag_path = config.inference.go_bag_path
        if not bag_path:
            raise ValueError("env.5w_lowerlock requires inference.go_bag_path")

        import rosbag

        with rosbag.Bag(bag_path, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=["/lb_leg_traj"]):
                positions = np.asarray(msg.position, dtype=float)
                if positions.shape != (4,) or not np.isfinite(positions).all():
                    raise ValueError(
                        "env.5w_lowerlock requires four finite positions in the "
                        "first /lb_leg_traj frame of inference.go_bag_path"
                    )
                self.lower_body_lock_target = np.deg2rad(positions)
                break
        if self.lower_body_lock_target is None:
            raise ValueError(
                "env.5w_lowerlock requires /lb_leg_traj in inference.go_bag_path"
            )

    def _set_config(self, config_kuavo_env):
        """设置配置参数"""
        self.platform_type = config_kuavo_env.platform_type
        self.use_5w_wholebody = config_kuavo_env.use_5w_wholebody
        self.use_5w_base_move = config_kuavo_env.use_5w_base_move
        self.arm_dof_per_side = config_kuavo_env.arm_dof_per_side
        self.arm_dof = self.arm_dof_per_side * 2
        self.eef_dof_per_side = config_kuavo_env.eef_dof_per_side
        self.ros_rate = config_kuavo_env.ros_rate
        self.control_mode = config_kuavo_env.control_mode
        self.obs_key_map = config_kuavo_env.obs_key_map
        self.eef_type = config_kuavo_env.eef_type
        self.which_arm = config_kuavo_env.which_arm
        self.direct_to_wbc = config_kuavo_env.direct_to_wbc
        self.qiangnao_dof_needed = config_kuavo_env.qiangnao_dof_needed
        self.sg100_dof_needed = config_kuavo_env.sg100_dof_needed
        self.sg100_dof_offset = config_kuavo_env.sg100_dof_offset
        self.sg100_open_pose = np.asarray(config_kuavo_env.sg100_open_pose, dtype=np.float64)
        self.sg100_close_pose = np.asarray(config_kuavo_env.sg100_close_pose, dtype=np.float64)
        self.sg100_init_pose = np.asarray(config_kuavo_env.sg100_init_pose, dtype=np.float64)
        self.sg100_joint_min = np.asarray(config_kuavo_env.sg100_joint_min, dtype=np.float64)
        self.sg100_joint_max = np.asarray(config_kuavo_env.sg100_joint_max, dtype=np.float64)
        self.sg100_force_threshold = config_kuavo_env.sg100_force_threshold
        self.sg100_force_kp = np.asarray(config_kuavo_env.sg100_force_kp, dtype=np.float64)
        self.sg100_force_kd = np.asarray(config_kuavo_env.sg100_force_kd, dtype=np.float64)
        self.sg100_force_torque_ff = np.asarray(config_kuavo_env.sg100_force_torque_ff, dtype=np.float64)
        self.sg100_force_output_limit = np.asarray(config_kuavo_env.sg100_force_output_limit, dtype=np.float64)
        self.sg100_force_velocities = np.asarray(config_kuavo_env.sg100_force_velocities, dtype=np.float64)
        self.is_binary_sg100_action = config_kuavo_env.is_binary_sg100_action
        self.sg100_action_threshold = config_kuavo_env.sg100_action_threshold
        self.control_rate_hz = getattr(config_kuavo_env, "control_rate", 100)
        self.enable_action_interpolation = getattr(config_kuavo_env, "enable_action_interpolation", True)
        self.interpolation_steps = (
            max(1, int(round(self.control_rate_hz / self.ros_rate)))
            if self.enable_action_interpolation
            else 1
        )

        self.is_binary = config_kuavo_env.is_binary
        self.head_init = config_kuavo_env.head_init
        self.arm_init = np.zeros(self.arm_dof, dtype=np.float64)


        # 从配置中获取limits部分
        self.limits = config_kuavo_env.limits
        self.obs_key_map = config_kuavo_env.obs_key_map
        self.obs_buffer = ObsBuffer(
            config=config_kuavo_env, 
            obs_key_map=self.obs_key_map,
            cache_full_arm=self.single_arm_dagger,
        )
        self.arm_state_keys = config_kuavo_env.arm_state_keys # observation.state 的key顺序
        self.ratio = config_kuavo_env.ratio
        self.frame_alignment = config_kuavo_env.frame_alignment
        self.execution_mode = getattr(config_kuavo_env, "execution_mode", "inline")
        # threaded 模式：由独立线程按 control_rate_hz 独占下发，step() 只投递最新目标。
        self.interp_threaded = (
            self.execution_mode == "threaded"
            and self.direct_to_wbc
            and self.enable_action_interpolation
        )
        self.control_loop = None
        if self.direct_to_wbc:
            self.last_predicted_action = None
            self.last_sent_action = None
            self.last_sent_arm_velocity = None
            self.is_first_step = True
            log_robot.info(
                f"Direct-to-WBC action interpolation: enabled={self.enable_action_interpolation}, "
                f"inference_rate={self.ros_rate}Hz, control_rate={self.control_rate_hz}Hz, "
                f"interpolation_steps={self.interpolation_steps}, "
                f"execution_mode={self.execution_mode}, threaded={self.interp_threaded}"
            )

    @property
    def _controlled_sides(self):
        return ("left", "right") if self.which_arm == "both" else (self.which_arm,)

    def _split_action(self, action):
        """Split the configured packed action without fixed 8D/16D offsets."""
        action = np.asarray(action)
        components = {}
        offset = 0
        for side in self._controlled_sides:
            joint_end = offset + self.arm_dof_per_side
            eef_end = joint_end + self.eef_dof_per_side
            components[side] = {
                "joints": action[offset:joint_end],
                "eef": action[joint_end:eef_end],
                "eef_slice": slice(joint_end, eef_end),
            }
            offset = eef_end
        if self.use_5w_wholebody:
            lower_dof = len(self.limits["lower_body"]["min"])
            lower_end = offset + lower_dof
            components["lower_body"] = action[offset:lower_end]
            offset = lower_end
        else:
            components["lower_body"] = np.array([])
        if self.use_5w_base_move:
            base_end = offset + BASE_VELOCITY_DOF
            components["base_velocity"] = action[offset:base_end]
        else:
            components["base_velocity"] = np.array([])
        return components

    def _full_arm_target(self, components):
        left = components.get("left", {}).get("joints", self.arm_init[:self.arm_dof_per_side])
        right = components.get("right", {}).get("joints", self.arm_init[self.arm_dof_per_side:])
        return np.concatenate((left, right), axis=0)

    def _set_observation_space(self):
        limits = self.limits
        obs_low, obs_high = [], []

        # -------- 构建 state 空间（joint_q + gripper） --------
        if 'joint_q' in self.obs_key_map:
            joint_min, joint_max = limits['joint_q']['min'], limits['joint_q']['max']
        else:
            joint_min, joint_max = [], []
        if 'gripper' in self.obs_key_map:
            grip_min, grip_max = limits['gripper']['min'], limits['gripper']['max']
        else:
            grip_min, grip_max = [], []
        for side_index, side in enumerate(("left", "right")):
            if side not in self._controlled_sides:
                continue
            arm_start = side_index * self.arm_dof_per_side
            arm_end = arm_start + self.arm_dof_per_side
            eef_start = side_index * self.eef_dof_per_side
            eef_end = eef_start + self.eef_dof_per_side
            obs_low.extend(joint_min[arm_start:arm_end] + grip_min[eef_start:eef_end])
            obs_high.extend(joint_max[arm_start:arm_end] + grip_max[eef_start:eef_end])
        if self.use_5w_wholebody:
            obs_low.extend(limits['lower_body']['min'])
            obs_high.extend(limits['lower_body']['max'])

        self.obs_low = np.array(obs_low)
        self.obs_high = np.array(obs_high)

        # -------- 构建图像空间 --------
        obs_spaces = {}
        for key, obs_name in self.obs_key_map.items():
            if any(tag in key for tag in ['cam', 'depth']):
                h, w = obs_name["handle"]["params"]["resize_wh"]
                if 'depth' in key:
                    low, high = obs_name['handle']['params']['depth_range']
                    obs_spaces[f"observation.{key}"] = gym.spaces.Box(
                        low=low, high=high, shape=(1, h, w), dtype=np.uint16
                    )
                else:  # cam 类键
                    obs_spaces[f"observation.images.{key}"] = gym.spaces.Box(
                        low=0, high=255, shape=(3, h, w), dtype=np.uint8
                    )

        # -------- 添加 state 空间 --------
        obs_spaces["observation.state"] = gym.spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32,
            shape=(len(self.obs_low),)
        )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _set_action_space(self):
        limits = self.limits
        if self.control_mode != 'joint':
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        arm_low, arm_high = [], []
        for side_index, side in enumerate(("left", "right")):
            if side not in self._controlled_sides:
                continue
            arm_start = side_index * self.arm_dof_per_side
            arm_end = arm_start + self.arm_dof_per_side
            eef_start = side_index * self.eef_dof_per_side
            eef_end = eef_start + self.eef_dof_per_side
            arm_low.extend(limits['joint_q']['min'][arm_start:arm_end] + limits['gripper']['min'][eef_start:eef_end])
            arm_high.extend(limits['joint_q']['max'][arm_start:arm_end] + limits['gripper']['max'][eef_start:eef_end])
        if self.use_5w_wholebody:
            arm_low.extend(limits['lower_body']['min'])
            arm_high.extend(limits['lower_body']['max'])
        if self.use_5w_base_move:
            # No project-level velocity limits are configured. Preserve model
            # outputs here; safe limits should be enforced by the base controller.
            arm_low.extend([-np.inf] * BASE_VELOCITY_DOF)
            arm_high.extend([np.inf] * BASE_VELOCITY_DOF)

        # ===============================
        # 创建 Gym Box 空间
        # ===============================
        self.action_space = gym.spaces.Box(
            low=np.array(arm_low, dtype=np.float64),
            high=np.array(arm_high, dtype=np.float64),
            dtype=np.float64,
        )

    def _init_kuavo_sdk(self):
        """初始化Kuavo SDK"""
        if not KuavoSDK().Init():
            log_robot.error("Init KuavoSDK failed, exit!")
            sys.exit(1)
        self.robot = KuavoRobot()
        self.robot_state = KuavoRobotState()
        self._cache_arm_joint_limits()
        if self.dagger_on:
            self._arm_velocity_transport = SDKArmVelocityTransport(self.robot)

    def _cache_arm_joint_limits(self):
        """把手臂关节限位缓存起来，避免每次下发都重新拉取并解析 URDF。

        SDK 的 ``control_arm_joint_positions()`` 每次下发都会调用
        ``KuavoRobotInfo.get_arm_joint_limits()``，而该方法无缓存，每次都重做
        ``rospy.has_param`` + ``rospy.get_param`` 两次**同步 XML-RPC** 到 ROS
        master（后者不填 rospy 缓存），外加一次 ``ET.fromstring(URDF)`` 与每个
        关节一次全树 find。实测拆分（N=100，交错测量）：

            whole get_arm_joint_limits           = 16.53 ms
              humanoid_description (2x RPC)      = 14.54 ms   <- 88%
              ET.fromstring + 14x find           =  2.54 ms   <- 15%

        主项是 RPC，且它随 master/CPU 负载剧烈波动（0.2ms ~ 14.5ms 都见过），
        这正是控制线程 dt_std 抖动（7~11ms）以及 inline 时快时慢的来源：
        control_rate=100Hz 下每次下发有 10ms 预算，RPC 一旦超过它，
        ``control_rate.sleep()`` 就吸收不掉，inline 的 3 次插值会串行变慢。
        URDF 运行期不变，故启动时解析一次即可；同时把这个方差一起消掉。

        解析失败/回退到 ±π 时不缓存，保持原行为以便后续可以重试。
        """
        try:
            info = self.robot._robot_info
            arm_min, arm_max = info.get_arm_joint_limits()
            if len(arm_min) != info.arm_joint_dof:
                log_robot.warning("[SDK] joint limits incomplete, keep stock path")
                return
            if np.allclose(arm_min, -np.pi) and np.allclose(arm_max, np.pi):
                # 这是 SDK 的默认回退值，说明没读到 URDF，不能固化。
                log_robot.warning("[SDK] joint limits fell back to ±pi, not caching")
                return
            info.get_arm_joint_limits = lambda: (list(arm_min), list(arm_max))
            log_robot.info(
                f"[SDK] arm joint limits cached at startup ({info.arm_joint_dof} joints)"
            )
        except Exception as e:
            log_robot.warning(f"[SDK] joint limit cache skipped: {e}")

    def _set_ros_topics(self):
        """设置ROS话题"""
        self.rate = rospy.Rate(self.ros_rate)
        self.control_rate = rospy.Rate(self.control_rate_hz)
        
        if self.eef_type == 'rq2f85':
            self.pub_eef_joint = self.ros_manager.register_publisher('/gripper/command', JointState, queue_size=10)
        elif self.eef_type == 'leju_claw':
            self.lejuclaw = LejuClaw()
        elif self.eef_type == 'qiangnao':
            self.qiangnao = DexterousHand()
        elif self.eef_type == 'sg100':
            self.sg100 = SG100Hand(ros_manager=self.ros_manager)
        if self.use_5w_base_move:
            self.cmd_vel_pub = self.ros_manager.register_publisher(
                '/cmd_vel', Twist, queue_size=10
            )
        # obs buffer 初始化
        self.obs_buffer.wait_buffer_ready()


    def prepare_inactive_hold(self):
        if not self.single_arm_dagger or self._held_arm is not None:
            return
        if self.eef_type not in ("qiangnao", "leju_claw"):
            raise ValueError("Single-arm DAgger supports qiangnao or leju_claw complete hand control")
        device = self.qiangnao if self.eef_type == "qiangnao" else self.lejuclaw
        if not callable(getattr(device, "control", None)):
            raise RuntimeError("SDK has no complete end-effector control; refusing motion")
        full = self.obs_buffer.full_arm_feedback(0.5)
        eef = self.obs_buffer.full_eef_feedback(0.5)
        inactive = slice(7, 14) if self.which_arm == "left" else slice(0, 7)
        low, high = (np.asarray(self.limits["joint_q"][bound]) for bound in ("min", "max"))
        if (full.shape != (14,) or not np.isfinite(full).all()
                or np.any(full[inactive] < low[inactive]) or np.any(full[inactive] > high[inactive])):
            raise ValueError("Inactive arm requires valid in-limit startup feedback")
        size = 12 if self.eef_type == "qiangnao" else 2
        if eef.shape != (size,) or not np.isfinite(eef).all() or np.any((eef < 0) | (eef > 100)):
            raise ValueError("Inactive hand requires complete startup feedback in 0..100")
        self._held_arm, self._held_eef = full.copy(), eef.copy()
        self.arm_init = full.copy()
        # Stop collecting complete arm/hand feedback after the startup snapshot.
        self.obs_buffer._cache_full_arm = False

    def reset(self, **kwargs):
        """重置机器人状态"""
        self.prepare_inactive_hold()
        self.stop_base()
        self._enter_external_control_mode()
        self._reset_head()
        self._reset_eef()

        # === 平均当前观测和位姿 ===
        avg_data = self._compute_average_state(average_num=10)

        # === 更新状态 ===
        self.cur_state = avg_data["state"]
        self.cur_joint_angles_action = avg_data["joint_action"]

        obs = self.get_obs()
        self.sleep_time = 0
        self.average_sleep_time = 0

        if self.direct_to_wbc:
            self.last_predicted_action = None
            self.last_sent_action = None
            self.last_sent_arm_velocity = None
            self.is_first_step = True

        if self.interp_threaded:
            self._start_control_loop()
        return obs, {}

    def _start_control_loop(self):
        """启动固定频率的下发线程，以当前实测状态为初值。"""
        if self.control_loop is not None and self.control_loop.running:
            self.control_loop.stop()
        self.control_loop = ControlLoop(
            self,
            control_hz=self.control_rate_hz,
            segment_dt=1.0 / self.ros_rate,
            log_every_n_ticks=500,  # 每 5s 打一次 dt 统计，便于判断线程是否被抢
        )
        # 优先复用最后一条实际下发值（threaded 模式下由下发线程写入），
        # 没有缓存时才退回实测状态，避免启动/重启瞬间出现位置跳变。
        with self._last_cmd_lock:
            cached_lower = None if self._last_lower_cmd is None else self._last_lower_cmd.copy()
            cached_base = None if self._last_base_cmd is None else self._last_base_cmd.copy()
        lower0 = None
        if self.use_5w_wholebody:
            lower0 = cached_lower
            if lower0 is None:
                lower0 = np.asarray(self.arm_state.get("lower_body", []), dtype=float)
                if lower0.shape != (len(self.limits["lower_body"]["min"]),):
                    lower0 = None
        base0 = cached_base if self.use_5w_base_move else None
        if base0 is None and self.use_5w_base_move:
            base0 = np.zeros(BASE_VELOCITY_DOF, dtype=float)
        # 末端以当前开合观测为初值，避免第一帧直接跳到模型目标。
        eef0 = self.arm_state.get("gripper")
        if eef0 is not None:
            eef0 = np.asarray(eef0, dtype=float)
            if eef0.shape != (2 * self.eef_dof_per_side,):
                eef0 = None
        self.control_loop.start(
            q0=self.cur_joint_angles_action,
            lower0=lower0,
            base0=base0,
            eef0=eef0,
        )

    def set_action_target(self, action):
        """threaded 模式下由 step() 调用：非阻塞地把最新目标交给下发线程。"""
        if self.control_loop is None or not self.control_loop.running:
            return
        session = self.dagger_session
        if session is not None and session.active:
            self.control_loop.set_target(action, generation=session.target_token())
        else:
            self.control_loop.set_target(action)

    # ==========================================================
    # 子函数 1. 外部控制模式设置
    # ==========================================================
    def _set_direct_to_wbc(self, control_mode):
        rospy.wait_for_service('/enable_wbc_arm_trajectory_control', timeout=5)
        try:
            change_mode = rospy.ServiceProxy('/enable_wbc_arm_trajectory_control', changeArmCtrlMode)
            req = changeArmCtrlModeRequest()
            req.control_mode = control_mode
            res = change_mode(req)
            if res.result:
                rospy.loginfo("wbc轨迹控制模式已更改为 %d", control_mode)
            else:
                rospy.logerr("无法将wbc轨迹控制模式更改为 %d", control_mode)
        except rospy.ServiceException as e:
            rospy.logerr("服务调用失败: %s", e)


    def _call_enable_arm_quick_mode(self, enable):
        """调用手臂快速模式切换服务"""
        try:
            # 等待服务可用
            rospy.loginfo(f"等待服务 /enable_lb_arm_quick_mode 可用...")
            rospy.wait_for_service('/enable_lb_arm_quick_mode', timeout=5.0)

            # 创建服务客户端
            service_client = rospy.ServiceProxy('/enable_lb_arm_quick_mode', changeLbQuickModeSrv)

            # 调用服务: quickMode 2 = 启动上肢快模式, 0 = 关闭
            rospy.loginfo(f"调用服务: {'启用' if enable else '禁用'}手臂快速模式")
            req = changeLbQuickModeSrvRequest()
            req.quickMode = 2 if enable else 0
            response = service_client(req)

            # 处理响应
            if response.success:
                rospy.loginfo(f"✓ 成功{'启用' if enable else '禁用'}手臂快速模式")
                if response.message:
                    rospy.loginfo(f"  消息: {response.message}")
                return True
            else:
                rospy.logwarn(f"✗ 服务调用失败")
                if response.message:
                    rospy.logwarn(f"  消息: {response.message}")
                return False

        except rospy.ROSException as e:
            rospy.logerr(f"服务不可用: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"服务调用异常: {e}")
            return False

    def _call_enable_arm_traj_interpolator(self, enable: bool) -> bool:
        """Enable the 500 Hz wheel-arm interpolator before quick-mode control."""
        service_name = "/enable_arm_traj_interpolator"
        try:
            rospy.loginfo(
                "等待服务 %s 可用，目标 enable=%s...", service_name, enable
            )
            rospy.wait_for_service(service_name, timeout=5.0)
            service_client = rospy.ServiceProxy(service_name, SetBool)
            response = service_client(enable)
            if response.success:
                rospy.loginfo("✓ wheel 手臂轨迹插补已设置为 %s", enable)
                if response.message:
                    rospy.loginfo("  消息: %s", response.message)
                return True
            rospy.logerr(
                "✗ wheel 手臂轨迹插补设置失败 (enable=%s): %s",
                enable,
                response.message,
            )
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr("wheel 手臂轨迹插补服务调用失败: %s", exc)
        return False

    def _link_wheel_arm_traj(self, transport: int) -> bool:
        """Select the wheel WBC arm-trajectory transport."""
        from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import SetIncrementalArmTrajLink, SetIncrementalArmTrajLinkRequest
        service_name = "/humanoid_wheel/set_incremental_arm_traj_link"
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            service_client = rospy.ServiceProxy(service_name, SetIncrementalArmTrajLink)
            request = SetIncrementalArmTrajLinkRequest()
            request.transport = transport
            response = service_client(request)
            if response.success:
                rospy.loginfo(
                    "✓ wheel 手臂动作 transport 已设置为 %d", transport
                )
                return True
            rospy.logerr(
                "wheel 手臂动作 transport=%d 设置失败: %s",
                transport,
                response.message,
            )
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr("wheel 手臂动作 transport 设置失败: %s", exc)
        return False

    def _restore_wheel_arm_control(self, *, force: bool = False) -> bool:
        """Restore safe wheel-arm defaults without exposing unfiltered quick mode."""
        if not force and not self._wheel_arm_cleanup_required:
            return True

        # Disable quick mode first. If this fails, keep interpolation and the
        # transport in place: removing either while quickMode=2 may still be
        # active would re-expose the step-and-velocity-spike path.
        if not self._call_enable_arm_quick_mode(enable=False):
            rospy.logerr(
                "无法确认 quickMode=0；为安全起见保留插补和 transport，请重启 KSC"
            )
            return False

        transport_reset = self._link_wheel_arm_traj(self.WHEEL_ARM_TRANSPORT_NONE)
        interpolator_reset = self._call_enable_arm_traj_interpolator(False)
        restored = transport_reset and interpolator_reset
        self._wheel_arm_control_configured = False
        self._wheel_arm_cleanup_required = not restored
        if restored:
            rospy.loginfo("✓ wheel 手臂 quick mode、transport 和插补已恢复默认值")
        else:
            rospy.logerr(
                "quickMode 已关闭，但 transport/插补未完全复位；下次启动前请检查 KSC"
            )
        return restored

    def _configure_wheel_arm_control(self) -> None:
        """Atomically prepare the smoothed /kuavo_arm_traj quick-mode path."""
        if self._wheel_arm_control_configured:
            return

        self._wheel_arm_cleanup_required = True
        steps = (
            (
                self._call_enable_arm_traj_interpolator,
                True,
                "无法启用 wheel 手臂轨迹插补",
            ),
            (
                self._link_wheel_arm_traj,
                self.WHEEL_ARM_TRANSPORT_ROS,
                "无法将 wheel 手臂输入绑定到 /kuavo_arm_traj",
            ),
            (
                self._call_enable_arm_quick_mode,
                True,
                "无法启用 wheel 上肢 quickMode=2",
            ),
        )
        for operation, value, error_message in steps:
            if operation(value):
                continue
            self._restore_wheel_arm_control(force=True)
            raise RuntimeError(f"{error_message}；为避免实机阶梯抖动，已中止控制初始化")

        self._wheel_arm_control_configured = True
        rospy.loginfo(
            "✓ wheel 手臂控制链初始化完成: interpolator=true, "
            "transport=/kuavo_arm_traj, quickMode=2"
        )

    def _enter_external_control_mode(self):
        self.robot.set_external_control_arm_mode()
        print("set_external_control_arm_mode", self.robot_state.arm_control_mode())
        # Wheel quick mode and the /kuavo_arm_traj -> WBC route are needed by
        # direct WBC execution and by DAgger's single-output arbitration path.
        # Keep the legacy 5w control architecture untouched otherwise.
        if self.platform_type == "5w" and self.dagger_on:
            self._configure_wheel_arm_control()
        if self.direct_to_wbc:
            if self.platform_type in ["4pro","5"]:
                self._set_direct_to_wbc(1)
            elif not self.dagger_on:
                # 100 Hz 位置流必须配合 500 Hz 插补，否则 quickMode=2 会阶梯抖动。
                self._wheel_arm_cleanup_required = True
                if not self._call_enable_arm_traj_interpolator(True):
                    raise RuntimeError("无法启用 wheel 手臂轨迹插补，已中止 quickMode=2")
                if not self._call_enable_arm_quick_mode(enable=True):
                    raise RuntimeError("无法启用 wheel 上肢 quickMode=2")


    # ==========================================================
    # 子函数 2. 头部复位
    # ==========================================================
    def _reset_head(self):
        if self.head_init is not None:
            self.robot.control_head(self.head_init[0], self.head_init[1])

    # ==========================================================
    # 子函数 3. 末端执行器（夹爪）复位
    # ==========================================================
    def _reset_eef(self):
        if self.single_arm_dagger:
            self.control_active_eef_positions([0, 100, 0, 0, 0, 0] if self.eef_type == "qiangnao" else [0])
            return
        if self.which_arm == 'both':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control(target_positions=[0, 100, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control(target_positions=[0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'sg100':
                self.sg100.control(self.sg100_init_pose.tolist())
        elif self.which_arm == 'left':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control_left(target_positions=[0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control_left(target_positions=[0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'sg100':
                self.sg100.control_left(self.sg100_init_pose[:11].tolist())
        elif self.which_arm == 'right':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control_right(target_positions=[0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control_right(target_positions=[0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'sg100':
                self.sg100.control_right(self.sg100_init_pose[11:].tolist())
        else:
            raise KeyError(f"Unsupported arm type: {self.which_arm}")

    # ==========================================================
    # 子函数 4. 求平均状态与末端姿态
    # ==========================================================
    def _compute_average_state(self, average_num=10):
        state_sum, joint_sum = None, None

        for i in range(average_num):
            state = self.get_obs()
            fk_joint_angles = self._get_init_joint_angles(self.arm_state["joint_q"])

            if i == 0:
                state_sum = np.array(state["observation.state"], dtype=float)
                joint_sum = np.array(fk_joint_angles, dtype=float)
            else:
                state_sum += np.array(state["observation.state"], dtype=float)
                joint_sum += fk_joint_angles
            time.sleep(0.001)

        # 平均计算
        avg_state = state_sum / average_num
        avg_joint = joint_sum / average_num

        return {
            "state": avg_state,
            "joint_action": avg_joint,
        }

    # ==========================================================
    # 子函数 5. 构造完整 FK 输入
    # ==========================================================
    def _get_init_joint_angles(self, joint_q):
        if self.which_arm == 'both':
            return np.array(joint_q)
        elif self.which_arm == 'left':
            return np.concatenate((joint_q, self.arm_init[self.arm_dof_per_side:]))
        elif self.which_arm == 'right':
            return np.concatenate((self.arm_init[:self.arm_dof_per_side], joint_q))
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
    
    def check_action(self, action, mode='default'):
        if mode == 'default':  # 比较 action_space
            if self.dagger_on:
                action = self.action_layout.validate(action)
            if len(action) != len(self.action_space.low):
                raise ValueError(f"action shape must be {len(self.action_space.low)}")
            if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
                log_robot.warning(
                    f"action out of range, action: {action}, "
                    f"low: {self.action_space.low}, high: {self.action_space.high}"
                )
                action = np.clip(action, self.action_space.low, self.action_space.high)
            return action

        raise ValueError(f"Unsupported mode: {mode}")

    def step(self, action):
        session = getattr(self, "dagger_session", None)
        session = session if session is not None and session.active else None
        if session is not None:
            session.check_interrupt()
        t0 = time.time()
        log_robot.info(f"action: {action}")
        # check clip action in action space
        action = self.check_action(action, mode='default')
        t1 = time.time()
        log_robot.info(f"clip action: {action}, check time: {t1 - t0:.3f}s")



        # === 4. 执行动作 ===
        t2 = time.time()
        action_components = self._split_action(action)
        self.cur_joint_angles_action = self._full_arm_target(action_components)

        if self.interp_threaded:
            # 目标交给下发线程，本帧立即返回，不再在推理线程里阻塞插值。
            self.set_action_target(action)
            # 下发线程已按 control_rate 独占下发，这里只需把“策略步进频率”限回
            # ros_rate。若推理本身慢于 ros_rate，rospy.Rate 会立即返回，于是自然
            # 变成“按模型推理时间更新”。少了这一句外层 rollout 会自由狂奔（实测
            # 63.77 step/s），策略闭环频率翻倍且不均匀，反而放大抖动。
            self.rate.sleep()
        elif not self.direct_to_wbc:
            self.exec_action(action)
            self.rate.sleep()
        else:
            # ==== 4.1 插值下发动作（direct_to_wbc 路径）
            # 第一帧从 current state 插值过去；之后每帧在 last_predicted_action -> action 之间插值。
            # Interpolate according to the configured packed action layout.

            current_q = self._get_init_joint_angles(self.arm_state["joint_q"])
            num_inter_points = self.interpolation_steps

            if self.is_first_step:
                self.is_first_step = False

                target_joints = self._full_arm_target(action_components)
                current_lower = np.asarray(self.arm_state.get("lower_body", []), dtype=float)
                target_lower = action_components["lower_body"]

                for i in range(num_inter_points):
                    alpha = (i + 1) / num_inter_points
                    inter_joints = (1 - alpha) * current_q + alpha * target_joints
                    inter_lower = ((1 - alpha) * current_lower + alpha * target_lower
                                   if self.use_5w_wholebody else None)
                    if session is not None:
                        point = action.copy()
                        for side in self._controlled_sides:
                            physical = slice(0, 7) if side == "left" else slice(7, 14)
                            section = self.action_layout.action_slice(side)
                            point[section.start:section.start + 7] = inter_joints[physical]
                        if self.use_5w_wholebody:
                            point[self.action_layout.lower_body_slice] = inter_lower
                        if self.use_5w_base_move:
                            point[self.action_layout.state_dim:] = alpha * action_components["base_velocity"]
                        session.submit(point, channels=("arm", "lower", "base"))
                    else:
                        self.safe_control_arm(inter_joints)
                        if self.use_5w_wholebody:
                            self.safe_control_lower_body(inter_lower)
                        if self.use_5w_base_move:
                            self.safe_control_base(alpha * action_components["base_velocity"])
                    self.control_rate.sleep()

                self.exec_action(action)
                self.last_predicted_action = action.copy()
            else:
                # 后续每一帧：在 last_predicted_action 与 action 之间按 action 维度插值，
                # 末端夹爪保持目标值不插（直接用本帧目标），然后调 exec_action 走对应的 left/right/both 分支
                if self.last_predicted_action is None:
                    self.last_predicted_action = action.copy()

                eef_targets = [
                    component["eef_slice"]
                    for side, component in action_components.items()
                    if side in ("left", "right")
                ]

                for i in range(num_inter_points):
                    alpha = (i + 1) / num_inter_points
                    inter_arm_action = (1 - alpha) * self.last_predicted_action + alpha * action
                    # End effectors retain the current target instead of interpolation.
                    for eef_slice in eef_targets:
                        inter_arm_action[eef_slice] = action[eef_slice]

                    self.exec_action(inter_arm_action)
                    self.control_rate.sleep()

                self.last_predicted_action = action.copy()
                # self.rate.sleep()
        # === 5. 延时与观测 ===
        
        self._record_sleep_time(t2)
        t3 = time.time()
        obs = self.get_obs()
        t4 = time.time()
        log_robot.info(f"get obs time: {t4 - t3:.3f}s")

        # === 6. 奖励与返回 ===
        reward = self.compute_reward()
        return obs, reward, False, False, {}

    def _record_sleep_time(self, t_start):
        self.sleep_time = time.time() - t_start
        self.average_sleep_time += self.sleep_time
        log_robot.info(f"rate.sleep time: {self.sleep_time:.3f}s")

    def safe_control_arm(self, target_position, *, arm_velocity=None):
        if arm_velocity is not None:
            arm_velocity = validate_arm_vector(arm_velocity, "Arm velocity")
        if self.single_arm_dagger:
            target_position = validate_arm_vector(target_position, "Arm position")
            inactive = slice(7, 14) if self.which_arm == "left" else slice(0, 7)
            target_position[inactive] = self._held_arm[inactive]
            if arm_velocity is not None:
                arm_velocity[inactive] = 0.0
        try:
            if arm_velocity is not None:
                if self._arm_velocity_transport is None:
                    raise RuntimeError("Arm velocity transport was not initialized")
                return self._arm_velocity_transport.send(target_position, arm_velocity)
            self.robot.control_arm_joint_positions(target_position)
        except RuntimeError as e:
            if arm_velocity is not None:
                raise
            # 当机器人处于 command_pose_world 状态（底盘移动）时，无法控制手臂
            if "must be in stance state" in str(e):
                log_robot.warning(f"⚠️  无法发送手臂命令：机器人当前状态不允许 (可能正在底盘移动)")
                log_robot.debug(f"   详细错误: {e}")
            else:
                raise

    def safe_control_lower_body(self, target_position):
        """Send model-space radians through the SDK's degree-based 5W API."""
        if self.lower_body_lock_target is not None:
            target_position = np.asarray(target_position, dtype=float).copy()
            target_position[self.lower_body_lock_mask] = self.lower_body_lock_target[
                self.lower_body_lock_mask
            ]
        # 下发线程也会调用本方法，缓存需加锁，避免与 step() 路径互相覆盖。
        with self._last_cmd_lock:
            self._last_lower_cmd = np.asarray(target_position, dtype=float).copy()
        try:
            accepted = self.robot.control_wheel_lower_joint(np.rad2deg(target_position).tolist())
            if self.dagger_on and accepted is False:
                raise RuntimeError("5W lower-body command was rejected; DAgger step is unconfirmed")
        except RuntimeError as e:
            if self.dagger_on:
                raise
            if "must be in stance state" in str(e):
                log_robot.warning("Unable to send 5W lower-body command in the current robot state")
            else:
                raise

    def safe_control_base(self, velocity):
        """Publish model-space [vx, vy, vyaw] to geometry_msgs/Twist."""
        velocity = np.asarray(velocity, dtype=float).reshape(-1)
        if velocity.size != BASE_VELOCITY_DOF or not np.isfinite(velocity).all():
            raise ValueError(
                f"Base velocity must contain {BASE_VELOCITY_DOF} finite values "
                f"[vx, vy, vyaw], got {velocity}"
            )
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.linear.y = float(velocity[1])
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(velocity[2])
        with self._last_cmd_lock:
            self._last_base_cmd = velocity.copy()
        self.cmd_vel_pub.publish(msg)

    def stop_base(self):
        """Publish a zero Twist when base control is active."""
        if getattr(self, 'use_5w_base_move', False) and hasattr(self, 'cmd_vel_pub'):
            self.safe_control_base(np.zeros(BASE_VELOCITY_DOF, dtype=float))

    def stop_arm_velocity(self):
        """Hold the last sent position with zero velocity on pause/exit."""
        if (getattr(self, '_arm_velocity_transport', None) is not None
                and self.last_sent_action is not None):
            target = self._full_arm_target(self._split_action(self.last_sent_action))
            _, velocity = self.safe_control_arm(target, arm_velocity=np.zeros(14))
            self.last_sent_arm_velocity = velocity.copy()

    def send_dagger_command(self, action, *, arm_velocity=None, channels):
        """Final selection uses the same checked SDK APIs as dev's publishers."""
        original = np.asarray(action).copy()
        action = self.check_action(original.copy(), mode='default')
        components = self._split_action(action)
        actual = (self.last_sent_action.copy() if self.last_sent_action is not None
                  else self.dagger_session._actual.copy())
        if "arm" in channels:
            target = self._full_arm_target(components)
            if arm_velocity is None:
                self.safe_control_arm(target)
                self.last_sent_arm_velocity = None
            else:
                velocity = validate_arm_vector(arm_velocity, "Arm velocity")
                unclipped = self._full_arm_target(self._split_action(original))
                velocity[np.abs(unclipped - target) > 1e-10] = 0.0
                target, velocity = self.safe_control_arm(target, arm_velocity=velocity)
                self.last_sent_arm_velocity = velocity.copy()
            for side in self._controlled_sides:
                section = self.action_layout.action_slice(side)
                physical = slice(0, 7) if side == "left" else slice(7, 14)
                actual[section.start:section.start + 7] = target[physical]
        def send_eef():
            if "eef" in channels:
                empty = np.zeros(self.eef_dof_per_side)
                self._control_eef(components.get("left", {}).get("eef", empty),
                                  components.get("right", {}).get("eef", empty))
                for side in self._controlled_sides:
                    section = self.action_layout.action_slice(side)
                    actual[section.start + 7:section.stop] = action[section.start + 7:section.stop]
        if not self.interp_threaded:
            send_eef()
        if "lower" in channels and self.use_5w_wholebody:
            self.safe_control_lower_body(components["lower_body"])
            actual[self.action_layout.lower_body_slice] = self._last_lower_cmd
        if "base" in channels and self.use_5w_base_move:
            self.safe_control_base(components["base_velocity"])
            actual[self.action_layout.state_dim:] = self._last_base_cmd
        if self.interp_threaded:
            send_eef()
        self.last_sent_action = actual.copy()
        return actual

    def exec_action(self, action):
        """执行机械臂与末端执行器动作"""
        session = getattr(self, "dagger_session", None)
        if session is not None and session.active:
            session.submit(action)
            return
        components = self._split_action(action)
        target_position = self._full_arm_target(components)
        self.safe_control_arm(target_position)
        empty_eef = np.zeros(self.eef_dof_per_side, dtype=float)
        left_eef = components.get("left", {}).get("eef", empty_eef)
        right_eef = components.get("right", {}).get("eef", empty_eef)
        self._control_eef(left_eef, right_eef)
        if self.use_5w_wholebody:
            self.safe_control_lower_body(components["lower_body"])
        if self.use_5w_base_move:
            self.safe_control_base(components["base_velocity"])



    def control_active_eef_positions(self, positions):
        """Compose native 0..100 commands with the locked complete inactive hand."""
        values = np.asarray(positions, dtype=np.float64)
        dim = 6 if self.eef_type == "qiangnao" else 1
        if values.shape != (dim,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {dim} finite end-effector positions")
        if np.any(values < 0) or np.any(values > 100):
            raise ValueError("End-effector positions must be within 0..100")
        device = self.qiangnao if self.eef_type == "qiangnao" else self.lejuclaw
        # Same-side SDK calls may fill the other side from a zero-initialized cache.
        # Always send an explicit complete target with the inactive pose preserved.
        full = self._held_eef.copy()
        start = 0 if self.which_arm == "left" else dim
        full[start:start + dim] = values
        device.control(target_positions=full.tolist(), target_velocities=None, target_torques=None)

    def _control_eef(self, left_eef, right_eef):
        """根据 eef_type 控制不同的末端执行器"""
        if self.single_arm_dagger:
            value = float(np.asarray(left_eef if self.which_arm == "left" else right_eef).reshape(-1)[0]) * 100
            positions = [value, 100, value, value, value, value] if self.eef_type == "qiangnao" else [value]
            self.control_active_eef_positions(positions)
            return

        left_eef = np.asarray(left_eef, dtype=float).reshape(-1)
        right_eef = np.asarray(right_eef, dtype=float).reshape(-1)
        if self.eef_type == 'rq2f85':
            eef_msg = JointState()
            try:
                eef_msg.name = ['left_gripper_joint', 'right_gripper_joint']
            except Exception as e:
                log_robot.info(f"_control_eef error! {e}")
            eef_msg.position = np.concatenate((left_eef, right_eef)) * 255
            self.pub_eef_joint.publish(eef_msg)

        elif self.eef_type == 'leju_claw':
            eef_msg = JointState()
            eef_msg.position = np.concatenate((left_eef, right_eef)) * 100
            self.lejuclaw.control(target_positions=eef_msg.position)

        elif self.eef_type == 'qiangnao':
            if self.qiangnao_dof_needed != 1:
                raise KeyError("qiangnao_dof_needed != 1 is not supported!")

            tem_left, tem_right = left_eef.item() * 100, right_eef.item() * 100
            target_positions = np.array([
                tem_left, 100, *([tem_left] * 4),
                tem_right, 100, *([tem_right] * 4)
            ])
            self.qiangnao.control(target_positions=target_positions)

        elif self.eef_type == 'sg100':
            left_eef = np.asarray(left_eef, dtype=np.float64).reshape(-1)
            right_eef = np.asarray(right_eef, dtype=np.float64).reshape(-1)
            if self.is_binary_sg100_action:
                left_eef = (left_eef > self.sg100_action_threshold).astype(np.float64)
                right_eef = (right_eef > self.sg100_action_threshold).astype(np.float64)

            def _expand(values: np.ndarray, hand_slice: slice):
                if values.size == 1:
                    # 模型输出是归一化开合量，先夹到 [0, 1]。
                    amount = float(np.clip(values[0], 0.0, 1.0))
                    positions = (
                        self.sg100_open_pose[hand_slice] * (1.0 - amount)
                        + self.sg100_close_pose[hand_slice] * amount
                    )
                elif values.size == 11:
                    # 逐关节动作已是弧度，直接透传；仅按各关节实测合法区间限幅。
                    positions = values
                else:
                    raise ValueError(f"SG100 action must have 1 or 11 values per hand, got {values.size}")
                # SG100 各关节量程不同（拇指 MCP 为负、小指 MCP_ABD 约 ±3.1），
                # 不能统一 clip 到 [0, 1.57]。
                positions = np.clip(
                    positions, self.sg100_joint_min[hand_slice], self.sg100_joint_max[hand_slice]
                )
                # SG100 驱动只支持关节位置模式，不使用阻抗模式。
                modes = [SG100HandCommand.MODE_JOINT_POSITION] * 11
                return positions.tolist(), modes

            left_positions, left_modes = _expand(left_eef, slice(0, 11))
            right_positions, right_modes = _expand(right_eef, slice(11, 22))
            self.sg100.control(
                target_positions=left_positions + right_positions,
                control_modes=left_modes + right_modes,
                kp=self.sg100_force_kp.tolist(),
                kd=self.sg100_force_kd.tolist(),
                torque_ff=self.sg100_force_torque_ff.tolist(),
                output_limit=self.sg100_force_output_limit.tolist(),
                target_velocities=self.sg100_force_velocities.tolist(),
                enable_left=self.which_arm in ('left', 'both'),
                enable_right=self.which_arm in ('right', 'both'),
            )

        else:
            raise KeyError(f"Unsupported eef_type: {self.eef_type}")

    def compute_reward(self):
        """计算奖励"""
        return 0

    def get_obs(self):
        """获取观测图像及state等"""
        obs = {}
        self.arm_state = {}

        if self.frame_alignment:
            obs_from_buffer = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=1/self.ros_rate,ratio=self.ratio)
            if obs_from_buffer is None or not all(v is not None for v in obs_from_buffer.values()):
                obs_from_buffer = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=float('inf'),ratio=self.ratio)
        else:
            obs_from_buffer = self.obs_buffer.get_latest_obs()

        for k,v in obs_from_buffer.items():
            # remap key
            if 'depth' in k:
                obs[f"observation.{k}"] = v
            elif 'cam' in k:
                obs[f"observation.images.{k}"] = v
            else:
                self.arm_state[f"{k}"] = v

        if self.is_binary and self.eef_type != "sg100":
            self.arm_state['gripper'] = np.where(self.arm_state['gripper']>0.5, 1, 0)

        assert len(self.arm_state.keys()) >= 2, f"arm_state must have exactly 2 elements, but got {len(self.arm_state.keys())}"

        state_keys = [k for k in self.arm_state_keys if k in self.arm_state and k != "lower_body"]

        arm_data = { "left": [], "right": [] }

        for key in state_keys:
            data = self.arm_state[key]
            if len(data) == 0:
                continue
            mid = len(data) // 2
            if self.which_arm == "both":
                arm_data["left"].append(data[:mid])
                arm_data["right"].append(data[mid:])
            elif self.which_arm == "left":
                arm_data["left"].append(data)
            elif self.which_arm == "right":
                arm_data["right"].append(data)
            else:
                raise KeyError(f"Unsupported which_arm: {self.which_arm}")

        # 拼接结果
        state_parts = arm_data["left"] + arm_data["right"]
        if self.use_5w_wholebody:
            state_parts.append(np.asarray(self.arm_state["lower_body"]))
        obs["observation.state"] = np.concatenate(state_parts, axis=0)
        log_robot.info(f"STATE: contained {state_keys}, concated value: {obs['observation.state']}")

        obs["observation.state"] = torch.from_numpy(obs["observation.state"]).float().unsqueeze(0)
        return obs

    def close(self):
        """关闭环境，释放资源"""
        log_robot.info("Closing KuavoBaseRosEnv...")
        try:
            if getattr(self, "dagger_session", None) is not None:
                self.dagger_session.end_episode()
            # 必须先停下发线程，再释放 ROS 资源，否则线程可能在 publisher 销毁后继续下发。
            if getattr(self, "control_loop", None) is not None:
                self.control_loop.stop()
                self.control_loop = None
            if self.dagger_on or self.single_arm_dagger:
                try:
                    self.stop_arm_velocity()
                except Exception as e:
                    log_robot.error(f"Unable to send zero arm velocity during close: {e}")
                if getattr(self, '_dagger_arbiter', None) is not None:
                    self._dagger_arbiter.close()
                if hasattr(self, 'lejuclaw'):
                    self.lejuclaw.close()
            if getattr(self, '_wheel_arm_cleanup_required', False):
                if not self._restore_wheel_arm_control():
                    log_robot.error(
                        "wheel 手臂控制模式未完全复位；再次启动 eval 前必须重启或检查 KSC"
                    )

            self.stop_base()
            if hasattr(self, 'obs_buffer'):
                self.obs_buffer.stop_subscribers()
                if hasattr(self.obs_buffer, 'obs_buffer_data'):
                    for k in self.obs_buffer.obs_buffer_data:
                        self.obs_buffer.obs_buffer_data[k]["data"].clear()
                        self.obs_buffer.obs_buffer_data[k]["timestamp"].clear()
                if hasattr(self.obs_buffer, 'ros_manager'):
                    self.obs_buffer.ros_manager = None
                if hasattr(self.obs_buffer, 'control_signal_manager'):
                    self.obs_buffer.control_signal_manager = None
                del self.obs_buffer
            
            if hasattr(self, 'ros_manager'):
                self.ros_manager.close()
            if hasattr(self, 'control_signal_manager'):
                self.control_signal_manager.close()
            log_robot.info("KuavoBaseRosEnv closed successfully.")
        except Exception as e:
            log_robot.error(f"Error closing KuavoBaseRosEnv: {e}")
            traceback.print_exc()
        finally:
            if getattr(self, '_arm_velocity_transport', None) is not None:
                self._arm_velocity_transport.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

class LejuClaw:
    """乐聚爪手控制器"""
    def __init__(self, ros_manager=None):
        self.ros_manager = ros_manager or ROSManager()
        self._pub_leju_claw_cmd = self.ros_manager.register_publisher('/leju_claw_command', lejuClawCommand, queue_size=10)

    def control(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制双手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 2)
        
        cmd = lejuClawCommand()
        cmd.data.name = ['left_claw', 'right_claw']
        
        target_positions = [max(0.0, min(100.0, float(pos))) for pos in target_positions]
        target_velocities = self._get_default_velocities(target_velocities, 2)
        target_torques = self._get_default_torques(target_torques, 2)
        
        cmd.data.position = target_positions
        cmd.data.velocity = target_velocities
        cmd.data.effort = target_torques
        self._pub_leju_claw_cmd.publish(cmd)

    def control_left(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制左手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 1)
        self.control(
            [target_positions[0], 0],
            [target_velocities[0] if target_velocities else 90, 0],
            [target_torques[0] if target_torques else 1.0, 0]
        )

    def control_right(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制右手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 1)
        self.control(
            [0, target_positions[0]],
            [0, target_velocities[0] if target_velocities else 90],
            [0, target_torques[0] if target_torques else 1.0]
        )

    def _validate_inputs(self, positions, velocities, torques, expected_len):
        """验证输入参数"""
        assert len(positions) == expected_len, f"target_positions must be a list of length {expected_len}"
        if velocities is not None:
            assert len(velocities) == expected_len, f"target_velocities must be a list of length {expected_len}"
        if torques is not None:
            assert len(torques) == expected_len, f"target_torques must be a list of length {expected_len}"

    def _get_default_velocities(self, velocities, length):
        """获取默认速度"""
        if velocities is None:
            return [90] * length
        return [max(0.0, min(100.0, vel)) for vel in velocities]

    def _get_default_torques(self, torques, length):
        """获取默认力矩"""
        if torques is None:
            return [1.0] * length
        return [max(0.0, min(10.0, torque)) for torque in torques]

    def close(self):
        """释放资源"""
        if hasattr(self, 'ros_manager'):
            self.ros_manager.close()


class SG100Hand:
    """Publisher for the repository-local SG100 ROS command message."""

    JOINTS_PER_HAND = 11
    ALL_JOINTS_MASK = 0x07FF

    def __init__(self, ros_manager=None):
        self.ros_manager = ros_manager or ROSManager()
        self._pub = self.ros_manager.register_publisher(
            '/sg100_hand_command', SG100HandCommand, queue_size=10
        )

    @staticmethod
    def _optional_22(name, values):
        if values is None:
            return None
        if len(values) != 22:
            raise ValueError(f"{name} must contain 22 values, got {len(values)}")
        return list(values)

    def control(
        self,
        target_positions,
        control_modes=None,
        kp=None,
        kd=None,
        torque_ff=None,
        output_limit=None,
        target_velocities=None,
        *,
        enable_left=True,
        enable_right=True,
    ):
        if len(target_positions) != 22:
            raise ValueError(f"target_positions must contain 22 values, got {len(target_positions)}")

        control_modes = self._optional_22("control_modes", control_modes)
        kp = self._optional_22("kp", kp)
        kd = self._optional_22("kd", kd)
        torque_ff = self._optional_22("torque_ff", torque_ff)
        output_limit = self._optional_22("output_limit", output_limit)
        target_velocities = self._optional_22("target_velocities", target_velocities)

        cmd = SG100HandCommand()
        cmd.header.stamp = rospy.Time.now()
        positions = [float(np.clip(value, -5, 5)) for value in target_positions]
        cmd.left_hand_positions = positions[:11]
        cmd.right_hand_positions = positions[11:]
        cmd.left_enable_mask = self.ALL_JOINTS_MASK if enable_left else 0
        cmd.right_enable_mask = self.ALL_JOINTS_MASK if enable_right else 0

        if control_modes is None:
            cmd.control_mode = SG100HandCommand.MODE_JOINT_POSITION
        else:
            cmd.control_mode = int(control_modes[0])
            cmd.left_hand_control_mode = [int(value) for value in control_modes[:11]]
            cmd.right_hand_control_mode = [int(value) for value in control_modes[11:]]

        for values, left_name, right_name in (
            (target_velocities, "left_hand_velocities", "right_hand_velocities"),
            (kp, "left_hand_kp", "right_hand_kp"),
            (kd, "left_hand_kd", "right_hand_kd"),
            (torque_ff, "left_hand_torque_ff", "right_hand_torque_ff"),
            (output_limit, "left_hand_output_limit", "right_hand_output_limit"),
        ):
            if values is not None:
                setattr(cmd, left_name, [float(value) for value in values[:11]])
                setattr(cmd, right_name, [float(value) for value in values[11:]])

        self._pub.publish(cmd)

    def control_left(self, target_positions):
        if len(target_positions) != 11:
            raise ValueError("SG100 left-hand command must contain 11 values")
        self.control(list(target_positions) + [0.0] * 11, enable_left=True, enable_right=False)

    def control_right(self, target_positions):
        if len(target_positions) != 11:
            raise ValueError("SG100 right-hand command must contain 11 values")
        self.control([0.0] * 11 + list(target_positions), enable_left=False, enable_right=True)

# 使用示例
if __name__ == "__main__":
    from kuavo_deploy.config import load_kuavo_config
    
    # 使用上下文管理器确保资源正确释放
    with KuavoBaseRosEnv(load_kuavo_config()) as env:
        obs, info = env.reset()

        for _ in range(1):
            obs = env.get_obs()
            env.rate.sleep()
            print(obs.keys())
            for k, v in obs.items():
                print(k, v.shape)
                print(v.max(), v.min())
