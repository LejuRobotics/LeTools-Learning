"""
机器人控制示例程序
提供机械臂运动控制、轨迹回放等功能

使用示例:
  python scripts.py --task go --config /path/to/custom_config.yaml"                   # 先插值到bag第一帧的位置，再回放bag包前往工作位置
  python scripts.py --task run --config /path/to/custom_config.yaml"                  # 从当前位置直接运行模型
  python scripts.py --task dagger --config /path/to/custom_config.yaml                # 沿用 dev 推理流程，支持 DAgger 人工接管
  python scripts.py --task go_run --config /path/to/custom_config.yaml"               # 到达工作位置直接运行模型
  python scripts.py --task here_run --config /path/to/custom_config.yaml"             # 插值至bag的最后一帧状态开始运行
  python scripts.py --task back_to_zero --config /path/to/custom_config.yaml"         # 中断模型推理后，倒放bag包回到0位
"""
import os
import rospy
import rosbag
import time
import argparse
import importlib
from pathlib import Path
from typing import List, Tuple, Optional

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.config import load_kuavo_config, KuavoConfig
import gymnasium as gym

import numpy as np
import signal
import threading
import subprocess
import traceback

from std_msgs.msg import Bool

# 配置日志
log_model = setup_logger("model", "DEBUG")  # 网络日志
log_robot = setup_logger("robot", "DEBUG")  # 机器人日志

# 控制变量
class ArmMoveController:
    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.lock = threading.Lock()
        
    def pause(self):
        with self.lock:
            self.paused = True
            log_robot.info("🔄 机械臂运动已暂停")
    
    def resume(self):
        with self.lock:
            self.paused = False
            log_robot.info("▶️ 机械臂运动已恢复")
    
    def stop(self):
        with self.lock:
            self.should_stop = True
            log_robot.info("⏹️ 机械臂运动已停止")
    
    def is_paused(self):
        with self.lock:
            return self.paused
    
    def should_exit(self):
        with self.lock:
            return self.should_stop

# 控制器实例
arm_controller = ArmMoveController()

# Ros发布暂停/停止信号
pause_pub = rospy.Publisher('/kuavo/pause_state', Bool, queue_size=1)
stop_pub = rospy.Publisher('/kuavo/stop_state', Bool, queue_size=1)

def signal_handler(signum, frame):
    """信号处理器"""
    log_robot.info(f"🔔 收到信号: {signum}")
    if signum == signal.SIGUSR1:  # 暂停/恢复
        if arm_controller.is_paused():
            log_robot.info("🔔 当前状态：已暂停，执行恢复")
            arm_controller.resume()
            pause_pub.publish(False)
        else:
            log_robot.info("🔔 当前状态：运行中，执行暂停")
            arm_controller.pause()
            pause_pub.publish(True)
    elif signum == signal.SIGUSR2:  # 停止
        log_robot.info("�� 执行停止")
        arm_controller.stop()
        stop_pub.publish(True)
    log_robot.info(f"🔔 信号处理完成，当前状态 - 暂停: {arm_controller.is_paused()}, 停止: {arm_controller.should_exit()}")

def setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGUSR1, signal_handler)  # 暂停/恢复
    signal.signal(signal.SIGUSR2, signal_handler)  # 停止
    log_robot.info("📡 信号处理器已设置:")
    log_robot.info("  SIGUSR1 (kill -USR1): 暂停/恢复机械臂运动")
    log_robot.info("  SIGUSR2 (kill -USR2): 停止机械臂运动")

def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

class ArmMove:
    """机械臂运动控制类"""
    
    def __init__(self, config: KuavoConfig):
        """
        初始化机械臂控制
        
        Args:
            bag_path: 轨迹文件路径
        """
        # Import/register robot environments only when motion is actually requested.
        importlib.import_module("kuavo_deploy.kuavo_env")
        self.config = config
        # 设置信号处理器
        self.shutdown_requested = False
        # 设置信号处理器
        setup_signal_handlers()
        
        # 输出当前进程ID，方便外部控制
        pid = os.getpid()
        log_robot.info(f"🆔 当前进程ID: {pid}")
        log_robot.info(f"💡 使用以下命令控制机械臂运动:")
        log_robot.info(f"   暂停/恢复: kill -USR1 {pid}")
        log_robot.info(f"   停止运动: kill -USR2 {pid}")

        self.inference_config = config.inference
        self.bag_path = self.inference_config.go_bag_path

        initial_topics = ["/control_robot_hand_position", "/leju_claw_command", "/kuavo_arm_traj", "/sg100_hand_command"]
        if config.env.use_5w_wholebody:
            initial_topics.append("/lb_leg_traj")
        self.msg_dict_of_list = self._read_topic_messages(
            bag_path=self.bag_path,
            topic_names=initial_topics,
        )

        rospy.init_node('kuavo_deploy', anonymous=True)
        self.env = gym.make(
            self.config.env.env_name,
            max_episode_steps=self.inference_config.max_episode_steps,
            config=self.config,
        )
        self.env = unwrap_env(self.env)


    def _check_control_signals(self):
        """检查控制信号"""
        # 检查暂停状态
        while arm_controller.is_paused():
            log_robot.info("🔄 机械臂运动已暂停")
            time.sleep(0.1)
            if arm_controller.should_exit():
                log_robot.info("🛑 机械臂运动被停止")
                return False
        
        # 检查是否需要停止
        if arm_controller.should_exit():
            log_robot.info("🛑 收到停止信号，退出机械臂运动")
            return False
            
        return True  # 正常继续
    
    def _read_topic_messages(self, bag_path, topic_names: list = None) -> dict:
        """
        读取bag包中指定话题的消息并转换为字典
        :param bag_path: bag文件路径
        :param topic_names: 话题名称列表
        :return: 消息字典，key为话题名称，value为该话题的消息列表
        """
        messages_dict = {}
        try:
            bag = rosbag.Bag(bag_path)
            for topic, msg, t in bag.read_messages(topics=topic_names):
                if topic not in messages_dict:
                    messages_dict[topic] = []
                messages_dict[topic].append(msg)
            bag.close()
            return messages_dict
        except Exception as e:
            rospy.logerr(f"Failed to read messages from bag: {e}")
            return {}

    def _pub_arm_traj(self, msg) -> None:
        """发布机械臂轨迹"""
        # 如果msg是list，则直接发布
        if isinstance(msg, list):
            position = msg
        else:
            position = np.array(msg.position)/180*np.pi
        if self.env.which_arm=="both":
            target_positions = position
        elif self.env.which_arm=="left":
            half = self.env.arm_dof_per_side
            target_positions = np.concatenate([position[:half], self.env.arm_init[half:]], axis=0)
        elif self.env.which_arm=="right":
            half = self.env.arm_dof_per_side
            target_positions = np.concatenate([self.env.arm_init[:half], position[half:]], axis=0)
        else:
            raise ValueError(f"Invalid which_arm: {self.env.which_arm}, must be 'left', 'right', or 'both'")
        self.env.robot.control_arm_joint_positions(target_positions)
    
    def _pub_leju_claw(self, msg) -> None:
        """发布夹爪"""
        if self.env.which_arm=="both":
            target_positions = msg.data.position
        elif self.env.which_arm=="left":
            target_positions = np.concatenate([msg.data.position[:1],[0]],axis=0)
        elif self.env.which_arm=="right":
            target_positions = np.concatenate([[0],msg.data.position[1:]],axis=0)
        else:
            raise ValueError(f"Invalid which_arm: {self.env.which_arm}, must be 'left', 'right', or 'both'")
        self.env.lejuclaw.control(target_positions)
    
    def _pub_qiangnao(self, msg) -> None:
        """发布灵巧手"""
        left_hand_position = np.frombuffer(msg.left_hand_position, dtype=np.uint8)
        right_hand_position = np.frombuffer(msg.right_hand_position, dtype=np.uint8)
        if self.env.which_arm=="both":
            target_positions = np.concatenate([left_hand_position,right_hand_position],axis=0)
        elif self.env.which_arm=="left":
            target_positions = np.concatenate([left_hand_position,[0,0,0,0,0,0]],axis=0)
        elif self.env.which_arm=="right":
            target_positions = np.concatenate([[0,0,0,0,0,0],right_hand_position],axis=0)
        else:
            raise ValueError(f"Invalid which_arm: {self.env.which_arm}, must be 'left', 'right', or 'both'")
        self.env.qiangnao.control(target_positions)

    def _pub_rq2f85(self,msg) -> None:
        self.env.pub_eef_joint.publish(msg)

    def _pub_lower_body(self, msg) -> None:
        """Replay a lower-body target using the bag's degree convention."""
        if isinstance(msg, list):
            target_position = np.asarray(msg, dtype=float)
        else:
            target_position = np.deg2rad(np.asarray(msg.position, dtype=float))

        expected_dof = len(self.env.limits["lower_body"]["min"])
        if target_position.size != expected_dof:
            raise ValueError(
                f"Expected {expected_dof} lower-body joint positions from /lb_leg_traj, "
                f"got {target_position.size}"
            )
        self.env.safe_control_lower_body(target_position)

    def _publish_bag_message(self, topic: str, msg) -> None:
        if topic == "/kuavo_arm_traj":
            self._pub_arm_traj(msg)
        elif topic == "/leju_claw_command":
            self._pub_leju_claw(msg)
        elif topic == "/control_robot_hand_position":
            self._pub_qiangnao(msg)
        elif topic == "/gripper_command":
            self._pub_rq2f85(msg)
        elif topic == "/sg100_hand_command":
            self._pub_sg100(msg)
        elif topic == "/lb_leg_traj":
            self._pub_lower_body(msg)

    def _pub_sg100(self, msg) -> None:
        positions = list(msg.left_hand_positions) + list(msg.right_hand_positions)
        self.env.sg100.control(
            target_positions=positions,
            kp=self.env.sg100_force_kp.tolist(),
            kd=self.env.sg100_force_kd.tolist(),
            torque_ff=self.env.sg100_force_torque_ff.tolist(),
            output_limit=self.env.sg100_force_output_limit.tolist(),
            target_velocities=self.env.sg100_force_velocities.tolist(),
            enable_left=self.env.which_arm in ('left', 'both'),
            enable_right=self.env.which_arm in ('right', 'both'),
        )

    def play_bag(self, go_bag, reverse=False):
        """
        将机械臂移动到工作姿态。均匀发布机械臂、手部位置和夹爪命令。
        
        Args:
            reverse (bool): 如果为True，则倒序播放命令序列
        """

        # topic_names = ["/joint_cmd", "/control_robot_hand_position", "/leju_claw_command"],
        if self.env.eef_type == 'leju_claw':
            topics = ["/kuavo_arm_traj", "/leju_claw_command"]
        elif self.env.eef_type == 'qiangnao':
            topics = ["/kuavo_arm_traj", "/control_robot_hand_position"]
        elif self.env.eef_type == 'rq2f85':
            topics = ["/kuavo_arm_traj", "/gripper_command"]
        elif self.env.eef_type == 'sg100':
            topics = ["/kuavo_arm_traj", "/sg100_hand_command"]
        else:
            raise ValueError(f"Invalid eef_type: {self.env.eef_type}, must be 'leju_claw', 'qiangnao', 'sg100', or 'rq2f85' ")
        if self.env.use_5w_wholebody:
            topics.append("/lb_leg_traj")
        
        msg_dict_of_list = self._read_topic_messages(
            bag_path = go_bag, 
            topic_names = topics
        )
        if self.env.use_5w_wholebody and not msg_dict_of_list.get("/lb_leg_traj"):
            raise ValueError(
                "5w_wholebody is enabled, but the bag contains no /lb_leg_traj messages"
            )
        if reverse:
            msg_dict_of_list = {topic: msg_dict_of_list[topic][::-1] for topic in msg_dict_of_list}
        log_robot.info(f"将回放 {go_bag} 中的 {[topic for topic in msg_dict_of_list.keys()]} 主题的消息")
        
        # 初始化消息字典，检查键值是否存在并获取消息列表
        msg_lists = {}
        for topic in msg_dict_of_list:
            msg_lists[topic] = {
                "msgs": msg_dict_of_list[topic],
                "total": len(msg_dict_of_list[topic]),
                "index": 0,
            }
                
        if not msg_lists:
            log_robot.warning("没有找到任何有效的消息数据可以播放")
            return

        if self.env.use_5w_wholebody:
            lower_info = msg_lists["/lb_leg_traj"]
            self._pub_lower_body(lower_info["msgs"][0])
            log_robot.info("发布 /lb_leg_traj 首帧")
        
        # 计算总步数为最长的消息列表的长度
        max_steps = max(info["total"] for info in msg_lists.values())
        log_robot.info(f"开始均匀播放 {max_steps} 步消息数据")
        
        # 均匀发布剩余数据
        rate = rospy.Rate(100)  # 100Hz，可根据需要调整
        for step in range(1, max_steps):
            # 检查控制信号
            if not self._check_control_signals():
                log_robot.info("🛑 轨迹播放被停止")
                return

            for topic, info in msg_lists.items():
                # 计算当前应该发布的索引
                # 使用浮点数计算保证均匀分布，然后取整
                target_index = min(int(step * info["total"] / max_steps), info["total"] - 1)
                
                # 只有当索引变化时才发布新消息
                if target_index > info["index"]:
                    self._publish_bag_message(topic, info["msgs"][target_index])
                log_robot.info(f"发布 {topic} 消息 {target_index+1}/{info['total']}")
            # 控制发布频率
            rate.sleep()
        
        # 确保最后一帧数据被发布
        for topic, info in msg_lists.items():
            if info["index"] < info["total"] - 1:
                target_index = info["total"] - 1
                self._publish_bag_message(topic, info["msgs"][target_index])
                log_robot.info(f"发布 {topic} 最终消息")
        
        log_robot.info("消息序列播放完成")

    def _get_current_joint_angles(self) -> List[float]:
        """获取当前关节角度(rad)"""
        return self.env.robot_state.arm_joint_state().position

    def _arm_interpolate_joint(self, q0: List[float], q1: List[float], steps: int = 100) -> List[List[float]]:
        """
        生成从 q0 到 q1 的平滑插值轨迹。
        
        Args:
            q0: 初始关节位置列表
            q1: 目标关节位置列表
            steps: 插值步数，默认为INTERPOLATION_STEPS
            
        Returns:
            包含插值位置的列表，每个元素是一个长度为NUM_JOINTS的列表
            
        Raises:
            ValueError: 如果输入关节位置数量不正确
        """
        NUM_JOINTS = 14  # 假设有14个关节
        if len(q0) != NUM_JOINTS or len(q1) != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} joint positions")
        
        return [
            [
                q0[j] + i / float(steps) * (q1[j] - q0[j])
                for j in range(NUM_JOINTS)
            ]
            for i in range(steps)
        ]

    def _move_to_joint_angles(self, target_angles: List[float], steps: int = 100) -> None:
        """
        移动到目标关节角度
        
        Args:
            target_angles: 目标关节角度列表
            steps: 插值步数
        """
        current_angles = self._get_current_joint_angles()
        log_robot.info(f"当前关节角度: {current_angles}")
        arm_inter = self._arm_interpolate_joint(
            current_angles, target_angles, steps=steps
        )
        
        for joint_angles in arm_inter:
            if not self._check_control_signals():
                log_robot.info("🛑 关节角度插值被停止")
                return
            log_robot.info(f"机械臂关节角度: {joint_angles}")
            self._pub_arm_traj(joint_angles)
            time.sleep(0.1)

    def go(self) -> None:
        """先插值到bag第一帧的位置，再回放bag包前往工作位置"""
        time.sleep(1)
        # 移动到轨迹起始位置
        start_angles = [float(j) for j in self.msg_dict_of_list.get("/kuavo_arm_traj", [])[0].position]
        start_angles = np.array(start_angles)/180*np.pi
        self._move_to_joint_angles(start_angles)
        # 播放轨迹
        self.play_bag(go_bag=self.bag_path)

    def here_run(self) -> None:
        """直接插值到bag最后一帧位置运行"""
        time.sleep(1)
        # 移动到轨迹结束位置
        end_angles = [float(j) for j in self.msg_dict_of_list.get("/kuavo_arm_traj", [])[-1].position]
        end_angles = np.array(end_angles)/180*np.pi
        self._move_to_joint_angles(end_angles)
        # 执行评估
        self.run()

    def back_to_zero(self) -> None:
        """回到零位"""
        time.sleep(1)
        # 移动到轨迹结束位置
        end_angles = [float(j) for j in self.msg_dict_of_list.get("/kuavo_arm_traj", [])[-1].position]
        end_angles = np.array(end_angles)/180*np.pi
        self._move_to_joint_angles(end_angles)
        # 反向播放轨迹
        self.play_bag(go_bag=self.bag_path,reverse=True)
        # 移动到零位
        zero_angles = [0.0] * 14
        self._move_to_joint_angles(zero_angles)

    def go_run(self) -> None:
        """执行前往并运行"""
        self.go()
        self.run()

    def run(self) -> None:
        """执行运行"""
        # 这里为了安全考虑，加一个stop base的函数
        try:
            from kuavo_deploy.src.eval.real_single_test import kuavo_eval
            kuavo_eval(config=self.config, env=self.env)
        finally:
            self.env.stop_base()

    def close(self) -> None:
        """Restore controller modes and release ROS resources exactly once."""
        if self.env is None:
            return
        try:
            self.env.close()
        finally:
            self.env = None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Kuavo机器人控制示例程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python scripts.py --task go --config /path/to/custom_config.yaml"                   # 先插值到bag第一帧的位置，再回放bag包前往工作位置
  python scripts.py --task run --config /path/to/custom_config.yaml"                  # 从当前位置直接运行模型
  python scripts.py --task dagger --config /path/to/custom_config.yaml                # 沿用 dev 推理流程，支持 DAgger 人工接管
  python scripts.py --task go_run --config /path/to/custom_config.yaml"               # 到达工作位置直接运行模型
  python scripts.py --task here_run --config /path/to/custom_config.yaml"             # 插值至bag的最后一帧状态开始运行
  python scripts.py --task back_to_zero --config /path/to/custom_config.yaml"         # 中断模型推理后，倒放bag包回到0位

任务说明:
  go          - 先插值到bag第一帧的位置，再回放bag包前往工作位置
  run         - 从当前位置直接运行模型
  dagger      - 沿用 dev 推理流程，启用 VR 接管与回切
  go_run      - 到达工作位置直接运行模型
  here_run    - 插值至bag的最后一帧状态开始运行
  back_to_zero - 中断模型推理后，倒放bag包回到0位
  auto_test   - 仿真中自动测试模型，执行eval_episodes次
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        choices=["go", "run", "dagger", "go_run", "here_run", "back_to_zero"],
        help="要执行的任务类型"
    )
    
    # 可选参数
    parser.add_argument(
        "--config", 
        type=str,
        required=True,
        help="配置文件路径(必须指定)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细输出"
    )
    
    parser.add_argument(
        "--dry_run", "--dry-run",
        dest="dry_run",
        action="store_true",
        help="干运行模式，只显示将要执行的操作但不实际执行"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        log_model.setLevel("DEBUG")
        log_robot.setLevel("DEBUG")
    
    # 确定配置文件路径
    config_path = Path(args.config)
    
    log_robot.info(f"使用配置文件: {config_path}")
    log_robot.info(f"执行任务: {args.task}")
    
    config = load_kuavo_config(config_path, task=args.task)
    extended_lifecycle = config.inference.dagger_on
    log_robot.info("任务模式: %s, intervention=%s, control_rate=%s, execution_mode=%s",
                   args.task, config.inference.dagger_on,
                   config.env.control_rate, config.env.execution_mode)
    if args.dry_run:
        log_robot.info("配置校验通过；干运行结束，未初始化机器人环境或执行任务")
        return

    # 初始化机械臂
    try:
        arm = ArmMove(config)
        log_robot.info("机械臂初始化成功")
    except Exception as e:
        log_robot.error(f"机械臂初始化失败: {e}")
        if extended_lifecycle:
            raise
        return

    # 任务映射
    task_map = {
        "go": arm.go,                    # 到达工作位置
        "run": arm.run,                  # 从当前位置直接运行模型
        "dagger": arm.run,            # dev 推理与 DAgger 仲裁
        "go_run": arm.go_run,           # 到达工作位置直接运行模型
        "here_run": arm.here_run,       # 从go_bag的最后一帧状态开始运行
        "back_to_zero": arm.back_to_zero, # 中断模型推理后，倒放bag包回到0位
    }
    
    # 执行任务
    try:
        log_robot.info(f"开始执行任务: {args.task}")
        task_map[args.task]()
        log_robot.info(f"任务 {args.task} 执行完成")
    except KeyboardInterrupt:
        log_robot.info("用户中断操作")
    except Exception as e:
        traceback.print_exc()
        log_robot.error(f"执行任务 {args.task} 时发生错误: {e}")
        if extended_lifecycle:
            raise
    finally:
        if extended_lifecycle:
            arm.close()

if __name__ == "__main__":
    main()
