#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""
Kuavo机器人控制示例脚本 (Python版)
完全等价于原 Bash 脚本，支持交互控制、暂停、恢复、停止、日志查看等功能。
"""

import os
import sys
import signal
import subprocess
import yaml
from pathlib import Path
from time import sleep
import select, time, threading, queue

# 全局变量
current_proc = None
LOG_DIR = None
_dagger_task = False


# ========== 信号处理 ==========
def cleanup(signum, frame):
    global current_proc
    if _dagger_task:
        if current_proc is not None and current_proc.poll() is None:
            current_proc.send_signal(signal.SIGUSR2)
            try:
                current_proc.wait(timeout=35)
            except subprocess.TimeoutExpired:
                current_proc.kill()
                current_proc.wait(timeout=5)
        # start_task's finally closes the bag after the controller exits.
        sys.exit(130)
    print("\n⏹️ 捕获到 Ctrl+C，开始终止任务")
    if current_proc and current_proc.poll() is None:
        print(f"⏹️ 正在终止任务 (PID: {current_proc.pid})...")
        current_proc.terminate()
        try:
            # Wholebody shutdown may retry a five-second controller switch
            # before clearing buffers and closing DDS.
            current_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            current_proc.kill()
        print("✅ 任务已终止")
    sys.exit(130)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


# ========== 工具函数 ==========
def print_header():
    print("=== Kuavo机器人控制示例 ===")
    print("此脚本展示如何使用命令行参数控制不同的任务")
    print("支持暂停、继续、停止功能\n")
    print("📋 控制功能说明:")
    print("  🔄 暂停/恢复: 发送 SIGUSR1 信号")
    print("  ⏹️  停止任务: 发送 SIGUSR2 信号")
    print("  📊 查看日志: tail -f log/kuavo_deploy/kuavo_deploy.log\n")


def get_script_paths():
    script_dir = Path(__file__).resolve().parent
    script = script_dir / "src" / "scripts" / "script.py"
    auto_test = script_dir / "src" / "scripts" / "script_auto_test.py"
    offline_bag_eval = script_dir / "offline_bag_eval.py"
    return script_dir, script, auto_test, offline_bag_eval


def ensure_log_dir(script_dir):
    log_root = script_dir.parent / "log" / "kuavo_deploy"
    log_root.mkdir(parents=True, exist_ok=True)
    return log_root


# ========== 交互控制 ==========
def input_listener(input_queue, stop_event):
    """后台线程：持续监听用户输入"""
    sys.stdout.write(f"\r🟢 任务运行中，输入命令以暂停/停止等(p/s/l/h): >")
    sys.stdout.flush()
    while not stop_event.is_set():
        # 显示提示符
        inputok, _, _ = select.select([sys.stdin], [], [], 0.1)
        if inputok:
            line = sys.stdin.readline()
            # print("line: ", line)  # 换行
            input_queue.put("s" if not line and _dagger_task else line.strip().lower())
            if not line and _dagger_task:
                return

def interactive_controller(stop_timeout=3, check_health=None):
    global current_proc, LOG_DIR

    print("🎮 交互式控制器已启动")
    print(f"任务PID: {current_proc.pid}\n")
    print("📋 可用命令:")
    print("  p/pause    - 暂停/恢复任务")
    print("  s/stop     - 停止任务")
    print("  l/log      - 查看实时日志")
    print("  h/help     - 显示帮助\n")

    input_queue = queue.Queue()
    stop_event = threading.Event()

    # 启动输入监听线程
    threading.Thread(
        target=input_listener, args=(input_queue, stop_event), daemon=True
    ).start()

    try:
        while True:
            # 检查子进程状态
            if current_proc.poll() is not None:
                retcode = current_proc.returncode  # 获取退出码

                if retcode == 0:
                    print("\n✅ 任务已正常结束")
                else:
                    print(f"\n❌ 任务异常退出，错误码：{retcode}")
                    print("📄 请查看日志文件：log/kuavo_deploy/kuavo_deploy.log")

                current_proc = None
                stop_event.set()
                return "task_exit"


            if check_health is not None:
                check_health()

            try:
                cmd = input_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # 无输入则继续检测进程
            # cmd = input(f"🟢 任务运行中 (PID: {current_proc.pid}) > ").strip().lower()  # 会阻塞等待输入
            # if cmd != "":
                # stop_event.set()
            if cmd in ("p", "pause"):
                print("🔄 发送暂停/恢复信号...")
                os.kill(current_proc.pid, signal.SIGUSR1)

            elif cmd in ("s", "stop"):
                print("⏹️  发送停止信号...")
                os.kill(current_proc.pid, signal.SIGUSR2)
                forced = False
                try:
                    current_proc.wait(timeout=stop_timeout)
                except subprocess.TimeoutExpired:
                    current_proc.kill()
                    current_proc.wait(timeout=5)
                    forced = True
                print("✅ 任务已强制停止" if forced else "✅ 任务已安全停止")
                stop_event.set()
                return "forced_stop" if forced else "user_stop"

            elif cmd in ("l", "log"):
                log_path = LOG_DIR / "kuavo_deploy.log"
                if log_path.exists():
                    print("📊 显示最新日志 (Ctrl+C 返回):")
                    os.system(f"tail -n 20 {log_path}")
                else:
                    print("❌ 日志文件不存在")

            elif cmd in ("h", "help"):
                print("📋 可用命令:")
                print("  p/pause    - 暂停/恢复任务")
                print("  s/stop     - 停止任务")
                print("  l/log      - 查看实时日志")
                print("  h/help     - 显示帮助\n")

            else:
                print(f"❌ 未知命令: {cmd}")
    finally:
        stop_event.set()


# ========== YAML 解析 ==========
def parse_config(config_path):
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        inf = cfg.get("inference", {})
        policy_type = str(inf.get("policy_type", "")).strip()

        print("📋 模型配置信息:")
        if policy_type == "client":
            print("   🔌 推理模式: Client（外置模型服务）")
            server_host = str(inf.get("server_host", "")).strip()
            server_port = inf.get("server_port")
            if server_host and server_port is not None:
                print(f"   🌐 服务地址: {server_host}:{server_port}")
            print("   ℹ️ Client 模式无需配置本地模型路径")
            return cfg

        pretrained_path = inf.get("pretrained_path", "")
        task = inf.get("task", "N/A")
        method = inf.get("method", "N/A")
        timestamp = inf.get("timestamp", "N/A")
        epoch = inf.get("epoch", "N/A")

        if pretrained_path:
            model_path = Path(pretrained_path)
        else:
            model_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")
        if pretrained_path:
            print(f"   Pretrained Path: {pretrained_path}")
        else:
            print(f"   Task: {task}")
            print(f"   Method: {method}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Epoch: {epoch}")
        print(f"📂 完整模型路径: {model_path}")
        if model_path.exists():
            print("✅ 模型路径存在")
        else:
            print("❌ 模型路径不存在")
        return cfg
    except Exception as e:
        print(f"❌ 解析配置文件失败: {e}")
        sys.exit(1)


def print_task_menu(config_path="<config_path>", use_color=True):
    """
    打印 Kuavo 任务菜单，说明在前，统一在最后输出命令行模板。
    
    Args:
        config_path (str): 默认配置文件路径，用于命令行模板显示。
        use_color (bool): 是否使用终端颜色，默认 True。
    """
    # 终端颜色定义
    GREEN  = "\033[32m" if use_color else ""
    BLUE   = "\033[34m" if use_color else ""
    YELLOW = "\033[33m" if use_color else ""
    RESET  = "\033[0m"  if use_color else ""

    tasks = [
        ("go", "普通任务: 先插值到bag第一帧的位置, 再回放bag包前往工作位置"),
        ("run", "普通任务: 从当前位置直接运行模型"),
        ("dagger", "DAgger任务: 沿用 dev 推理流程，支持 VR 人工接管与数据采集"),
        ("auto_test", "自动测试任务：仿真中自动测试模型，执行 eval_episodes 次"),
        ("offline_bag", "离线任务：从一条 bag 读取 obs，跑完整推理链路并和 bag action 做误差验证"),
        ("退出", ""),
    ]

    print(f"\n🟢 可选择的任务示例如下:")
    for idx, (name, desc) in enumerate(tasks, 1):
        if desc:
            print(f"{GREEN}{idx}. {name:<15}{RESET} : {BLUE}{desc}{RESET}")
        else:
            print(f"{GREEN}{idx}. {name}{RESET}")

    # 统一输出命令行模板
    print(f"📋 数字选择后，会自动执行的命令示例:{RESET}")
    print(f"普通任务:{RESET}")
    print(f"{YELLOW}  python kuavo_deploy/src/scripts/script.py --task <chosen_task> --config {config_path}{RESET}")
    print(f"DAgger任务:{RESET}")
    print(f"{YELLOW}  python kuavo_deploy/src/scripts/script.py --task dagger --config {config_path}{RESET}")
    print(f"自动测试任务:{RESET}")
    print(f"{YELLOW}  python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config {config_path}{RESET}")
    print(f"离线bag推理:{RESET}")
    print(f"{YELLOW}  python kuavo_deploy/offline_bag_eval.py --config {config_path} --bag <bag_path> --plot{RESET}")


def is_roban_config(cfg):
    """Keep the existing Kuavo entry unchanged unless Roban is explicit."""
    platform_type = cfg.get("platform_type")
    if platform_type is None and isinstance(cfg.get("env"), dict):
        platform_type = cfg["env"].get("platform_type")
    return str(platform_type).strip().lower() == "roban"


def run_roban_wholebody_menu(config_path, runner_path):
    """Wholebody HEFT deployment intentionally has no bag replay path."""
    global current_proc, LOG_DIR

    while True:
        print("\n🟢 Roban Wholebody 可选择的任务:")
        print("1. run : 运行 HEFT wholebody 模型推理")
        print("2. 退出")
        choice = input("请选择要执行的任务 (1-2): ").strip()
        if choice == "2":
            return
        if choice != "1":
            print(f"❌ 无效选择: {choice}")
            continue

        cmd = [
            "python3",
            "-u",
            str(runner_path),
            "--task",
            "run",
            "--config",
            config_path,
        ]
        log_path = LOG_DIR / "kuavo_deploy.log"
        with open(log_path, "w") as log_file:
            current_proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT
            )
        print(f"任务已启动，PID: {current_proc.pid}")
        interactive_controller(stop_timeout=15)



# ========== 主逻辑 ==========
def main():
    global current_proc, LOG_DIR

    print_header()
    script_dir, script, auto_test, offline_bag_eval = get_script_paths()
    LOG_DIR = ensure_log_dir(script_dir)

    if not script.exists():
        print(f"错误: 找不到 script.py 文件: {script}")
        sys.exit(1)
    if not auto_test.exists():
        print(f"错误: 找不到 script_auto_test.py 文件: {auto_test}")
        sys.exit(1)
    if not offline_bag_eval.exists():
        print(f"错误: 找不到 offline_bag_eval.py 文件: {offline_bag_eval}")
        sys.exit(1)

    # print("1. 执行: python script.py --help")
    # print("2. 执行: python script_auto_test.py --help")
    # print("3. 进一步选择示例\n")

    # choice = input("请选择要执行的示例 (1-3) 或按 Enter 退出: ").strip()
    # if choice == "1":
    #     subprocess.run(["python3", str(script), "--help"])
    #     return
    # elif choice == "2":
    #     subprocess.run(["python3", str(auto_test), "--help"])
    #     return
    # elif choice == "":
    #     print("退出")
    #     return
    # elif choice != "3":
    #     print("无效选择")
    #     return

    # config_path = input("请输入自定义配置文件路径: 例如: configs/deploy/deploy.yaml").strip()
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/deploy/deploy.yaml"
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    cfg = parse_config(config_path)

    if is_roban_config(cfg):
        wholebody_runner = (
            script_dir / "roban_wholebody_deploy" / "runner.py"
        )
        if not wholebody_runner.exists():
            print(f"错误: 找不到 Roban wholebody runner: {wholebody_runner}")
            sys.exit(1)
        run_roban_wholebody_menu(config_path, wholebody_runner)
        return

    while True:
        print_task_menu(config_path=config_path, use_color=True)

        sub_choice = input("请选择要执行的示例 (1-6): ").strip()

        def start_task(cmd, record_run=False, task=None):
            global current_proc, _dagger_task
            if task != "dagger":
                with (LOG_DIR / "kuavo_deploy.log").open("w") as output:
                    current_proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
                print(f"任务已启动，PID: {current_proc.pid}")
                interactive_controller()
                return

            recorder, process, reason = None, None, "task_error"
            child_env = os.environ.copy()
            child_env.pop("DAGGER_EPISODE_ID", None)
            root = str(script_dir.parent)
            child_env["PYTHONPATH"] = os.pathsep.join([root, child_env.get("PYTHONPATH", "")])
            previous_path = sys.path[:]
            sys.path.insert(0, root)
            _dagger_task = True
            try:
                if record_run:
                    from kuavo_deploy.config import load_kuavo_config
                    from kuavo_deploy.utils.episode_recorder import EpisodeRecorder
                    recorder = EpisodeRecorder(load_kuavo_config(config_path, task="dagger"), root)
                    child_env["DAGGER_EPISODE_ID"] = recorder.start()
                with (LOG_DIR / "kuavo_deploy.log").open("w") as output:
                    process = current_proc = subprocess.Popen(
                        [sys.executable, *cmd[1:]], stdout=output, stderr=subprocess.STDOUT,
                        env=child_env, start_new_session=True)
                print(f"任务已启动，PID: {process.pid}")
                reason = interactive_controller(
                    stop_timeout=35, check_health=recorder.check_health if recorder else None)
            except (KeyboardInterrupt, SystemExit):
                reason = "interrupted"
                raise
            except Exception as exc:
                print(f"❌ DAgger 任务/录包错误: {exc}")
            finally:
                try:
                    if process is not None and process.poll() is None:
                        process.send_signal(signal.SIGUSR2)
                        try:
                            process.wait(timeout=35)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                    if recorder is not None:
                        try:
                            recorder.finish(reason, process.returncode if process else None)
                        except Exception as exc:
                            print(f"❌ 录包未正常结束: {exc}")
                finally:
                    current_proc = None
                    _dagger_task = False
                    sys.path[:] = previous_path

        if sub_choice == "1":
            start_task(["python3", str(script), "--task", "go", "--config", config_path])
        elif sub_choice == "2":
            start_task(["python3", str(script), "--task", "run", "--config", config_path])
        elif sub_choice == "3":
            print("\n是否录制 rosbag？\n1 - 不录制\n2 - 录制")
            record_choice = input("请选择 (1-2): ").strip()
            if record_choice not in ("1", "2"):
                print("❌ 无效选择: ", record_choice)
                continue
            start_task(["python3", str(script), "--task", "dagger", "--config", config_path],
                       record_run=record_choice == "2", task="dagger")
        elif sub_choice == "4":
            start_task(["python3", str(auto_test), "--task", "auto_test", "--config", config_path])
        elif sub_choice == "5":
            bag_path = input("请输入 bag 路径（留空则使用 inference.go_bag_path）: ").strip()
            # plot_choice = input("是否保存 action 对比曲线？(y/N): ").strip().lower()
            cmd = ["python3", "-u", str(offline_bag_eval), "--config", config_path, "--log-step-times"]
            if bag_path:
                cmd.extend(["--bag", bag_path])
            # if plot_choice in ("y", "yes"):
            cmd.append("--plot")
            start_task(cmd)
        elif sub_choice == "6":
            print("退出")
            break
        else:
            print("❌ 无效选择: ", sub_choice)


if __name__ == "__main__":
    main()
