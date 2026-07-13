<div align="center">

<h1 align="center">LeTools-Learning</h1>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/LeRobot-0.5.2-green.svg)](https://github.com/huggingface/lerobot)
[![ROS](https://img.shields.io/badge/ROS-Noetic-22314E.svg)](http://wiki.ros.org/noetic)
[![Leju](https://img.shields.io/badge/Leju-Robotics-orange)](https://www.lejurobot.com/zh)

📖 [完整文档](https://www.letools.lejurobot.com)

</div>

---
## 🚀 News
- **[2026-05-30]** : 支持lerobot0.5.2内置的10种模型与原版模型(lingbotvla, pi0, pi0fast, pi05, gr00tN1.7)
- **[2026-06-13]** : 新增离线推理、异步推理与RTC

## ✨ 核心特性

| 特性 | 说明 |
|:---:|:---|
| 📦 **数据转换** | Rosbag → LeRobot Dataset v3 格式 |
| 🧠 **lerobot集成** | 模仿学习（ACT、DPT、Multi-task DIT）+ VLA（PI0、PI0_FAST、PI0.5、GR00T N1.5、WALL-X、XVLA、SmolVLA）|
| 🔌 **外部模型** | Pi0 / Pi0.5 / GR00T N1.7 / LingbotVla 等|
| 🚀 **仿真&真机** | Kuavo 仿真与真机部署评测流程 |
| 🦾 **多平台支持** | Kuavo 4 Pro · Kuavo 5 · Kuavo 5W |

---

## 📋 环境要求

推荐环境：

- Ubuntu 20.04
- Python 3.12
- ROS Noetic
- 支持 CUDA 的 NVIDIA GPU（训练需要）
- uv
- Docker 与 NVIDIA Container Toolkit（仿真或容器化 ROS 工作流需要）

---

## 🛠️ 安装

**1. 克隆仓库**

```bash
git clone https://github.com/LejuRobotics/LeTools-Learning.git
cd LeTools-Learning
```

**2. 确认 uv 已安装**

```bash
uv --version
```

**3. 创建 `.venv` 并一键安装依赖**

```bash
source /opt/ros/noetic/setup.bash
chmod +x setup_env.sh
bash setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` 会通过 uv 自动准备 Python 3.12，并将所有 Python 依赖安装到项目内的 `.venv`。
ROS、ffmpeg 和 PortAudio 属于系统依赖，不由 uv 管理；脚本会检测 ffmpeg，并在缺少 PortAudio 时给出安装命令。

> 📘 完整安装步骤见 [安装指南](https://www.letools.lejurobot.com/docs.html#get_started/installation.md)

---

## 🚀 快速开始

### 📊 Step 1 — 数据转换：Rosbag → LeRobot

编辑 `configs/data/KuavoRosbag2Lerobot.yaml`：

```yaml
rosbag:
  mode: normal
  rosbag_dir: /path/to/rosbags
  target_dir: /path/to/output_parent

dataset:
  platform_type: "4pro"      # 4pro, 5, or 5w
  eef_type: leju_claw        # leju_claw, rq2f85, or qiangnao
  which_arm: both            # left, right, or both
  use_videos: false          # false saves frames as images; true encodes MP4
  task_description: "Pick and Place"
```

```bash
python kuavo_data/CvtRosbag2Lerobot.py
```

> 输出目录：`/path/to/output_parent/lerobot` · 详情见 [数据准备](https://www.letools.lejurobot.com/docs.html#tutorials/data_preparation.md)

---

### 🧠 Step 2 — 策略训练

编辑配置文件，例如 `configs/train/lerobot/act.yaml`：

```yaml
dataset:
  repo_id: "lerobot/your_dataset"  # 标识名，可任意起
  root: /path/to/your/lerobot      # 本地 lerobot 数据集目录

training:
  resume: false
  output_dir: "outputs/train/act"
  job_name: "act"
  batch_size: 32
  steps: 100000
  save_freq: 20000
  num_workers: 8
  seed: 1000
```

```bash
python kuavo_model/train.py --policy act
```

<details>
<summary>📝 常用训练参数</summary>

| 参数 | 说明 |
|:---|:---|
| `--policy` | **必填**，支持：`act` `diffusion` `pi0` `pi0_fast` `pi05` `gr00t` `smolvla` `xvla` `multi_task_dit` `wall_x` |
| `--mode simple` | 默认模式，先读 `total/<policy>_total.yaml`，再用 `<policy>.yaml` 覆盖常用字段 |
| `--mode total` | 只读取完整配置 |
| `--launcher python` | 默认，单机单卡 |
| `--launcher accelerate` | 单机多卡，需先配置 `configs/accelerate/accelerate_config.yaml` |
| `--dry-run` | 只打印解析后的命令和配置，不真正启动训练 |
| `--no-timestamp` | 不为输出目录追加时间戳 |

</details>

> 📘 更多说明见 [LeRobot 模型训练](https://www.letools.lejurobot.com/docs.html#tutorials/lerobot_training.md) · [原生模型训练](https://www.letools.lejurobot.com/docs.html#tutorials/model_training.md)

---

### 🚁 Step 3 — 推理部署

编辑 `configs/deploy/deploy.yaml`：

```yaml
env:
  inference_env: sim                      # sim=仿真, real=真机
  platform_type: "4pro"
  which_arm: both
  eef_type: rq2f85
  ros_rate: 10

inference:
  policy_type: act                        # 策略名称
  pretrained_path: /path/to/checkpoint    # 权重路径
  task_prompt: "robot manipulation"       # VLA 策略需指定 prompt
```

```bash
python kuavo_deploy/eval.py
```

<details>
<summary>🔌 外部模型部署</summary>

外部模型位于 `kuavo_model/external_models/`，当前集成：

- `openpi` · `gr00tn1d7` · `lingbot-vla`

均可通过 `kuavo_server` adapter 接入统一部署流程。使用外部模型时：
1. 先启动模型服务器
2. 在部署配置中设置 `inference.policy_type: client`

</details>

> 📘 详情见 [推理与部署](https://www.letools.lejurobot.com/docs.html#tutorials/inference.md)

---

## 🗂️ 项目结构

```text
.
├── configs/                    # 数据、训练、部署、平台配置
│   ├── data/                   # Rosbag -> LeRobot 数据转换配置
│   ├── train/lerobot/          # LeRobot simple/total 训练配置
│   ├── deploy/                 # 仿真 / 真机部署配置
│   ├── platform/               # Kuavo 平台关节与机器人映射
│   └── accelerate/             # 多卡训练配置
├── kuavo_data/                 # 数据转换与数据处理工具
├── kuavo_model/                # 训练入口与外部模型目录
├── kuavo_deploy/               # ROS 部署、评测、仿真/真机环境
├── kuavo_server/               # 标准化模型服务 adapter
├── third_party/                # LeRobot 等第三方子模块
├── lerobot_patches/            # 上游兼容补丁
└── outputs/                    # 训练与评测输出
```

## 📚 文档导航


| 主题           | 文档                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------ |
| 项目介绍         | [查看文档](https://www.letools.lejurobot.com/docs.html#get_started/intro.md)          |
| 安装指南         | [查看文档](https://www.letools.lejurobot.com/docs.html#get_started/installation.md)   |
| 快速开始         | [查看文档](https://www.letools.lejurobot.com/docs.html#tutorials/quick_start.md)      |
| 数据准备         | [查看文档](https://www.letools.lejurobot.com/docs.html#tutorials/data_preparation.md) |
| LeRobot 模型训练 | [查看文档](https://www.letools.lejurobot.com/docs.html#tutorials/lerobot_training.md) |
| 外挂模型训练       | [查看文档](https://www.letools.lejurobot.com/docs.html#tutorials/model_training.md)   |
| 推理与部署        | [查看文档](https://www.letools.lejurobot.com/docs.html#tutorials/inference.md)        |
| 新策略拓展        | [查看文档](https://www.letools.lejurobot.com/docs.html#tutorials/bring_policies.md)   |



## 💬 支持与反馈

我们鼓励通过 GitHub 渠道反馈问题，使之可被搜索、归档，也方便后来者复用。


| 我想要...           | 推荐渠道                                                                                                                   |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 报告 Bug 或运行时报错    | [GitHub Issues](https://github.com/LejuRobotics/LeTools-Learning/issues/new?labels=bug)（请附环境信息、复现步骤、完整 traceback） |
| 提交新功能建议 / 模型接入需求 | [GitHub Issues](https://github.com/LejuRobotics/LeTools-Learning/issues/new?labels=enhancement)                   |                   |
| 国内用户交流群          | QQ 群 / 微信群入群方式见 [社区交流](https://www.letools.lejurobot.com/docs.html#troubleshooting/community.md)        |
| 项目合作 / 企业级支持     | [lejurobot@lejurobot.com](mailto:lejurobot@lejurobot.com)                                                              |


**提交 Issue 前请先：**

1. 搜索 [已有 Issues](https://github.com/LejuRobotics/LeTools-Learning/issues?q=is%3Aissue) 和 [常见问题 FAQ](https://www.letools.lejurobot.com/docs.html#troubleshooting/faq.md)，避免重复。
2. 阅读 [贡献指南](https://github.com/LejuRobotics/LeTools-Learning/blob/main/CONTRIBUTING.md) 了解分支策略与 PR 流程。
3. 使用对应的 [Issue 模板](https://github.com/LejuRobotics/LeTools-Learning/issues/new/choose)，填全环境与复现信息。

**响应预期：** 本项目由乐聚机器人算法团队在工作时间维护。我们会尽力响应所有公开渠道的反馈，但**不对修复时间作正式 SLA 承诺**；如需企业级保障，请走商业合作邮箱。

**乐聚内部成员：内部沟通渠道**（外部用户可忽略）

内部沟通流程不公开维护在本仓库，请通过以下入口：

- **KDC 用户群**（飞书）：日常使用答疑
- **飞书表格**：缺陷提报 / 意见反馈 填入飞书表格，将会按严重等级流转（严重缺陷 → 紧急修复；普通缺陷 → 算法部临时方案 + 正式版本修复；意见建议 → 评审后纳入计划）
- **飞书项目**：如需新增全新的功能（如新模型接入）走正式需求流程

详细流程请见公司内部 Wiki。

## 🙏 致谢

本项目构建在以下开源项目和生态之上：

- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [Lingbot-VLA](https://github.com/robbyant/lingbot-vla)
- Kuavo 人形机器人软件生态

## 📄 许可证

本仓库基于 [GNU General Public License v3.0](LICENSE) 开源。
