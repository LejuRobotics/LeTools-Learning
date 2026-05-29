# 贡献指南 Contributing Guide

感谢你愿意为 **Kuavo Learning Studio** 贡献代码、文档或想法！本指南适用于所有外部和内部贡献者。

> 提交贡献即表示你同意遵守本仓库的 [行为准则 (Code of Conduct)](CODE_OF_CONDUCT.md)。

## 在开始之前 Before You Start

- **小改动**（typo、格式、文档补充）：直接发 PR，无需先开 Issue。
- **新功能 / 新模型接入 / 架构变更**：请**先开 Issue 讨论**，避免做完才发现方向不合适。
- **Bug 修复**：建议先开 Issue 描述复现路径，再 PR 关联。

## 开发流程 Development Workflow

### 1. Fork & Clone

```bash
git clone https://github.com/<your-name>/kuavo_learning_studio.git
cd kuavo_learning_studio
git remote add upstream https://github.com/LejuRobotics/kuavo_learning_studio.git
git submodule update --init --recursive
```

### 2. 创建特性分支 Branch Strategy

**禁止直接在 `master` 上提交。** 请基于 `master` 切新分支：

| 分支类型 | 命名规范 | 用途 |
| --- | --- | --- |
| 功能 | `feat/<short-name>` | 新功能 / 新模型 |
| 修复 | `fix/<short-name>` | Bug 修复 |
| 文档 | `docs/<short-name>` | 仅改文档 |
| 重构 | `refactor/<short-name>` | 不改变行为的内部重构 |

示例：

```bash
git checkout master && git pull upstream master
git checkout -b feat/add-newmodel-adapter
```

### 3. 提交规范 Commit Message

采用 [Conventional Commits](https://www.conventionalcommits.org/) 风格，便于自动生成 CHANGELOG：

```
<type>(<scope>): <subject>

<body, 可选>
```

`type` 取值：

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档变更
- `refactor`: 重构（不改行为）
- `perf`: 性能优化
- `test`: 测试相关
- `build`: 构建 / 依赖
- `ci`: CI 配置
- `chore`: 其它

示例：

```
feat(kuavo_server): add wall_x adapter
fix(kuavo_data): handle empty depth topic in rosbag converter
docs(readme): clarify Python 3.10 requirement for AGX Orin
```

### 4. 代码风格 Code Style

- **Python**：遵循 PEP 8；使用 `ruff` / `black` 格式化（行宽 100）。
- **YAML 配置**：缩进 2 空格，键名 snake_case。
- **文档**：中文文档与英文文档可并存；面向用户的命令、参数、路径必须可直接复制运行。
- **不要提交：** `.pyc`、`__pycache__/`、`outputs/`、本地训练日志、个人 IDE 配置、大文件 (>10MB)。

### 5. 提交 Pull Request

PR 标题遵循同样的 Conventional Commits 风格。PR 正文请包含：

- **What**: 这个 PR 做了什么？
- **Why**: 为什么需要这个改动？
- **How tested**: 如何验证？（命令、数据集、checkpoint 路径）
- **Breaking changes**: 是否破坏现有接口？
- **Related issues**: `Closes #123` / `Refs #456`

PR 提交后 CI 会运行基础检查。**请在自己本地确认通过后再请求 review**。

## 接入新模型 Adding a New Model

请遵循 [kuavo_server/docs/add_new_model.md](kuavo_server/docs/add_new_model.md) 中的完整 7 步流程：

1. 在 `kuavo_server/adapters/` 下创建 adapter
2. 仅在 adapter 内做 obs / action 转换
3. 在 `kuavo_server/builtin_adapters.py` 注册
4. 更新 `kuavo_server/README.md` 添加启动命令
5. （如果带训练）在 `configs/train/lerobot/` 或 `kuavo_model/external_models/` 下补全配置和文档
6. 提供至少一份 smoke test 命令
7. 在 PR 描述里说明已用何种数据 / checkpoint 跑通

## 文档贡献 Docs Contribution

文档源在 `docs/` 与各模块 `readme.md`。请：

- 中文为主，专有名词保留英文（如 `LeRobot`、`PI0`、`ROS`）。
- 提供可直接复制运行的命令（含完整路径占位符）。
- 任何 `<placeholder>` 必须在文末解释或在命令中标注。
- 不要在公开文档中暴露内部 IP / 内部 SaaS / 个人邮箱。

## 报告 Bug Reporting Bugs

请通过 [GitHub Issues](https://github.com/LejuRobotics/kuavo_learning_studio/issues/new) 提交，附：

- 环境信息：OS、Python 版本、CUDA 版本、ROS 版本、GPU 型号、commit hash
- 完整复现命令
- 完整 traceback
- 相关配置文件（必要时脱敏）

## 沟通渠道 Communication

- **公开异步讨论**：[GitHub Discussions](https://github.com/LejuRobotics/kuavo_learning_studio/discussions)
- **Bug / 功能请求**：[GitHub Issues](https://github.com/LejuRobotics/kuavo_learning_studio/issues)
- **安全漏洞**：见 [SECURITY.md](SECURITY.md)
- **乐聚内部成员**：内部沟通流程见公司 Wiki，不在本仓库维护

## 许可证 License

提交贡献即表示你同意你的贡献以 [GNU General Public License v3.0](LICENSE) 发布。
