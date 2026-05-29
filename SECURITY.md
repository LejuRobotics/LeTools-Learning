# 安全政策 Security Policy

> 本项目欢迎安全研究者负责任地披露安全漏洞。
> Responsible security disclosure is welcomed.

## 报告漏洞 Reporting a Vulnerability

**请勿在公开的 GitHub Issue 中提交安全漏洞。**
**Please do NOT report security vulnerabilities through public GitHub issues.**

请将漏洞详情发送至本仓库任一 maintainer 邮箱。

报告中请包含：

- 漏洞所在的文件 / 模块 / 子系统
- 影响版本（commit hash 或 release tag）
- 复现步骤（最好附 PoC）
- 你认为的潜在影响（信息泄露 / RCE / 拒绝服务 / 物理安全等）
- 你希望的署名方式（公开致谢 / 匿名）

## 我们的承诺 Our Commitment

| 阶段 | 目标响应时间 |
| --- | --- |
| 首次确认收到 | 5 个工作日内 |
| 初步影响评估 | 10 个工作日内 |
| 修复方案 / 缓解措施 | 视严重程度而定 |
| 公开披露窗口 | 与报告者协商，通常 30 ~ 90 天 |

我们不提供漏洞赏金 (Bug Bounty)，但会在 release notes 中公开致谢（除非你要求匿名）。

## 支持的版本 Supported Versions

仅 `master` / `main` 分支接受安全修复。历史 release tag **不再** 反向移植安全补丁，请用户自行升级到最新版本。

## 范围 Scope

**纳入** 安全披露的内容：

- 本仓库下的全部源码（不含 `third_party/` 子模块、不含 `kuavo_model/external_models/` 外部模型仓库）
- 默认配置 (`configs/`)
- 部署脚本与服务 (`kuavo_server/`, `kuavo_deploy/`)

**不在范围内**：

- 第三方仓库（请向上游报告：LeRobot / OpenPI / NVIDIA GR00T / LingBot-VLA 等）
- 已被 EoL（End of Life）的依赖版本
- 仅在用户自行修改默认配置后才可触发的问题
- 机器人物理硬件的固件安全（请联系硬件团队）

## 关于人形机器人的物理安全

⚠️ **本项目控制的是物理机器人。** 任何可能导致机器人异常运动、人身伤害的漏洞，请按 **严重缺陷** 等级紧急报告，我们会优先处理。
