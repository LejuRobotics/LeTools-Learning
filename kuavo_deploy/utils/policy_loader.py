from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from kuavo_deploy.utils.xvla_florence_pad_token import install_xvla_florence_pad_token_dict_patch


def _progress(msg: str) -> None:
    """打点：写入 model logger 日志文件"""
    logging.getLogger("model").info(msg)


def _is_pretrained_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() and (path / "model.safetensors").exists()


def resolve_pretrained_model_dir(pretrained_path: str | Path) -> Path:
    path = Path(pretrained_path).expanduser().resolve()

    if _is_pretrained_model_dir(path):
        return path

    nested_pretrained = path / "pretrained_model"
    if _is_pretrained_model_dir(nested_pretrained):
        return nested_pretrained

    checkpoints_dir = path / "checkpoints"
    if checkpoints_dir.is_dir():
        checkpoint_dirs = sorted(
            (
                p
                for p in checkpoints_dir.iterdir()
                if p.is_dir() and p.name.isdigit() and _is_pretrained_model_dir(p / "pretrained_model")
            ),
            key=lambda p: int(p.name),
        )
        if checkpoint_dirs:
            return checkpoint_dirs[-1] / "pretrained_model"

    raise FileNotFoundError(
        f"Could not resolve a LeRobot pretrained model directory from: {path}"
    )


def resolve_eval_output_dir(pretrained_model_dir: str | Path, output_root: str | Path) -> Path:
    pretrained_model_dir = Path(pretrained_model_dir).resolve()
    output_root = Path(output_root).resolve()

    checkpoints_dir = pretrained_model_dir.parent
    if checkpoints_dir.name == "checkpoints":
        run_dir = checkpoints_dir.parent
        checkpoint_name = pretrained_model_dir.parent.name
        return output_root / run_dir.name / checkpoint_name

    return output_root / pretrained_model_dir.parent.name


def load_native_policy_bundle(
    pretrained_path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> tuple[Any, Any, Any, Path]:

    wall = time.perf_counter()
    _progress(
        f"[策略加载 1/5] 解析预训练路径(LeRobot checkpoint)… "
        f"raw={pretrained_path}"
    )
    pretrained_model_dir = resolve_pretrained_model_dir(pretrained_path)
    _progress(
        f"[策略加载 1/5] 解析预训练路径完成: resolved_dir={pretrained_model_dir} (+{time.perf_counter() - wall:.1f}s)"
    )

    wall = time.perf_counter()
    _progress("[策略加载 2/5] 读取配置文件 config.json ")
    policy_cfg = PreTrainedConfig.from_pretrained(
        pretrained_model_dir,
        device=str(device),
    )
    policy_cfg.device = str(device)
    _progress(
        f"[策略加载 2/5] 读取配置文件完成: type={policy_cfg.type} (+{time.perf_counter() - wall:.1f}s)"
    )

    if getattr(policy_cfg, "type", None) == "xvla":    # 当使用XVLA模型时,需要用此补丁在Florence2Config顶层补加pad_token_id的attribute以保证训推正常运行
        install_xvla_florence_pad_token_dict_patch()

    wall = time.perf_counter()
    policy_cls = get_policy_class(policy_cfg.type)
    _progress(
        f"[策略加载 3/5] 加载策略ckpt(构图->读model.safetensors->按需从HF下载backbone): from_pretrained … class={policy_cls.__name__} "
        "若等待时间过长,请检查HF cache路径体积是否增长中"
    )
    policy = policy_cls.from_pretrained(
        pretrained_model_dir,
        config=policy_cfg,
        strict=strict,
    )
    _progress(
        f"[策略加载 3/5] 加载策略checkpoint完成: (+{time.perf_counter() - wall:.1f}s)"
    )

    wall = time.perf_counter()
    policy.eval()
    policy.to(device)
    policy.reset()
    _progress(
        f"[策略加载 4/5] 评估与重置完成: policy.eval()+to({device})+reset 完成 (+{time.perf_counter() - wall:.1f}s)"
    )

    wall = time.perf_counter()
    _progress("[策略加载 5/5] 构建预处理器与后处理器: make_pre_post_processors…")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(pretrained_model_dir),
    )
    _progress(
        f"[策略加载 5/5] 构建预处理器与后处理器完成: (+{time.perf_counter() - wall:.1f}s) · "
        "load_native_policy_bundle 完成"
    )
    return policy, preprocessor, postprocessor, pretrained_model_dir


def inject_task_prompt(
    observation: dict[str, Any],
    task_prompt: str | None,
) -> dict[str, Any]:
    """
    Inject a language task into a native LeRobot observation.

    LeRobot's processor pipelines expect language under the top-level ``task`` key.
    The batch-to-transition converter will move it into complementary_data, where
    tokenizer and VLA-specific processor steps can consume it.
    """
    if not isinstance(observation, dict):
        return observation

    if "task" in observation and observation["task"]:
        return observation

    prompt = (task_prompt or "").strip()
    if not prompt:
        return observation

    observation_with_prompt = dict(observation)
    observation_with_prompt["task"] = prompt
    return observation_with_prompt
