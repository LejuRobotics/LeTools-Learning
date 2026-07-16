from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from kuavo_deploy.utils.xvla_florence_pad_token import install_xvla_florence_pad_token_dict_patch


def _progress(msg: str) -> None:
    """打点：写入 model logger 日志文件"""
    logging.getLogger("model").info(msg)


class _CheckpointKind(str, Enum):
    FULL = "full"
    PEFT = "peft"


def _checkpoint_kind(path: Path) -> _CheckpointKind | None:
    """Return the kind of a local LeRobot checkpoint directory, if valid."""
    if not (path / "config.json").is_file():
        return None

    if (path / "model.safetensors").is_file():
        return _CheckpointKind.FULL

    adapter_weights = (
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
    )
    if (path / "adapter_config.json").is_file() and any(
        file.is_file() for file in adapter_weights
    ):
        return _CheckpointKind.PEFT

    return None


def _is_pretrained_model_dir(path: Path) -> bool:
    return _checkpoint_kind(path) is not None


def resolve_pretrained_model_dir(pretrained_path: str | Path) -> Path:
    path = Path(pretrained_path).expanduser().resolve()

    if _is_pretrained_model_dir(path):
        return path

    nested_pretrained = path / "pretrained_model"
    if _is_pretrained_model_dir(nested_pretrained):
        return nested_pretrained

    checkpoints_dir = path / "checkpoints"
    if checkpoints_dir.is_dir():
        last_pretrained = checkpoints_dir / "last" / "pretrained_model"
        if _is_pretrained_model_dir(last_pretrained):
            return last_pretrained.resolve()

        checkpoint_dirs = sorted(
            (
                p
                for p in checkpoints_dir.iterdir()
                if p.is_dir()
                and p.name.isdigit()
                and _is_pretrained_model_dir(p / "pretrained_model")
            ),
            key=lambda p: int(p.name),
        )
        if checkpoint_dirs:
            return checkpoint_dirs[-1] / "pretrained_model"

    raise FileNotFoundError(
        "Could not resolve a LeRobot pretrained model directory from: "
        f"{path}. Expected config.json plus either model.safetensors, or "
        "adapter_config.json plus adapter_model.safetensors/adapter_model.bin."
    )


def _resolve_peft_base_model(adapter_dir: Path, base_model_name_or_path: str | None) -> str:
    """Resolve a PEFT base model while preserving Hugging Face repository IDs."""
    if not base_model_name_or_path:
        raise ValueError(
            f"PEFT adapter at {adapter_dir} does not define base_model_name_or_path in adapter_config.json."
        )

    base_model = Path(base_model_name_or_path).expanduser()
    if base_model.exists():
        return str(base_model.resolve())

    # PEFT may store a relative local path. Resolve it relative to the adapter
    # directory when possible; otherwise leave it unchanged as a potential Hub ID.
    if not base_model.is_absolute():
        relative_to_adapter = adapter_dir / base_model
        if relative_to_adapter.exists():
            return str(relative_to_adapter.resolve())

    if base_model.is_absolute():
        raise FileNotFoundError(
            f"PEFT adapter references a local base model that no longer exists: {base_model}. "
            "Update base_model_name_or_path in adapter_config.json or restore the base checkpoint."
        )

    return base_model_name_or_path


def _resolve_model_weights_file(base_model_name_or_path: str) -> Path:
    """Resolve model.safetensors locally without loading its tensors."""
    base_model = Path(base_model_name_or_path)
    if base_model.is_dir():
        weights_file = base_model / "model.safetensors"
        if not weights_file.is_file():
            raise FileNotFoundError(
                f"PEFT base model directory does not contain model.safetensors: {base_model}"
            )
        return weights_file.resolve()

    try:
        from transformers.utils import cached_file

        # Deployment commonly runs offline. Prefer an existing Hub cache entry
        # and only touch the network when the weights are not cached yet.
        resolved_file = cached_file(
            base_model_name_or_path,
            "model.safetensors",
            local_files_only=True,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if resolved_file is None:
            resolved_file = cached_file(base_model_name_or_path, "model.safetensors")
    except Exception as exc:
        raise FileNotFoundError(
            "Could not load model.safetensors for the PEFT base model "
            f"'{base_model_name_or_path}'. Ensure it exists locally/in the Hugging Face cache "
            "or that the machine can access the model repository."
        ) from exc

    if resolved_file is None:
        raise FileNotFoundError(
            f"model.safetensors was not found for PEFT base model '{base_model_name_or_path}'."
        )
    return Path(resolved_file).resolve()


def _construct_empty_pi(policy_cls: type, policy_cfg: PreTrainedConfig) -> Any:
    """Construct PI0/PI05 on the meta device, then allocate final tensors.

    PI0 normally initializes billions of random FP32 parameters before casting
    most of them to BF16. That transient initialization is the dominant host-RAM
    peak. The meta device runs the unchanged LeRobot constructors without backing
    storage; ``to_empty`` then allocates each parameter once, in its final dtype.
    """
    target_device = str(policy_cfg.device)
    policy_cfg.device = "meta"
    try:
        with torch.device("meta"):
            policy = policy_cls(policy_cfg)
    finally:
        policy_cfg.device = target_device

    # Restore the deployment device in both config references before PEFT wraps
    # the policy. to_empty preserves the dtype selected by LeRobot's constructor.
    policy.config.device = target_device
    policy.to_empty(device=torch.device(target_device))
    return policy


def _prepare_policy_config_for_inference(policy_cfg: PreTrainedConfig) -> None:
    """Disable training-only PI0/PI05 features before model construction."""
    if getattr(policy_cfg, "type", None) not in {"pi0", "pi05"}:
        return

    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False
    if hasattr(policy_cfg, "gradient_checkpointing"):
        policy_cfg.gradient_checkpointing = False


def _pi_target_keys(source_key: str, policy_type: str) -> tuple[str, ...]:
    """Map OpenPI/LeRobot checkpoint keys to PI0 or PI05 policy keys."""
    bare_key = source_key.removeprefix("model.")
    if policy_type == "pi0":
        if bare_key.startswith("time_mlp_in."):
            bare_key = bare_key.replace("time_mlp_in.", "action_time_mlp_in.", 1)
        elif bare_key.startswith("time_mlp_out."):
            bare_key = bare_key.replace("time_mlp_out.", "action_time_mlp_out.", 1)
    elif policy_type == "pi05":
        if bare_key.startswith("action_time_mlp_in."):
            bare_key = bare_key.replace("action_time_mlp_in.", "time_mlp_in.", 1)
        elif bare_key.startswith("action_time_mlp_out."):
            bare_key = bare_key.replace("action_time_mlp_out.", "time_mlp_out.", 1)

        # PI05 has no state projection. Its action expert uses adaRMS, so the
        # legacy non-adaRMS norm weights are intentionally ignored, matching
        # PI05Policy._fix_pytorch_state_dict_keys.
        expert_key = "paligemma_with_expert.gemma_expert.model."
        if bare_key.startswith("state_proj."):
            return ()
        if bare_key.startswith(expert_key) and bare_key.endswith(
            ("input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight")
        ):
            return ()

    target_keys = [f"model.{bare_key}"]
    if bare_key == "paligemma_with_expert.paligemma.lm_head.weight":
        # The official checkpoint stores the tied language embedding once.
        target_keys.append(
            "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        )
    return tuple(target_keys)


def _load_pi_streaming(
    policy_cls: type,
    policy_cfg: PreTrainedConfig,
    weights_file: Path,
    *,
    strict: bool,
) -> Any:
    """Build PI0/PI05 without storage and copy one tensor at a time."""
    from safetensors import safe_open

    policy = _construct_empty_pi(policy_cls, policy_cfg)
    policy_type = str(policy_cfg.type)

    target_state = policy.state_dict()
    loaded_keys: set[str] = set()
    unexpected_keys: list[str] = []

    with safe_open(weights_file, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = list(checkpoint.keys())

    with torch.no_grad():
        for source_key in checkpoint_keys:
            # Open one tensor at a time. Keeping one safe_open handle for the
            # whole checkpoint leaves every touched mmap page resident and can
            # make a 14 GiB FP32 file appear as an equally large RSS increase.
            # Closing here allows the OS to reclaim each source tensor as soon
            # as it has been cast/copied into the final model storage.
            with safe_open(weights_file, framework="pt", device="cpu") as checkpoint:
                source_tensor = checkpoint.get_tensor(source_key)
            target_keys = _pi_target_keys(source_key, policy_type)
            if not target_keys:
                del source_tensor
                continue

            matched = False
            for target_key in target_keys:
                target_tensor = target_state.get(target_key)
                if target_tensor is None:
                    continue
                if tuple(target_tensor.shape) != tuple(source_tensor.shape):
                    raise ValueError(
                        f"PI0 weight shape mismatch for {source_key} -> {target_key}: "
                        f"checkpoint={tuple(source_tensor.shape)}, model={tuple(target_tensor.shape)}"
                    )
                # copy_ casts directly into the target dtype/device and avoids a
                # second full-size converted tensor in host memory.
                target_tensor.copy_(source_tensor)
                loaded_keys.add(target_key)
                matched = True
            if not matched:
                unexpected_keys.append(source_key)
            del source_tensor

    missing_keys = sorted(set(target_state) - loaded_keys)
    if strict and (missing_keys or unexpected_keys):
        missing_preview = missing_keys[:20]
        unexpected_preview = unexpected_keys[:20]
        raise RuntimeError(
            f"Streaming {policy_type} weight loading did not match the model exactly. "
            f"Missing ({len(missing_keys)}): {missing_preview}; "
            f"unexpected ({len(unexpected_keys)}): {unexpected_preview}"
        )
    if missing_keys or unexpected_keys:
        logging.warning(
            "%s streaming load completed with %d missing and %d unexpected keys.",
            policy_type,
            len(missing_keys),
            len(unexpected_keys),
        )
    return policy


def _load_base_policy(
    policy_cls: type,
    policy_cfg: PreTrainedConfig,
    pretrained_name_or_path: str | Path,
    *,
    strict: bool,
) -> Any:
    if getattr(policy_cfg, "type", None) in {"pi0", "pi05"}:
        weights_file = _resolve_model_weights_file(str(pretrained_name_or_path))
        return _load_pi_streaming(policy_cls, policy_cfg, weights_file, strict=strict)

    return policy_cls.from_pretrained(
        pretrained_name_or_path,
        config=policy_cfg,
        strict=strict,
    )


def _load_policy_from_checkpoint(
    pretrained_model_dir: Path,
    policy_cfg: PreTrainedConfig,
    *,
    strict: bool,
) -> Any:
    """Load either a full LeRobot model or a PEFT adapter checkpoint."""
    checkpoint_kind = _checkpoint_kind(pretrained_model_dir)
    if checkpoint_kind is None:
        raise FileNotFoundError(f"Invalid LeRobot checkpoint directory: {pretrained_model_dir}")

    policy_cls = get_policy_class(policy_cfg.type)
    if checkpoint_kind is _CheckpointKind.FULL:
        _progress(
            "[策略加载 3/5] 加载完整策略ckpt(model.safetensors): "
            f"from_pretrained … class={policy_cls.__name__}"
        )
        return _load_base_policy(
            policy_cls,
            policy_cfg,
            pretrained_model_dir,
            strict=strict,
        )

    # Keep PEFT optional for users who only deploy full checkpoints.
    try:
        from peft import PeftConfig, PeftModel
    except ImportError as exc:
        raise ImportError(
            "Loading this checkpoint requires PEFT because adapter_model.safetensors was found. "
            "Install the project's LoRA/PEFT dependencies and retry."
        ) from exc

    peft_config = PeftConfig.from_pretrained(pretrained_model_dir)
    base_model_name_or_path = _resolve_peft_base_model(
        pretrained_model_dir,
        peft_config.base_model_name_or_path,
    )
    _resolve_model_weights_file(base_model_name_or_path)
    _progress(
        "[策略加载 3/5] 加载PEFT策略ckpt: "
        f"base={base_model_name_or_path}, adapter={pretrained_model_dir}, "
        f"class={policy_cls.__name__}"
    )

    # The checkpoint's config.json contains the dataset-derived feature shapes
    # and inference settings. Use it while loading the original base weights,
    # then attach the trained adapter (including modules_to_save, if present).
    base_policy = _load_base_policy(
        policy_cls,
        policy_cfg,
        base_model_name_or_path,
        strict=strict,
    )
    policy = PeftModel.from_pretrained(
        base_policy,
        pretrained_model_dir,
        config=peft_config,
        is_trainable=False,
    )
    policy.config.use_peft = True
    policy.config.pretrained_path = pretrained_model_dir
    return policy


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
    _prepare_policy_config_for_inference(policy_cfg)
    _progress(
        f"[策略加载 2/5] 读取配置文件完成: type={policy_cfg.type} (+{time.perf_counter() - wall:.1f}s)"
    )

    if getattr(policy_cfg, "type", None) == "xvla":    # 当使用XVLA模型时,需要用此补丁在Florence2Config顶层补加pad_token_id的attribute以保证训推正常运行
        install_xvla_florence_pad_token_dict_patch()

    wall = time.perf_counter()
    policy = _load_policy_from_checkpoint(
        pretrained_model_dir,
        policy_cfg,
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
