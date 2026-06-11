from __future__ import annotations

import hashlib
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ..runtime import register_adapter
from .base import ModelServerAdapter, kuavo_repo_root, resolve_model_repo_root


def _resolve_lingbot_root(lingbot_root: str | None = None) -> Path:
    return resolve_model_repo_root("lingbot_vla", lingbot_root)


def _resolve_required_path(path: str, argument: str, *, directory: bool) -> Path:
    if not path:
        raise ValueError(f"Please provide `{argument}`.")

    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        expected = "directory" if directory else "file"
        raise FileNotFoundError(f"{argument} {expected} not found at: {resolved}")
    return resolved


def _resolve_training_config_path(lingbot_root: Path, training_config_path: str) -> Path:
    if training_config_path:
        return _resolve_required_path(training_config_path, "--training_config_path", directory=False)

    default_path = lingbot_root / "configs" / "vla" / "real_load20000h.yaml"
    if not default_path.is_file():
        raise FileNotFoundError(
            f"Default real-world LingBot-VLA config not found at: {default_path}. "
            "Please provide `--training_config_path`."
        )
    return default_path


def _resolve_robot_config_path(robot_config_path: str) -> Path:
    if robot_config_path:
        return _resolve_required_path(robot_config_path, "--robot_config_path", directory=False)
    return Path(__file__).resolve().parents[1] / "configs" / "lingbotvla" / "kuavo.yaml"


def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _as_hwc_uint8(img: Any) -> np.ndarray:
    arr = _to_numpy(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Image must be HWC with 3 channels, got shape={arr.shape}")
    return arr


def _build_runtime_model_dir(
    *,
    lingbot_root: Path,
    checkpoint: Path,
    training_config_path: Path,
    qwen25_path: Path,
    robot_norm_path: Path,
) -> Path:
    config = yaml.safe_load(training_config_path.read_text(encoding="utf-8")) or {}
    config.setdefault("model", {})
    config.setdefault("data", {})
    config.setdefault("train", {})

    model_json = yaml.safe_load(checkpoint.joinpath("config.json").read_text(encoding="utf-8")) or {}
    config["model"]["model_path"] = str(lingbot_root)
    config["model"]["tokenizer_path"] = str(qwen25_path)
    config["data"]["norm_stats_file"] = str(robot_norm_path)
    config["data"].setdefault("norm_type", "meanstd")
    config["train"].setdefault("chunk_size", model_json.get("chunk_size", 50))
    config["train"].setdefault("action_dim", model_json.get("action_dim", 16))
    config["train"].setdefault("max_action_dim", model_json.get("max_action_dim", 75))
    config["train"].setdefault("max_state_dim", model_json.get("max_state_dim", 75))

    fingerprint = hashlib.sha1(
        "|".join(
            [
                str(checkpoint.resolve()),
                str(training_config_path.resolve()),
                str(qwen25_path.resolve()),
                str(robot_norm_path.resolve()),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    runtime_model_dir = (
        kuavo_repo_root() / "kuavo_model" / ".runtime_staging" / "lingbot_vla" / fingerprint / "model"
    )
    runtime_model_dir.mkdir(parents=True, exist_ok=True)

    for file_path in checkpoint.iterdir():
        if file_path.name == "lingbotvla_cli.yaml":
            continue
        target = runtime_model_dir / file_path.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(file_path.resolve(), target_is_directory=file_path.is_dir())
        except OSError as exc:
            raise OSError(
                f"Failed to create symlink for LingBot runtime staging: {file_path} -> {target}. "
                "Use a filesystem that supports symlinks."
            ) from exc

    config_path = runtime_model_dir / "lingbotvla_cli.yaml"
    if config_path.is_symlink():
        config_path.unlink()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return runtime_model_dir


@register_adapter
class LingBotVlaAdapter(ModelServerAdapter):
    """Serve the current upstream LingBot-VLA policy through the Kuavo ZMQ protocol."""

    name = "lingbot_vla"

    def __init__(
        self,
        *,
        checkpoint: str,
        model_repo_root: str,
        which_arm: str,
        execution_horizon: int,
        qwen25_path: str,
        training_config_path: str,
        robot_config_path: str,
        robot_norm_path: str,
        num_denoising_step: int,
        use_compile: bool,
        use_fp32: bool,
    ) -> None:
        if execution_horizon == 0 or execution_horizon < -1:
            raise ValueError("--execution_horizon must be -1 or a positive integer.")

        self.which_arm = which_arm
        self.execution_horizon = execution_horizon
        self.checkpoint = _resolve_required_path(checkpoint, "--checkpoint", directory=True)
        if not (self.checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"LingBot-VLA config.json not found under checkpoint: {self.checkpoint}")
        self.model_repo_root = _resolve_lingbot_root(model_repo_root)
        self.qwen25_path = _resolve_required_path(qwen25_path, "--qwen25_path", directory=True)
        self.training_config_path = _resolve_training_config_path(self.model_repo_root, training_config_path)
        self.robot_config_path = _resolve_robot_config_path(robot_config_path)
        self.robot_norm_path = _resolve_required_path(robot_norm_path, "--robot_norm_path", directory=False)
        self._pending_actions: list[np.ndarray] = []

        if str(self.model_repo_root) not in sys.path:
            sys.path.insert(0, str(self.model_repo_root))
        os.environ["QWEN25_PATH"] = str(self.qwen25_path)

        runtime_model_dir = _build_runtime_model_dir(
            lingbot_root=self.model_repo_root,
            checkpoint=self.checkpoint,
            training_config_path=self.training_config_path,
            qwen25_path=self.qwen25_path,
            robot_norm_path=self.robot_norm_path,
        )
        self.runtime_model_path = str(runtime_model_dir)

        from deploy.lingbot_vla_policy import LingbotVLAServer  # type: ignore

        self.model = LingbotVLAServer(
            path_to_pi_model=self.runtime_model_path,
            use_length=self.execution_horizon,
            use_bf16=not use_fp32,
            use_fp32=use_fp32,
            robot_norm_path=str(self.robot_norm_path),
            num_denoising_step=num_denoising_step,
            use_compile=use_compile,
        )
        self._reset_model_transform()

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--checkpoint", type=str, required=True, help="Path to a LingBot-VLA hf_ckpt directory")
        parser.add_argument(
            "--model_repo_root",
            type=str,
            default="",
            help="Path to the current LingBot-VLA source checkout. Defaults to kuavo_model/external_models/lingbot-vla.",
        )
        parser.add_argument("--robot_norm_path", type=str, required=True, help="Path to Kuavo norm stats JSON")
        parser.add_argument(
            "--training_config_path",
            type=str,
            default="",
            help="LingBot-VLA YAML; defaults to configs/vla/real_load20000h.yaml in the source checkout.",
        )
        parser.add_argument(
            "--robot_config_path",
            type=str,
            default="",
            help="Kuavo feature mapping YAML; defaults to kuavo_server/configs/lingbotvla/kuavo.yaml.",
        )
        parser.add_argument("--qwen25_path", type=str, required=True, help="Path to Qwen2.5-VL tokenizer/model dir")
        parser.add_argument("--which_arm", type=str, default="both", choices=["left", "right", "both"])
        parser.add_argument("--execution_horizon", type=int, default=25, help="-1 returns the model's full chunk")
        parser.add_argument("--num_denoising_step", type=int, default=10)
        parser.add_argument("--use_compile", action="store_true")
        parser.add_argument("--use_fp32", action="store_true", help="Use fp32 inference instead of bf16")

    @classmethod
    def from_args(cls, args: Namespace) -> "LingBotVlaAdapter":
        return cls(
            checkpoint=args.checkpoint,
            model_repo_root=args.model_repo_root,
            which_arm=args.which_arm,
            execution_horizon=args.execution_horizon,
            qwen25_path=args.qwen25_path,
            training_config_path=args.training_config_path,
            robot_config_path=args.robot_config_path,
            robot_norm_path=args.robot_norm_path,
            num_denoising_step=args.num_denoising_step,
            use_compile=args.use_compile,
            use_fp32=args.use_fp32,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "which_arm": self.which_arm,
            "checkpoint": str(self.checkpoint),
            "model_repo_root": str(self.model_repo_root),
            "runtime_model_path": self.runtime_model_path,
            "training_config_path": str(self.training_config_path),
            "robot_config_path": str(self.robot_config_path),
            "robot_norm_path": str(self.robot_norm_path),
            "execution_horizon": self.execution_horizon,
        }

    def _reset_model_transform(self) -> None:
        from lingbotvla.data.vla_data.utils import FeatureTransform  # type: ignore

        self.model.global_step = 0
        self.model.last_action_chunk = None
        self.model.vla.feature_transform = FeatureTransform(
            str(self.robot_config_path),
            self.model.data_config,
            self.model.language_tokenizer,
            self.model.processor.image_processor,
            chunk_size=self.model.config.chunk_size,
            norm_stats_path=str(self.robot_norm_path),
        )

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        self._reset_model_transform()
        return {"status": "ok", "message": "adapter state cleared"}

    def _build_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation.images.head_cam_h": _as_hwc_uint8(obs["observation.images.head_cam_h"]),
            "observation.images.wrist_cam_l": _as_hwc_uint8(
                obs.get("observation.images.wrist_cam_l", obs["observation.images.head_cam_h"])
            ),
            "observation.images.wrist_cam_r": _as_hwc_uint8(
                obs.get("observation.images.wrist_cam_r", obs["observation.images.head_cam_h"])
            ),
            "observation.state": _to_numpy(obs["observation.state"]).astype(np.float32).reshape(-1),
            "task": str(obs.get("prompt", "")),
        }

    def _convert_action(self, action: Any) -> np.ndarray:
        action_np = _to_numpy(action).reshape(-1).astype(np.float64)
        if action_np.shape[0] == 16:
            if self.which_arm == "both":
                return action_np
            if self.which_arm == "left":
                return np.concatenate([action_np[:7], action_np[7:8]])
            if self.which_arm == "right":
                return np.concatenate([action_np[8:15], action_np[15:16]])
        if action_np.shape[0] == 8 and self.which_arm in ("left", "right"):
            return action_np
        raise ValueError(f"Unsupported action shape {action_np.shape} for which_arm={self.which_arm}")

    def _convert_action_chunk(self, chunk: Any) -> np.ndarray:
        chunk_np = _to_numpy(chunk)
        if chunk_np.ndim == 1:
            return self._convert_action(chunk_np)[None, :]
        if chunk_np.ndim != 2:
            raise ValueError(f"Unexpected action chunk ndim={chunk_np.ndim}, shape={chunk_np.shape}")
        return np.stack([self._convert_action(step) for step in chunk_np], axis=0)

    def _infer_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        out = self.model.infer(self._build_model_obs(obs))
        if not isinstance(out, dict) or "action" not in out:
            raise ValueError(f"Unexpected LingBot-VLA model output keys: {list(out) if isinstance(out, dict) else type(out)}")
        return self._convert_action_chunk(out["action"])

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if not self._pending_actions:
            chunk = self._infer_chunk(obs)
            self._pending_actions.extend(chunk)
        return self._pending_actions.pop(0)

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        return self._infer_chunk(obs)

    def select_action_chunk_rtc(
        self,
        obs: dict[str, Any],
        *,
        prev_chunk_leftover: np.ndarray | None,
        inference_delay: int,
        execution_horizon: int,
        rtc_options: dict[str, Any],
    ) -> dict[str, Any]:
        self._pending_actions.clear()
        options = dict(rtc_options)
        mode = str(options["mode"])
        options["mode"] = mode
        model_horizon = int(
            getattr(self.model.vla.model.config, "n_action_steps", self.model.config.chunk_size)
        )
        target_horizon = max(0, int(options.get("overlap_steps", 0)))
        options["prefix_attention_horizon"] = min(model_horizon, target_horizon)
        out = self.model.infer_rtc(
            self._build_model_obs(obs),
            prev_chunk_leftover=prev_chunk_leftover,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            rtc_options=options,
        )
        if not isinstance(out, dict) or "action" not in out or "rtc_original_actions" not in out:
            raise ValueError("Unexpected LingBot-VLA RTC model output")
        processed = self._convert_action_chunk(out["action"])
        original = _to_numpy(out["rtc_original_actions"])
        if processed.shape[0] != original.shape[0]:
            raise ValueError(
                "LingBot-VLA RTC returned misaligned processed/original chunks: "
                f"{processed.shape[0]} and {original.shape[0]}"
            )
        return {
            "processed_actions": processed,
            "original_actions": original,
            "metadata": {"backend": f"lingbot_rtc_{mode}", "rtc_mode": mode},
        }
