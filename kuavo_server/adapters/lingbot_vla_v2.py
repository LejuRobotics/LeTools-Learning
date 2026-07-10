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
    return resolve_model_repo_root("lingbot_vla_v2", lingbot_root)


def _resolve_required_path(path: str, argument: str, *, directory: bool) -> Path:
    if not path:
        raise ValueError(f"Please provide `{argument}`.")

    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        expected = "directory" if directory else "file"
        raise FileNotFoundError(f"{argument} {expected} not found at: {resolved}")
    return resolved


def _resolve_optional_dir(path: str, argument: str) -> Path | None:
    if not path:
        return None
    return _resolve_required_path(path, argument, directory=True)


def _resolve_training_config_path(lingbot_root: Path, training_config_path: str) -> Path:
    if training_config_path:
        return _resolve_required_path(training_config_path, "--training_config_path", directory=False)

    default_path = lingbot_root / "configs" / "vla" / "kuavo" / "kuavo_real_depth.yaml"
    if not default_path.is_file():
        raise FileNotFoundError(
            f"Default Kuavo LingBot-VLA v2 config not found at: {default_path}. "
            "Please provide `--training_config_path`."
        )
    return default_path


def _resolve_robot_config_path(robot_config_path: str) -> Path:
    if robot_config_path:
        return _resolve_required_path(robot_config_path, "--robot_config_path", directory=False)
    return Path(__file__).resolve().parents[1] / "configs" / "lingbotvla_v2" / "kuavo.yaml"


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


def _checkpoint_has_weights(checkpoint: Path) -> bool:
    return any(checkpoint.glob("*.safetensors"))


def _stringify_mapping_list(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [str(item) if isinstance(item, dict) else item for item in value]


def _build_runtime_model_dir(
    *,
    lingbot_root: Path,
    checkpoint: Path,
    training_config_path: Path,
    qwen3vl_path: Path | None,
    robot_norm_path: Path,
) -> Path:
    config = yaml.safe_load(training_config_path.read_text(encoding="utf-8")) or {}
    config.setdefault("model", {})
    config.setdefault("data", {})
    config.setdefault("train", {})

    config["model"]["model_path"] = str(checkpoint)
    if qwen3vl_path is not None:
        config["model"]["tokenizer_path"] = str(qwen3vl_path)
    config["data"]["norm_stats_file"] = str(robot_norm_path)
    config["data"]["joints"] = _stringify_mapping_list(config["data"].get("joints"))
    config["data"]["norm_type"] = _stringify_mapping_list(config["data"].get("norm_type"))
    config["data"].setdefault(
        "robot_config_root",
        str(lingbot_root / "configs" / "robot_configs"),
    )

    fingerprint = hashlib.sha1(
        "|".join(
            [
                str(checkpoint.resolve()),
                str(training_config_path.resolve()),
                str(qwen3vl_path.resolve()) if qwen3vl_path is not None else "",
                str(robot_norm_path.resolve()),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    runtime_root = kuavo_repo_root() / "kuavo_model" / ".runtime_staging" / "lingbot_vla_v2" / fingerprint
    runtime_model_dir = runtime_root / "checkpoint" / "hf_ckpt" / "model"
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
                f"Failed to create symlink for LingBot-VLA v2 runtime staging: {file_path} -> {target}. "
                "Use a filesystem that supports symlinks."
            ) from exc

    config_path = runtime_root / "lingbotvla_cli.yaml"
    if config_path.is_symlink():
        config_path.unlink()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return runtime_model_dir


@register_adapter
class LingBotVlaV2Adapter(ModelServerAdapter):
    """Serve LingBot-VLA v2 through the Kuavo ZMQ protocol."""

    name = "lingbot_vla_v2"

    def __init__(
        self,
        *,
        checkpoint: str,
        model_repo_root: str,
        which_arm: str,
        execution_horizon: int,
        qwen3vl_path: str,
        training_config_path: str,
        robot_config_path: str,
        robot_norm_path: str,
        use_compile: bool,
        use_fp32: bool,
    ) -> None:
        if execution_horizon == 0 or execution_horizon < -1:
            raise ValueError("--execution_horizon must be -1 or a positive integer.")

        self.which_arm = which_arm
        self.execution_horizon = execution_horizon
        self.checkpoint = _resolve_required_path(checkpoint, "--checkpoint", directory=True)
        if not _checkpoint_has_weights(self.checkpoint):
            raise FileNotFoundError(f"No .safetensors weights found under LingBot-VLA v2 checkpoint: {self.checkpoint}")
        self.model_repo_root = _resolve_lingbot_root(model_repo_root)
        self.qwen3vl_path = _resolve_optional_dir(qwen3vl_path, "--qwen3vl_path")
        self.training_config_path = _resolve_training_config_path(self.model_repo_root, training_config_path)
        self.robot_config_path = _resolve_robot_config_path(robot_config_path)
        self.robot_norm_path = _resolve_required_path(robot_norm_path, "--robot_norm_path", directory=False)
        self._pending_actions: list[np.ndarray] = []

        if str(self.model_repo_root) not in sys.path:
            sys.path.insert(0, str(self.model_repo_root))
        if self.qwen3vl_path is not None:
            os.environ["QWEN3VL_PATH"] = str(self.qwen3vl_path)

        runtime_model_dir = _build_runtime_model_dir(
            lingbot_root=self.model_repo_root,
            checkpoint=self.checkpoint,
            training_config_path=self.training_config_path,
            qwen3vl_path=self.qwen3vl_path,
            robot_norm_path=self.robot_norm_path,
        )
        self.runtime_model_path = str(runtime_model_dir)

        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server  # type: ignore

        self.model = LingbotVLAv2Server(
            path_to_pi_model=self.runtime_model_path,
            robot_norm_path=str(self.robot_norm_path),
            use_length=self.execution_horizon,
            chunk_ret=True,
            use_bf16=not use_fp32,
            use_fp32=use_fp32,
            use_compile=use_compile,
        )
        self._reset_model_transform()

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--checkpoint", type=str, required=True, help="Path to a LingBot-VLA v2 hf_ckpt/model dir")
        parser.add_argument(
            "--model_repo_root",
            type=str,
            default="",
            help="Path to LingBot-VLA v2 source checkout. Defaults to kuavo_model/external_models/lingbot-vla-v2.",
        )
        parser.add_argument("--robot_norm_path", type=str, required=True, help="Path to Kuavo norm stats JSON")
        parser.add_argument(
            "--training_config_path",
            type=str,
            default="",
            help="LingBot-VLA v2 YAML; defaults to configs/vla/kuavo/kuavo_real_depth.yaml in the source checkout.",
        )
        parser.add_argument(
            "--robot_config_path",
            type=str,
            default="",
            help="Kuavo feature mapping YAML; defaults to kuavo_server/configs/lingbotvla_v2/kuavo.yaml.",
        )
        parser.add_argument(
            "--qwen3vl_path",
            type=str,
            default="",
            help="Optional local Qwen3-VL tokenizer/model dir. If empty, uses training_config_path model.tokenizer_path.",
        )
        parser.add_argument("--which_arm", type=str, default="both", choices=["left", "right", "both"])
        parser.add_argument("--execution_horizon", type=int, default=25, help="-1 returns the model's full chunk")
        parser.add_argument("--use_compile", action="store_true")
        parser.add_argument("--use_fp32", action="store_true", help="Use fp32 inference instead of bf16")

    @classmethod
    def from_args(cls, args: Namespace) -> "LingBotVlaV2Adapter":
        return cls(
            checkpoint=args.checkpoint,
            model_repo_root=args.model_repo_root,
            which_arm=args.which_arm,
            execution_horizon=args.execution_horizon,
            qwen3vl_path=args.qwen3vl_path,
            training_config_path=args.training_config_path,
            robot_config_path=args.robot_config_path,
            robot_norm_path=args.robot_norm_path,
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
            "qwen3vl_path": str(self.qwen3vl_path) if self.qwen3vl_path is not None else "",
            "execution_horizon": self.execution_horizon,
        }

    def _reset_model_transform(self) -> None:
        from lingbotvla.data.vla_data.utils import FeatureTransform  # type: ignore

        self.model.global_step = 0
        self.model.last_action_chunk = None
        self.model.last_normalized_action_chunk = None
        self.model.vla.feature_transform = FeatureTransform(
            str(self.robot_config_path),
            self.model.data_config,
            self.model.config,
            self.model.processor,
            chunk_size=self.model.config.chunk_size,
            norm_stats_path=str(self.robot_norm_path),
        )
        self.model.action_key = self.model.vla.feature_transform.org_features["actions"]

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

    def _compose_action_from_dict(self, out: dict[str, Any]) -> Any:
        if "action" in out:
            return out["action"]
        if "action.arm.position" in out and "action.effector.position" in out:
            arm = _to_numpy(out["action.arm.position"])
            effector = _to_numpy(out["action.effector.position"])
            if arm.ndim == 1:
                arm = arm[None, :]
            if effector.ndim == 1:
                effector = effector[None, :]
            if arm.shape[0] != effector.shape[0] or arm.shape[-1] != 14 or effector.shape[-1] != 2:
                raise ValueError(
                    "Unexpected split LingBot-VLA v2 action shapes: "
                    f"arm={arm.shape}, effector={effector.shape}"
                )
            return np.concatenate([arm[:, :7], effector[:, :1], arm[:, 7:14], effector[:, 1:2]], axis=-1)
        raise ValueError(f"Unexpected LingBot-VLA v2 model output keys: {list(out)}")

    def _convert_action_chunk(self, chunk: Any) -> np.ndarray:
        chunk_np = _to_numpy(chunk)
        if chunk_np.ndim == 1:
            return self._convert_action(chunk_np)[None, :]
        if chunk_np.ndim != 2:
            raise ValueError(f"Unexpected action chunk ndim={chunk_np.ndim}, shape={chunk_np.shape}")
        return np.stack([self._convert_action(step) for step in chunk_np], axis=0)

    def _infer_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        out = self.model.infer(self._build_model_obs(obs))
        if not isinstance(out, dict):
            raise ValueError(f"Unexpected LingBot-VLA v2 model output type: {type(out)}")
        return self._convert_action_chunk(self._compose_action_from_dict(out))

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if not self._pending_actions:
            chunk = self._infer_chunk(obs)
            self._pending_actions.extend(chunk)
        return self._pending_actions.pop(0)

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        return self._infer_chunk(obs)
