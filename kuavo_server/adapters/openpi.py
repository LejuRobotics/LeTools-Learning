from __future__ import annotations

import sys
import dataclasses
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

from ..runtime import register_adapter
from .base import ModelServerAdapter, resolve_model_repo_root

def _resolve_repo_root(model_repo_root: str | None = None) -> Path:
    return resolve_model_repo_root("openpi", model_repo_root)


def _ensure_repo_import_paths(repo_root: Path) -> None:
    candidates = [
        repo_root,
        repo_root / "src",
        repo_root / "packages" / "openpi-client" / "src",
        repo_root / "third_party" / "lerobot" / "src",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


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


class _OpenPiRuntime:
    def __init__(
        self,
        *,
        repo_root: Path,
        policy_config_name: str,
        checkpoint_dir: Path,
        execution_horizon: int,
        pytorch_device: str,
        asset_id: str,
        repack_structure: dict[str, Any] | None = None,
    ) -> None:
        print(f"[openpi] repo_root={repo_root}", flush=True)
        print(f"[openpi] checkpoint_dir={checkpoint_dir}", flush=True)
        _ensure_repo_import_paths(repo_root)
        print("[openpi] import paths ready", flush=True)

        from openpi import transforms as _transforms
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        from openpi_client.action_chunk_broker import ActionChunkBroker
        print("[openpi] modules imported", flush=True)

        self.config = _config.get_config(policy_config_name)
        if asset_id:
            self.config = dataclasses.replace(
                self.config,
                data=dataclasses.replace(
                    self.config.data,
                    assets=dataclasses.replace(self.config.data.assets, asset_id=asset_id),
                ),
            )
        broker_horizon = execution_horizon or int(getattr(self.config.model, "action_horizon", 1))
        print(
            f"[openpi] config={policy_config_name} asset_id={asset_id or 'default'} "
            f"model_action_horizon={getattr(self.config.model, 'action_horizon', 'unknown')}",
            flush=True,
        )

        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    repack_structure
                    or {
                        "cam_h": "observation.images.head_cam_h",
                        "cam_r": "observation.images.wrist_cam_r",
                        "cam_l": "observation.images.wrist_cam_l",
                        "state": "observation.state",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        is_pytorch = checkpoint_dir.joinpath("model.safetensors").exists()
        checkpoint_kind = "pytorch(model.safetensors)" if is_pytorch else "jax(params/)"
        print(f"[openpi] creating trained policy from {checkpoint_kind}", flush=True)

        policy = _policy_config.create_trained_policy(
            self.config,
            str(checkpoint_dir),
            repack_transforms=repack_transforms,
            pytorch_device=pytorch_device or None,
        )
        print("[openpi] trained policy created", flush=True)
        self.raw_policy = policy
        self.policy = ActionChunkBroker(policy, action_horizon=broker_horizon)
        self.execution_horizon = broker_horizon
        print(f"[openpi] action broker ready horizon={self.execution_horizon}", flush=True)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.policy.infer(obs)

    def infer_chunk(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.raw_policy.infer(obs)

    def reset(self) -> None:
        self.policy.reset()
        self.raw_policy.reset_rtc()


@register_adapter
class OpenPiJaxLejuAdapter(ModelServerAdapter):
    """Adapter for serving upstream openpi checkpoints through the standardized Kuavo ZMQ runtime."""

    name = "openpi"

    def __init__(
        self,
        *,
        checkpoint: str,
        model_repo_root: str,
        policy_config_name: str,
        which_arm: str,
        eef_dof: int,
        execution_horizon: int,
        device: str,
        asset_id: str,
        arm_joint_dim: int = 7,
        end_effector_dim: int | None = None,
    ) -> None:
        self.model_repo_root = _resolve_repo_root(model_repo_root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint dir does not exist: {self.checkpoint}")

        self.policy_config_name = policy_config_name
        self.which_arm = which_arm
        self.eef_dof = eef_dof
        self.device = device
        self.arm_joint_dim = arm_joint_dim
        self.end_effector_dim = eef_dof if end_effector_dim is None else end_effector_dim
        if self.arm_joint_dim <= 0 or self.end_effector_dim < 0:
            raise ValueError("arm_joint_dim must be positive and end_effector_dim must be non-negative")
        self.asset_id = asset_id or self._detect_asset_id(self.checkpoint)
        self.expected_state_dim = self._detect_norm_dim(self.checkpoint, self.asset_id, "state")
        self.expected_action_dim = self._detect_norm_dim(self.checkpoint, self.asset_id, "actions")
        self._pending_actions: list[np.ndarray] = []

        print(f"[openpi] initializing adapter={self.name}", flush=True)
        print(
            f"[openpi] detected asset_id={self.asset_id or 'default'} "
            f"state_dim={self.expected_state_dim} action_dim={self.expected_action_dim}",
            flush=True,
        )
        self.model = _OpenPiRuntime(
            repo_root=self.model_repo_root,
            policy_config_name=policy_config_name,
            checkpoint_dir=self.checkpoint,
            execution_horizon=execution_horizon,
            pytorch_device=device,
            asset_id=self.asset_id,
        )
        self.configured_action_dim = getattr(self.model.config.data, "physical_action_dim", None)
        if (
            self.expected_action_dim is not None
            and self.configured_action_dim is not None
            and self.expected_action_dim != self.configured_action_dim
        ):
            raise ValueError(
                f"OpenPI config {self.policy_config_name!r} emits {self.configured_action_dim} physical action dims, "
                f"but checkpoint norm stats contain {self.expected_action_dim}. For a 5W whole-body checkpoint, "
                "use policy_config_name=pi0_kuavo_5w_wholebody."
            )
        print("[openpi] adapter initialization finished", flush=True)

    @staticmethod
    def _detect_asset_id(checkpoint: Path) -> str:
        assets_dir = checkpoint / "assets"
        if not assets_dir.is_dir():
            return ""
        children = sorted(p.name for p in assets_dir.iterdir() if p.is_dir())
        if len(children) == 1:
            return children[0]
        return ""

    @staticmethod
    def _detect_norm_dim(checkpoint: Path, asset_id: str, key: str) -> int | None:
        if not asset_id:
            return None
        norm_stats_path = checkpoint / "assets" / asset_id / "norm_stats.json"
        if not norm_stats_path.is_file():
            return None
        try:
            data = json.loads(norm_stats_path.read_text(encoding="utf-8"))
            value = data.get("norm_stats", {}).get(key, {}).get("mean")
            if isinstance(value, list):
                return len(value)
        except Exception:
            return None
        return None

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--model_repo_root",
            type=str,
            default="",
            help="Optional path to openpi repo root. Defaults to kuavo_model/external_models/openpi.",
        )
        parser.add_argument("--checkpoint", type=str, required=True, help="Path to openpi checkpoint dir")
        parser.add_argument(
            "--policy_config_name",
            type=str,
            default="pi0_kuavo",
            help="openpi training config name used to construct the policy",
        )
        parser.add_argument("--which_arm", type=str, default="right", choices=["left", "right", "both"])
        parser.add_argument("--eef_dof", type=int, default=1, choices=[1, 11])
        parser.add_argument(
            "--execution_horizon",
            type=int,
            default=0,
            help="Override ActionChunkBroker horizon. 0 means use config.model.action_horizon.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="",
            help="Optional PyTorch device override, for example `cuda`, `cuda:0`, or `cpu`.",
        )
        parser.add_argument(
            "--asset_id",
            type=str,
            default="",
            help="Optional checkpoint asset id used to load norm stats. Default is auto-detect from checkpoint/assets/*.",
        )
        parser.add_argument("--arm_joint_dim", type=int, default=7)
        parser.add_argument("--end_effector_dim", type=int, default=None)

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenPiJaxLejuAdapter":
        return cls(
            checkpoint=args.checkpoint,
            model_repo_root=args.model_repo_root,
            policy_config_name=args.policy_config_name,
            which_arm=args.which_arm,
            eef_dof=args.eef_dof,
            execution_horizon=args.execution_horizon,
            device=args.device,
            asset_id=args.asset_id,
            arm_joint_dim=args.arm_joint_dim,
            end_effector_dim=args.end_effector_dim,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "model_repo_root": str(self.model_repo_root),
            "checkpoint": str(self.checkpoint),
            "policy_config_name": self.policy_config_name,
            "which_arm": self.which_arm,
            "eef_dof": self.eef_dof,
            "execution_horizon": self.model.execution_horizon,
            "asset_id": self.asset_id,
            "state_dim": self.expected_state_dim,
            "action_dim": self.expected_action_dim,
            "configured_action_dim": self.configured_action_dim,
            "arm_joint_dim": self.arm_joint_dim,
            "end_effector_dim": self.end_effector_dim,
        }

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        self.model.reset()
        return {"status": "ok", "message": "adapter state cleared"}

    def _build_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = _to_numpy(obs["observation.state"]).astype(np.float32).reshape(-1)
        state = self._adapt_state_dim(state)

        model_obs = {
            "observation.images.head_cam_h": _as_hwc_uint8(obs["observation.images.head_cam_h"]),
            "observation.images.wrist_cam_l": _as_hwc_uint8(
                obs.get("observation.images.wrist_cam_l", obs["observation.images.head_cam_h"])
            ),
            "observation.images.wrist_cam_r": _as_hwc_uint8(
                obs.get("observation.images.wrist_cam_r", obs["observation.images.head_cam_h"])
            ),
            "observation.state": state,
        }
        prompt = str(obs.get("prompt", ""))
        if prompt:
            model_obs["prompt"] = prompt
        return model_obs

    def _adapt_state_dim(self, state: np.ndarray) -> np.ndarray:
        expected = self.expected_state_dim
        if expected is None or state.shape[0] == expected:
            return state

        per_arm_dim = self.arm_joint_dim + self.end_effector_dim
        full_arm_dim = 2 * per_arm_dim
        lower_body_dim = state.shape[0] - full_arm_dim
        expected_lower_body_dim = expected - per_arm_dim
        if (
            self.which_arm in ("left", "right")
            and lower_body_dim >= 0
            and expected_lower_body_dim == lower_body_dim
        ):
            if self.which_arm == "left":
                arm_state = state[:per_arm_dim]
            if self.which_arm == "right":
                arm_state = state[per_arm_dim:full_arm_dim]
            return np.concatenate([arm_state, state[full_arm_dim:]], axis=0)

        raise ValueError(
            f"Unsupported state shape {state.shape} for expected_state_dim={expected} "
            f"under which_arm={self.which_arm}. Check that deploy 5w_wholebody and the "
            "OpenPI checkpoint/config use the same physical state layout."
        )

    def _convert_action(self, action: Any) -> np.ndarray:
        action_np = _to_numpy(action).reshape(-1).astype(np.float64)

        if self.expected_action_dim is not None and action_np.shape[0] != self.expected_action_dim:
            raise ValueError(
                f"OpenPI returned {action_np.shape[0]} action dims, but checkpoint norm stats expect "
                f"{self.expected_action_dim}. Use policy_config_name=pi0_kuavo_5w_wholebody for a "
                "20D 5W whole-body checkpoint."
            )

        if self.which_arm == "both":
            return action_np

        per_arm_dim = self.arm_joint_dim + self.end_effector_dim
        full_arm_dim = 2 * per_arm_dim
        if action_np.shape[0] >= full_arm_dim:
            arm_offset = 0 if self.which_arm == "left" else per_arm_dim
            arm_action = action_np[arm_offset : arm_offset + per_arm_dim]
            return np.concatenate([arm_action, action_np[full_arm_dim:]], axis=0)

        if action_np.shape[0] >= per_arm_dim:
            return action_np

        raise ValueError(
            f"Unsupported action shape {action_np.shape} for which_arm={self.which_arm} "
            f"under config={self.policy_config_name}"
        )

    def _predict_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        out = self.model.infer_chunk(self._build_model_obs(obs))
        if not isinstance(out, dict) or "actions" not in out:
            raise ValueError(f"Unexpected model output: {type(out)}")

        action_chunk = _to_numpy(out["actions"])
        if action_chunk.ndim == 3 and action_chunk.shape[0] == 1:
            action_chunk = action_chunk[0]
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected openpi actions shape [D] or [T, D], got {action_chunk.shape}")

        action_chunk = action_chunk[: self.model.execution_horizon]
        return np.stack([self._convert_action(step) for step in action_chunk], axis=0)

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if self._pending_actions:
            return self._pending_actions.pop(0)

        action_chunk = self._predict_action_chunk(obs)
        if action_chunk.shape[0] == 0:
            raise ValueError("openpi returned empty action chunk.")

        self._pending_actions = [np.asarray(step) for step in action_chunk[1:]]
        return np.asarray(action_chunk[0])

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        return self._predict_action_chunk(obs)

    def select_action_chunk_rtc(self, payload):
        from kuavo_server.rtc import parse_request, reconcile_absolute_prefix, response

        self._pending_actions.clear()
        policy = self.model.raw_policy
        maximum = policy.rtc_training_max_delay
        obs, rtc, original, absolute = parse_request(payload, maximum)
        prefix = None
        if original is not None:
            native = policy.decode_rtc_prefix(original, context=payload.get("previous_context"))
            prefix = reconcile_absolute_prefix(native, absolute, self._convert_action)
        out = policy.infer(self._build_model_obs(obs), rtc_options=rtc, rtc_prefix_absolute=prefix)
        processed = np.stack([self._convert_action(step) for step in out["actions"]])
        limit = self.model.execution_horizon
        return response(out["original_actions"][:limit], processed[:limit], maximum, out["action_context"])
