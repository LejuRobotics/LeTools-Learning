from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

from ..runtime import register_adapter
from .base import ModelServerAdapter


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
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got {arr.shape}")
    return arr


def _resolve_repo_root(repo_root: str | None) -> Path:
    if repo_root:
        path = Path(repo_root).expanduser().resolve()
        if path.is_dir():
            return path
    raise FileNotFoundError("Set --model_repo_root to the external model repo checkout.")


@register_adapter
class TemplateExternalModelAdapter(ModelServerAdapter):
    """Reference template for adding a new external model repo.

    Copy this file to a real adapter name and replace the TODO blocks.
    Do not import this template in `builtin_adapters.py`.
    """

    name = "template_external_model"

    def __init__(
        self,
        *,
        model_repo_root: str,
        checkpoint: str,
        which_arm: str,
    ) -> None:
        self.model_repo_root = str(_resolve_repo_root(model_repo_root))
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.which_arm = which_arm
        self._pending_actions: list[np.ndarray] = []

        repo_root = Path(self.model_repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        # TODO: import the model repo's native inference wrapper here.
        # Example:
        # from deploy.some_model_policy import SomeInferenceServer
        #
        # self.model = SomeInferenceServer(...)

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--model_repo_root", type=str, required=True)
        parser.add_argument("--checkpoint", type=str, required=True)
        parser.add_argument("--which_arm", type=str, default="both", choices=["left", "right", "both"])

    @classmethod
    def from_args(cls, args: Namespace) -> "TemplateExternalModelAdapter":
        return cls(
            model_repo_root=args.model_repo_root,
            checkpoint=args.checkpoint,
            which_arm=args.which_arm,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "model_repo_root": self.model_repo_root,
            "checkpoint": self.checkpoint,
            "which_arm": self.which_arm,
        }

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        # TODO: reset model-side caches if needed.
        return {"status": "ok", "message": "adapter state cleared"}

    def _build_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        # TODO: map standard Kuavo payload keys into model-native input keys.
        return {
            "state": _to_numpy(obs["observation.state"]).astype(np.float32).reshape(-1),
            "head_image": _as_hwc_uint8(obs["observation.images.head_cam_h"]),
            "left_wrist_image": _as_hwc_uint8(
                obs.get("observation.images.wrist_cam_l", obs["observation.images.head_cam_h"])
            ),
            "right_wrist_image": _as_hwc_uint8(
                obs.get("observation.images.wrist_cam_r", obs["observation.images.head_cam_h"])
            ),
            "prompt": str(obs.get("prompt", "")),
        }

    def _convert_action(self, action: Any) -> np.ndarray:
        action_np = _to_numpy(action).reshape(-1).astype(np.float64)
        # TODO: implement both/left/right slicing rules for the target model.
        return action_np

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if self._pending_actions:
            return self._convert_action(self._pending_actions.pop(0))

        model_obs = self._build_model_obs(obs)

        # TODO: replace this section with the real model inference call.
        raise NotImplementedError(
            "Copy template_adapter.py to a real adapter name and implement the model-specific inference call."
        )
