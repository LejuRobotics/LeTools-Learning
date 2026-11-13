from __future__ import annotations

import os
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
from .base import ModelServerAdapter, resolve_model_repo_root


def _resolve_repo_root(model_repo_root: str | None = None) -> Path:
    return resolve_model_repo_root("isaac_gr00t_n17", model_repo_root)


def _ensure_repo_import_paths(repo_root: Path) -> None:
    candidates = [repo_root]
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
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got {arr.shape}")
    return arr


def _normalize_key(key: str) -> str:
    return key.lower().replace(".", "_").replace("-", "_")


def _fit_dim(vec: np.ndarray, dim: int, fill: float = 0.0) -> np.ndarray:
    arr = vec.astype(np.float32).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    if arr.shape[0] > dim:
        return arr[:dim]
    pad = np.full((dim - arr.shape[0],), fill, dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def _kuavo_state(raw_state: Any, which_arm: str, eef_dof: int) -> np.ndarray:
    state = _to_numpy(raw_state).astype(np.float32).reshape(-1)
    full_dim = 14 + 2 * eef_dof
    single_dim = 7 + eef_dof
    if state.shape[0] == full_dim:
        return state
    if state.shape[0] == 14:
        return np.concatenate([
            state[:7], np.zeros(eef_dof, dtype=np.float32),
            state[7:14], np.zeros(eef_dof, dtype=np.float32),
        ])
    if state.shape[0] == single_dim:
        if which_arm == "left":
            return np.concatenate([state, np.zeros(single_dim, dtype=np.float32)])
        if which_arm == "right":
            return np.concatenate([np.zeros(single_dim, dtype=np.float32), state])
    if state.shape[0] > full_dim:
        raise ValueError(
            f"Legacy GR00T checkpoint expects at most {full_dim} arm/eef state dims, got {state.shape[0]}. "
            "Use a checkpoint whose modality/norm metadata includes lower_body for 5W whole-body control."
        )
    if state.shape[0] < full_dim:
        return _fit_dim(state, full_dim)
    return state


def _is_left_key(name: str) -> bool:
    tokens = ("left", "l_", "_l", "arm_l", "zarm_l", "larm")
    return any(token in name for token in tokens)


def _is_right_key(name: str) -> bool:
    tokens = ("right", "r_", "_r", "arm_r", "zarm_r", "rarm")
    return any(token in name for token in tokens)


def _is_gripper_key(name: str) -> bool:
    tokens = ("gripper", "claw", "hand", "effector", "sg100")
    return any(token in name for token in tokens)


def _is_lower_body_key(name: str) -> bool:
    tokens = ("lower_body", "lowerbody", "wheel", "leg", "waist")
    return any(token in name for token in tokens)


def _is_arm_like(name: str) -> bool:
    tokens = ("arm", "joint", "qpos", "zarm", "link", "state", "position")
    return any(token in name for token in tokens)


def _map_video_source_key(video_key: str) -> str:
    key = _normalize_key(video_key)
    if "left" in key or "_l" in key:
        return "observation.images.wrist_cam_l"
    if "right" in key or "_r" in key:
        return "observation.images.wrist_cam_r"
    if "wrist" in key and "left" not in key and "right" not in key:
        return "observation.images.wrist_cam_r"
    return "observation.images.head_cam_h"


class _Gr00tRuntime:
    def __init__(
        self,
        *,
        repo_root: Path,
        model_path: Path,
        embodiment_tag_raw: str,
        device: str,
        strict: bool,
    ) -> None:
        _ensure_repo_import_paths(repo_root)
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        tag = self._parse_embodiment_tag(embodiment_tag_raw, EmbodimentTag)
        self.policy = Gr00tPolicy(
            embodiment_tag=tag,
            model_path=str(model_path),
            device=device,
            strict=strict,
        )
        self.embodiment_value = tag.value
        self.modality = self.policy.get_modality_config()
        self.state_keys = list(self.modality["state"].modality_keys)
        self.action_keys = list(self.modality["action"].modality_keys)
        self.video_keys = list(self.modality["video"].modality_keys)
        self.language_key = self.modality["language"].modality_keys[0]
        self.action_horizon = len(self.modality["action"].delta_indices)

        norm_params = self.policy.processor.state_action_processor.norm_params[self.embodiment_value]
        self.state_dims = {
            key: int(norm_params["state"][key]["dim"].item()) for key in self.state_keys
        }
        self.action_dims = {
            key: int(norm_params["action"][key]["dim"].item()) for key in self.action_keys
        }

    @staticmethod
    def _parse_embodiment_tag(raw: str, enum_cls: Any) -> Any:
        normalized = raw.strip()
        if not normalized:
            return enum_cls.NEW_EMBODIMENT
        if hasattr(enum_cls, normalized):
            return getattr(enum_cls, normalized)
        upper_key = normalized.upper()
        if hasattr(enum_cls, upper_key):
            return getattr(enum_cls, upper_key)
        for item in enum_cls:
            if item.value == normalized:
                return item
        known = ", ".join([item.name for item in enum_cls])
        raise ValueError(f"Unknown embodiment_tag `{raw}`. Available: {known}")

    def infer(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        actions, _ = self.policy.get_action(observation)
        return actions

    def reset(self) -> None:
        self.policy.reset()
        self.policy._rtc_previous_states = None


@register_adapter
class IsaacGr00tN17Adapter(ModelServerAdapter):
    name = "isaac_gr00t_n17"

    def __init__(
        self,
        *,
        model_repo_root: str,
        checkpoint: str,
        embodiment_tag: str,
        which_arm: str,
        eef_dof: int,
        execution_horizon: int | None,
        device: str,
        strict: bool,
    ) -> None:
        self.model_repo_root = _resolve_repo_root(model_repo_root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint dir does not exist: {self.checkpoint}")

        self.which_arm = which_arm
        self.eef_dof = eef_dof
        self.execution_horizon = execution_horizon
        self._pending_actions: list[np.ndarray] = []
        self._last_state: np.ndarray = np.zeros(14 + 2 * eef_dof, dtype=np.float32)

        print(f"[isaac-gr00t-n17] initializing adapter={self.name}", flush=True)
        print(f"[isaac-gr00t-n17] repo_root={self.model_repo_root}", flush=True)
        print(f"[isaac-gr00t-n17] checkpoint={self.checkpoint}", flush=True)

        self.model = _Gr00tRuntime(
            repo_root=self.model_repo_root,
            model_path=self.checkpoint,
            embodiment_tag_raw=embodiment_tag,
            device=device,
            strict=strict,
        )
        lower_dims = [
            dims[key]
            for keys, dims in (
                (self.model.state_keys, self.model.state_dims),
                (self.model.action_keys, self.model.action_dims),
            )
            for key in keys
            if _is_lower_body_key(_normalize_key(key))
        ]
        packed_arm_dim = (7 + self.eef_dof) * (2 if self.which_arm == "both" else 1)
        if not lower_dims and len(self.model.state_keys) == len(self.model.action_keys) == 1:
            state_dim = self.model.state_dims[self.model.state_keys[0]]
            action_dim = self.model.action_dims[self.model.action_keys[0]]
            if state_dim == action_dim and state_dim > packed_arm_dim:
                lower_dims = [state_dim - packed_arm_dim]
        self.lower_body_dim = max(lower_dims, default=0)
        if any(dim != self.lower_body_dim for dim in lower_dims):
            raise ValueError(f"Inconsistent GR00T lower-body dimensions: {lower_dims}")
        self._last_lower_body = np.zeros(self.lower_body_dim, dtype=np.float32)
        print(
            f"[isaac-gr00t-n17] embodiment={self.model.embodiment_value} "
            f"state_keys={self.model.state_keys} action_keys={self.model.action_keys}",
            flush=True,
        )
        print(
            f"[isaac-gr00t-n17] model_action_chunk_size={self.model.action_horizon} "
            f"execution_steps={self.execution_horizon if self.execution_horizon is not None else self.model.action_horizon} "
            f"which_arm={self.which_arm}",
            flush=True,
        )

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--model_repo_root",
            type=str,
            default="",
            help="Optional path to Isaac-GR00T-N17 repo root. Defaults to kuavo_model/external_models/gr00tn1d7.",
        )
        parser.add_argument("--checkpoint", type=str, required=True, help="Path to Isaac-GR00T-N17 checkpoint dir")
        parser.add_argument(
            "--embodiment_tag",
            type=str,
            default="NEW_EMBODIMENT",
            help="Embodiment tag name or value (e.g., NEW_EMBODIMENT or new_embodiment)",
        )
        parser.add_argument("--which_arm", type=str, default="both", choices=["left", "right", "both"])
        parser.add_argument("--eef_dof", type=int, default=1, choices=[1, 11])
        parser.add_argument("--execution_horizon", type=int, default=16, help="Number of actions to execute per chunk (receding horizon). Defaults to full model prediction.")
        parser.add_argument("--device", type=str, default="cuda", help="Torch device passed to Gr00tPolicy")
        parser.add_argument("--strict", action="store_true", help="Enable strict input/output checks in Gr00tPolicy")

    @classmethod
    def from_args(cls, args: Namespace) -> "IsaacGr00tN17Adapter":
        return cls(
            model_repo_root=args.model_repo_root,
            checkpoint=args.checkpoint,
            embodiment_tag=args.embodiment_tag,
            which_arm=args.which_arm,
            eef_dof=args.eef_dof,
            execution_horizon=args.execution_horizon,
            device=args.device,
            strict=args.strict,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "model_repo_root": str(self.model_repo_root),
            "checkpoint": str(self.checkpoint),
            "embodiment_tag": self.model.embodiment_value,
            "which_arm": self.which_arm,
            "lower_body_dim": self.lower_body_dim,
            "eef_dof": self.eef_dof,
        }

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        self.model.reset()
        return {"status": "ok", "message": "adapter state cleared"}

    def _split_input_state(self, raw_state: Any) -> tuple[np.ndarray, np.ndarray]:
        state = _to_numpy(raw_state).astype(np.float32).reshape(-1)
        packed_arm_dim = (7 + self.eef_dof) * (2 if self.which_arm == "both" else 1)
        expected = packed_arm_dim + self.lower_body_dim
        if self.lower_body_dim == 0:
            return _kuavo_state(state, self.which_arm, self.eef_dof), np.zeros(0, dtype=np.float32)
        if state.shape[0] != expected:
            raise ValueError(
                f"GR00T {self.which_arm} whole-body checkpoint expects {expected} state dims "
                f"({packed_arm_dim} arm/eef + {self.lower_body_dim} lower body), got {state.shape[0]}"
            )
        return _kuavo_state(state[:packed_arm_dim], self.which_arm, self.eef_dof), state[packed_arm_dim:]

    def _packed_state(self, kuavo_state: np.ndarray, lower_body: np.ndarray) -> np.ndarray:
        if self.which_arm == "left":
            arm_eef = kuavo_state[:7 + self.eef_dof]
        elif self.which_arm == "right":
            arm_eef = kuavo_state[7 + self.eef_dof:14 + 2 * self.eef_dof]
        else:
            arm_eef = kuavo_state
        return np.concatenate((arm_eef, lower_body), axis=0)

    def _state_for_key(
        self,
        key: str,
        dim: int,
        kuavo_state: np.ndarray,
        lower_body: np.ndarray,
    ) -> np.ndarray:
        full_dim = 14 + 2 * self.eef_dof
        left_arm = kuavo_state[:7]
        left_gripper = kuavo_state[7:7 + self.eef_dof]
        right_arm = kuavo_state[7 + self.eef_dof:14 + self.eef_dof]
        right_gripper = kuavo_state[14 + self.eef_dof:full_dim]
        both_arms = np.concatenate([left_arm, right_arm], axis=0)

        name = _normalize_key(key)
        if _is_lower_body_key(name):
            return _fit_dim(lower_body, dim)
        if _is_gripper_key(name):
            if _is_left_key(name):
                return _fit_dim(left_gripper, dim)
            if _is_right_key(name):
                return _fit_dim(right_gripper, dim)
            if self.which_arm == "left":
                return _fit_dim(left_gripper, dim)
            if self.which_arm == "right":
                return _fit_dim(right_gripper, dim)
            return _fit_dim(np.concatenate([left_gripper, right_gripper], axis=0), dim)

        if _is_arm_like(name):
            if _is_left_key(name):
                return _fit_dim(left_arm, dim)
            if _is_right_key(name):
                return _fit_dim(right_arm, dim)
            if "single_arm" in name and self.which_arm in ("left", "right"):
                return _fit_dim(left_arm if self.which_arm == "left" else right_arm, dim)
            if dim == 14:
                return _fit_dim(both_arms, dim)
            if dim == full_dim:
                return _fit_dim(kuavo_state, dim)

        packed = self._packed_state(kuavo_state, lower_body)
        return _fit_dim(packed, dim)

    def _build_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        kuavo_state, lower_body = self._split_input_state(obs["observation.state"])
        self._last_state = kuavo_state
        self._last_lower_body = lower_body
        prompt = str(obs.get("prompt", ""))

        video: dict[str, np.ndarray] = {}
        for key in self.model.video_keys:
            source = _map_video_source_key(key)
            image = _as_hwc_uint8(obs.get(source, obs["observation.images.head_cam_h"]))
            video[key] = image[None, None, ...]

        state: dict[str, np.ndarray] = {}
        for key in self.model.state_keys:
            state_dim = self.model.state_dims[key]
            vec = self._state_for_key(key, state_dim, kuavo_state, lower_body)
            state[key] = vec[None, None, ...].astype(np.float32)

        language = {self.model.language_key: [[prompt]]}
        return {"video": video, "state": state, "language": language}

    def _inject_action_piece(
        self,
        *,
        key: str,
        vec: np.ndarray,
        slots: dict[str, np.ndarray | None],
    ) -> bool:
        name = _normalize_key(key)
        dim = vec.shape[0]

        packed_arm_dim = (7 + self.eef_dof) * (2 if self.which_arm == "both" else 1)
        packed_dim = packed_arm_dim + self.lower_body_dim
        if dim == packed_dim:
            arm_eef = vec[:packed_arm_dim]
            if self.which_arm == "both":
                slots["left_arm"] = arm_eef[:7]
                slots["left_gripper"] = arm_eef[7:7 + self.eef_dof]
                slots["right_arm"] = arm_eef[7 + self.eef_dof:14 + self.eef_dof]
                slots["right_gripper"] = arm_eef[14 + self.eef_dof:14 + 2 * self.eef_dof]
            else:
                slots[f"{self.which_arm}_arm"] = arm_eef[:7]
                slots[f"{self.which_arm}_gripper"] = arm_eef[7:7 + self.eef_dof]
            if self.lower_body_dim:
                slots["lower_body"] = vec[packed_arm_dim:]
            return True

        if _is_lower_body_key(name):
            slots["lower_body"] = _fit_dim(vec, self.lower_body_dim)
            return True

        if _is_gripper_key(name):
            if _is_left_key(name):
                slots["left_gripper"] = _fit_dim(vec, self.eef_dof)
                return True
            if _is_right_key(name):
                slots["right_gripper"] = _fit_dim(vec, self.eef_dof)
                return True
            if self.which_arm == "left":
                slots["left_gripper"] = _fit_dim(vec, self.eef_dof)
                return True
            if self.which_arm == "right":
                slots["right_gripper"] = _fit_dim(vec, self.eef_dof)
                return True
            pair = _fit_dim(vec, 2 * self.eef_dof)
            slots["left_gripper"] = pair[:self.eef_dof]
            slots["right_gripper"] = pair[self.eef_dof:]
            return True

        if _is_arm_like(name):
            if _is_left_key(name):
                if dim <= 1:
                    return False
                slots["left_arm"] = _fit_dim(vec, 7)
                return True
            if _is_right_key(name):
                if dim <= 1:
                    return False
                slots["right_arm"] = _fit_dim(vec, 7)
                return True
            if "single_arm" in name and self.which_arm == "left":
                slots["left_arm"] = _fit_dim(vec, 7)
                return True
            if "single_arm" in name and self.which_arm == "right":
                slots["right_arm"] = _fit_dim(vec, 7)
                return True
            full_dim = 14 + 2 * self.eef_dof
            single_dim = 7 + self.eef_dof
            if dim >= full_dim:
                base = vec[:full_dim]
                slots["left_arm"] = base[:7]
                slots["left_gripper"] = base[7:7 + self.eef_dof]
                slots["right_arm"] = base[7 + self.eef_dof:14 + self.eef_dof]
                slots["right_gripper"] = base[14 + self.eef_dof:full_dim]
                if self.lower_body_dim and dim >= full_dim + self.lower_body_dim:
                    slots["lower_body"] = vec[full_dim:full_dim + self.lower_body_dim]
                return True
            if dim == single_dim and self.which_arm in ("left", "right"):
                slots[f"{self.which_arm}_arm"] = vec[:7]
                slots[f"{self.which_arm}_gripper"] = vec[7:single_dim]
                return True
            single_whole_dim = single_dim + self.lower_body_dim
            if self.lower_body_dim and dim == single_whole_dim and self.which_arm in ("left", "right"):
                slots[f"{self.which_arm}_arm"] = vec[:7]
                slots[f"{self.which_arm}_gripper"] = vec[7:single_dim]
                slots["lower_body"] = vec[single_dim:]
                return True
            if dim >= 14:
                base = _fit_dim(vec, 14)
                slots["left_arm"] = base[:7]
                slots["right_arm"] = base[7:14]
                return True
        return False

    def _compose_kuavo_action(self, action_step: dict[str, np.ndarray]) -> np.ndarray:
        slots: dict[str, np.ndarray | None] = {
            "left_arm": None,
            "left_gripper": None,
            "right_arm": None,
            "right_gripper": None,
            "lower_body": None,
        }
        unknown: list[np.ndarray] = []

        for key in self.model.action_keys:
            vec = _to_numpy(action_step[key]).astype(np.float32).reshape(-1)
            handled = self._inject_action_piece(key=key, vec=vec, slots=slots)
            if not handled:
                unknown.append(vec)

        if slots["left_arm"] is None and len(unknown) > 0:
            slots["left_arm"] = _fit_dim(unknown.pop(0), 7)
        if slots["right_arm"] is None and len(unknown) > 0:
            slots["right_arm"] = _fit_dim(unknown.pop(0), 7)
        if slots["left_gripper"] is None and len(unknown) > 0:
            slots["left_gripper"] = _fit_dim(unknown.pop(0), self.eef_dof)
        if slots["right_gripper"] is None and len(unknown) > 0:
            slots["right_gripper"] = _fit_dim(unknown.pop(0), self.eef_dof)

        if slots["left_arm"] is None:
            slots["left_arm"] = self._last_state[:7].astype(np.float32)
        if slots["right_arm"] is None:
            slots["right_arm"] = self._last_state[7 + self.eef_dof:14 + self.eef_dof].astype(np.float32)
        if slots["left_gripper"] is None:
            slots["left_gripper"] = self._last_state[7:7 + self.eef_dof].astype(np.float32)
        if slots["right_gripper"] is None:
            slots["right_gripper"] = self._last_state[14 + self.eef_dof:14 + 2 * self.eef_dof].astype(np.float32)
        if slots["lower_body"] is None:
            slots["lower_body"] = self._last_lower_body.astype(np.float32)

        full = np.concatenate(
            [
                slots["left_arm"],
                slots["left_gripper"],
                slots["right_arm"],
                slots["right_gripper"],
            ],
            axis=0,
        ).astype(np.float64)

        if self.which_arm == "left":
            out = full[:7 + self.eef_dof]
        elif self.which_arm == "right":
            out = full[7 + self.eef_dof:]
        else:
            out = full
        return np.concatenate((out, slots["lower_body"]), axis=0)

    def _convert_action_chunk(self, action_dict: dict[str, np.ndarray]) -> list[np.ndarray]:
        if not self.model.action_keys:
            raise ValueError("Isaac-GR00T-N17 returned empty action key list.")

        first_key = self.model.action_keys[0]
        base = _to_numpy(action_dict[first_key])
        if base.ndim != 3 or base.shape[0] != 1:
            raise ValueError(
                f"Expected action[{first_key}] shape [1, T, D], got {base.shape}"
            )
        horizon = int(base.shape[1])
        chunk: list[np.ndarray] = []
        for t in range(horizon):
            step = {
                key: _to_numpy(action_dict[key])[0, t]
                for key in self.model.action_keys
            }
            chunk.append(self._compose_kuavo_action(step))
        return chunk

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if self._pending_actions:
            return self._pending_actions.pop(0)

        model_obs = self._build_model_obs(obs)
        action_dict = self.model.infer(model_obs)
        if not isinstance(action_dict, dict):
            raise ValueError(f"Unexpected Isaac-GR00T-N17 output type: {type(action_dict)}")

        action_chunk = self._convert_action_chunk(action_dict)
        if not action_chunk:
            raise ValueError("Isaac-GR00T-N17 returned empty action chunk.")

        if self.execution_horizon is not None:
            action_chunk = action_chunk[:self.execution_horizon]

        self._pending_actions = [np.asarray(step) for step in action_chunk[1:]]
        return np.asarray(action_chunk[0])

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        model_obs = self._build_model_obs(obs)
        action_dict = self.model.infer(model_obs)
        if not isinstance(action_dict, dict):
            raise ValueError(f"Unexpected Isaac-GR00T-N17 output type: {type(action_dict)}")

        action_chunk = self._convert_action_chunk(action_dict)
        if not action_chunk:
            raise ValueError("Isaac-GR00T-N17 returned empty action chunk.")

        if self.execution_horizon is not None:
            action_chunk = action_chunk[:self.execution_horizon]
        return np.stack([np.asarray(step) for step in action_chunk], axis=0)

    def select_action_chunk_rtc(self, payload):
        from kuavo_server.rtc import flatten_groups, parse_request, reconcile_absolute_prefix, response

        self._pending_actions.clear()
        policy = self.model.policy
        maximum = int(getattr(policy.model.config, "rtc_training_max_delay", 0))
        obs, rtc, original, absolute = parse_request(payload, maximum)
        model_obs = self._build_model_obs(obs)
        if original is not None:
            groups = policy.decode_rtc_prefix(original, context=payload.get("previous_context"))
            native, split = flatten_groups(groups, self.model.action_keys)
            def pack(step):
                return self._compose_kuavo_action(split(step))
            native = reconcile_absolute_prefix(native[0], absolute, pack)
            rtc["absolute_prefix"] = split(native[None])
        groups, info = policy.get_action(model_obs, options=rtc)
        processed = np.stack(self._convert_action_chunk(groups))
        original = info["original_actions"][0]
        if self.execution_horizon is not None:
            original, processed = original[:self.execution_horizon], processed[:self.execution_horizon]
        return response(original, processed, maximum, info["action_context"])
