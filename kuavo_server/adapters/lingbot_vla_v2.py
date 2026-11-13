from __future__ import annotations

import hashlib
import os
import sys
import time
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


def _feature_source_dim(config_path: Path, section: str) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ends = []
    for feature in config.get(section, []):
        for spec in feature.values():
            for source in spec.get("origin_keys", []):
                for source_spec in source.values():
                    ends.append(int(source_spec["end"]))
    if not ends:
        raise ValueError(f"No origin_keys found in {section!r} of {config_path}")
    return max(ends)


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


def _resolve_robot_config_path(robot_config_path: str, eef_dof: int) -> Path:
    if robot_config_path:
        return _resolve_required_path(robot_config_path, "--robot_config_path", directory=False)
    filename = "kuavo_sg100.yaml" if eef_dof == 11 else "kuavo.yaml"
    return Path(__file__).resolve().parents[1] / "configs" / "lingbotvla_v2" / filename


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


def _deploy_config():
    """Load the exact deploy config the client will run, so warmup can mimic it.

    `load_kuavo_config` applies the total-config merge, `${...}` templates and
    `inference_env` overrides, so the returned values are what the client
    actually sends. Returns None if it cannot be read: warming built-in defaults
    would compile the wrong token shapes, which is the very cost this avoids.
    """
    path = os.environ.get("KUAVO_DEPLOY_CONFIG") or kuavo_repo_root() / "configs" / "deploy" / "deploy.yaml"
    try:
        from kuavo_deploy.config import load_kuavo_config  # type: ignore

        return load_kuavo_config(str(path))
    except Exception as exc:  # warmup is best-effort: never block startup
        print(f"[warmup] deploy config unavailable ({exc}); skipping warmup", flush=True)
        return None


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
    # Trained RTC support is a checkpoint property, not a deployment override.
    config["model"]["rtc_training_max_delay"] = 0
    config["train"].pop("rtc_training_max_delay", None)
    checkpoint_config = checkpoint / "config.json"
    if checkpoint_config.is_file():
        checkpoint_metadata = yaml.safe_load(checkpoint_config.read_text(encoding="utf-8")) or {}
        config["model"]["rtc_training_max_delay"] = int(checkpoint_metadata.get("rtc_training_max_delay", 0))

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
        eef_dof: int,
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
        self.eef_dof = eef_dof
        self.execution_horizon = execution_horizon
        self.checkpoint = _resolve_required_path(checkpoint, "--checkpoint", directory=True)
        if not _checkpoint_has_weights(self.checkpoint):
            raise FileNotFoundError(f"No .safetensors weights found under LingBot-VLA v2 checkpoint: {self.checkpoint}")
        self.model_repo_root = _resolve_lingbot_root(model_repo_root)
        self.qwen3vl_path = _resolve_optional_dir(qwen3vl_path, "--qwen3vl_path")
        self.training_config_path = _resolve_training_config_path(self.model_repo_root, training_config_path)
        self.robot_config_path = _resolve_robot_config_path(robot_config_path, eef_dof)
        self.expected_action_dim = _feature_source_dim(self.robot_config_path, "actions")
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
        self._warmup()

    def _warmup(self, max_iters: int = 4) -> None:
        """Pay the JIT/autotune cost of the endpoint that will actually be served.

        `rtc_enabled` decides which single path is warmed. Warming both would
        double the startup cost and leave a graph/allocator high-water mark that
        does not match serving; the plain `select_action_chunk` forward in
        particular degraded steady-state RTC from ~130 ms to ~200 ms.

        Each path still pays two cold costs: the first call compiles the Triton
        MoE and attention kernels, and for RTC the first call with a prefix plus
        nonzero delay compiles the frozen-prefix mask path.

        Shape, prompt and RTC options come from the deploy config so the token
        counts match what the client will send; any difference re-triggers
        compilation on the first real request.
        """
        if os.environ.get("KUAVO_SERVER_WARMUP", "1") == "0":
            return

        config = _deploy_config()
        if config is None:
            return
        env, inf = config.env, config.inference
        width, height = (int(value) for value in env.image_size)
        shape = (height, width, 3)  # deploy.yaml is [width, height]; the model takes HWC
        prompt = os.environ.get("KUAVO_SERVER_WARMUP_PROMPT") or str(inf.task_prompt or "")
        obs = {
            "observation.images.head_cam_h": np.zeros(shape, dtype=np.uint8),
            "observation.state": np.zeros(
                _feature_source_dim(self.robot_config_path, "states"), dtype=np.float32
            ),
            "prompt": prompt,
        }
        self._warmup_signature = (shape, len(prompt))

        rtc = None
        delay = 0
        if inf.rtc_enabled:
            # A trained checkpoint is required for trained mode; otherwise the
            # RTC endpoint would reject the request, so fall back to guided.
            max_delay = int(getattr(self.model.config, "rtc_training_max_delay", 0))
            mode = "guided" if inf.rtc_mode == "trained" and max_delay <= 0 else inf.rtc_mode
            horizon = max(1, int(inf.rtc_execution_horizon))
            # Mirror the client's clamp so the prefixed request is accepted.
            delay = min(max(1, int(inf.rtc_initial_delay_steps)), horizon, max_delay or horizon)
            rtc = {
                "mode": mode,
                "inference_delay": 0,
                "execution_horizon": horizon,
                "prefix_attention_schedule": inf.rtc_prefix_attention_schedule,
                "max_guidance_weight": float(inf.rtc_max_guidance_weight),
            }
        # Only the endpoint the client will actually call is warmed: warming both
        # leaves a graph/allocator high-water mark that does not match serving
        # and degraded steady-state RTC from ~130 ms to ~200 ms.
        detail = f"mode={rtc['mode']} horizon={rtc['execution_horizon']} delay={delay}" if rtc else "mode=chunk"
        print(f"[warmup] {self.name} shape={shape} prompt_len={len(prompt)} {detail}", flush=True)
        try:
            for index in range(max_iters):
                started = time.monotonic()
                if rtc is None:
                    self.select_action_chunk(obs)
                else:
                    # No outer inference_mode: the guided branch builds an
                    # autograd graph and inference tensors cannot be saved for backward.
                    first = self.select_action_chunk_rtc({
                        "observation": obs, "previous_original": None,
                        "previous_absolute": None, "previous_context": None,
                        "rtc": dict(rtc),
                    })
                    self.select_action_chunk_rtc({
                        "observation": obs,
                        "previous_original": first["original_actions"],
                        "previous_absolute": first["processed_actions"],
                        "previous_context": first["action_context"],
                        "rtc": {**rtc, "inference_delay": delay},
                    })
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                print(f"[warmup] {self.name} {index + 1}/{max_iters} {elapsed * 1000:.0f} ms", flush=True)
                if elapsed < 1.0:
                    break
        except Exception as exc:  # never block startup on a warmup failure
            print(f"[warmup] {self.name} aborted: {exc}", flush=True)
        finally:
            self.reset()

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
        parser.add_argument("--eef_dof", type=int, default=1, choices=[1, 11])
        parser.add_argument("--execution_horizon", type=int, default=25, help="-1 returns the model's full chunk")
        parser.add_argument("--use_compile", action="store_true")
        parser.add_argument("--use_fp32", action="store_true", help="Use fp32 inference instead of bf16")

    @classmethod
    def from_args(cls, args: Namespace) -> "LingBotVlaV2Adapter":
        return cls(
            checkpoint=args.checkpoint,
            model_repo_root=args.model_repo_root,
            which_arm=args.which_arm,
            eef_dof=args.eef_dof,
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
            "eef_dof": self.eef_dof,
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
        self.model._rtc_previous_applied = None
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
        full_dim = 14 + 2 * self.eef_dof
        single_dim = 7 + self.eef_dof
        if action_np.shape[0] == full_dim:
            if self.which_arm == "both":
                return action_np
            if self.which_arm == "left":
                return action_np[:single_dim]
            if self.which_arm == "right":
                return action_np[single_dim:full_dim]
        if action_np.shape[0] == self.expected_action_dim:
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
            if arm.shape[0] != effector.shape[0] or arm.shape[-1] != 14 or effector.shape[-1] != 2 * self.eef_dof:
                raise ValueError(
                    "Unexpected split LingBot-VLA v2 action shapes: "
                    f"arm={arm.shape}, effector={effector.shape}"
                )
            return np.concatenate([arm[:, :7], effector[:, :self.eef_dof], arm[:, 7:14], effector[:, self.eef_dof:2 * self.eef_dof]], axis=-1)
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
        self._warn_on_warmup_mismatch(obs)
        return self._infer_chunk(obs)

    def _warn_on_warmup_mismatch(self, obs: dict[str, Any]) -> None:
        """Report once if the live request no longer matches what warmup compiled.

        A different image size or prompt length changes the vision/language
        token counts, so `torch.compile` re-traces and the first real request
        pays the cold cost again. Surfacing it avoids a mystery slowdown.
        """
        if getattr(self, "_warmup_mismatch_reported", False) or not hasattr(self, "_warmup_signature"):
            return
        live = (tuple(_to_numpy(obs["observation.images.head_cam_h"]).shape), len(str(obs.get("prompt", ""))))
        if live == self._warmup_signature:
            return
        self._warmup_mismatch_reported = True
        print(
            f"[warmup] {self.name} request differs from warmup: "
            f"image {self._warmup_signature[0]} -> {live[0]}, "
            f"prompt_len {self._warmup_signature[1]} -> {live[1]}; recompiling",
            flush=True,
        )

    def select_action_chunk_rtc(self, payload):
        from kuavo_server.rtc import flatten_groups, parse_request, reconcile_absolute_prefix, response

        self._pending_actions.clear()
        t0 = time.perf_counter()
        maximum = int(getattr(self.model.config, "rtc_training_max_delay", 0))
        obs, rtc, original, absolute = parse_request(payload, maximum)
        self._warn_on_warmup_mismatch(obs)
        prefix = None
        print(f"[RTC-TIMING] has_prefix={original is not None} mode={rtc.get('mode')} delay={rtc.get('inference_delay')}", flush=True)
        if original is not None:
            t = time.perf_counter()
            groups = self.model.decode_rtc_prefix(original, context=payload.get("previous_context"))
            print(f"[RTC-TIMING] decode_rtc_prefix {time.perf_counter() - t:.3f}s", flush=True)
            native, split = flatten_groups(groups, list(self.model.action_key))
            def pack(step):
                # The existing adapter composer accepts [T,D], whereas the
                # model decoder returns [B,T,D]. Remove only the batch axis.
                return self._convert_action(self._compose_action_from_dict(split(step[None]))[0])
            t = time.perf_counter()
            native = reconcile_absolute_prefix(native[0], absolute, pack)
            prefix = split(native[None])
            print(f"[RTC-TIMING] reconcile_prefix {time.perf_counter() - t:.3f}s", flush=True)
        t = time.perf_counter()
        groups, original, context = self.model.infer_rtc(self._build_model_obs(obs), rtc_options=rtc, absolute_prefix=prefix)
        print(f"[RTC-TIMING] infer_rtc {time.perf_counter() - t:.3f}s", flush=True)
        unbatched = {key: np.asarray(value)[0] for key, value in groups.items()}
        processed = self._convert_action_chunk(self._compose_action_from_dict(unbatched))
        limit = self.execution_horizon if self.execution_horizon > 0 else len(processed)
        print(f"[RTC-TIMING] total {time.perf_counter() - t0:.3f}s", flush=True)
        return response(original[:limit], processed[:limit], maximum, context)
