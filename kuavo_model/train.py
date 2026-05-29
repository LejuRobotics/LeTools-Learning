from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_ROOT = REPO_ROOT / "configs" / "train" / "lerobot"
DEFAULT_ACCELERATE_CONFIG = REPO_ROOT / "configs" / "accelerate" / "accelerate_config.yaml"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"

POLICY_CHOICES = (
    "act",
    "diffusion",
    "pi0",
    "pi0_fast",
    "pi05",
    "gr00t",
    "smolvla",
    "xvla",
    "wall_x",
    "multi_task_dit",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected mapping config in {path}, got {type(cfg)}")
    return cfg


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    return value


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _deep_merge(*configs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for cfg in configs:
        merged = _deep_merge_dicts(merged, cfg)
    return merged


def _resolve_config_paths(policy: str, mode: str, config_root: Path) -> tuple[Path, Path | None]:
    total_path = config_root / "total" / f"{policy}_total.yaml"
    simple_path = config_root / f"{policy}.yaml"

    if mode == "total":
        return total_path, None
    return total_path, simple_path


def _convert_to_lerobot_train_config(merged_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = _to_builtin(merged_cfg)
    if not isinstance(merged, dict):
        raise TypeError(f"Expected merged config to be dict, got {type(merged)}")

    output: dict[str, Any] = {}

    training = merged.pop("training", {})
    if not isinstance(training, dict):
        raise TypeError(f"Expected 'training' section to be dict, got {type(training)}")
    output.update(training)

    for section in ("dataset", "eval", "wandb", "peft", "policy", "env"):
        if section in merged:
            output[section] = merged.pop(section)

    if merged:
        # Keep unknown top-level keys instead of dropping them silently.
        output.update(merged)

    return output


def _write_temp_json(config: dict[str, Any]) -> Path:
    fd, path = tempfile.mkstemp(prefix="kuavo_lerobot_train_", suffix=".json")
    os.close(fd)
    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path


def _resolve_output_dir(config: dict[str, Any]) -> Path | None:
    output_dir = config.get("output_dir")
    if output_dir in (None, ""):
        return None
    path = Path(output_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _append_timestamp_suffix(resolved_config: dict[str, Any], stamp: str) -> None:
    output_dir = resolved_config.get("output_dir")
    if output_dir not in (None, ""):
        output_path = Path(output_dir)
        resolved_config["output_dir"] = str(output_path.parent / f"{output_path.name}_{stamp}")

    job_name = resolved_config.get("job_name")
    if isinstance(job_name, str) and job_name.strip():
        resolved_config["job_name"] = f"{job_name}_{stamp}"


def _write_resolved_config(output_dir: Path, config: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_train_config.json"
    with resolved_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return resolved_path


def _write_resolved_config_when_available(
    output_dir: Path | None,
    config: dict[str, Any],
    process: subprocess.Popen,
    *,
    poll_interval_s: float = 0.2,
    timeout_s: float = 30.0,
) -> Path | None:
    if output_dir is None:
        return None

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if output_dir.exists():
            return _write_resolved_config(output_dir, config)
        if process.poll() is not None:
            break
        time.sleep(poll_interval_s)

    if output_dir.exists():
        return _write_resolved_config(output_dir, config)
    return None


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_resume_config_path(output_dir: Path) -> Path:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Resume requested, but checkpoints directory not found: {checkpoints_dir}")

    last_dir = checkpoints_dir / "last"
    candidate_dirs: list[Path] = []
    if last_dir.exists():
        candidate_dirs.append(last_dir.resolve())

    numeric_dirs = sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
        reverse=True,
    )
    candidate_dirs.extend(numeric_dirs)

    seen: set[Path] = set()
    for checkpoint_dir in candidate_dirs:
        checkpoint_dir = checkpoint_dir.resolve()
        if checkpoint_dir in seen:
            continue
        seen.add(checkpoint_dir)
        config_path = checkpoint_dir / "pretrained_model" / "train_config.json"
        if config_path.is_file():
            return config_path

    raise FileNotFoundError(
        f"Resume requested, but no checkpoint train_config.json found under {checkpoints_dir}"
    )


def _build_lerobot_train_command(config_path: Path, passthrough_args: list[str]) -> list[str]:
    return ["-m", "lerobot.scripts.lerobot_train", f"--config_path={config_path}", *passthrough_args]


def _build_command(
    config_path: Path,
    passthrough_args: list[str],
    launcher: str,
    accelerate_config: Path | None,
) -> list[str]:
    lerobot_cmd = _build_lerobot_train_command(config_path, passthrough_args)
    if launcher == "python":
        return [sys.executable, *lerobot_cmd]

    if launcher == "accelerate":
        if accelerate_config is None:
            raise ValueError("launcher=accelerate requires an accelerate config path")
        return [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(accelerate_config),
            *lerobot_cmd,
        ]

    raise ValueError(f"Unsupported launcher: {launcher}")


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    if current:
        env["PYTHONPATH"] = f"{LEROBOT_SRC}{os.pathsep}{current}"
    else:
        env["PYTHONPATH"] = str(LEROBOT_SRC)
    return env


def _print_summary(
    policy: str,
    mode: str,
    total_path: Path,
    simple_path: Path | None,
    config_path: Path,
    output_dir: Path | None,
    command: list[str],
    launcher: str,
    accelerate_config: Path | None = None,
    resume_from: Path | None = None,
) -> None:
    print(f"Policy: {policy}")
    print(f"Mode: {mode}")
    print(f"Launcher: {launcher}")
    print(f"Total config: {total_path}")
    if simple_path is not None:
        print(f"Simple config: {simple_path}")
    print(f"Resolved config: {config_path}")
    if output_dir is not None:
        print(f"Training output dir: {output_dir}")
    if accelerate_config is not None:
        print(f"Accelerate config: {accelerate_config}")
    if resume_from is not None:
        print(f"Resume from: {resume_from}")
    print("Command:")
    print(" ".join(command))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Thin Kuavo wrapper around upstream LeRobot training."
    )
    parser.add_argument(
        "--policy",
        required=True,
        choices=POLICY_CHOICES,
        help="Policy config family to load.",
    )
    parser.add_argument(
        "--mode",
        default="simple",
        choices=("simple", "total"),
        help="simple: total + simple override; total: total only.",
    )
    parser.add_argument(
        "--config-root",
        default=str(DEFAULT_CONFIG_ROOT),
        help="Root directory that contains lerobot policy YAMLs.",
    )
    parser.add_argument(
        "--launcher",
        default="python",
        choices=("python", "accelerate"),
        help="python: single-process launch; accelerate: single/multi-GPU via accelerate launch.",
    )
    parser.add_argument(
        "--accelerate-config",
        default=str(DEFAULT_ACCELERATE_CONFIG),
        help="Accelerate config YAML used when --launcher accelerate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command and exit without starting training.",
    )
    parser.add_argument(
        "--keep-config",
        action="store_true",
        help="Keep the resolved temporary JSON config on disk after launch.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable automatic timestamp suffix for output_dir and job_name.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> int:
    args, passthrough_args = parse_args()

    config_root = Path(args.config_root).resolve()
    accelerate_config = Path(args.accelerate_config).resolve() if args.launcher == "accelerate" else None
    total_path, simple_path = _resolve_config_paths(args.policy, args.mode, config_root)

    configs = [_load_yaml(total_path)]
    if simple_path is not None:
        configs.append(_load_yaml(simple_path))

    merged_cfg = _deep_merge(*configs)
    resolved = _convert_to_lerobot_train_config(merged_cfg)
    if not args.no_timestamp and not _is_truthy(resolved.get("resume")):
        _append_timestamp_suffix(resolved, datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = _resolve_output_dir(resolved)
    resume_from: Path | None = None

    if _is_truthy(resolved.get("resume")):
        if output_dir is None:
            raise ValueError("resume=true requires training.output_dir to be set")
        resume_from = _resolve_resume_config_path(output_dir)
        config_path = resume_from
    else:
        config_path = _write_temp_json(resolved)

    # If  train_config.json keeps resume=false in Checkpoint, --resume=true sets cfg.resume so that validate + state could reload and run.
    passthrough = list(passthrough_args)
    if resume_from is not None:
        passthrough.append("--resume=true")

    command = _build_command(config_path, passthrough, args.launcher, accelerate_config)
    _print_summary(
        args.policy,
        args.mode,
        total_path,
        simple_path,
        config_path,
        output_dir,
        command,
        args.launcher,
        accelerate_config=accelerate_config,
        resume_from=resume_from,
    )

    if args.dry_run:
        if output_dir is not None and output_dir.exists():
            resolved_path = _write_resolved_config(output_dir, resolved)
            print(f"Saved resolved config: {resolved_path}")
        if resume_from is None and not args.keep_config:
            config_path.unlink(missing_ok=True)
        return 0

    env = _build_env()
    try:
        process = subprocess.Popen(command, env=env, cwd=str(REPO_ROOT))
        resolved_path = _write_resolved_config_when_available(output_dir, resolved, process)
        if resolved_path is not None:
            print(f"Saved resolved config: {resolved_path}")
        return process.wait()
    finally:
        if resume_from is None and not args.keep_config:
            config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
