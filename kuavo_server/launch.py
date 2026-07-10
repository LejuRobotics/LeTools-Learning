#!/usr/bin/env python3
"""Unified launcher for kuavo_server.

Reads kuavo_server/configs/launcher.yaml and dispatches to the right
environment (uv project / conda env / current shell) plus the right
serve.py command line for the chosen adapter.

Usage:
    python kuavo_server/launch.py <adapter>
    python kuavo_server/launch.py <adapter> --config /path/to/other.yaml
    python kuavo_server/launch.py <adapter> --dry-run
    python kuavo_server/launch.py --list

Extra CLI flags after the adapter name are appended verbatim to serve.py,
so you can override any single value without editing the YAML, e.g.:

    python kuavo_server/launch.py openpi --checkpoint /tmp/ckpt --port 6000
    python kuavo_server/launch.py groot --execution_horizon 8
    python kuavo_server/launch.py lingbotvla2 --checkpoint /tmp/v2/hf_ckpt/model
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "server" / "launcher.yaml"
SERVE_SCRIPT = REPO_ROOT / "kuavo_server" / "serve.py"

# args that belong to serve.py's base parser (host/port/api_token), not to an adapter
COMMON_KEYS = {"host", "port", "api_token"}


def expand(value: Any) -> Any:
    """Expand ${REPO_ROOT} and env vars inside string values."""
    if isinstance(value, str):
        v = value.replace("${REPO_ROOT}", str(REPO_ROOT))
        return os.path.expandvars(os.path.expanduser(v))
    return value


def args_to_cli(args: dict[str, Any]) -> list[str]:
    """Turn an args dict into ['--key', 'value', ...]. Drops empty/None/False."""
    out: list[str] = []
    for key, raw in args.items():
        val = expand(raw)
        if val is None or val == "":
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                out.append(flag)
            # False -> skip
            continue
        out.extend([flag, str(val)])
    return out


def build_serve_argv(short_name: str, cfg: dict[str, Any], extra: list[str]) -> list[str]:
    common = dict(cfg.get("common") or {})
    block = (cfg.get("adapters") or {}).get(short_name)
    if not block:
        sys.exit(f"[launch] adapter '{short_name}' not found in config")
    real_adapter = block.get("adapter") or short_name
    args = dict(block.get("args") or {})

    # common defaults, but adapter args win
    merged = {**common, **args}

    argv = [sys.executable, str(SERVE_SCRIPT), "--adapter", real_adapter]
    argv += args_to_cli(merged)
    argv += extra  # user overrides win because argparse keeps the last value
    return argv


def wrap_with_env(cmd: list[str], env_block: dict[str, Any]) -> list[str]:
    mode = (env_block or {}).get("mode", "none")

    if mode == "none":
        return cmd

    if mode == "uv":
        project = expand(env_block.get("project") or "")
        if not project:
            sys.exit("[launch] env.mode=uv requires env.project")
        # uv run --project <project> python serve.py ...
        # cmd[0] is the python we don't need — uv picks the project's python.
        return ["uv", "run", "--project", project, "python", *cmd[1:]]

    if mode == "conda":
        env_name = env_block.get("conda_env") or ""
        if not env_name:
            sys.exit("[launch] env.mode=conda requires env.conda_env")
        # conda run -n <env> --no-capture-output python serve.py ...
        # Drop cmd[0] (parent's sys.executable) so conda picks the env's own python.
        return ["conda", "run", "-n", env_name, "--no-capture-output", "python", *cmd[1:]]

    sys.exit(f"[launch] unknown env.mode '{mode}' (expected uv|conda|none)")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"[launch] config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified launcher for kuavo_server adapters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Extra args after <adapter> are forwarded verbatim to serve.py.\n"
            "Example: launch.py openpi --port 6000 --checkpoint /tmp/ckpt"
        ),
    )
    ap.add_argument("adapter", nargs="?", help="adapter name (openpi / isaac_gr00t_n17 / lingbot_vla / lingbotvla2 / ...)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"YAML config (default: {DEFAULT_CONFIG})")
    ap.add_argument("--list", action="store_true", help="list adapters defined in the config and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved command without executing")

    # Split known args from forwarded args. argparse can't easily do "consume until --",
    # so we hand-split on the first unknown flag.
    argv = sys.argv[1:]
    own: list[str] = []
    extra: list[str] = []
    seen_adapter = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--config", "--list", "--dry-run") or (not seen_adapter and not tok.startswith("-")):
            own.append(tok)
            if tok == "--config" and i + 1 < len(argv):
                own.append(argv[i + 1])
                i += 2
                continue
            if not tok.startswith("-"):
                seen_adapter = True
            i += 1
            continue
        # everything else is forwarded
        extra.append(tok)
        i += 1

    ns = ap.parse_args(own)
    cfg = load_config(ns.config)

    if ns.list:
        adapters = sorted((cfg.get("adapters") or {}).keys())
        print("Adapters defined in", ns.config)
        for name in adapters:
            block = cfg["adapters"][name]
            env = (block.get("env") or {}).get("mode", "none")
            real = block.get("adapter") or name
            suffix = "" if real == name else f"  -> {real}"
            print(f"  - {name:<14} (env: {env}){suffix}")
        return

    if not ns.adapter:
        ap.error("adapter is required (or pass --list)")

    block = (cfg.get("adapters") or {}).get(ns.adapter)
    if block is None:
        adapters = sorted((cfg.get("adapters") or {}).keys())
        sys.exit(f"[launch] adapter '{ns.adapter}' not in config. Known: {adapters}")

    serve_cmd = build_serve_argv(ns.adapter, cfg, extra)
    full_cmd = wrap_with_env(serve_cmd, block.get("env") or {})

    print("[launch] cwd     :", REPO_ROOT)
    print("[launch] command :", " ".join(shlex.quote(c) for c in full_cmd))
    if ns.dry_run:
        return

    os.chdir(REPO_ROOT)
    try:
        os.execvp(full_cmd[0], full_cmd)
    except FileNotFoundError:
        sys.exit(f"[launch] executable not found: {full_cmd[0]} (is uv/conda installed and on PATH?)")


if __name__ == "__main__":
    main()
