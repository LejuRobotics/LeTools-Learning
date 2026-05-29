from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from kuavo_server.builtin_adapters import ensure_adapter_loaded, list_builtin_adapters
    from kuavo_server.runtime import ModelInferenceServer, get_adapter_class
else:
    from .builtin_adapters import ensure_adapter_loaded, list_builtin_adapters
    from .runtime import ModelInferenceServer, get_adapter_class


def build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardized Kuavo model server. Robot-side kuavo_deploy connects through PolicyClient."
    )
    parser.add_argument("--adapter", type=str, required=True, choices=list_builtin_adapters())
    parser.add_argument("--host", type=str, default="*")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api_token", type=str, default="")
    return parser


def build_full_parser() -> argparse.ArgumentParser:
    base = build_base_parser()
    known_args, _ = base.parse_known_args()
    ensure_adapter_loaded(known_args.adapter)
    adapter_cls = get_adapter_class(known_args.adapter)
    adapter_cls.add_cli_args(base)
    return base


def main() -> None:
    parser = build_full_parser()
    args = parser.parse_args()

    ensure_adapter_loaded(args.adapter)
    adapter_cls = get_adapter_class(args.adapter)
    adapter = adapter_cls.from_args(args)

    server = ModelInferenceServer(
        adapter,
        host=args.host,
        port=args.port,
        api_token=args.api_token or None,
    )
    server.run()


if __name__ == "__main__":
    main()
