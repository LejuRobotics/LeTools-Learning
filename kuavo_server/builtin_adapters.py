"""Built-in adapter registry metadata with lazy imports."""

from __future__ import annotations

import importlib

ADAPTER_MODULES: dict[str, str] = {
    "lingbot_vla": "kuavo_server.adapters.lingbot_vla",
    "openpi": "kuavo_server.adapters.openpi",
    "wall_x": "kuavo_server.adapters.wall_x",
    "isaac_gr00t": "kuavo_server.adapters.isaac_gr00t",
    "isaac_gr00t_n17": "kuavo_server.adapters.isaac_gr00t_n17",
}


def list_builtin_adapters() -> list[str]:
    return sorted(ADAPTER_MODULES)


def ensure_adapter_loaded(name: str) -> None:
    try:
        module_name = ADAPTER_MODULES[name]
    except KeyError as exc:
        known = ", ".join(sorted(ADAPTER_MODULES)) or "<none>"
        raise KeyError(f"Unknown adapter '{name}'. Available adapters: {known}") from exc

    importlib.import_module(module_name)


__all__ = ["ADAPTER_MODULES", "list_builtin_adapters", "ensure_adapter_loaded"]
