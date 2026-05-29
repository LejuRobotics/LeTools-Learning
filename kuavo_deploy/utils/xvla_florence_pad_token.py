"""Ensure Florence2 config dict carries top-level ``pad_token_id`` (Kuavo-local fix).

Upstream Florence-2+VLM stacks expect ``config.pad_token_id`` while ``florence_config``
YAML often only nests it under ``text_config``. Lerobot submodule previously patched
``Florence2Config``; here we inject the same field into ``config_dict`` before
``Florence2Config(**config_dict)`` so the submodule can stay vanilla.

Called from ``policy_loader.load_native_policy_bundle`` only when ``policy_cfg.type == "xvla"``
so non-XVLA policies do not monkey-patch LeRobot globally.
"""

from __future__ import annotations

from typing import Any


_PATCH_INSTALLED = False


def inject_pad_token_id_into_florence_config_dict(config_dict: dict[str, Any]) -> None:
    """Mutate ``config_dict`` like the old ``Florence2Config`` block in the submodule."""
    if config_dict.get("pad_token_id") is not None:
        return
    tc = config_dict.get("text_config")
    if isinstance(tc, dict):
        pad = tc.get("pad_token_id")
        config_dict["pad_token_id"] = 1 if pad is None else pad
    else:
        config_dict["pad_token_id"] = 1


def install_xvla_florence_pad_token_dict_patch() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    from lerobot.policies.xvla.configuration_florence2 import Florence2Config
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    if getattr(XVLAConfig.get_florence_config, "_kuavo_pad_token_patch", False):
        _PATCH_INSTALLED = True
        return

    _orig = XVLAConfig.get_florence_config

    def get_florence_config_patched(self):
        if self._florence_config_obj is None:
            config_dict = dict(self.florence_config)
            if "vision_config" not in config_dict or config_dict["vision_config"] is None:
                raise ValueError("vision_config is required")
            if "text_config" not in config_dict or config_dict["text_config"] is None:
                raise ValueError("text_config is required")
            inject_pad_token_id_into_florence_config_dict(config_dict)
            self._florence_config_obj = Florence2Config(**config_dict)
        return self._florence_config_obj

    get_florence_config_patched._kuavo_pad_token_patch = True  # type: ignore[attr-defined]
    XVLAConfig.get_florence_config = get_florence_config_patched
    _PATCH_INSTALLED = True
