"""Standardized model-server runtime for Kuavo deployment."""

from .runtime import get_adapter_class, list_adapters, register_adapter

__all__ = ["get_adapter_class", "list_adapters", "register_adapter"]
