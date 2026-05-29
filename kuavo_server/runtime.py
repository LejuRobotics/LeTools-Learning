from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Type

if TYPE_CHECKING:
    from .adapters.base import ModelServerAdapter


class TorchSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        import torch

        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        import torch

        buffer = BytesIO(data)
        return torch.load(buffer, weights_only=False)


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """Minimal REP server used by model adapters."""

    def __init__(self, host: str = "*", port: int = 5555, api_token: str | None = None):
        import zmq

        self.running = True
        self.context = zmq.Context()
        self._zmq = zmq
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self) -> dict[str, str]:
        self.running = False
        return {"status": "ok", "message": "server will stop"}

    def _handle_ping(self) -> dict[str, str]:
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True) -> None:
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict[str, Any]) -> bool:
        if self.api_token is None:
            return True
        return request.get("api_token") == self.api_token

    def run(self) -> None:
        addr = self.socket.getsockopt_string(self._zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")

        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)

                if not self._validate_token(request):
                    self.socket.send(TorchSerializer.to_bytes({"error": "Unauthorized: Invalid API token"}))
                    continue

                endpoint = request.get("endpoint", "select_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as exc:
                print(f"Error in server: {exc}")
                self.socket.send(TorchSerializer.to_bytes({"error": str(exc)}))


class ModelInferenceServer(BaseInferenceServer):
    """Standard server wrapper that exposes adapter endpoints."""

    def __init__(
        self,
        adapter: ModelServerAdapter,
        *,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
    ):
        super().__init__(host=host, port=port, api_token=api_token)
        self.adapter = adapter

        self.register_endpoint("metadata", adapter.metadata, requires_input=False)
        self.register_endpoint("reset", adapter.reset, requires_input=False)
        self.register_endpoint("select_action", adapter.select_action, requires_input=True)
        self.register_endpoint("select_action_chunk", adapter.select_action_chunk, requires_input=True)


_ADAPTER_REGISTRY: dict[str, Type[Any]] = {}


def register_adapter(adapter_cls: Type[Any]) -> Type[Any]:
    if not adapter_cls.name:
        raise ValueError("Adapter class must define a non-empty `name`.")
    _ADAPTER_REGISTRY[adapter_cls.name] = adapter_cls
    return adapter_cls


def get_adapter_class(name: str) -> Type[Any]:
    try:
        return _ADAPTER_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTER_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown adapter '{name}'. Available adapters: {known}") from exc


def list_adapters() -> list[str]:
    return sorted(_ADAPTER_REGISTRY)
