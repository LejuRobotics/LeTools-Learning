# Copyright (C) 2025-2026 LejuRobotics.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ---
#
# This project includes code from LeRobot (https://github.com/huggingface/lerobot),
# which is licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict

import torch
import zmq
import numpy as np


class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555, api_token: str = None):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        TorchSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
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
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(TorchSerializer.to_bytes({"error": str(e)}))


class BaseInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def reset_server(self):
        """
        Reset server-side adapter state if supported.
        """
        return self.call_endpoint("reset", requires_input=False)

    def call_endpoint(
        self, endpoint, data = None, requires_input = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        
        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        response = TorchSerializer.from_bytes(message)
        
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def select_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        response = self.call_endpoint("select_action", observations)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"{response['error']} | request_keys={list(observations.keys())}")
        return response

    def select_action_chunk(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get an action chunk from the server.
        """
        response = self.call_endpoint("select_action_chunk", observations)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"{response['error']} | request_keys={list(observations.keys())}")
        return response
        

# policy client
class PolicyClient:
    def __init__(self, host="localhost", port=5555, task_prompt="robot manipulation", api_token=None):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.task_prompt = str(task_prompt).strip()

    def eval(self):
        return self

    def to(self, _device):
        return self

    def reset(self):
        try:
            response = self.policy.reset_server()
            if isinstance(response, dict) and response.get("error"):
                return self
        except Exception:
            # Keep compatibility with older servers that do not expose a reset endpoint.
            return self
        return self
    
    @staticmethod
    def _to_action_tensor(action: Any) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            tensor = action
        elif isinstance(action, np.ndarray):
            tensor = torch.from_numpy(action)
        else:
            tensor = torch.as_tensor(action)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _to_action_chunk_tensor(actions: Any) -> torch.Tensor:
        tensor = PolicyClient._to_action_tensor(actions)
        if tensor.ndim != 2:
            raise ValueError(f"Expected action chunk with shape [T, D], got {tuple(tensor.shape)}")
        return tensor

    def _prepare_obs(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.task_prompt or obs_dict.get("prompt"):
            return obs_dict
        payload = dict(obs_dict)
        payload["prompt"] = self.task_prompt
        return payload

    def select_action(self, obs_dict):
        return self._to_action_tensor(self.policy.select_action(self._prepare_obs(obs_dict)))

    def select_action_chunk(self, obs_dict):
        try:
            actions = self.policy.select_action_chunk(self._prepare_obs(obs_dict))
        except RuntimeError as exc:
            if "Unknown endpoint: select_action_chunk" not in str(exc):
                raise
            actions = self.policy.select_action(self._prepare_obs(obs_dict))
        return self._to_action_chunk_tensor(actions)

    def predict_action_chunk(self, obs_dict):
        return self.select_action_chunk(obs_dict)


# # convert hardware observations to policy's observation dict
# def hardware_obses_to_policy_obs_dict(robot_qpos, head_cam_h, wrist_cam_l, wrist_cam_r):
#     obs_dict = {
#             "video.head_cam_h": head_cam_h.reshape(1, 256, 256, 3),
#             "video.wrist_cam_l": wrist_cam_l.reshape(1, 256, 256, 3),
#             "video.wrist_cam_r": wrist_cam_r.reshape(1, 256, 256, 3),
#             "state.state": robot_qpos.reshape(1, -1).astype(np.float64),
#             "annotation.human.action.task_description": ["DEBUG"],
#         }
#     return obs_dict
