from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_repack_transforms: Sequence[_transforms.DataTransformFn] = (),
        rtc_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rtc_repack = _transforms.compose(rtc_repack_transforms)
        self._rtc_input_transform = _transforms.compose(rtc_transforms)
        self._rtc_last_state = None
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions,
                **({"static_argnames": ("rtc_mode", "rtc_schedule", "rtc_max_guidance_weight")}
                   if hasattr(model, "rtc_training_max_delay") else {}),
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, rtc_options=None, rtc_prefix_absolute=None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc_prefix = None
        if rtc_options is not None and not hasattr(self._model, "rtc_training_max_delay") and not hasattr(getattr(self._model, "config", None), "rtc_training_max_delay"):
            raise ValueError("OpenPI RTC requires a Pi0/Pi0.5 flow-matching model")
        if rtc_options is not None and rtc_prefix_absolute is not None:
            inputs = self._rtc_repack(inputs)
            inputs["actions"] = np.asarray(rtc_prefix_absolute).copy()
            # Same physical->relative->normalized->padded pipeline as training,
            # anchored at THIS observation, not at the previous inference.
            inputs = self._rtc_input_transform(inputs)
            rtc_prefix = inputs.pop("actions")
        else:
            inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_fn = self._sample_actions
        if rtc_options is not None:
            maximum = self.rtc_training_max_delay
            if rtc_options["mode"] == "trained" and (maximum <= 0 or rtc_options["inference_delay"] > maximum):
                raise ValueError(f"RTC checkpoint supports maximum trained delay {maximum}")
            if self._is_pytorch_model:
                sample_fn = getattr(sample_fn, "_torchdynamo_orig_callable", sample_fn)
                rtc_prefix = None if rtc_prefix is None else torch.as_tensor(rtc_prefix, device=self._pytorch_device)[None]
            else:
                rtc_prefix = None if rtc_prefix is None else jnp.asarray(rtc_prefix)[None]
            sample_kwargs.update(
                rtc_prefix=rtc_prefix, rtc_delay=rtc_options["inference_delay"],
                rtc_horizon=rtc_options["execution_horizon"], rtc_mode=rtc_options["mode"],
                rtc_schedule=rtc_options["prefix_attention_schedule"],
                rtc_max_guidance_weight=rtc_options["max_guidance_weight"],
            )
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": sample_fn(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        original = np.asarray(outputs["actions"]).copy()
        if rtc_options is not None:
            self._rtc_last_state = np.asarray(outputs["state"]).copy()
        outputs = self._output_transform(outputs)
        if rtc_options is not None:
            outputs["original_actions"] = original
            outputs["action_context"] = {"state": self._rtc_last_state.copy()}
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def rtc_training_max_delay(self):
        return int(getattr(self._model, "rtc_training_max_delay", getattr(getattr(self._model, "config", None), "rtc_training_max_delay", 0)))

    def decode_rtc_prefix(self, original, context=None):
        state = self._rtc_last_state if context is None else np.asarray(context["state"])
        if state is None:
            raise ValueError("RTC prefix has no previous observation anchor")
        return self._output_transform({"actions": np.asarray(original).copy(), "state": state.copy()})["actions"]

    def reset_rtc(self):
        self._rtc_last_state = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
