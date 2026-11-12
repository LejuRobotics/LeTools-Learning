"""Action chunk prefetching inspired by LeRobot's async robot client.

Only the control thread touches the environment. A single worker owns the
policy, processors and remote RPCs. Actions are aligned by control timestep;
waiting for an empty queue does not advance the robot's action timeline.
"""

from collections import deque
from copy import deepcopy
import logging
import math
import threading
import time

import torch

logger = logging.getLogger("model")
NEW_ACTION_WEIGHTS = {
    "latest_only": 1.0,
    "weighted_average": 0.7,
    "average": 0.5,
    "conservative": 0.3,
}


def _overlap_weights(count, mode):
    """New-action weights over the remaining, shared control steps.

    Curve modes gradually hand over from old to new. Cosine uses the deleted
    RTC-Lite ramp's sampling: phase=(index+1)/count, ending at a full handover.
    Exponential decay is normalized so the old weight also reaches zero.
    """
    if count <= 0:
        return []
    if mode in NEW_ACTION_WEIGHTS:
        return [NEW_ACTION_WEIGHTS[mode]] * count
    if mode not in ("exp", "cosine"):
        raise ValueError(f"Unknown aggregate mode: {mode}")
    if count == 1:
        return [1.0]
    phases = [(index + 1) / count for index in range(count)]
    if mode == "cosine":
        return [0.5 - 0.5 * math.cos(math.pi * phase) for phase in phases]
    return [-math.expm1(-phase) / -math.expm1(-1.0) for phase in phases]


def _make_remote_policy(config):
    from kuavo_deploy.kuavo_service.client import PolicyClient

    return PolicyClient(task_prompt=config.task_prompt)


class AsyncInference:
    """One episode of asynchronous inference, used as a context manager.

    Inputs are raw environment observations. Outputs have already passed the
    policy postprocessor and are CPU tensors with shape [1, action_dim].
    """

    def __init__(self, policy, preprocessor, postprocessor, config, *, policy_factory=None, record_chunks=False, control_hz=None, policy_ready=None):
        config.validate()
        if policy is None and policy_factory is None and config.policy_type == "client":
            policy_factory = lambda: _make_remote_policy(config)
        if policy is not None and policy_factory is not None:
            raise ValueError("Provide either a policy or a policy_factory")
        if policy_factory is None and not callable(getattr(policy, "predict_action_chunk", None)):
            raise ValueError("Async inference requires policy.predict_action_chunk()")
        self.policy = policy
        self._policy_factory = policy_factory
        self._policy_ready = policy_ready
        self._record_chunks = record_chunks
        self._chunk_history = []
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.config = config
        self.control_hz = control_hz
        if config.rtc_enabled and (control_hz is None or not math.isfinite(control_hz) or control_hz <= 0):
            raise ValueError("RTC requires a positive control_hz (the model action frequency)")
        self._original_actions = {}
        self._action_context = None
        self._rtc_latencies = deque(maxlen=30)
        self._rtc_training_max_delay = None
        self._rtc_discards = 0
        policy_config = getattr(policy, "config", None)
        self.chunk_limit = config.actions_per_chunk or getattr(policy_config, "n_action_steps", None)
        self._history = deque(maxlen=getattr(policy_config, "n_obs_steps", 1))
        self._condition = threading.Condition()
        self._actions = {}
        self._next_step = 0
        self._generation = 0
        self._chunk_size = 0
        self._request = None
        self._busy = False
        self._busy_since = 0.0
        self._error = None
        self._closed = False
        self._action_dim = None
        self._original_dim = None
        self._thread = threading.Thread(target=self._worker, name="kuavo-inference", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def queue_size(self):
        with self._condition:
            return len(self._actions)

    @property
    def chunk_history(self):
        with self._condition:
            return [dict(record) for record in self._chunk_history]

    def invalidate(self, *, preserve_error=False):
        """Discard queued and in-flight results after a pause."""
        with self._condition:
            if preserve_error and self._error is not None:
                raise RuntimeError("Async inference failed") from self._error
            self._generation += 1
            self._actions.clear()
            self._original_actions.clear()
            self._action_context = None
            self._rtc_latencies.clear()
            self._rtc_discards = 0
            self._history.clear()
            self._chunk_size = 0
            self._error = None
            self._condition.notify_all()

    def _submit(self):
        # Called with the condition held; keep at most one RPC in flight.
        if self._busy:
            return
        if self._actions and len(self._actions) > self.config.chunk_size_threshold * self._chunk_size:
            return
        rtc_request = None
        if self.config.rtc_enabled:
            steps = sorted(self._actions)
            if steps and steps != list(range(self._next_step, self._next_step + len(steps))):
                raise RuntimeError("RTC action queues must have a contiguous shared timeline")
            previous = torch.cat([self._original_actions[step] for step in steps]) if steps else None
            absolute = torch.cat([self._actions[step] for step in steps]) if steps else None
            delay = max(self._rtc_latencies) if self._rtc_latencies else self.config.rtc_initial_delay_steps
            delay = min(delay, len(steps), self.config.rtc_execution_horizon)
            if self.config.rtc_mode == "trained" and self._rtc_training_max_delay is not None:
                delay = min(delay, self._rtc_training_max_delay)
            rtc_request = {
                "previous_original": previous, "previous_absolute": absolute,
                "previous_context": deepcopy(self._action_context) if steps else None,
                "rtc": {
                    "mode": self.config.rtc_mode, "inference_delay": delay,
                    "execution_horizon": self.config.rtc_execution_horizon,
                    "prefix_attention_schedule": self.config.rtc_prefix_attention_schedule,
                    "max_guidance_weight": self.config.rtc_max_guidance_weight,
                },
            }
        self._request = (self._generation, self._next_step, list(self._history), rtc_request)
        self._busy = True
        self._busy_since = time.monotonic()
        self._condition.notify_all()

    def select_action(self, observation, *, stop_event=None, pause_event=None, observation_fn=None, on_pause=None, check_interrupt=None):
        """Prefetch from the latest observation and consume one aligned action.

        Returns None on stop. Pause clears old results and refreshes the
        observation on resume. Both signals are checked while awaiting inference.
        The environment supplies the execution cadence through env.step().
        """
        if not self._thread.is_alive():
            if self._error is not None:
                raise RuntimeError("Async policy inference failed") from self._error
            raise RuntimeError("Async inference worker is not running")
        with self._condition:
            self._history.append(deepcopy(observation))
        deadline = time.monotonic() + self.config.async_timeout_s
        while True:
            if check_interrupt is not None:
                check_interrupt()
            if stop_event is not None and stop_event.is_set():
                return None
            if pause_event is not None and pause_event.is_set():
                self.invalidate(preserve_error=check_interrupt is not None)
                if on_pause is not None:
                    on_pause()
                logger.info("Async inference paused; queued actions discarded")
                while pause_event.is_set():
                    if stop_event is not None and stop_event.wait(0.05):
                        return None
                    if stop_event is None:
                        time.sleep(0.05)
                if observation_fn is None:
                    raise ValueError("Async resume requires observation_fn to obtain fresh observations")
                observation = observation_fn()
                with self._condition:
                    self._history.append(deepcopy(observation))
                deadline = time.monotonic() + self.config.async_timeout_s
            with self._condition:
                if self._closed:
                    raise RuntimeError("Async inference is closed")
                if self._error is not None:
                    raise RuntimeError("Async policy inference failed") from self._error
                if self._busy and time.monotonic() - self._busy_since >= self.config.async_timeout_s:
                    raise TimeoutError("Async policy inference timed out")
                self._submit()
                action = self._actions.pop(self._next_step, None)
                if action is not None:
                    self._original_actions.pop(self._next_step, None)
                    self._next_step += 1
                    return action
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for an async action chunk")
                self._condition.wait(timeout=0.05)

    def _prepare(self, observation):
        # Processors may mutate their inputs. Keep retained raw history intact
        # when successive requests reuse the same observation snapshots.
        observation = deepcopy(observation)
        if self.config.policy_type != "client":
            from kuavo_deploy.utils.policy_loader import inject_task_prompt

            observation = inject_task_prompt(observation, self.config.task_prompt)
        return self.preprocessor(observation)

    def _predict(self, history):
        if getattr(getattr(self.policy, "config", None), "type", None) == "diffusion":
            # LeRobot DiffusionPolicy.predict_action_chunk reads observation
            # queues normally filled by select_action. Populate them from the
            # latest control-step history, without consuming its action queue.
            for key, queue in self.policy._queues.items():
                if key != "action":
                    queue.clear()
            for raw in history:
                observation = self._prepare(raw)
                if self.policy.config.image_features:
                    observation = dict(observation)
                    observation["observation.images"] = torch.stack(
                        [observation[key] for key in self.policy.config.image_features], dim=-4
                    )
                for key, queue in self.policy._queues.items():
                    if key == "action" or key not in observation:
                        continue
                    if not queue:
                        queue.extend([observation[key]] * queue.maxlen)
                    else:
                        queue.append(observation[key])
        else:
            observation = self._prepare(history[-1])
        chunk = torch.as_tensor(self.policy.predict_action_chunk(observation))
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] == 0:
            raise ValueError(f"Expected a nonempty [T, D] or [1, T, D] chunk, got {tuple(chunk.shape)}")
        if self.chunk_limit is not None:
            chunk = chunk[:self.chunk_limit]
        # LeRobot processors expect [B, D], not an entire [B, T, D] chunk.
        actions = [torch.as_tensor(self.postprocessor(row.unsqueeze(0))).detach().cpu().clone() for row in chunk]
        for action in actions:
            if action.ndim != 2 or action.shape[0] != 1 or not torch.isfinite(action).all():
                raise ValueError("Postprocessed actions must be finite tensors with shape [1, D]")
            if self._action_dim is None:
                self._action_dim = action.shape[1]
            if action.shape[1] != self._action_dim:
                raise ValueError("Policy action dimension changed during the episode")
        return actions

    def _merge(self, start_step, actions):
        # Exclude consumed steps and new-only tail actions from the ramp. This
        # keeps handover tied to the overlap that can actually still execute.
        queue_before = len(self._actions)
        overlap_steps = [
            timestep for timestep in range(start_step, start_step + len(actions))
            if timestep >= self._next_step and timestep in self._actions
        ]
        overlap_weights = dict(zip(
            overlap_steps, _overlap_weights(len(overlap_steps), self.config.aggregate_fn_name),
        ))
        for offset, action in enumerate(actions):
            timestep = start_step + offset
            if timestep < self._next_step:
                continue
            old = self._actions.get(timestep)
            if old is None:
                self._actions[timestep] = action
            else:
                weight = overlap_weights[timestep]
                self._actions[timestep] = action if weight == 1 else (1 - weight) * old + weight * action
        self._chunk_size = len(actions)
        dropped = min(len(actions), max(0, self._next_step - start_step))
        return {
            "start_step": start_step,
            "next_step_at_merge": self._next_step,
            "produced": len(actions),
            "dropped": dropped,
            "overlap_steps": len(overlap_steps),
            "new_only_steps": len(actions) - dropped - len(overlap_steps),
            "queue_before": queue_before,
            "queue_after": len(self._actions),
        }

    def _predict_rtc(self, history, request):
        if not callable(getattr(self.policy, "predict_action_chunk_rtc", None)):
            raise ValueError("RTC requires the select_action_chunk_rtc model adapter endpoint")
        result = self.policy.predict_action_chunk_rtc(self._prepare(history[-1]), **request)
        original = torch.as_tensor(result["original_actions"]).detach().cpu().clone()
        processed = torch.as_tensor(result["processed_actions"]).detach().cpu().clone()
        if (original.ndim != 2 or processed.ndim != 2 or len(original) != len(processed)
                or not len(original) or not original.shape[1] or not processed.shape[1]
                or not torch.isfinite(original).all() or not torch.isfinite(processed).all()):
            raise ValueError("RTC response must contain finite paired [T,D] action spaces")
        if result.get("action_context") is None:
            raise ValueError("RTC response requires the action chunk's observation context")
        if self.chunk_limit is not None:
            original, processed = original[:self.chunk_limit], processed[:self.chunk_limit]
        maximum = int(result.get("rtc_training_max_delay", 0))
        if self.config.rtc_mode == "trained" and maximum <= 0:
            raise ValueError("RTC trained mode requires a trained RTC checkpoint")
        self._rtc_training_max_delay = maximum
        actions = [torch.as_tensor(self.postprocessor(row[None])).detach().cpu().clone() for row in processed]
        for action in actions:
            if action.shape != (1, processed.shape[1]) or not torch.isfinite(action).all():
                raise ValueError("Invalid postprocessed RTC action")
        if self._action_dim is not None and self._action_dim != processed.shape[1]:
            raise ValueError("Policy action dimension changed during the episode")
        self._action_dim = processed.shape[1]
        if self._original_dim is not None and self._original_dim != original.shape[1]:
            raise ValueError("Model action dimension changed during the episode")
        self._original_dim = original.shape[1]
        return original, actions, result["action_context"]

    def _merge_rtc(self, start_step, original, actions, context, request, elapsed):
        consumed = max(0, self._next_step - start_step)
        has_prefix = request["previous_original"] is not None
        conditioned = request["rtc"]["inference_delay"]
        measured = math.ceil(elapsed * self.control_hz)
        # Retain the old queue if latency outruns a trained clean prefix. Its
        # original anchor remains attached even though the server has computed
        # another (discarded) chunk in the meantime.
        discard = self.config.rtc_mode == "trained" and has_prefix and (
            consumed > conditioned or measured > self._rtc_training_max_delay)
        before = len(self._actions)
        overlap = sum(start_step + offset in self._actions for offset in range(consumed, len(actions)))
        if discard:
            self._rtc_discards += 1
            if self._rtc_discards >= 3:
                raise RuntimeError("RTC latency repeatedly exceeds the trained prefix; increase training delay or use guided mode")
        else:
            self._rtc_discards = 0
            self._actions = {start_step + i: action for i, action in enumerate(actions) if i >= consumed}
            self._original_actions = {start_step + i: original[i:i+1] for i in range(len(actions)) if i >= consumed}
            self._action_context = deepcopy(context)
            self._chunk_size = len(actions)
        if has_prefix:
            self._rtc_latencies.append(max(measured, consumed))
        return {
            "start_step": start_step, "next_step_at_merge": self._next_step,
            "produced": len(actions), "dropped": len(actions) if discard else min(consumed, len(actions)),
            "overlap_steps": overlap, "new_only_steps": max(0, len(actions) - consumed - overlap),
            "queue_before": before, "queue_after": len(self._actions),
            "rtc_mode": self.config.rtc_mode, "conditioned_delay": conditioned,
            "consumed_delay": consumed, "measured_delay": measured, "discarded_chunk": discard,
        }

    def _worker(self):
        try:
            if self._policy_factory is not None:
                # Use the existing client unchanged. Construct, call and release
                # it in this worker so its socket never crosses thread boundaries.
                self.policy = self._policy_factory()
                if not callable(getattr(self.policy, "predict_action_chunk", None)):
                    raise ValueError("Async inference requires policy.predict_action_chunk()")
            if self._policy_ready is not None:
                prepared = self._policy_ready(self.policy, self.preprocessor, self.postprocessor)
                if prepared is not None:
                    self.policy = prepared
            self._run_worker()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            if self._policy_factory is not None:
                self.policy = None

    def _run_worker(self):
        generation = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._request is not None)
                if self._closed:
                    return
                request_generation, start_step, history, rtc_request = self._request
                self._request = None
            try:
                started = time.monotonic()
                with torch.inference_mode():
                    if generation != request_generation:
                        self.policy.reset()
                        for processor in (self.preprocessor, self.postprocessor):
                            if callable(getattr(processor, "reset", None)):
                                processor.reset()
                        generation = request_generation
                    if rtc_request is None:
                        actions = self._predict(history)
                    else:
                        original, actions, context = self._predict_rtc(history, rtc_request)
                with self._condition:
                    if not self._closed and request_generation == self._generation:
                        elapsed = time.monotonic() - started
                        record = self._merge(start_step, actions) if rtc_request is None else self._merge_rtc(
                            start_step, original, actions, context, rtc_request, elapsed)
                        if self._record_chunks:
                            record["infer_ms"] = elapsed * 1000.0
                            self._chunk_history.append(record)
                        logger.info(
                            "Async chunk: step=%d, actions=%d, elapsed=%.3fs, queue=%d",
                            start_step, len(actions), elapsed, len(self._actions),
                        )
            except Exception as exc:
                with self._condition:
                    if request_generation == self._generation:
                        self._error = exc
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._actions.clear()
            self._original_actions.clear()
            self._action_context = None
            self._condition.notify_all()
        if self._thread.ident is not None:
            self._thread.join(timeout=self.config.async_timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("Async worker did not stop; do not reuse this policy")
