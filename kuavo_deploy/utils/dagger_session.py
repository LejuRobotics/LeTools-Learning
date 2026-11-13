"""Optional intervention hooks for the ordinary dev rollout and publishers.

This module never calls a policy or interpolates model actions. Inline commands
are acknowledged one at a time by a single sender; the existing threaded
publisher calls the same selector directly. Both keep checking VR while the
rollout is waiting for inference.
"""
from dataclasses import dataclass, replace
import threading
import time

import numpy as np


class PolicyRefreshRequested(Exception):
    """An intervention invalidated the observation/action currently in use."""


@dataclass
class _Command:
    action: np.ndarray
    generation: int
    channels: frozenset
    done: threading.Event
    error: Exception | None = None


class DaggerSession:
    def __init__(self, config, env, *, arbiter=None, recorder=None, clock=time.monotonic):
        self.config = config
        self.env = env
        self.layout = env.action_layout
        self.clock = clock
        self.period = 1.0 / config.env.control_rate
        self.arbiter = arbiter
        self.recorder = recorder
        self.active = False
        self.refresh_event = threading.Event()
        self._condition = threading.Condition()
        self._thread = None
        self._pending = None
        self._error = None
        self._generation = 0
        self._producer_generation = 0
        self._model_generation = -1
        self._paused = False
        self._last_source = "policy"
        self._model = None
        self._actual = None
        self._eef_sent = None
        self._ticks = 0
        self._last_tick = None
        self._max_interval = 0.0
        self._stats_start = clock()

    def __enter__(self):
        if self.arbiter is None:
            from kuavo_deploy.utils.dagger_action_arbiter import create_ros_dagger_arbiter
            self.arbiter = create_ros_dagger_arbiter(self.config, self.env)
        try:
            if self.recorder is None:
                from kuavo_deploy.utils.episode_recorder import DecisionRecorder
                self.recorder = DecisionRecorder()
            self.env.dagger_session = self
        except Exception:
            self.arbiter.close()
            raise
        return self

    def __exit__(self, *exc):
        try:
            self.end_episode()
        finally:
            self.env.dagger_session = None
            try:
                if self.recorder is not None:
                    self.recorder.close()
            finally:
                self.arbiter.close()

    def prepare_policy(self, policy, preprocessor, postprocessor):
        # Called in the existing policy owner's thread, including its ZMQ socket.
        client = getattr(policy, "policy", None)
        if callable(getattr(client, "call_endpoint", None)) and hasattr(policy, "_pending_actions"):
            # Scope strict errors/reset to this DAgger instance, retaining dev's
            # action cache, tensor conversion and socket ownership.
            if not getattr(client, "_dagger_checked", False):
                call_endpoint = client.call_endpoint

                def checked_endpoint(endpoint, *args, **kwargs):
                    reply = call_endpoint(endpoint, *args, **kwargs)
                    if isinstance(reply, dict) and "error" in reply:
                        raise RuntimeError(f"DAgger {endpoint} failed: {reply['error'] or 'empty server error'}")
                    return reply

                def reset():
                    policy._pending_actions.clear()
                    client.reset_server()
                    return policy

                client.call_endpoint = checked_endpoint
                client._dagger_checked = True
                policy.reset = reset
        if getattr(client, "socket", None) is not None:
            import zmq
            timeout = max(1, round(self.config.inference.async_timeout_s * 1000))
            client.socket.setsockopt(zmq.RCVTIMEO, timeout)
            client.socket.setsockopt(zmq.SNDTIMEO, timeout)
            client.socket.setsockopt(zmq.LINGER, 0)
        if callable(getattr(client, "call_endpoint", None)) and (
                self.config.env.which_arm != "both" or self.layout.lower_body_dof):
            metadata = client.call_endpoint("metadata", requires_input=False)
            expected = dict(which_arm=self.config.env.which_arm,
                            state_dim=self.layout.state_dim, action_dim=self.layout.dim)
            if not isinstance(metadata, dict) or metadata.get("which_arm") != expected["which_arm"]:
                raise ValueError("Model metadata must match deployment which_arm")
            for key in ("state_dim", "action_dim"):
                if metadata.get(key) is not None and metadata[key] != expected[key]:
                    raise ValueError(f"Model {key}={metadata[key]} != deployment {expected[key]}")
        return policy

    def begin_episode(self, observation):
        self.end_episode()
        state = observation["observation.state"]
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state = np.asarray(state).reshape(-1)
        if state.shape != (self.layout.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"DAgger observation must contain {self.layout.state_dim} finite values")
        initial = self.env.check_action(np.concatenate((state, np.zeros(self.layout.base_velocity_dof))))
        with self._condition:
            self._generation += 1
            self._producer_generation = self._generation
            self._model_generation = -1
            self._model = initial.copy()
            self._actual = initial.copy()
            self._eef_sent = None
            self._pending = None
            self._error = None
            self._paused = False
            self._last_source = "policy"
            self.refresh_event.clear()
            self.arbiter.reset()
            self.arbiter.acknowledge_action(initial)
            self.active = True
            self._ticks = 0
            self._last_tick = None
            self._max_interval = 0.0
            self._stats_start = self.clock()
        if self.env.interp_threaded:
            # reset() has already started dev's publisher. It owns all sends.
            if self.env.control_loop is None or not self.env.control_loop.running:
                self.env._start_control_loop()
        else:
            self._thread = threading.Thread(target=self._run_inline, name="dagger-command-selector", daemon=True)
            self._thread.start()

    def end_episode(self):
        was_active = self.active
        with self._condition:
            self.active = False
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("DAgger sender did not stop; refusing environment reuse")
            self._thread = None
        elif was_active and self.env.interp_threaded:
            self.env.control_loop.stop()
            if self.env.control_loop.running:
                raise RuntimeError("DAgger control loop did not stop")
            self.env.stop_arm_velocity()
            self.env.stop_base()
            self.recorder.record("stopped")
        self.check_error()

    def check_error(self):
        if self._error is not None:
            raise RuntimeError("DAgger command dispatch failed") from self._error

    def check_interrupt(self):
        self.check_error()
        if self.refresh_event.is_set():
            raise PolicyRefreshRequested()

    def inference_token(self):
        self.check_interrupt()
        with self._condition:
            return self._generation

    def accept_policy(self, token):
        with self._condition:
            self.check_interrupt()
            if token != self._generation:
                raise PolicyRefreshRequested()
            self._producer_generation = token

    def start_refresh(self):
        """The rollout consumes a refresh and resets its own policy/processors."""
        with self._condition:
            self.refresh_event.clear()
            # Inline interpolation must start at the actual command after release.
            self.env.last_predicted_action = self._actual.copy()
            self.env.is_first_step = False
            return self._generation

    def _invalidate(self):
        with self._condition:
            self._generation += 1
            self._model = self._actual.copy()
            self._model_generation = -1
            self.refresh_event.set()
            self._condition.notify_all()

    def target_token(self):
        with self._condition:
            self.check_interrupt()
            return self._producer_generation

    def submit(self, action, *, channels=("arm", "eef", "lower", "base")):
        """Hand off one dev-generated inline point, without dropping/reordering it."""
        command = _Command(self.layout.validate(action).copy(), self.target_token(),
                           frozenset(channels), threading.Event())
        with self._condition:
            self.check_error()
            if self._pending is not None:
                raise RuntimeError("Multiple inline command producers")
            self._pending = command
            self._condition.notify_all()
        deadline = self.clock() + self.config.inference.async_timeout_s
        while not command.done.wait(min(self.period, 0.02)):
            self.check_error()
            if not self.active or self._stopped():
                raise PolicyRefreshRequested()
            if self.clock() >= deadline:
                raise TimeoutError("DAgger command was not acknowledged")
        self.check_error()
        if command.error is not None:
            raise command.error

    def _stopped(self):
        return self.env.control_signal_manager.stop_flag.is_set()

    def _control_allowed(self):
        self.check_error()
        paused = self.env.control_signal_manager.pause_flag.is_set()
        if paused and not self._paused:
            self._paused = True
            self.env.stop_arm_velocity()
            self.env.stop_base()
            self.recorder.record("paused")
            self._invalidate()
        elif not paused and self._paused:
            self._paused = False
            self.arbiter.resume_after_pause()
        return self.active and not paused and not self._stopped()

    def threaded_ready(self):
        """Called by dev's control thread before tracking or sending anything."""
        return self._control_allowed()

    def anchor_for_target(self, token):
        with self._condition:
            if token != self._generation:
                return False, None
            anchor = self._actual.copy() if self._model_generation != token else None
            return True, anchor

    def threaded_tick(self, action=None, *, token=None):
        try:
            if not self._control_allowed():
                return
            if action is not None and token == self._generation:
                self._model = self.layout.validate(action).copy()
                self._model_generation = token
            self._dispatch(new_model=action is not None and token == self._generation,
                           channels=frozenset(("arm", "eef", "lower", "base")))
        except Exception as exc:
            self._error = exc
            self.env.stop_arm_velocity()
            self.env.stop_base()
            raise

    def _run_inline(self):
        deadline = self.clock()
        command = None
        try:
            while self.active and not self._stopped():
                with self._condition:
                    remaining = deadline - self.clock()
                    if self._pending is None and remaining > 0:
                        self._condition.wait(timeout=remaining)
                    # A dev inline point may arrive just after an idle VR tick.
                    # Keep human commands at the control cadence instead of
                    # emitting a second command a fraction of a millisecond later.
                    while (self.active and self._pending is not None
                           and self._last_source != "policy" and self.clock() < deadline):
                        self._condition.wait(timeout=deadline - self.clock())
                    if not self.active:
                        break
                    command, self._pending = self._pending, None
                allowed = self._control_allowed()
                if command is not None:
                    if not allowed or command.generation != self._generation:
                        command.error = PolicyRefreshRequested()
                    else:
                        self._model = command.action.copy()
                        self._model_generation = command.generation
                if allowed:
                    self._dispatch(new_model=command is not None and command.error is None,
                                   channels=(command.channels if command is not None else frozenset()))
                if command is not None:
                    command.done.set()
                    command = None
                # Never burst to catch up after a slow SDK call.
                deadline = self.clock() + self.period
        except Exception as exc:
            self._error = exc
        finally:
            with self._condition:
                for pending in (command, self._pending):
                    if pending is not None:
                        pending.error = self._error or PolicyRefreshRequested()
                        pending.done.set()
                self._pending = None
            try:
                self.env.stop_arm_velocity()
                self.env.stop_base()
                self.recorder.record("stopped")
            except Exception as exc:
                self._error = self._error or exc

    def _eef(self, action):
        return np.concatenate([action[self.layout.action_slice(side)][self.layout.arm_dof:]
                               for side in self.layout.sides])

    def _dispatch(self, *, new_model, channels):
        now = self.clock()
        if self._last_tick is not None:
            self._max_interval = max(self._max_interval, now - self._last_tick)
        self._last_tick = now
        self._ticks += 1
        policy = self._model.copy()
        fresh = self._model_generation == self._generation
        decision = self.arbiter.select(policy, policy_fresh=fresh)
        previous_source = self._last_source
        self._last_source = decision.source
        if decision.source == "policy":
            decision = replace(decision, arm_velocity=None)
        intervention = decision.source != "policy" or previous_source != "policy"
        if new_model or intervention:
            selected_channels = (frozenset(("arm", "eef", "lower", "base"))
                                 if intervention else channels)
            # Preserve dev threaded EEF deduplication, but compare the final
            # selected value so a VR command cannot poison the model's cache.
            eef = self._eef(decision.action)
            if self.env.interp_threaded and "eef" in selected_channels:
                from kuavo_deploy.utils.interp_thread import EEF_EPSILON
                if self._eef_sent is not None and np.allclose(eef, self._eef_sent, atol=EEF_EPSILON):
                    selected_channels = selected_channels - {"eef"}

            actual = self.env.send_dagger_command(decision.action, arm_velocity=decision.arm_velocity,
                                                 channels=selected_channels)
            self._actual = actual.copy()
            if "eef" in selected_channels:
                self._eef_sent = eef.copy()
            self.arbiter.acknowledge_action(actual, self.env.last_sent_arm_velocity)
            self.recorder.record(decision.source)
        else:
            # select() also advances its anchor on idle ticks. Keep that anchor
            # at the physical command, including channels not sent on this tick.
            self.arbiter.acknowledge_action(self._actual, self.env.last_sent_arm_velocity)
        if decision.require_fresh_policy:
            # Anchor the new generation to the command just sent. The other
            # arm may have advanced on this release tick; retaining the prior
            # command would move it backwards while fresh inference is pending.
            self._invalidate()
        elapsed = now - self._stats_start
        if elapsed >= 1.0:
            from kuavo_deploy.utils.logging_utils import setup_logger
            setup_logger("robot").info("DAgger arbitration: %.1fHz, max_interval=%.1fms, source=%s",
                                       self._ticks / elapsed, self._max_interval * 1000, decision.source)
            self._stats_start = now
            self._ticks = 0
            self._max_interval = 0.0
