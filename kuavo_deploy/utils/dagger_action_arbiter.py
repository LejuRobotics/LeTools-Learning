"""Thread-safe DAgger action arbitration for Quest3 intervention.

The pure :class:`DaggerActionArbiter` has no ROS dependency and is unit-testable.
ROS subscribers are installed lazily by :func:`create_ros_dagger_arbiter`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from kuavo_deploy.utils.dagger_layout import DaggerLayout


@dataclass(frozen=True)
class ArbitrationDecision:
    action: np.ndarray
    source: str
    require_fresh_policy: bool = False
    arm_velocity: np.ndarray | None = None  # full left7/right7, radians/s


class DaggerActionArbiter:
    """Select policy or human candidates independently for arms and optional waist."""

    def __init__(
        self,
        *,
        action_timeout: float = 0.2,
        signal_timeout: float = 0.2,
        handover_duration: float = 0.3,
        policy_handover_duration: float = 0.5,
        layout: DaggerLayout,
        waist_on: bool = False,
        lower_body_lock_mask=None,
        arm_position_limits=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.layout = layout
        self.waist_on = bool(waist_on)
        if self.waist_on and self.layout.lower_body_dof != 4:
            raise ValueError("Waist arbitration requires a four-dimensional lower-body action")
        self.lower_body_lock_mask = np.asarray(
            [False] * 4 if lower_body_lock_mask is None else lower_body_lock_mask, dtype=bool)
        if self.lower_body_lock_mask.shape != (4,):
            raise ValueError("lower_body_lock_mask must contain four values")
        self.action_timeout = float(action_timeout)
        self.signal_timeout = float(signal_timeout)
        self.handover_duration = max(0.0, float(handover_duration))
        self.policy_handover_duration = max(0.0, float(policy_handover_duration))
        self.arm_position_limits = None if arm_position_limits is None else np.asarray(arm_position_limits, float)
        if self.arm_position_limits is not None:
            if (self.arm_position_limits.shape != (2, 14) or not np.isfinite(self.arm_position_limits).all()
                    or np.any(self.arm_position_limits[0] >= self.arm_position_limits[1])):
                raise ValueError("Arm position limits require finite ordered lower/upper vectors")
        self._clock = clock
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """Start a new episode without carrying over VR samples or transition anchors."""
        with self._lock:
            self._vr_arm: np.ndarray | None = None
            self._vr_arm_velocity: np.ndarray | None = None
            self._vr_eef: np.ndarray | None = None
            self._vr_arm_time = float("-inf")
            self._vr_eef_time = float("-inf")
            self._left_requested = False
            self._right_requested = False
            self._signal_time = float("-inf")
            self._left_active = False
            self._right_active = False
            self._left_rising_time = float("inf")
            self._right_rising_time = float("inf")
            self._left_blend_start_time = float("-inf")
            self._right_blend_start_time = float("-inf")
            self._left_blend_anchor: np.ndarray | None = None
            self._right_blend_anchor: np.ndarray | None = None
            self._last_executed: np.ndarray | None = None
            self._last_arm_velocity = np.zeros(14)
            self._handover_offsets = {"left": None, "right": None}
            self._candidate_status = None
            self._candidate_last_token = {"left": None, "right": None}
            self._candidate_blocked_token = {"left": None, "right": None}
            self._waist_requested = False
            self._waist_config_session = None
            self._waist_config_acknowledged = False
            self._waist_ready = False
            self._waist_token = None
            self._waist_last_token = None
            self._waist_blocked_token = None
            self._waist_rising_time = float("inf")
            self._waist_active = False
            self._vr_waist = None
            self._vr_waist_time = float("-inf")
            self._vr_waist_source_time = float("-inf")
            self._waist_blend_start_time = float("-inf")
            self._waist_blend_anchor = None
            self._policy_waiting = {}
            self._policy_blending = {}
        callback = getattr(self, "_waist_config_reset", None)
        if callback:
            callback()

    def update_vr_arm(self, arm_radians, velocity_radians, *, candidate_status) -> bool:
        """Accept a complete arm candidate with velocity and takeover readiness."""
        arm = np.asarray(arm_radians, dtype=np.float64)
        if arm.shape != (14,):
            raise ValueError(f"VR arm action must have shape (14,), got {arm.shape}")
        if not np.all(np.isfinite(arm)):
            raise ValueError("VR arm action contains non-finite values")
        velocity = np.asarray(velocity_radians, dtype=np.float64)
        if velocity.shape != (14,) or not np.isfinite(velocity).all():
            raise ValueError("VR arm velocity must contain 14 finite radians/s values")
        if not isinstance(candidate_status, dict):
            raise ValueError("VR arm candidate requires takeover status")
        status = dict(candidate_status)
        if (type(status.get("schema_version")) is not int or status["schema_version"] != 1
                or not isinstance(status.get("session"), str) or not status["session"]
                or type(status.get("sequence")) is not int or status["sequence"] < 0):
            raise ValueError("Invalid DAgger candidate protocol/session/sequence")
        for side in ("left", "right"):
            if (type(status.get(side + "_generation")) is not int or status[side + "_generation"] < 0
                    or type(status.get(side + "_ready")) is not bool
                    or type(status.get(side + "_requested")) is not bool):
                raise ValueError("Invalid DAgger candidate generation/readiness")
        with self._lock:
            previous = self._candidate_status
            if (previous is not None and status["session"] == previous["session"]
                    and status["sequence"] <= previous["sequence"]):
                return False  # queued/out-of-order packets cannot restore old readiness
            self._candidate_status = status
            now = self._clock()
            self._vr_arm = arm.copy()
            self._vr_arm_velocity = velocity.copy()
            self._vr_arm_time = now
            return True

    def close(self):
        """Stop VR callbacks before the environment releases its ROS buffers."""
        callback = getattr(self, "_waist_config_close", None)
        if callback:
            callback()
        for subscriber in getattr(self, "_ros_subscribers", ()):
            subscriber.unregister()
        self._ros_subscribers = []

    def update_vr_eef(self, eef) -> None:
        value = np.asarray(eef, dtype=np.float64)
        if value.shape != (2,):
            raise ValueError(f"VR eef action must have shape (2,), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("VR eef action contains non-finite values")
        with self._lock:
            self._vr_eef = np.clip(value, 0.0, 1.0)
            self._vr_eef_time = self._clock()

    def configure_waist(self, session):
        """Require KRC to acknowledge this episode's limits and takeover scope."""
        with self._lock:
            self._waist_config_session = session
            self._waist_config_acknowledged = False
            self._waist_ready = False

    def update_vr_waist(self, packet, *, session, sequence, source_age) -> None:
        """Accept a KRC joint candidate; source age must include upstream delay."""
        if (not isinstance(packet, dict) or not isinstance(packet.get("config_session"), str)
                or type(packet.get("ready")) is not bool or type(packet.get("requested")) is not bool
                or type(packet.get("generation")) is not int or packet["generation"] < 0
                or type(packet.get("stamp_ns")) is not int or packet["stamp_ns"] < 0):
            raise ValueError("Invalid KRC waist candidate status")
        value = np.asarray(packet["position"], dtype=np.float64)
        if value.shape != (4,) or not np.all(np.isfinite(value)):
            raise ValueError("VR waist candidate must contain four finite radians")
        with self._lock:
            status = self._candidate_status
            if status is None or (session, sequence) != (status["session"], status["sequence"]):
                return
            self._waist_ready = False
            if packet["config_session"] != self._waist_config_session:
                return
            self._waist_config_acknowledged = True
            token = (session, packet["config_session"], packet["generation"])
            if (self._waist_token is not None and token[:2] == self._waist_token[:2]
                    and token[2] < self._waist_token[2]):
                return
            self._waist_token = token
            self._waist_ready = (packet["ready"] and packet["requested"] and packet["stamp_ns"] > 0
                                 and np.isfinite(source_age) and -0.05 <= source_age <= self.action_timeout)
            self._vr_waist = value.copy()
            self._vr_waist_time = self._clock()
            self._vr_waist_source_time = self._vr_waist_time - source_age

    def update_intervention(self, left=None, right=None) -> None:
        with self._lock:
            now = self._clock()
            arms_updated = left is not None or right is not None
            left = (self._left_requested if left is None else bool(left)) and "left" in self.layout.sides
            right = (self._right_requested if right is None else bool(right)) and "right" in self.layout.sides
            if left and not self._left_requested:
                self._left_rising_time = now
                self._candidate_blocked_token["left"] = self._candidate_last_token["left"]
            if right and not self._right_requested:
                self._right_rising_time = now
                self._candidate_blocked_token["right"] = self._candidate_last_token["right"]
            self._left_requested = left
            self._right_requested = right
            if arms_updated:
                self._signal_time = now
            requested = self.waist_on and (left or right)
            if requested and not self._waist_requested:
                self._waist_rising_time = now
                self._waist_blocked_token = self._waist_last_token
            self._waist_requested = requested

    def acknowledge_action(self, action, arm_velocity=None):
        """Use the dispatched command as the next takeover/return anchor."""
        value = self.layout.validate(action)
        velocity = np.zeros(14) if arm_velocity is None else np.asarray(arm_velocity, float)
        if velocity.shape != (14,) or not np.isfinite(velocity).all():
            raise ValueError("Executed arm velocity must contain 14 finite values")
        with self._lock:
            self._last_executed = value.copy()
            self._last_arm_velocity = velocity.copy()

    def resume_after_pause(self):
        """Restart the handover from the position held during a pause."""
        with self._lock:
            self._left_active = self._right_active = self._waist_active = False
            self._handover_offsets = {"left": None, "right": None}
            self._last_arm_velocity.fill(0.0)

    def _candidate_ready_locked(self, side):
        status = self._candidate_status
        if status is None or not status[side + "_ready"] or not status[side + "_requested"]:
            return False
        token = (status["session"], status[side + "_generation"])
        return token != self._candidate_blocked_token[side]

    def _start_handover_locked(self, side, anchor):
        status = self._candidate_status
        self._candidate_last_token[side] = (status["session"], status[side + "_generation"])
        self._handover_offsets[side] = None
        if self.handover_duration > 0:
            joints = slice(0, 7) if side == "left" else slice(7, 14)
            q, v = self._clip_arm_target_locked(self._vr_arm[joints], self._vr_arm_velocity[joints], joints)
            # Capture the mismatch ONCE. Later operator motion does not
            # increase this correction or postpone the end of the handover.
            self._handover_offsets[side] = (anchor[:7] - q, self._last_arm_velocity[joints] - v)

    @staticmethod
    def _smooth_weight(elapsed: float, duration: float) -> float:
        if duration <= 0.0:
            return 1.0
        s = min(1.0, max(0.0, elapsed / duration))
        return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))

    @staticmethod
    def _smooth_weight_derivative(elapsed: float, duration: float) -> float:
        if duration <= 0.0 or elapsed <= 0.0 or elapsed >= duration:
            return 0.0
        s = elapsed / duration
        return 30.0 * s * s * (1.0 - s) ** 2 / duration

    def _clip_arm_target_locked(self, position, velocity, joints):
        position, velocity = position.copy(), velocity.copy()
        if self.arm_position_limits is not None:
            low, high = self.arm_position_limits[:, joints]
            clipped = np.clip(position, low, high)
            outside = abs(clipped - position) > 1e-10
            outward = ((clipped >= high - 1e-10) & (velocity > 0)) | ((clipped <= low + 1e-10) & (velocity < 0))
            velocity[outside | outward] = 0.0
            position = clipped
        return position, velocity

    def _blend_side_locked(
        self,
        *,
        now: float,
        target: np.ndarray,
        is_left: bool,
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        start_time = self._left_blend_start_time if is_left else self._right_blend_start_time
        anchor = self._left_blend_anchor if is_left else self._right_blend_anchor
        joints = slice(0, 7) if is_left else slice(7, 14)
        velocity = self._vr_arm_velocity[joints].copy()
        target = target.copy()
        target[:7], velocity = self._clip_arm_target_locked(target[:7], velocity, joints)
        offset = self._handover_offsets["left" if is_left else "right"]
        elapsed = now - start_time
        if offset is None or elapsed >= self.handover_duration:
            return target, False, velocity
        s = max(0.0, elapsed / self.handover_duration)
        delta_q, delta_v = offset
        # Fade the fixed initial mismatch: match the dispatched q/v initially,
        # then make the correction and its first two derivatives zero at T.
        # q_out(t) = q_vr(t) + correction(t), never a moving-target chase.
        hq = 1.0 - self._smooth_weight(elapsed, self.handover_duration)
        hv = s - 6*s**3 + 8*s**4 - 3*s**5
        dhq = -self._smooth_weight_derivative(elapsed, self.handover_duration)
        dhv = 1.0 - 18*s**2 + 32*s**3 - 15*s**4
        target[:7] += hq * delta_q + self.handover_duration * hv * delta_v
        velocity += dhq * delta_q + dhv * delta_v
        if anchor is not None:
            target[7:] = hq * anchor[7:] + (1.0 - hq) * target[7:]
        return target, True, velocity

    def _vr_action_locked(self) -> np.ndarray | None:
        if self._vr_arm is None or self._vr_eef is None:
            return None
        return self.layout.vr_action(self._vr_arm, self._vr_eef)

    def select(self, policy_action, *, policy_fresh: bool = True) -> ArbitrationDecision:
        policy = self.layout.validate(policy_action)

        with self._lock:
            now = self._clock()
            signal_fresh = now - self._signal_time <= self.signal_timeout
            arm_fresh = now - self._vr_arm_time <= self.action_timeout
            eef_fresh = now - self._vr_eef_time <= self.action_timeout
            vr = self._vr_action_locked()
            common_ready = signal_fresh and arm_fresh and eef_fresh and vr is not None
            # Never execute a candidate cached before the grip rising edge. Both
            # arm and EEF samples must causally follow the takeover request.
            left_ready = (
                common_ready
                and self._left_requested
                and self._vr_arm_time >= self._left_rising_time
                and self._vr_eef_time >= self._left_rising_time
                and self._candidate_ready_locked("left")
            )
            right_ready = (
                common_ready
                and self._right_requested
                and self._vr_arm_time >= self._right_rising_time
                and self._vr_eef_time >= self._right_rising_time
                and self._candidate_ready_locked("right")
            )
            waist_ready = (
                self.waist_on and self._waist_requested and self._waist_ready and self._vr_waist is not None
                and not self.lower_body_lock_mask.all()
                and now - self._signal_time <= self.signal_timeout
                and now - self._vr_waist_time <= self.action_timeout
                and now - self._vr_waist_source_time <= self.action_timeout
                and self._waist_token != self._waist_blocked_token
                and self._vr_waist_time >= self._waist_rising_time
            )

            if self._candidate_status is not None:
                # A restarted robot node or a missed OFF/ON pair is a new
                # handover even when the latest Bool is still true.
                for side in ("left", "right"):
                    token = (self._candidate_status["session"], self._candidate_status[side + "_generation"])
                    ready = left_ready if side == "left" else right_ready
                    if ready and token != self._candidate_last_token[side]:
                        setattr(self, "_" + side + "_active", False)
            if left_ready and not self._left_active:
                self._left_active = True
                self._left_blend_start_time = now
                base = self._last_executed if self._last_executed is not None else policy
                self._left_blend_anchor = base[self.layout.action_slice("left")].copy()
                self._start_handover_locked("left", self._left_blend_anchor)
            if right_ready and not self._right_active:
                self._right_active = True
                self._right_blend_start_time = now
                base = self._last_executed if self._last_executed is not None else policy
                self._right_blend_anchor = base[self.layout.action_slice("right")].copy()
                self._start_handover_locked("right", self._right_blend_anchor)

            left_active = self._left_active and left_ready
            right_active = self._right_active and right_ready
            if waist_ready and self._waist_token != self._waist_last_token:
                self._waist_active = False
            if waist_ready and not self._waist_active:
                self._waist_last_token = self._waist_token
                self._waist_blend_start_time = now
                base = self._last_executed if self._last_executed is not None else policy
                self._waist_blend_anchor = base[self.layout.lower_body_slice].copy()
            released_parts = {name for name, released in (
                ("left", self._left_active and not left_active),
                ("right", self._right_active and not right_active),
                ("waist", self._waist_active and not waist_ready)) if released}
            released = bool(released_parts)
            self._left_active = left_active
            self._right_active = right_active
            self._waist_active = waist_ready
            for side in ("left", "right"):
                if not getattr(self, "_" + side + "_active"):
                    self._handover_offsets[side] = None

            selected = policy.copy()
            # Preserve the existing zero-velocity convention on model/held arms.
            # VR velocity is side-specific and never applied to the other arm.
            arm_velocity = np.zeros(14)
            sources: list[str] = []
            if left_active:
                selected[self.layout.action_slice("left")], blending, arm_velocity[:7] = self._blend_side_locked(
                    now=now, target=vr[self.layout.action_slice("left")], is_left=True
                )
                sources.append("left_blend" if blending else "left_vr")
            if right_active:
                selected[self.layout.action_slice("right")], blending, arm_velocity[7:] = self._blend_side_locked(
                    now=now, target=vr[self.layout.action_slice("right")], is_left=False
                )
                sources.append("right_blend" if blending else "right_vr")
            if waist_ready:
                weight = self._smooth_weight(now - self._waist_blend_start_time, self.handover_duration)
                candidate = (1 - weight) * self._waist_blend_anchor + weight * self._vr_waist
                # Preserve the policy segment for locked joints; env applies the go-bag lock target.
                section = selected[self.layout.lower_body_slice]
                section[~self.lower_body_lock_mask] = candidate[~self.lower_body_lock_mask]
                sources.append("waist_blend" if weight < 1 else "waist_vr")
            active_parts = {name for name, active in (
                ("left", left_active), ("right", right_active), ("waist", waist_ready)) if active}
            for name in active_parts:
                self._policy_waiting.pop(name, None)
                self._policy_blending.pop(name, None)
            base = self._last_executed if self._last_executed is not None else policy
            for name in released_parts:
                section = (self.layout.lower_body_slice if name == "waist"
                           else self.layout.action_slice(name))
                self._policy_waiting[name] = base[section].copy()
                self._policy_blending.pop(name, None)
            for name, anchor in list(self._policy_waiting.items()):
                section = (self.layout.lower_body_slice if name == "waist"
                           else self.layout.action_slice(name))
                # The current policy predates this release even if fresh.
                if policy_fresh and not released:
                    self._policy_blending[name] = (now, anchor)
                    del self._policy_waiting[name]
                else:
                    selected[section] = anchor
                    sources.append(name + "_wait_policy")
            for name, (start, anchor) in list(self._policy_blending.items()):
                section = (self.layout.lower_body_slice if name == "waist"
                           else self.layout.action_slice(name))
                weight = self._smooth_weight(now - start, self.policy_handover_duration)
                selected[section] = (1.0 - weight) * anchor + weight * policy[section]
                if weight < 1.0:
                    sources.append(name + "_policy_blend")
                else:
                    del self._policy_blending[name]
            self._last_executed = selected.copy()
            return ArbitrationDecision(selected, "+".join(sources) if sources else "policy",
                                       require_fresh_policy=released, arm_velocity=arm_velocity)


def create_ros_dagger_arbiter(config, env) -> DaggerActionArbiter:
    """Create an arbiter and subscribe to lower-computer candidate topics."""
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, String

    cfg, env_config = config.inference, config.env
    limits = env.robot._robot_info.get_arm_joint_limits()

    arbiter = DaggerActionArbiter(
        action_timeout=cfg.dagger_vr_action_timeout,
        signal_timeout=cfg.dagger_signal_timeout,
        handover_duration=getattr(cfg, "dagger_handover_duration", 0.3),
        policy_handover_duration=getattr(cfg, "dagger_policy_handover_duration", 0.5),
        arm_position_limits=limits,
        layout=env.action_layout,
        waist_on=getattr(cfg, "dagger_waist_on", False),
        lower_body_lock_mask=getattr(env_config, "lowerlock_5w", None),
    )

    def on_eef(msg: JointState) -> None:
        try:
            arbiter.update_vr_eef(msg.position)
        except ValueError as exc:
            rospy.logwarn_throttle(1.0, f"Invalid DAgger VR eef candidate: {exc}")

    def on_candidate(msg: String) -> None:
        import json
        try:
            packet = json.loads(msg.data)
            if not isinstance(packet, dict):
                raise ValueError("VR arm candidate must be a JSON object")
            stamp = packet["stamp_ns"]
            if type(stamp) is not int or stamp <= 0:
                raise ValueError("Missing candidate source timestamp")
            age = rospy.Time.now().to_sec() - stamp / 1e9
            if not -0.05 <= age <= cfg.dagger_vr_action_timeout:
                raise ValueError(f"Stale candidate packet ({age:.3f}s)")
            accepted = arbiter.update_vr_arm(packet["position"], packet["velocity"], candidate_status=packet)
            if accepted and arbiter.waist_on:
                try:
                    waist = packet["waist"]
                    source_age = rospy.Time.now().to_sec() - waist["stamp_ns"] / 1e9
                    arbiter.update_vr_waist(waist, session=packet["session"],
                                           sequence=packet["sequence"], source_age=source_age)
                except (ValueError, KeyError, TypeError) as exc:
                    with arbiter._lock:
                        arbiter._waist_ready = False
                    rospy.logwarn_throttle(1.0, f"Invalid KRC waist candidate: {exc}")
        except (ValueError, KeyError, TypeError) as exc:
            rospy.logwarn_throttle(1.0, f"Invalid DAgger ready candidate: {exc}")

    def update_side(side: str, value: bool) -> None:
        arbiter.update_intervention(**{side: value})

    arbiter._ros_subscribers = [
        rospy.Subscriber(cfg.dagger_vr_eef_topic, JointState, on_eef, queue_size=1, tcp_nodelay=True),
        rospy.Subscriber(cfg.dagger_vr_candidate_topic, String, on_candidate, queue_size=1, tcp_nodelay=True),
    ]
    try:
        on_candidate(rospy.wait_for_message(cfg.dagger_vr_candidate_topic, String, timeout=3.0))
        if arbiter._candidate_status is None:
            raise RuntimeError("Robot candidate protocol did not provide a valid packet")
    except Exception:
        arbiter.close()
        raise RuntimeError("Ready VR candidate unavailable; restart the updated lower-computer VR node")
    env._dagger_arbiter = arbiter
    for side in arbiter.layout.sides:
        arbiter._ros_subscribers.append(rospy.Subscriber(
            getattr(cfg, f"dagger_{side}_intervention_topic"), Bool,
            lambda msg, side=side: update_side(side, msg.data),
            queue_size=1, tcp_nodelay=True,
        ))
    if arbiter.waist_on:
        import json
        import uuid

        # Internal configuration handshake; no duplicate launch/YAML settings on KRC.
        publisher = rospy.Publisher("/dagger/vr/waist_config", String, queue_size=1, latch=True)
        arbiter._waist_config_close = publisher.unregister

        def configure_waist():
            session = uuid.uuid4().hex
            arbiter.configure_waist(session)
            lock_position = env.lower_body_lock_target
            packet = dict(schema_version=1, session=session, which_arm=env_config.which_arm,
                          lower=list(env_config.limits["lower_body"]["min"]),
                          upper=list(env_config.limits["lower_body"]["max"]),
                          lock_mask=arbiter.lower_body_lock_mask.tolist(),
                          lock_position=np.zeros(4).tolist() if lock_position is None else lock_position.tolist(),
                          max_speed=cfg.dagger_waist_pose_max_speed,
                          action_timeout=cfg.dagger_vr_action_timeout,
                          signal_timeout=cfg.dagger_signal_timeout)
            publisher.publish(String(data=json.dumps(packet, allow_nan=False)))

        arbiter._waist_config_reset = configure_waist
        try:
            configure_waist()
            deadline = time.monotonic() + 3.0
            while not arbiter._waist_config_acknowledged and not rospy.is_shutdown():
                if time.monotonic() >= deadline:
                    raise RuntimeError("KRC did not acknowledge waist configuration within 3s; "
                                       "build and restart the updated lower-computer VR node")
                time.sleep(0.01)
        except Exception:
            arbiter.close()
            raise
    return arbiter


def validate_dagger_vr_mode(config, *, ros=None, master=None):
    if not config.inference.dagger_on or config.env.inference_env != "real":
        return
    if ros is None:
        import rospy as ros

    restart_hint = (
        "请停止普通 VR launch，重新启动 launch_quest3_ik.launch，加入 "
        "dagger_candidate_only:=true use_cpp_incremental_ik:=true "
        "reset_joint_to_default:=false；只修改 rosparam 不会改变已运行 IK 节点的模式。"
    )
    if ros.get_param("/quest3/dagger_candidate_only", None) is not True:
        raise RuntimeError(
            "DAgger 实机部署要求 /quest3/dagger_candidate_only=true。"
            "普通 VR 会与模型同时下发手臂、夹爪、腰部命令，造成抖动和错误标签。"
            + restart_hint
        )

    if master is None:
        import rosgraph
        master = rosgraph.Master(ros.get_name())
    try:
        publishers, subscribers, services = master.getSystemState()
    except Exception as exc:
        raise RuntimeError("无法核对 DAgger 的 ROS 控制路由，已中止实机初始化") from exc

    # Wheel IK advertises actuator publishers in BOTH modes. Its cached
    # candidate flag disables actual actuator writes and enables these two
    # intervention subscriptions at initialization. Check that signature on the
    # SAME node instead of treating advertised publishers as command traffic.
    publisher_nodes = {topic: set(nodes) for topic, nodes in publishers}
    subscriber_nodes = {topic: set(nodes) for topic, nodes in subscribers}
    wheel_candidate_nodes = (
        publisher_nodes.get(config.inference.dagger_vr_candidate_topic, set())
        & subscriber_nodes.get(config.inference.dagger_left_intervention_topic, set())
        & subscriber_nodes.get(config.inference.dagger_right_intervention_topic, set())
    )
    conflicts = []
    for topic, nodes in publishers:
        if topic not in ("/kuavo_arm_traj", "/kuavo_arm_traj_cpp"):
            continue
        for node in nodes:
            if (node.rsplit("/", 1)[-1] in ("ik_ros_uni_cpp_node", "ik_ros_uni")
                    and node not in wheel_candidate_nodes):
                conflicts.append(f"{node} -> {topic}")
    registered_nodes = {
        node for group in (publishers, subscribers, services)
        for _, nodes in group for node in nodes
    }
    for node in registered_nodes:
        if node.rsplit("/", 1)[-1] == "humanoid_quest_control_with_arm":
            conflicts.append(node)
    if conflicts:
        raise RuntimeError(
            "DAgger 检测到普通 VR 控制节点仍在运行："
            + ", ".join(sorted(conflicts)) + "。" + restart_hint
        )
