"""Fixed-rate interpolation thread for direct-to-WBC deployment.

In ``direct_to_wbc`` mode the old code interpolated and published inside
``KuavoBaseRosEnv.step()``, which serialized inference and command publishing on
one thread. The robot therefore saw "silence for one inference interval, then a
burst of points", which the WBC turns into visible jitter.

This module owns every robot command publish at a fixed rate. The inference
thread only calls :meth:`ControlLoop.set_target`, which is non-blocking and
overwrites the previous target ("newest wins"). Each new target starts a segment
of duration ``segment_dt`` (= 1 / inference_rate) that linearly interpolates from
the last published command to the target, giving constant velocity within a segment.
"""

import threading
import time

import numpy as np

from kuavo_deploy.utils.logging_utils import setup_logger

log_robot = setup_logger("robot")

# 默认线性插值段时长（秒），env 会传入 1 / ros_rate。
SEGMENT_DT = 1.0 / 30.0
# 目标进入该死区后直接吸附，保证最终精确到达。
EMA_SNAP_EPS = 2e-3
# 安全兜底限速（rad/s）：正常跟踪远达不到，只防止首帧跳变或模型偶发野值。
MAX_JOINT_VEL = 3.0
MAX_LOWER_VEL = 1.5
MAX_BASE_VEL_ACC = 3.0
# 末端是归一化开合量（0=张开，1=闭合），量程只有 1，所以限速按“全量程/秒”取。
MAX_EEF_VEL = 2.0       # 0 -> 1 至少 0.5s
# 末端执行器只在指令变化超过该阈值时才下发，避免 100Hz 重复发同一帧。
# 阈值远小于 0.5，不会吞掉 SG100 的二值跳变。
EEF_EPSILON = 1e-4
# 落后超过这么多个 tick 就重置时间基准：只放慢，不追赶。
MAX_LAG_TICKS = 3


class ControlLoop:
    """Publish robot commands at a fixed rate from a dedicated thread."""

    def __init__(
        self,
        env,
        control_hz=100.0,
        segment_dt=SEGMENT_DT,
        max_joint_vel=MAX_JOINT_VEL,
        max_joint_vel_lower=MAX_LOWER_VEL,
        max_eef_vel=MAX_EEF_VEL,
        log_every_n_ticks=0,
    ):
        self.env = env
        self.control_hz = float(control_hz)
        self.dt = 1.0 / self.control_hz
        self.segment_dt = float(segment_dt)
        self.max_joint_vel = float(max_joint_vel)
        self.max_joint_vel_lower = float(max_joint_vel_lower)
        self.max_eef_vel = float(max_eef_vel)
        self._eef_max_step = self.max_eef_vel * self.dt
        self.log_every_n_ticks = int(log_every_n_ticks)

        self._lock = threading.Lock()
        self._pending_action = None
        self._pending_generation = None
        self._target_generation = None
        self._thread = None
        self._running = False

        # 每个通道一条线性段：从段起点插值到段终点，frac 只在未暂停的 tick 推进。
        self._frac = 1.0
        self._seg_start = {}
        self._seg_goal = {}
        self._cmd = {}          # 上一条实际下发值
        self._target_components = None
        self._eef_sent = None   # 上一条真正下发的末端值（去重用）

        self._ticks = 0
        self._targets = 0
        self._dt_sum = 0.0
        self._dt_sq_sum = 0.0
        self._dt_max = 0.0

    # ------------------------------------------------------------------ API
    @property
    def running(self):
        return self._running

    def start(self, q0, lower0=None, base0=None, eef0=None):
        """Start the loop from the currently measured robot state."""
        if self._running:
            return
        env = self.env
        # 优先复用 env 缓存的最后一条实际下发值：重连/重启时从指令位置继续，
        # 不会因为拿测量值当初值而产生一次跳变。
        if lower0 is None:
            lower0 = getattr(env, "_last_lower_cmd", None)
        if base0 is None:
            base0 = getattr(env, "_last_base_cmd", None)
        # 末端从当前开合量播种，避免第一帧就跳到模型目标。
        seeds = {"arm": q0, "lower": lower0, "base": base0, "eef": eef0}
        self._cmd = {
            k: np.asarray(v, dtype=float).copy() for k, v in seeds.items() if v is not None
        }
        self._seg_start = {}
        self._seg_goal = {}
        self._frac = 1.0
        self._eef_sent = None
        self._target_components = None
        with self._lock:
            self._pending_action = None
            self._pending_generation = None
        self._target_generation = None
        self._ticks = 0
        self._targets = 0
        self._dt_sum = 0.0
        self._dt_sq_sum = 0.0
        self._dt_max = 0.0
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="kuavo-control-loop", daemon=True
        )
        self._thread.start()
        log_robot.info(
            f"[CTRL] control loop started at {self.control_hz:.1f}Hz "
            f"(linear segment_dt={self.segment_dt * 1000:.1f}ms)"
        )

    def set_target(self, action, *, generation=None):
        """Non-blocking handoff. Any older un-consumed target is dropped."""
        with self._lock:
            self._pending_action = np.asarray(action, dtype=float).copy()
            self._pending_generation = generation

    def stop(self, timeout=2.0):
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive() and getattr(self.env, "dagger_session", None) is not None:
                raise RuntimeError("Control publisher did not stop; refusing another writer")
        self._thread = None
        if thread is not None:
            log_robot.info(f"[CTRL] control loop stopped: {self.stats()}")

    def stats(self):
        ticks = self._ticks
        mean = self._dt_sum / ticks if ticks else 0.0
        var = max(self._dt_sq_sum / ticks - mean * mean, 0.0) if ticks else 0.0
        return {
            "ticks": ticks,
            "targets": self._targets,
            "dt_mean": mean,
            "dt_std": var ** 0.5,
            "dt_max": self._dt_max,
        }

    # ------------------------------------------------------------- tracking
    def _step(self, key, max_step):
        """Move channel `key` toward its linear segment point, clamped per tick."""
        goal = self._seg_goal[key]
        start = self._seg_start[key]
        desired = start + (goal - start) * self._frac
        cmd = self._cmd[key]
        cmd = cmd + np.clip(desired - cmd, -max_step, max_step)
        cmd = np.where(np.abs(goal - cmd) < EMA_SNAP_EPS, goal, cmd)
        self._cmd[key] = cmd
        return cmd

    def _begin_segment(self, goals):
        """Start a new linear segment per channel, from the last published command."""
        for key, goal in goals.items():
            goal = np.asarray(goal, dtype=float).copy()
            if key not in self._cmd:
                self._cmd[key] = goal.copy()
            self._seg_start[key] = self._cmd[key].copy()
            self._seg_goal[key] = goal
        self._frac = 0.0

    # ----------------------------------------------------------- thread body
    def _run(self):
        env = self.env
        signals = getattr(env, "control_signal_manager", None)
        next_t = time.perf_counter()
        last_t = next_t

        while self._running:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(next_t - now, self.dt))
                continue

            # A stalled tick must not trigger a catch-up burst.
            if now - next_t > MAX_LAG_TICKS * self.dt:
                next_t = now

            if signals is not None and signals.stop_flag.is_set():
                log_robot.info("[CTRL] stop flag received, exiting control loop")
                break

            # Never call check_control_signals() here: it blocks while paused.
            paused = signals is not None and signals.pause_flag.is_set()
            session = getattr(env, "dagger_session", None)
            if session is not None:
                try:
                    if session.active and session.threaded_ready():
                        self._take_target()
                        if self._target_components is None:
                            session.threaded_tick()
                        else:
                            self._publish(paused=False)
                except Exception as exc:
                    session._error = exc
                    break
                next_t += self.dt
                continue
            if not paused:
                self._take_target()

            if self._target_components is not None:
                self._publish(paused=paused)

            tick_dt = now - last_t
            last_t = now
            self._ticks += 1
            self._dt_sum += tick_dt
            self._dt_sq_sum += tick_dt * tick_dt
            if tick_dt > self._dt_max:
                self._dt_max = tick_dt

            if self.log_every_n_ticks and self._ticks % self.log_every_n_ticks == 0:
                s = self.stats()
                log_robot.info(
                    f"[CTRL] ticks={s['ticks']} targets={s['targets']} "
                    f"dt_mean={s['dt_mean'] * 1000:.2f}ms "
                    f"dt_std={s['dt_std'] * 1000:.2f}ms "
                    f"dt_max={s['dt_max'] * 1000:.2f}ms"
                )

            next_t += self.dt

        session = getattr(env, "dagger_session", None)
        if session is not None and session.active:
            try:
                env.stop_arm_velocity()
                env.stop_base()
                session.recorder.record("stopped")
            except Exception as exc:
                session._error = session._error or exc
        self._running = False

    # ------------------------------------------------------------- internals
    def _take_target(self):
        with self._lock:
            action = self._pending_action
            generation = self._pending_generation
            self._pending_action = None
        if action is None:
            return
        env = self.env
        session = getattr(env, "dagger_session", None)
        if session is not None and session.active:
            accepted, anchor = session.anchor_for_target(generation)
            if not accepted:
                return
            if anchor is not None:
                # A fresh policy generation starts at the last command actually
                # sent by arbitration, including any preceding VR intervention.
                parts = env._split_action(anchor)
                empty = np.zeros(env.eef_dof_per_side)
                self._cmd = {
                    "arm": env._full_arm_target(parts).copy(),
                    "eef": np.concatenate((parts.get("left", {}).get("eef", empty),
                                           parts.get("right", {}).get("eef", empty))),
                }
                if env.use_5w_wholebody:
                    self._cmd["lower"] = np.asarray(parts["lower_body"]).copy()
                if env.use_5w_base_move:
                    self._cmd["base"] = np.asarray(parts["base_velocity"]).copy()
        components = env._split_action(action)
        self._target_action = action.copy()
        self._target_generation = generation
        self._target_components = components
        self._targets += 1

        goals = {"arm": env._full_arm_target(components)}
        if env.use_5w_wholebody:
            goals["lower"] = components["lower_body"]
        if env.use_5w_base_move:
            goals["base"] = components["base_velocity"]
        empty_eef = np.zeros(env.eef_dof_per_side, dtype=float)
        left_eef = components.get("left", {}).get("eef", empty_eef)
        right_eef = components.get("right", {}).get("eef", empty_eef)
        goals["eef"] = np.concatenate(
            (np.asarray(left_eef, dtype=float), np.asarray(right_eef, dtype=float))
        )
        self._begin_segment(goals)

    def _publish(self, paused=False):
        env = self.env
        session = getattr(env, "dagger_session", None)
        if session is not None and not session.active:
            return
        if not paused:
            self._frac = min(1.0, self._frac + self.dt / self.segment_dt)

        arm_cmd = self._step("arm", self.max_joint_vel * self.dt)
        if session is None:
            env.safe_control_arm(arm_cmd)

        if env.use_5w_wholebody:
            lower_cmd = self._step("lower", self.max_joint_vel_lower * self.dt)
            if session is None:
                env.safe_control_lower_body(lower_cmd)

        if env.use_5w_base_move:
            base_cmd = self._step("base", MAX_BASE_VEL_ACC * self.dt)
            if session is None:
                env.safe_control_base(base_cmd)

        eef_now = self._step("eef", self._eef_max_step)
        # 去重比较的是插值之后的值，仍是穿越阈值时才发。
        if session is not None:
            action = self._target_action.copy()
            for side in env.action_layout.sides:
                section = env.action_layout.action_slice(side)
                i = 0 if side == "left" else 1
                action[section.start:section.start + env.arm_dof_per_side] = arm_cmd[i * env.arm_dof_per_side:(i + 1) * env.arm_dof_per_side]
                action[section.start + env.arm_dof_per_side:section.stop] = eef_now[i * env.eef_dof_per_side:(i + 1) * env.eef_dof_per_side]
            if env.use_5w_wholebody:
                action[env.action_layout.lower_body_slice] = lower_cmd
            if env.use_5w_base_move:
                action[env.action_layout.state_dim:] = base_cmd
            session.threaded_tick(action, token=self._target_generation)
        elif self._eef_sent is None or not np.allclose(
            eef_now, self._eef_sent, atol=EEF_EPSILON
        ):
            self._eef_sent = eef_now.copy()
            per_side = env.eef_dof_per_side
            env._control_eef(eef_now[:per_side], eef_now[per_side:])
