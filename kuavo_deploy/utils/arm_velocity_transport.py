"""Add arm velocity to the SDK's existing, checked JointState publication.

The SDK position API still owns mode/collision checks and physical joint
clipping. Only its final publication is extended, on the calling thread, so
there is no second actuator publisher and no change to other SDK callers.
"""

from threading import RLock, local

import numpy as np


def validate_arm_vector(value, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (14,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain 14 finite arm joint values")
    return vector.copy()


class SDKArmVelocityTransport:
    def __init__(self, robot):
        # This SDK version has no public position+velocity API. Keep this
        # compatibility hook in the repository rather than editing SDK caches.
        try:
            self.control = robot._kuavo_core._control.kuavo_arm_control
            self.original_publish = self.control.pub_control_robot_arm_traj
            if not callable(self.original_publish):
                raise AttributeError("pub_control_robot_arm_traj")
            if isinstance(getattr(self.original_publish, "__self__", None), SDKArmVelocityTransport):
                raise RuntimeError("SDK arm velocity transport already has an owner")
            self.robot_info = robot._robot_info
            for name in ("_pub_ctrl_arm_traj", "_pub_ctrl_arm_traj_arm_collision"):
                if not callable(getattr(self.control, name).publish):
                    raise AttributeError(name)
        except AttributeError as exc:
            raise RuntimeError("SDK arm transport cannot preserve joint velocity") from exc
        self.robot = robot
        self.context = local()
        self.lock = RLock()
        self.closed = False
        self.publish_hook = self._publish
        self.control.pub_control_robot_arm_traj = self.publish_hook

    def send(self, position, velocity):
        position = validate_arm_vector(position, "Arm position")
        velocity = validate_arm_vector(velocity, "Arm velocity")
        with self.lock:
            if self.closed:
                raise RuntimeError("SDK arm velocity transport is closed")
            if getattr(self.context, "command", None) is not None:
                raise RuntimeError("Nested SDK arm velocity command")
            command = {"requested": position, "velocity": velocity, "actual": None}
            self.context.command = command
            try:
                accepted = self.robot.control_arm_joint_positions(position.tolist())
                if accepted is False or command["actual"] is None:
                    raise RuntimeError("SDK rejected the arm position/velocity command")
                return command["actual"]
            finally:
                del self.context.command

    def _publish(self, position):
        command = getattr(self.context, "command", None)
        if command is None:
            return self.original_publish(position)
        if command["actual"] is not None:
            raise RuntimeError("SDK published an arm command more than once")

        import rospy
        from sensor_msgs.msg import JointState

        # Position has already passed the real SDK's physical-limit clipping.
        position = validate_arm_vector(position, "SDK arm position")
        velocity = command["velocity"].copy()
        low, high = (np.asarray(v, dtype=float) for v in self.robot_info.get_arm_joint_limits())
        clipped = np.abs(position - command["requested"]) > 1e-10
        outward = (((position >= high - 1e-10) & (velocity > 0))
                   | ((position <= low + 1e-10) & (velocity < 0)))
        velocity[clipped | outward] = 0.0

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["arm_joint_" + str(i) for i in range(14)]
        # /kuavo_arm_traj uses degrees and degrees/s; VR candidates use radians.
        msg.position = np.rad2deg(position).tolist()
        msg.velocity = np.rad2deg(velocity).tolist()
        publisher = (self.control._pub_ctrl_arm_traj_arm_collision
                     if self.control.arm_collision_enable else self.control._pub_ctrl_arm_traj)
        publisher.publish(msg)
        command["actual"] = (position.copy(), velocity.copy())
        return True

    def close(self):
        with self.lock:
            if not self.closed:
                if self.control.pub_control_robot_arm_traj is self.publish_hook:
                    self.control.pub_control_robot_arm_traj = self.original_publish
                self.closed = True
