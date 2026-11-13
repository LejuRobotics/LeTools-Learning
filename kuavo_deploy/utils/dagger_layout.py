"""DAgger slices derived from the environment's existing action splitter."""
import numpy as np


class DaggerLayout:
    def __init__(self, env):
        self.dim = env.action_space.shape[0]
        self.arm_dof = env.arm_dof_per_side
        parts = env._split_action(np.zeros(self.dim))
        self.sides = tuple(env._controlled_sides)
        self._arms = {
            side: slice(parts[side]["eef_slice"].start - len(parts[side]["joints"]),
                        parts[side]["eef_slice"].stop)
            for side in self.sides
        }
        self.base_velocity_dof = len(parts["base_velocity"])
        self.state_dim = self.dim - self.base_velocity_dof
        self.lower_body_dof = len(parts["lower_body"])
        self.lower_body_slice = slice(self.state_dim - self.lower_body_dof, self.state_dim)

    def action_slice(self, side):
        return self._arms[side]

    def validate(self, action):
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (self.dim,) or not np.isfinite(value).all():
            raise ValueError(f"DAgger action must contain {self.dim} finite values, got {value.shape}")
        return value

    def vr_action(self, joints, eef):
        parts = []
        for side in self.sides:
            i = 0 if side == "left" else 1
            parts.extend((joints[i * 7:(i + 1) * 7], eef[i:i + 1]))
        return np.concatenate(parts)
