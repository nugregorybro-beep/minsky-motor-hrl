import numpy as np
from src.primitives.primitive_base import Option

class ReachPrimitive(Option):
    def __init__(self, env):
        super().__init__(name="reach", env=env)
        self.success_threshold = 0.05

    def get_reward(self, state, next_state):
        gripper_pos = np.array(next_state[0:3])
        target_pos = np.array(next_state[13:16])
        dist = np.linalg.norm(gripper_pos - target_pos)
        reward = -dist
        if dist < self.success_threshold:
            reward += 5.0
        return reward