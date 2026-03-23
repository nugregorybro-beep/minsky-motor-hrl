import numpy as np
from src.primitives.primitive_base import Option

class PushOption(Option):
    def __init__(self, env, weight_path="models/push_expert.pth"):
        super().__init__(name="Push", env=env, weight_path=weight_path)

    def get_reward(self, state, next_state):
        """
        Intrinsic Reward for Pushing:
        Based on the horizontal displacement of the block toward a goal.
        """
        block_pos = next_state[3:6]
        goal_pos = np.array([0.6, 0.0, 0.05]) # Example goal center
        
        distance = np.linalg.norm(block_pos[:2] - goal_pos[:2])
        reward = -distance
        
        # Check if block is still on the table (z height ~ 0.05)
        if block_pos[2] < 0.04: # Block fell off
            reward -= 50.0
            
        return reward