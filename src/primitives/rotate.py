import numpy as np
from src.primitives.primitive_base import Option

class RotateOption(Option):
    def __init__(self, env, weight_path="models/rotate_expert.pth"):
        super().__init__(name="Rotate", env=env, weight_path=weight_path)

    def get_reward(self, state, next_state):
        """
        Intrinsic Reward for Rotating:
        Reward is based on the alignment of the block's Euler angles with a target orientation.
        """
        # Assuming indices 7-10 are the block's orientation quaternion
        current_orientation = next_state[7:11]
        target_orientation = np.array([0, 0, 0, 1]) # Identity quaternion
        
        # Dot product of quaternions gives a measure of alignment
        alignment = np.abs(np.dot(current_orientation, target_orientation))
        return 10.0 * alignment