from src.primitives.primitive_base import Option

class ReleaseOption(Option):
    def __init__(self, env, weight_path="models/release_expert.pth"):
        super().__init__(name="Release", env=env, weight_path=weight_path)

    def get_reward(self, state, next_state):
        """
        Intrinsic Reward for Releasing:
        Positive reward for opening the gripper while the block remains stable.
        """
        gripper_width = next_state[6] # 0.0 is closed, 1.0 is fully open
        block_velocity = np.linalg.norm(next_state[11:14]) # Velocity indices
        
        reward = 0.0
        if gripper_width > 0.8:
            reward += 5.0
            
        # Penalty if the block is still moving (unstable release)
        if block_velocity > 0.1:
            reward -= 2.0
            
        return reward