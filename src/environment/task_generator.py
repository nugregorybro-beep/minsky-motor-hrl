import numpy as np

class TaskGenerator:
    """
    Handles the randomization of workspace objects to ensure 
    statistical robustness during Phase 4 Validation.
    """
    def __init__(self, workspace_limits):
        self.x_limits = workspace_limits['x']
        self.y_limits = workspace_limits['y']
        self.base_z = workspace_limits['z']

    def get_random_goal(self, noise_cm=10.0):
        """
        Generates a randomized block position within the workspace.
        noise_cm: The range of randomization (default 10cm).
        """
        # Convert cm to meters for PyBullet
        noise_m = noise_cm / 100.0
        
        # Calculate base center of workspace
        center_x = sum(self.x_limits) / 2
        center_y = sum(self.y_limits) / 2
        
        # Apply uniform noise
        target_x = center_x + np.random.uniform(-noise_m, noise_m)
        target_y = center_y + np.random.uniform(-noise_m, noise_m)
        
        return np.array([target_x, target_y, self.base_z])