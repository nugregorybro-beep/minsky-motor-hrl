import numpy as np
import pybullet as p
from src.primitives.primitive_base import Option

class GraspPrimitive(Option):
    """
    Motor Primitive for grasping and lifting an object.
    """
    def __init__(self, env):
        super().__init__(name="grasp", env=env)
        self.success_threshold = 0.02 # 2cm grip tolerance

    def get_reward(self, state, next_state):
        # next_state[7:9] = Finger joint positions
        # next_state[13:16] = Object Position (XYZ)
        
        # 1. Contact Reward: Check if the robot is actually touching the object
        contacts = p.getContactPoints(bodyA=self.env.robot_id)
        contact_reward = 2.0 if len(contacts) > 0 else 0.0
        
        # 2. Lifting Reward: If the object's Z-height is above the table (0.02m)
        obj_z = next_state[15]
        lift_reward = 10.0 * obj_z if obj_z > 0.03 else 0.0
        
        # 3. Finger Closure: Reward for keeping fingers closed (low values for finger joints)
        finger_pos = next_state[7] + next_state[8]
        closure_reward = 1.0 - finger_pos 

        return contact_reward + lift_reward + closure_reward