"""
robotic_manipulation_env.py — PyBullet Gymnasium environment for robotic manipulation.

Observation space (16-D):
    obs[0:3]   — gripper XYZ position (link 11)
    obs[3:7]   — gripper orientation quaternion
    obs[7:13]  — joint angles for joints 0-5
    obs[13:16] — block XYZ position

Action space (3-D, continuous [-1, 1]):
    action[0]  — delta X (scaled by speed=0.02 m/step)
    action[1]  — delta Y (scaled by speed=0.02 m/step)
    action[2]  — gripper command (>0.5 = attach, <-0.5 = release)

Notes:
    - Gripper Z is internally managed: 0.08 m normally, 0.20 m when attached
      and task='grasp'.
    - The block (cube.urdf, globalScaling=0.5) settles to Z≈0.25–0.50 m after
      the 50-step physics warmup. Its half-extents are ~0.25 m, so the gripper
      at Z=0.08 sits inside the block volume when XY-aligned.
    - The magnet trigger (action[2]>0.5, 3D dist<0.05) is unreachable via
      env.step() alone because gripper Z=0.08 and block centre Z≈0.25–0.50
      gives a minimum 3D distance of ~0.17 m. Use direct PyBullet constraints
      (as in run_minsky.py) to reliably attach the block.
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class RoboticManipulationEnv(gym.Env):

    # Maps high-level primitive index (0–4) to a [dx, dy, gripper] action vector.
    # Used when an integer action is passed to step() instead of a continuous vector.
    PRIMITIVE_ACTIONS = {
        0: np.array([ 1.0,  0.0,  0.0]),   # reach:   move forward (+X)
        1: np.array([ 0.0,  0.0,  1.0]),   # grasp:   close gripper
        2: np.array([ 1.0,  0.0,  0.0]),   # push:    push forward (+X)
        3: np.array([ 0.0,  1.0,  0.0]),   # rotate:  lateral sweep (+Y)
        4: np.array([ 0.0,  0.0, -1.0]),   # release: open gripper
    }

    def __init__(self, render_mode=p.DIRECT, task='reach'):
        super().__init__()
        self.client = p.connect(render_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.task = task
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        self.robot_id = None
        self.block_id = None
        self.cid      = None
        self.attached = False
        self.steps    = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.attached = False
        self.cid      = None
        self.steps    = 0

        # Park arm at safe home position
        safe_start_pos = [0.3, 0.0, 0.2]
        joint_poses = p.calculateInverseKinematics(self.robot_id, 11, safe_start_pos)
        for i in range(len(joint_poses)):
            p.resetJointState(self.robot_id, i, joint_poses[i])

        # Spawn block with ±10 cm spatial noise (X: 0.3–0.45, Y: ±0.1)
        rx = np.random.uniform(0.3, 0.45)
        ry = np.random.uniform(-0.1, 0.1)
        self.block_id = p.loadURDF("cube.urdf", [rx, ry, 0.03], globalScaling=0.5)

        # Physics warmup — block settles to Z≈0.25–0.50 depending on spawn position
        for _ in range(50):
            p.stepSimulation()

        return self._get_obs(), {}

    def _get_obs(self):
        state = p.getLinkState(self.robot_id, 11)
        pos, orn = state[0], state[1]
        joints = [p.getJointState(self.robot_id, i)[0] for i in range(6)]
        block_pos, _ = p.getBasePositionAndOrientation(self.block_id)
        return np.concatenate([pos, orn, joints, block_pos]).astype(np.float32)

    def step(self, action):
        # Accept a primitive index (int) or a continuous action vector
        if isinstance(action, (int, np.integer)):
            action = self.PRIMITIVE_ACTIONS.get(int(action), np.zeros(3))

        self.steps += 1
        obs       = self._get_obs()
        curr_pos  = obs[0:3]
        block_pos = obs[13:16]

        speed    = 0.02
        target_z = 0.2 if (self.attached and self.task == 'grasp') else 0.08

        new_pos = [
            np.clip(curr_pos[0] + action[0] * speed, 0.2, 0.5),
            np.clip(curr_pos[1] + action[1] * speed, -0.2, 0.2),
            target_z,
        ]

        joint_poses = p.calculateInverseKinematics(self.robot_id, 11, new_pos)
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL, joint_poses[i]
            )

        # Magnet logic
        dist_to_block = np.linalg.norm(curr_pos - block_pos)
        if action[2] > 0.5 and dist_to_block < 0.05 and not self.attached:
            self.cid = p.createConstraint(
                self.robot_id, 11, self.block_id, -1,
                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0]
            )
            self.attached = True

        if action[2] < -0.5 and self.attached:
            p.removeConstraint(self.cid)
            self.attached = False

        p.stepSimulation()

        new_obs    = self._get_obs()
        reward     = self._calculate_reward(new_obs, dist_to_block)
        terminated = bool(self.attached and new_obs[15] > 0.1)
        truncated  = bool(self.steps >= 200)

        return new_obs, reward, terminated, truncated, {}

    def _calculate_reward(self, obs, dist):
        block_z = obs[15]

        if self.task == 'reach':
            return -dist

        elif self.task == 'grasp':
            reward = -dist
            if self.attached:
                reward += 10.0
                if block_z > 0.1:
                    reward += 50.0
            return reward

        elif self.task == 'push':
            return obs[13] * 10.0

        elif self.task == 'rotate':
            return 5.0 * (1.0 - np.abs(obs[3]))

        elif self.task == 'release':
            return 20.0 if (not self.attached and block_z < 0.06) else -1.0

        return 0.0

    def close(self):
        p.disconnect()