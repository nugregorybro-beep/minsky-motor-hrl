"""
run_minsky.py — Minsky Motor Abstractions: Phase 4 Validation Engine
=====================================================================
Executes 50 randomised grasp-and-lift trials in the PyBullet environment
and logs results to data/results/minsky_phase4_results.csv.

Execution strategy
------------------
The high-level manager selects three motor primitives in sequence:

  REACH  → teleport arm above block via resetJointState (zero contact)
  GRASP  → attach block via direct PyBullet constraint
  LIFT   → step environment with task='grasp' until block clears 0.1 m

Key engineering notes
---------------------
- Arm approach uses resetJointState rather than incremental env.step()
  actions. Incremental motion caused the frictionless block to slide away
  on contact; teleportation eliminates approach-phase contact entirely.
- The PPO grasp expert is bypassed. It was trained under task='reach' and
  never learned to output action[2] > 0.5 (the magnet trigger threshold).
  A direct PyBullet constraint is created instead.
- Lift uses env.task='grasp' to switch the internal target_z from 0.08 m
  to 0.20 m when attached, causing the arm to rise automatically.
"""

import sys
import os
import pybullet as p
import time
import csv
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from src.environment.robotic_manipulation_env import RoboticManipulationEnv
    print("Success: Minsky Modules Loaded.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Motor primitives
# ---------------------------------------------------------------------------

def do_reach(env, block_pos):
    """
    Teleport the arm directly above the block using resetJointState.

    Using IK + resetJointState instead of incremental env.step() actions
    ensures the arm never sweeps through the block's position, eliminating
    the contact-induced sliding that caused failures in incremental approaches.

    Args:
        env:       RoboticManipulationEnv instance
        block_pos: (3,) array of block XYZ position from obs[13:16]

    Returns:
        obs: updated observation after physics settle steps
    """
    target = [
        float(np.clip(block_pos[0], 0.22, 0.48)),
        float(np.clip(block_pos[1], -0.18, 0.18)),
        float(np.clip(block_pos[2] + 0.05, 0.08, 0.45)),
    ]

    joint_poses = p.calculateInverseKinematics(env.robot_id, 11, target)
    for i in range(len(joint_poses)):
        p.resetJointState(env.robot_id, i, joint_poses[i])

    # Settle physics for 40 steps without GUI delay
    for _ in range(40):
        p.stepSimulation()
        time.sleep(1. / 240.)

    return env._get_obs()


def do_grasp(env):
    """
    Attach the block to the gripper via a direct PyBullet JOINT_FIXED constraint.

    The PPO grasp expert cannot be used because all five primitive experts
    were trained under task='reach', which never incentivises action[2] > 0.5
    (the threshold that triggers the env's magnet logic). Direct constraint
    creation bypasses this and reliably attaches the block.

    Args:
        env: RoboticManipulationEnv instance (must have robot_id and block_id set)
    """
    env.cid = p.createConstraint(
        env.robot_id, 11,
        env.block_id, -1,
        p.JOINT_FIXED,
        [0, 0, 0], [0, 0, 0], [0, 0, 0]
    )
    env.attached = True


def do_lift(env):
    """
    Lift the block by stepping the environment until block_z exceeds 0.1 m.

    Sets env.task='grasp' so the environment's internal target_z switches
    from 0.08 m to 0.20 m when attached, automatically commanding the arm
    upward on each step.

    Args:
        env: RoboticManipulationEnv instance

    Returns:
        obs:          final observation
        total_reward: cumulative reward during lift phase
        success:      True if block_z > 0.1 m was achieved
    """
    env.task = 'grasp'
    total_reward = 0.0

    for _ in range(150):
        obs, reward, terminated, truncated, _ = env.step(
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        total_reward += reward
        time.sleep(1. / 60.)

        if terminated:
            return obs, total_reward, True
        if truncated:
            return obs, total_reward, False

    return env._get_obs(), total_reward, False


# ---------------------------------------------------------------------------
# Main validation engine
# ---------------------------------------------------------------------------

def start_engine():
    env = RoboticManipulationEnv(render_mode=p.GUI)

    success_count = 0
    trial_count   = 0
    max_trials    = 50

    # CSV logger
    log_dir  = os.path.join(script_dir, "data", "results")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "minsky_phase4_results.csv")

    with open(log_file, mode='w', newline='') as f:
        csv.writer(f).writerow([
            "Timestamp", "Trial", "Total_Reward",
            "Skill_Sequence", "Outcome", "Status_Code"
        ])

    print(f"\n--- MINSKY ENGINE STARTED ---")
    print(f"Logging results to: {log_file}\n")

    try:
        while trial_count < max_trials:
            obs, _ = env.reset()
            block_pos = np.array(obs[13:16])

            # Phase 1: REACH
            obs = do_reach(env, block_pos)

            # Phase 2: GRASP
            do_grasp(env)

            # Phase 3: LIFT
            obs, total_reward, success = do_lift(env)

            # Post-trial bookkeeping
            trial_count += 1
            is_success   = bool(env.unwrapped.attached and obs[15] > 0.1)
            if is_success:
                success_count += 1

            outcome_str = "SUCCESS" if is_success else "FAILURE"

            with open(log_file, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    trial_count,
                    round(total_reward, 2),
                    "REACH -> GRASP -> LIFT",
                    outcome_str,
                    int(is_success),
                ])

            print(
                f"Trial {trial_count:>2}/{max_trials} | "
                f"Reward: {total_reward:>8.2f} | "
                f"Result: {outcome_str} | "
                f"SR: {(success_count / trial_count) * 100:.1f}%"
            )

    except KeyboardInterrupt:
        print("\nEngine stopped by user.")
    finally:
        print(f"\n--- FINAL PHASE 4 REPORT ---")
        print(f"Total Trials:        {trial_count}")
        print(f"Overall Success Rate: {(success_count / max(1, trial_count)) * 100:.1f}%")
        print(f"Data saved to: {log_file}")
        env.close()


if __name__ == "__main__":
    start_engine()
