"""
metrics.py — Standalone evaluation script for the MMA framework.
Loads trained expert models and runs N evaluation episodes,
reporting success rate and average steps to success.

Run from the project root:
    python src/evaluation/metrics.py
"""
import os
import torch
import numpy as np
import pandas as pd
import pybullet as p
from stable_baselines3 import PPO
from src.environment.robotic_manipulation_env import RoboticManipulationEnv  # fixed import
from src.policies.high_level_policy import HighLevelManager


def run_evaluation(num_episodes: int = 50) -> None:
    env = RoboticManipulationEnv(render_mode=p.DIRECT)

    # Load expert library
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    expert_paths = {
        0: os.path.join(model_dir, "reach_expert.pth"),
        1: os.path.join(model_dir, "grasp_expert.pth"),
    }
    primitive_library = {}
    for idx, path in expert_paths.items():
        if os.path.exists(path):
            primitive_library[idx] = PPO.load(path)
        else:
            print(f"Warning: expert weights not found at {path}")

    state_dim = env.observation_space.shape[0]
    manager = HighLevelManager(state_dim, num_primitives=5)

    results = []
    print(f"--- EVALUATION: {num_episodes} EPISODES ---")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        success = False

        while not done and steps < 200:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                prim_probs = manager(state_tensor)
            selected_prim = torch.argmax(prim_probs, dim=1).item()

            expert = primitive_library.get(selected_prim)
            for _ in range(80):
                if expert:
                    action, _ = expert.predict(obs, deterministic=True)
                else:
                    action = np.zeros(3)
                obs, reward, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated:
                    success = True
                    done = True
                    break
                if truncated:
                    done = True
                    break

        results.append({"episode": ep, "success": success, "steps": steps})
        if ep % 10 == 0:
            print(f"Episode {ep}/{num_episodes} evaluated...")

    env.close()

    df = pd.DataFrame(results)
    success_rate = df["success"].mean() * 100
    successful = df[df["success"]]
    avg_steps = successful["steps"].mean() if len(successful) > 0 else float("nan")

    print("\n--- FINAL RESULTS ---")
    print(f"Task Success Rate:         {success_rate:.2f}%")
    print(f"Average Steps to Success:  {avg_steps:.1f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "results", "experiment_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    run_evaluation()
