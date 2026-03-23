import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from src.environment.robotic_manipulation_env import RoboticManipulationEnv
from src.policies.high_level_policy import HighLevelManager

# 1. Setup Environment, Logging, and Dimensions
env = RoboticManipulationEnv()
state_dim = env.observation_space.shape[0]
writer = SummaryWriter("logs/hierarchy/run_final")

# Names for the Sequence Tracker
PRIM_NAMES = {0: "REACH", 1: "GRASP", 2: "PUSH", 3: "ROTATE", 4: "RELEASE"}

# Initialize Manager
manager = HighLevelManager(state_dim, num_primitives=5)

# 2. Load the Society of Mind Library
primitive_library = {
    0: PPO.load("models/reach_expert.pth"),
    1: PPO.load("models/grasp_expert.pth"),
    2: PPO.load("models/push_expert.pth"),
    3: PPO.load("models/rotate_expert.pth"),
    4: PPO.load("models/release_expert.pth")
}
def train_manager():
    print("--- PHASE 3: ACTIVE HIERARCHICAL LEARNING (REWARD HARDENING) ---")
    
    for episode in range(3000): 
        obs, _ = env.reset()
        done = False
        total_episode_reward = 0
        decisions = [] 
        
        # Skill sequence tracking
        skill_count = 0
        max_skills = 3
        last_prim_idx = -1  
        used_skills = set() 
        
        while not done and skill_count < max_skills:
            skill_count += 1
            state_tensor = torch.FloatTensor(obs).unsqueeze(0) 
            
            # Manager selects a primitive
            prim_probs = manager(state_tensor)

            # --- ADVANCED ACTION MASKING ---
            mask = torch.ones_like(prim_probs)
            if last_prim_idx != -1:
                mask[0, last_prim_idx] = 0.0 
            
            if 0 in used_skills:
                mask[0, 0] = 0.0
                
            prim_probs = (prim_probs * mask) + 1e-8
            prim_probs = prim_probs / prim_probs.sum()
            
            # Numerical safety check
            if torch.isnan(prim_probs).any():
                prim_probs = torch.ones_like(prim_probs) / 5.0

            dist = torch.distributions.Categorical(prim_probs)
            selected_prim_idx = dist.sample()
            last_prim_idx = selected_prim_idx.item()
            used_skills.add(last_prim_idx) 
            
            prim_name = PRIM_NAMES[last_prim_idx]
            decisions.append(prim_name)
            
            # Pre-execution state for shaping
            block_pos_before = obs[13:16].copy()
            dist_before = np.linalg.norm(obs[0:3] - block_pos_before)
            
            # Execute chosen primitive (80 step horizon)
            active_primitive = primitive_library[last_prim_idx]
            cumulative_env_reward = 0
            
            for _ in range(80): 
                action, _ = active_primitive.predict(obs, deterministic=True)
                action = action.flatten() 
                if action.shape[0] == 2:
                    action = np.append(action, 0.0) 
                
                obs, reward, terminated, truncated, _ = env.step(action)
                cumulative_env_reward += reward
                
                if terminated or truncated:
                    done = True
                    break
            
            # --- MANAGER REWARD HARDENING & SHAPING ---
            # DSC FIX: Cap the environment penalty to prevent gradient explosion
            env_reward_capped = max(cumulative_env_reward, -50.0)
            shaped_bonus = 0
            
            dist_after = np.linalg.norm(obs[0:3] - obs[13:16])
            is_attached = env.unwrapped.attached
            
            if prim_name == "REACH":
                if dist_after < dist_before and dist_after > 0.05:
                    shaped_bonus += 20.0 
                elif dist_after <= 0.05:
                    shaped_bonus -= 10.0 
            
            elif prim_name == "GRASP":
                if is_attached:
                    shaped_bonus += 150.0 
                    print(f"!!! SUCCESS AT EPISODE {episode}: {decisions} !!!")
                elif dist_after <= 0.05:
                    shaped_bonus += 30.0  
                else:
                    shaped_bonus -= 15.0 
            
            elif prim_name == "ROTATE":
                if is_attached:
                    height_gain = obs[15] - block_pos_before[2]
                    if height_gain > 0.01:
                        shaped_bonus += 200.0  
                        print(f"--> LIFT! Ep {episode} | Gain: {height_gain:.3f}")
                else:
                    shaped_bonus -= 20.0 

            elif prim_name in ["RELEASE", "PUSH"]:
                if not is_attached:
                    shaped_bonus -= 25.0 

            # Efficiency Tax
            if skill_count == max_skills and not is_attached:
                shaped_bonus -= 30.0 

            # --- STABLE BACKPROPAGATION ---
            # Total signal: Capped Env + Shaped Bonus normalized by 100
            manager_signal = (env_reward_capped + shaped_bonus) / 100.0
            
            loss_val = manager.update(state_tensor, selected_prim_idx, manager_signal)
            
            if np.isnan(loss_val):
                print(f"CRITICAL: NaN at Ep {episode}. Terminating.")
                return

            total_episode_reward += cumulative_env_reward 

        # LOGGING
        writer.add_scalar("Reward/Episode", total_episode_reward, episode)
        writer.add_scalar("Loss/Manager", loss_val, episode)
        
        if episode % 5 == 0:
            sequence_str = " -> ".join(decisions)
            print(f"Ep {episode} | Env Reward: {total_episode_reward:.1f} | Seq: [{sequence_str}]")
            
        if episode % 100 == 0:
            # Checkpoint with episode number for DSc evaluation
            torch.save(manager.state_dict(), f"models/hierarchy/minsky_mgr_ep{episode}.pth")

    writer.close()

if __name__ == "__main__":
    os.makedirs("models/hierarchy", exist_ok=True)
    train_manager()