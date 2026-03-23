import os
import yaml
import pybullet as p  # <--- ADD THIS LINE
from stable_baselines3 import PPO
from src.environment.robotic_manipulation_env import RoboticManipulationEnv

def load_config():
    with open("config/hrl_config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_experts():
    config = load_config()
    env = RoboticManipulationEnv(render_mode=p.DIRECT) # No GUI for faster training
    
    # List of all 5 Minskyan primitives
    skills = ["reach", "grasp", "push", "rotate", "release"]
    
    print("--- STARTING PHASE 2: PRIMITIVE TRAINING ---")
    
    for skill in skills:
        print(f"\nTraining Expert: {skill.upper()}")
        
        # Fetch hyperparameters from YAML
        lr = config['primitives'][skill]['learning_rate']
        steps = config['primitives'][skill]['timesteps']
        save_path = config['primitives'][skill]['expert_weight_path']
        
        # Initialize PPO
        model = PPO("MlpPolicy", env, learning_rate=lr, verbose=1)
        
        # Train
        model.learn(total_timesteps=steps)
        
        # Save
        os.makedirs("models", exist_ok=True)
        model.save(save_path)
        print(f"Done! {skill} saved to {save_path}")

    print("\n--- ALL EXPERTS TRAINED SUCCESSFULLY ---")

if __name__ == "__main__":
    train_experts()