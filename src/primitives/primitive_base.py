from abc import ABC, abstractmethod
from stable_baselines3 import PPO

class Option(ABC):
    """Base class for motor primitives."""
   def __init__(self, name, env, weight_path=None):
        self.name = name
        self.env = env
        self.weight_path = weight_path
        
        # Load pre-trained expert if weights exist, otherwise create a new one
        if weight_path and (os.path.exists(weight_path + ".zip") or os.path.exists(weight_path)):
            self.policy = PPO.load(weight_path, env=self.env)
            print(f"[{self.name.upper()}] Expert loaded from {weight_path}")
        else:
            self.policy = PPO("MlpPolicy", env, verbose=1)

    @abstractmethod
    def get_reward(self, state, next_state):
        pass

    def execute(self, state):
        action, _ = self.policy.predict(state)
        return action