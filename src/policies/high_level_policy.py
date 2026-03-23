import torch
import torch.nn as nn
import torch.optim as optim

class HighLevelManager(nn.Module):
    def __init__(self, state_dim, num_primitives):
        super(HighLevelManager, self).__init__()
        
        # Minsky-inspired Society of Mind Manager
        # We use nn.Sequential directly as the top-level attribute.
        # This matches the "0.weight", "2.weight" keys in your .pth file.
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_primitives),
            nn.Softmax(dim=-1)
        )
        
        # Optimizer: Adam with hierarchical stability
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        # Ensure input is a FloatTensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        # Pass directly into the sequential block
        # We use [0] because the state_dict keys '0.weight' imply 
        # the model was saved as the sequential object itself.
        return self.network(x)

    def load_state_dict(self, state_dict, strict=True):
        """
        Overriding load_state_dict to handle the case where the model 
        was saved as a raw Sequential container.
        """
        try:
            # Try loading into self.network if the keys match
            self.network.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            # Fallback if names are slightly off
            super().load_state_dict(state_dict, strict=False)

    def select_action(self, state):
        """
        Used during Phase 4 validation to pick the best motor skill.
        Returns the index of the primitive (0-4).
        """
        self.eval() 
        with torch.no_grad(): 
            if isinstance(state, torch.Tensor):
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
            probs = self.forward(state_tensor)
            # Pick the skill with the highest probability
            action_idx = torch.argmax(probs, dim=1).item()
        return action_idx

    def update(self, state, action_idx, reward):
        self.train()
        self.optimizer.zero_grad()
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        
        if not isinstance(action_idx, torch.Tensor):
            action_idx = torch.tensor(action_idx)
            
        log_prob = dist.log_prob(action_idx)
        loss = -(log_prob * reward)
        loss_val = loss.mean()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss_val.item()