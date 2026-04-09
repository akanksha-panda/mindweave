# server/rl/ppo_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    

    def forward(self, x):
        # . ADD THIS GUARD CLAUDE
        if isinstance(x, dict):
            # Use your encoder to turn the dictionary into a 388-dim list
            from server.rl.state_encoder import encode_state
            x = encode_state(x)
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure input is a tensor and has a batch dimension
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Now the Linear layers will work
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def get_action(self, state):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value, dist.entropy()