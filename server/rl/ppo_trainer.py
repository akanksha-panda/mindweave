# server/rl/ppo_trainer.py

import torch
import torch.optim as optim


class PPOTrainer:
    def __init__(self, model, lr=3e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.entropy_coef = 0.1
        self.value_coef = 0.5

    # =========================
    # . GAE ADVANTAGE
    # =========================
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return advantages, returns

    # =========================
    # . PPO UPDATE
    # =========================
    def update(self, states, actions, old_log_probs, returns, advantages):

        states = torch.stack(states)
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(old_log_probs).detach()
        advantages = torch.tensor(advantages).detach()
        returns = torch.tensor(returns).detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # . multiple epochs
            self.optimizer.zero_grad(set_to_none=True)
            logits, values = self.model(states)
            probs = torch.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values.squeeze()).pow(2).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()