import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from server.environment import MentalHealthEnv
from server.rl.state_encoder import encode_state
from server.rl.ppo_model import PPOPolicy
from server.rl.ppo_trainer import PPOTrainer

# Import your emotion list for live variety
from server.emotions.emotion_data import ALL_EMOTIONS

# =========================
# . PRE-TRAINING (IMITATION)
# =========================
def pretrain_from_logs(model, trainer, log_path, epochs=30): # Increased epochs for better cloning
    if not os.path.exists(log_path):
        print(f". Log file {log_path} not found. Skipping imitation phase.")
        return

    print(f". Phase 1: Imitation Learning from {log_path}...")
    
    states_data = []
    actions_data = []
    ACTION_TO_ID = {"behavioral": 0, "cognitive": 1, "emotional": 2}

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            s_vec = encode_state(item["state"])
            a_idx = ACTION_TO_ID[item["action"]["type"]]
            
            states_data.append(torch.tensor(s_vec, dtype=torch.float32))
            actions_data.append(a_idx)

    X = torch.stack(states_data)
    Y = torch.tensor(actions_data).long()

    model.train()
    for epoch in range(epochs):
        trainer.optimizer.zero_grad()
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        trainer.optimizer.step()
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch:2} | Imitation Loss: {loss.item():.4f}")

    print(".Phase 1 Complete. Model has 'cloned' the expert rules.")

# =========================
# . LIVE RL TRAINING
# =========================
def train():
    env = MentalHealthEnv()
    state_dim = 388 
    model = PPOPolicy(state_dim)
    trainer = PPOTrainer(model)
    
    os.makedirs("models", exist_ok=True)

    # 1. Start with Imitation Learning
    log_file = "logs/trajectories_v3.jsonl"
    pretrain_from_logs(model, trainer, log_file, epochs=40)

    # 2. Transition to PPO Reinforcement Learning
    ACTION_MAP = {0: "behavioral", 1: "cognitive", 2: "emotional"}
    reward_history = []

    print("\n. Phase 2: Live PPO Reinforcement Learning...")

    # ... inside your train() function ...
    print("\n. Phase 2: Live PPO Reinforcement Learning with Curriculum...")

    for episode in range(1001):
        # . CURRICULUM LOGIC: Force specific scenarios to teach the model
        if episode % 3 == 0:
            start_phrase = "Why does this always happen to me?" # Force Question (Cognitive)
        elif episode % 3 == 1:
            start_phrase = "I am so tired I can't even move." # Force Low Energy (Behavioral)
        else:
            start_phrase = random.choice(ALL_EMOTIONS) # Random (Variety)

        state = env.reset(start_phrase)
        total_reward = 0

        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        for step in range(20):
            s_vec = torch.tensor(encode_state(state), dtype=torch.float32)

            # Sample Action
            action, log_prob, value, entropy = model.get_action(s_vec)

            # --- UPDATED STEP CALL ---
            # We explicitly pass the 'task' so the environment returns a reward
            next_state, reward, done = env.step({
                "type": ACTION_MAP[action], 
                "intensity": 1, 
                "text": "",
                "task": "agent_selection" 
            })
            # -------------------------

            states.append(s_vec)
            actions.append(action)
            log_probs.append(log_prob.detach())
            rewards.append(reward)
            values.append(value.item())
            dones.append(1 if (done or step == 19) else 0)

            total_reward += reward
            state = next_state
            if done: break

        if len(rewards) > 0:
            advantages, returns = trainer.compute_gae(rewards, values, dones)
            loss = trainer.update(states, actions, log_probs, returns, advantages)

        reward_history.append(total_reward)

        if episode % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Epi {episode:4} | Start: {start_phrase[:15]:15} | Avg Reward: {avg_reward:6.2f} | PPO Loss: {loss:.4f}")
            torch.save(model.state_dict(), "models/ppo_mental_health.pt")

    print(".All Training Phases Complete.")
    torch.save(model.state_dict(), "models/ppo_mental_health_final.pt")

if __name__ == "__main__":
    train()