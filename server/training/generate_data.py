import json
import random
import os
import numpy as np

# Import your emotion datasets
# Assuming your file is in server/emotions/emotion_data.py
from server.emotions.emotion_data import ALL_EMOTIONS, FLAT_EMOTION_MAP, CATEGORY_STATE_HINTS

LOG_FILE = "logs/trajectories_v3.jsonl"
os.makedirs("logs", exist_ok=True)

# =========================
# . SMART STATE GENERATION
# =========================
def generate_state():
    # 1. Pick a random word or phrase from your massive list
    emotion_word = random.choice(ALL_EMOTIONS)
    
    # 2. Get the metadata for that word
    meta = FLAT_EMOTION_MAP.get(emotion_word, {
        "category": "misc_positive", 
        "polarity": "positive"
    })
    
    category = meta["category"]
    
    # 3. Get the "Hint" for this category (e.g., how it affects mood/energy)
    hint = CATEGORY_STATE_HINTS.get(category, {"mood": 0, "energy": 0, "distortion": 0, "sentiment": 0})
    
    # 4. Generate base values with a bit of "noise" so every 'sad' state isn't identical
    # Neutral starting point is 5/10
    mood = clip(5 + hint["mood"] + random.randint(-1, 1), 1, 10)
    energy = clip(1 + hint["energy"], 0, 2) # Energy is 0, 1, or 2
    distortion = clip(5 + hint["distortion"] + random.randint(-1, 1), 0, 10)
    
    # Sentiment is a float between -1.0 and 1.0
    sentiment = np.clip(hint["sentiment"] + random.uniform(-0.1, 0.1), -1.0, 1.0)

    return {
        "mood": int(mood),
        "energy": int(energy),
        "distortion": int(distortion),
        "sentiment": float(sentiment),
        "emotion": emotion_word,    # The specific word (e.g., "blue monday")
        "category": category        # The parent category (e.g., "sadness")
    }

def clip(val, low, high):
    return max(low, min(high, val))

# =========================
# . REFINED ACTION POLICY
# =========================
def choose_action(state):
    cat = state["category"]
    
    # Rule 1: High Distortion -> Cognitive
    if state["distortion"] > 6:
        return {"type": "cognitive", "intensity": 2 if state["distortion"] > 8 else 1}
    
    # Rule 2: Low Energy -> Behavioral
    if state["energy"] == 0:
        return {"type": "behavioral", "intensity": 1}
    
    # Rule 3: Negative Emotions (Afraid, Angry, Sad) -> Emotional Support
    if cat in ["afraid", "angry", "sad", "pain", "sadness", "anxiety"]:
        return {"type": "emotional", "intensity": 2}
    
    # Rule 4: Already Positive/Love -> Maintain
    if cat in ["love", "joy", "positive", "gratitude"]:
        return {"type": "emotional", "intensity": 1}

    return {"type": "cognitive", "intensity": 1}

# =========================
# . ENV STEP SIMULATION
# =========================
def simulate_step(state, action):
    new_state = state.copy()
    
    if action["type"] == "behavioral":
        new_state["energy"] = clip(state["energy"] + 1, 0, 2)
        new_state["mood"] = clip(state["mood"] + 1, 1, 10)
        
    elif action["type"] == "cognitive":
        # Cognitive reduces distortion
        reduction = 2 if action["intensity"] == 2 else 1
        new_state["distortion"] = clip(state["distortion"] - reduction, 0, 10)
        new_state["mood"] = clip(state["mood"] + 1, 1, 10)
        
    elif action["type"] == "emotional":
        new_state["mood"] = clip(state["mood"] + 2, 1, 10)
        new_state["distortion"] = clip(state["distortion"] - 1, 0, 10)

    return new_state

# =========================
# . REWARD FUNCTION
# =========================
def compute_reward(state, action, new_state):
    reward = 0.0
    
    # Reward for lifting mood
    reward += (new_state["mood"] - state["mood"]) * 1.0
    
    # Reward for reducing distortion
    reward += (state["distortion"] - new_state["distortion"]) * 0.5
    
    # Reward for energy gain
    reward += (new_state["energy"] - state["energy"]) * 0.5
    
    # Penalty: Don't give "Cognitive" advice to someone who is "Exhausted" (Energy 0)
    if state["energy"] == 0 and action["type"] == "cognitive":
        reward -= 2.0
        
    return round(reward, 2)

# =========================
# . GENERATE DATASET
# =========================
def generate_dataset(n=2000):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for _ in range(n):
            state = generate_state()
            action = choose_action(state)
            new_state = simulate_step(state, action)
            reward = compute_reward(state, action, new_state)

            log_entry = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": new_state
            }
            f.write(json.dumps(log_entry) + "\n")

    print(f".Generated {n} interactions using ALL_EMOTIONS and CATEGORY_STATE_HINTS")

if __name__ == "__main__":
    generate_dataset(2000)