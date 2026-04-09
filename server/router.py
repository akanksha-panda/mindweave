from server.agents.cognitive import cognitive_agent
from server.agents.behavioral import behavioral_agent
from server.agents.emotional import emotional_agent
from server.agents.adaptive import adaptive_agent
import random

from server.rl.state_encoder import encode_state

ACTION_MAP = {
    0: behavioral_agent,
    1: cognitive_agent,
    2: emotional_agent
}

def route(state, user_input, model=None):
    text = user_input.lower().replace("'", "")
    
    # Extract State Variables
    emotion = state.get("emotion") or state.get("emotion_category") or "neutral"
    intent = state.get("intent", "unknown")
    sentiment = state.get("sentiment", 0)
    energy = state.get("energy", 1)
    distortion = state.get("distortion", 0)

    # ==========================================
    # 1. . HARD RULES & SPECIAL INTENTS
    # ==========================================
    if energy == 0:
            return behavioral_agent(state, user_input) 
    # --- GREETINGS ---
    if intent == "greeting":
        return {"type": "emotional", "text": random.choice(["Hi!", "Hey 🙂", "Hello!"])}

    # --- PHILOSOPHICAL ---
    PHILOSOPHICAL_PATTERNS = ["meaning", "point", "why do we", "deserve", "worthy"]
    if any(p in text for p in PHILOSOPHICAL_PATTERNS):
        return adaptive_agent(state, user_input, mode="philosophical")

    # --- . EMOTIONAL SUPPORT OVERRIDE ---
    # Forces empathy if the user explicitly asks for support or intent is detected as emotional
    if intent == "emotional" or "support" in text:
        if model is not None:
            try:
                action_id, *_ = model.get_action(state)
                # If PPO tries to be too logical (1) during an emotional crisis, override to Emotional (2)
                if action_id == 1: 
                    return emotional_agent(state, user_input)
                return ACTION_MAP.get(action_id, emotional_agent)(state, user_input)
            except:
                return emotional_agent(state, user_input)
        return emotional_agent(state, user_input)

    # ==========================================
    # 2. . RL AGENT (PPO) - PRIMARY DECISION MAKER
    # ==========================================
    if model is not None:
        try:
            # Check for high-intensity states
            is_emotional = emotion in ["sadness", "anxiety", "fear", "pain", "vulnerable"]
            is_distorted = distortion >= 5
            
            # . CHIT-CHAT GATE: Bypass RL for purely neutral statements
            if not is_emotional and not is_distorted and sentiment > -0.1:
                return adaptive_agent(state, user_input)

            # Get RL Action from PPO Policy
            action_id, *_ = model.get_action(state)
            
            # --- . INTELLIGENT CORRECTIONS (The "Safety Net") ---
            
            # A. LOW ENERGY OVERRIDE (Behavioral Superpower)
            # If energy is 0, brain-heavy tasks (Cognitive) are often too much.
            if energy == 0:
                # Force Behavioral Agent to suggest small physical steps
                return behavioral_agent(state, user_input)

            # B. SADNESS VS COGNITIVE
            if emotion == "sadness" and action_id == 1:
                # If PPO picked logic for sadness, only allow if energy is decent
                if energy <= 1: 
                    return emotional_agent(state, user_input)
            
            return ACTION_MAP.get(action_id, emotional_agent)(state, user_input)

        except Exception as e:
            print(f". PPO Routing Error: {e}")

    # ==========================================
    # 3. . RULE-BASED FALLBACK
    # ==========================================
    if distortion >= 7:
        return cognitive_agent(state, user_input)
    
    if energy == 0:
        return behavioral_agent(state, user_input)

    if emotion in ["sadness", "anxiety", "fear", "vulnerable"]:
        return emotional_agent(state, user_input)

    return adaptive_agent(state, user_input)