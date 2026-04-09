# server/environment.py


from server.emotions.emotion_mapper import NEGATIONS
from server.emotions.emotion_data import FLAT_EMOTION_MAP
import re

QUESTION_WORDS = ["how", "why", "what", "where", "which", "when"]


# =========================
# . INTENT DETECTION
# =========================
def detect_intent(text):
    text = text.lower().strip()

    if any(x in text for x in ["i feel", "i am", "i'm", "feeling"]):
        return "emotional"

    if text.endswith("?") or text.startswith(tuple(QUESTION_WORDS)):
        return "question"

    return "statement"


# =========================
# . EMOTION FEATURES
# =========================
def build_emotion_features(user_input):
    
    text = user_input.lower()
    tokens = re.findall(r"\b\w+\b", text)

    keyword_emotion = None

    for i, word in enumerate(tokens):
        if word in FLAT_EMOTION_MAP:
            window = tokens[max(0, i - 3):i]
            is_negated = any(w in NEGATIONS for w in window)

            if not is_negated:
                keyword_emotion = word
                break

    if keyword_emotion:
        final_emotion = keyword_emotion
        source = "keyword"
    else:
        from server.emotions.embedding_detector import detect_emotions
        embedding_emotion = detect_emotions(user_input)

        if embedding_emotion and embedding_emotion in tokens:
            final_emotion = embedding_emotion
            source = "embedding"
        else:
            final_emotion = "neutral"
            source = "none"

    # intensity
    if final_emotion in ["failure", "worthless", "useless"]:
        intensity = 1.0
    elif final_emotion in ["anxious", "upset"]:
        intensity = 0.7
    elif final_emotion == "tired":
        intensity = 0.6
    elif final_emotion in ["motivated", "happy"]:
        intensity = 0.3
    else:
        intensity = 0.2

    return {
        "emotion": final_emotion,
        "emotion_source": source,
        "emotion_intensity": intensity
    }


# =========================
# . ENVIRONMENT
# =========================
class MentalHealthEnv:
    
    def __init__(self):
        self.state = None
        self.history = []

    def _base_state(self):
        return {
            "mood": 5,
            "energy": 1,
            "distortion": 5,
            "sentiment": 0,
            "done": False,
            "intent": "statement"
        }

    def reset(self, user_input: str = None):
        self.state = self._base_state()
        self.history = []

        # . BASELINE (Crucial for Trajectory Rewards)
        self.baseline = {
            "mood": self.state["mood"],
            "distortion": self.state["distortion"],
            "energy": self.state["energy"]
        }

        if user_input:
            self._update_state_from_input(user_input)
            # Update baseline again after initial input affects state
            self.baseline = {
                "mood": self.state["mood"],
                "distortion": self.state["distortion"],
                "energy": self.state["energy"]
            }

        return self.state

    def _update_state_from_input(self, user_input):
        if self.state is None:
            self.state = self._base_state()

        # . DEFINE TEXT HERE TO FIX THE ERROR
        text = user_input.lower().strip()

        self.history.append(f"user: {text}")
        self.history = self.history[-5:]

        # 1. Detect intent & emotion
        self.state["intent"] = detect_intent(user_input)
        emotion_features = build_emotion_features(user_input)
        self.state.update(emotion_features)
        
        # 2. . EXPLICIT ENERGY & MOOD LOGIC
        # Adding 'no energy' and 'exhausted' keywords specifically
        low_energy_words = ["no energy", "exhausted", "tired", "can't get up", "fatigue", "drained", "low energy"]
        
        # Check the text for energy depletion
        if any(w in text for w in low_energy_words) or self.state["emotion"] == "tired":
            self.state["energy"] = 0
            self.state["mood"] = max(1, self.state["mood"] - 1)
        else:
            # If they aren't tired, keep energy at a baseline (e.g., 1 or 2)
            self.state["energy"] = 1
        # State drops based on detected negative emotions
        emotion = self.state["emotion"]
        if emotion in ["failure", "worthless", "upset", "pain", "sad"]:
            self.state["mood"] = max(1, self.state["mood"] - 2)
            self.state["distortion"] = min(10, self.state["distortion"] + 2)
        elif emotion == "anxious":
            self.state["distortion"] = min(10, self.state["distortion"] + 3)
        elif emotion in ["motivated", "happy", "joy"]:
            self.state["mood"] = min(10, self.state["mood"] + 1)

    def step(self, action: dict):
        import copy
        if self.state is None: self.reset()
        
        current_state = copy.deepcopy(self.state)
        reward = 0.0

        task = action.get("task", "agent_selection") 
        intent = current_state.get("intent", "statement")

        if task in ["emotion_classification", "intent_detection"]:
            return self.state, 0.0, self.state["done"]

        if task == "agent_selection":
            agent_type = action.get("type")

            # ==========================================
            # 1. UNIQUE AGENT EFFECTS & BASE REWARDS
            # ==========================================
            
            if agent_type == "cognitive":
                # Superpower: Best at reducing distortion
                self.state["distortion"] -= 3 
                self.state["mood"] += 1
                
                if intent == "question":
                    reward += 5.0  # Massive bonus for answering questions with logic
                else:
                    reward += 2.0

            elif agent_type == "behavioral":
                # Superpower: Best at fixing low energy
                if current_state["energy"] == 0:
                    self.state["energy"] += 2
                    reward += 6.0  # Huge reward for "getting them out of bed"
                else:
                    self.state["energy"] = min(2, self.state["energy"] + 1)
                    reward += 1.0
                
                self.state["mood"] += 1

            elif agent_type == "emotional":
                # Superpower: Best at pure mood boosting (but bad for logic/distortion)
                self.state["mood"] += 2
                reward += 1.5

                # --- PENALTIES FOR MISUSE ---
                if intent == "question":
                    reward -= 4.0  # Harsh penalty: Don't just give sympathy when a user asks 'Why?'
                
                if current_state["distortion"] > 6:
                    reward -= 2.0  # Penalty: Sympathy alone won't fix high cognitive distortion

            # ==========================================
            # 2. DENSE TRAJECTORY REWARDS (Feedback Loop)
            # ==========================================
            mood_gain = self.state["mood"] - current_state["mood"]
            dist_reduction = current_state["distortion"] - self.state["distortion"]

            # We reward the model for the POSITIVE change it created
            reward += max(0, mood_gain) * 2.0
            reward += max(0, dist_reduction) * 2.0

            # ==========================================
            # 3. SAFETY & LOGIC CHECK
            # ==========================================
            if current_state["energy"] == 0 and agent_type == "cognitive":
                reward -= 5.0  # Don't give "brain work" to someone with 0 energy

        # ==========================================
        # 4. CLAMP & RETURN
        # ==========================================
        self.state["mood"] = max(1, min(10, self.state["mood"]))
        self.state["distortion"] = max(0, min(10, self.state["distortion"]))
        self.state["energy"] = max(0, min(2, self.state["energy"]))

        return self.state, float(reward), self.state["done"]
    
    