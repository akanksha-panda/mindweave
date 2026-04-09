# server/agents/emotional.py

def emotional_agent(state, user_input):
    emotion = state.get("emotion_category", "neutral")

    # . intensity levels
    strong = ["sadness", "pain", "vulnerable", "fear"]
    medium = ["confusion", "yearning", "fatigue"]
    positive = ["joy", "love", "gratitude", "calm"]

    if emotion in strong:
        text = "Acknowledge their feelings with strong empathy."

    elif emotion in medium:
        text = "Respond with light empathy and gentle understanding."

    elif emotion in positive:
        text = "Respond positively without emotional cushioning."

    else:
        text = "Respond naturally without emotional framing."

    return {
        "type": "emotional",
        "intensity": 1,
        "text": text
    }