# server/agents/safety.py

def safety_check(state, action):
    text = action.get("text", "").lower()
    energy = state.get("energy", 1)

    # . toxic positivity
    if "just be happy" in text or "others have it worse" in text:
        return {
            "type": "emotional",
            "intensity": 1,
            "text": "I hear you. Let’s take this one step at a time."
        }

    # . too hard when exhausted
    if energy == 0 and action["type"] == "behavioral" and action.get("intensity", 1) > 2:
        return {
            "type": "behavioral",
            "intensity": 1,
            "text": "Let’s start very small—maybe just take a deep breath."
        }

    return action