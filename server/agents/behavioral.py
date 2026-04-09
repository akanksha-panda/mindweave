# server/agents/behavioral.py

def behavioral_agent(state, user_input):
    energy = state.get("energy", 1)

    if energy == 0:
        return {
            "type": "behavioral",
            "intensity": 1,
            "text": "Try something very small like drinking water or stretching."
        }

    elif energy == 1:
        return {
            "type": "behavioral",
            "intensity": 2,
            "text": "Maybe take a short walk or do a small task."
        }

    else:
        return {
            "type": "behavioral",
            "intensity": 3,
            "text": "You could try a focused session or a light workout."
        }