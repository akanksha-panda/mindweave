# server/agents/adaptive.py

def adaptive_agent(state, user_input, mode="normal"):
    return {
        "type": "adaptive",
        "mode": mode,   # .FIXED
        "text": "Respond naturally and intelligently to the user. Understand intent from context. Keep it conversational and relevant."
    }