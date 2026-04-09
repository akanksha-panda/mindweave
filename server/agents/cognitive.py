# server/agents/cognitive.py

def cognitive_agent(state, user_input):
    distortion = state.get("distortion", 5)
    sentiment = state.get("sentiment", 0)

    if sentiment < -0.8:
        text = f"If a friend said '{user_input}', what would you tell them?"

    elif distortion > 8:
        text = "It sounds like you're being really hard on yourself.What evidence do you have that this is completely true?"

    else:
        text = "Is there a more balanced way to look at this situation?"

    return {
        "type": "cognitive",
        "intensity": 2,
        "text": text
    }