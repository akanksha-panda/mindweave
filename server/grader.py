# server/grader.py

def clamp_score(score: float) -> float:
    return max(0.10, min(0.99, score))  # ← 0.10 to 0.99

# =========================
# TOP LEVEL GRADER FUNCTIONS
# matched by openenv.yaml grader field
# =========================
def grade_emotion(env_or_input, action=None, *args, **kwargs) -> float:
    from server.environment import MentalHealthEnv
    # handle both dict and (str, str) calling conventions
    if isinstance(env_or_input, dict):
        input_text = env_or_input.get("input", "") if isinstance(env_or_input.get("input"), str) else ""
        pred = env_or_input.get("action", "").strip().lower()
    else:
        input_text = env_or_input or ""
        pred = (action or "").strip().lower()
    
    e = MentalHealthEnv()
    state = e.reset(input_text)
    gt = state.get("emotion", "neutral")
    raw = 0.99 if pred == gt else 0.10  # ← never exactly 0.0 or 1.0!
    return raw

def grade_intent(env_or_input, action=None, *args, **kwargs) -> float:
    from server.environment import MentalHealthEnv
    if isinstance(env_or_input, dict):
        input_text = env_or_input.get("input", "") if isinstance(env_or_input.get("input"), str) else ""
        pred = env_or_input.get("action", "").strip().lower()
    else:
        input_text = env_or_input or ""
        pred = (action or "").strip().lower()

    e = MentalHealthEnv()
    state = e.reset(input_text)
    gt = state.get("intent", "statement")
    raw = 0.99 if pred == gt else 0.10  # ← never exactly 0.0 or 1.0!
    return raw

def grade_agent(env_or_input, action=None, *args, **kwargs) -> float:
    import asyncio
    from server.environment2 import MindweaveEnvironment
    from models import MindweaveAction

    if isinstance(env_or_input, dict):
        input_text = env_or_input.get("input", "") if isinstance(env_or_input.get("input"), str) else ""
        action_text = env_or_input.get("action", "emotional").strip().lower()
    else:
        input_text = env_or_input or ""
        action_text = (action or "emotional").strip().lower()

    env_instance = MindweaveEnvironment()
    env_instance.reset()
    env_instance.env.reset(input_text)
    env_instance.initial_state = env_instance.env.state.copy()
    env_instance.current_task_index = 2

    obs = asyncio.run(env_instance.step_async(
        MindweaveAction(message=action_text, task="agent_selection")
    ))

    return clamp_score(obs.reward)



class MindweaveGrader:
    def emotionGrader(self, env, *args, **kwargs) -> float:
        return grade_emotion(env, *args, **kwargs)

    def intentGrader(self, env, *args, **kwargs) -> float:
        return grade_intent(env, *args, **kwargs)

    def agentGrader(self, env, *args, **kwargs) -> float:
        return grade_agent(env, *args, **kwargs)

    def grade(self, env, *args, **kwargs) -> float:
        task_id = env.get("task_id", "emotion_classification")
        if task_id == "emotion_classification":
            return self.emotionGrader(env, *args, **kwargs)
        elif task_id == "intent_detection":
            return self.intentGrader(env, *args, **kwargs)
        elif task_id == "agent_selection":
            return self.agentGrader(env, *args, **kwargs)
        return 0.10  # ← was 0.001, now strictly within (0, 1)
