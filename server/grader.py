# server/grader.py

def clamp_score(score: float) -> float:
    return max(0.10, min(0.99, score))

# =========================
# TOP LEVEL GRADER FUNCTIONS
# matched by openenv.yaml grader field
# =========================
def grade_emotion(env, *args, **kwargs) -> float:
    from server.environment import MentalHealthEnv
    e = MentalHealthEnv()
    input_text = env.get("input", "") if isinstance(env.get("input"), str) else ""
    state = e.reset(input_text)
    gt = state.get("emotion", "neutral")
    pred = env.get("action", "").strip().lower()
    raw = 1.0 if pred == gt else 0.0
    return clamp_score(raw)

def grade_intent(env, *args, **kwargs) -> float:
    from server.environment import MentalHealthEnv
    e = MentalHealthEnv()
    input_text = env.get("input", "") if isinstance(env.get("input"), str) else ""
    state = e.reset(input_text)
    gt = state.get("intent", "statement")
    pred = env.get("action", "").strip().lower()
    raw = 1.0 if pred == gt else 0.0
    return clamp_score(raw)

def grade_agent(env, *args, **kwargs) -> float:
    from server.environment import MentalHealthEnv
    e = MentalHealthEnv()
    input_text = env.get("input", "") if isinstance(env.get("input"), str) else ""
    state = e.reset(input_text)
    _, raw, _ = e.step({
        "task": "agent_selection",
        "type": env.get("action", "emotional").strip().lower(),
        "message": env.get("action", "emotional").strip().lower(),
    })
    return clamp_score((raw + 5.0) / 20.0)

def compute_task_score(task_id: str, state: dict, action: str) -> float:
    if task_id == "emotion_classification":
        return grade_emotion({"input": state.get("input", ""), "action": action})
    elif task_id == "intent_detection":
        return grade_intent({"input": state.get("input", ""), "action": action})
    elif task_id == "agent_selection":
        return grade_agent({"input": state.get("input", ""), "action": action})
    return 0.001

# =========================
# CONSOLIDATED GRADER CLASS
# =========================
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
        return 0.001
