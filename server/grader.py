# server/grader.py

def clamp_score(score: float) -> float:
    return max(0.001, min(0.999, score))

def compute_task_score(task_id: str, state: dict, action: str) -> float:
    from server.environment import MentalHealthEnv
    env = MentalHealthEnv()
    parsed_state = env.reset(state.get("input", ""))

    if task_id == "emotion_classification":
        gt = parsed_state.get("emotion", "neutral")
        raw = 1.0 if action.strip().lower() == gt else 0.0
        return clamp_score(raw)

    elif task_id == "intent_detection":
        gt = parsed_state.get("intent", "statement")
        raw = 1.0 if action.strip().lower() == gt else 0.0
        return clamp_score(raw)

    elif task_id == "agent_selection":
        _, raw, _ = env.step({
            "task": "agent_selection",
            "type": action.strip().lower(),
            "message": action.strip().lower(),
        })
        return clamp_score((raw + 5.0) / 20.0)

    return 0.001

# =========================
# CONSOLIDATED GRADER CLASS
# =========================
class MindweaveGrader:

    def emotionGrader(self, env, *args, **kwargs) -> float:
        from server.environment import MentalHealthEnv
        e = MentalHealthEnv()
        # ← input is already state dict from environment2
        input_text = env.get("input", "") if isinstance(env.get("input"), str) else ""
        state = e.reset(input_text)
        gt = state.get("emotion", "neutral")
        pred = env.get("action", "").strip().lower()
        raw = 1.0 if pred == gt else 0.0
        return clamp_score(raw)

    def intentGrader(self, env, *args, **kwargs) -> float:
        from server.environment import MentalHealthEnv
        e = MentalHealthEnv()
        input_text = env.get("input", "") if isinstance(env.get("input"), str) else ""
        state = e.reset(input_text)
        gt = state.get("intent", "statement")
        pred = env.get("action", "").strip().lower()
        raw = 1.0 if pred == gt else 0.0
        return clamp_score(raw)

    def agentGrader(self, env, *args, **kwargs) -> float:
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

    def grade(self, env, *args, **kwargs) -> float:
        task_id = env.get("task_id", "emotion_classification")
        if task_id == "emotion_classification":
            return self.emotionGrader(env, *args, **kwargs)
        elif task_id == "intent_detection":
            return self.intentGrader(env, *args, **kwargs)
        elif task_id == "agent_selection":
            return self.agentGrader(env, *args, **kwargs)
        return 0.001
