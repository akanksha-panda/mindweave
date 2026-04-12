class EmotionGrader:
    def grade(self, env, *args, **kwargs) -> float:
        from server.environment import MentalHealthEnv
        e = MentalHealthEnv()
        state = e.reset(env.get("input", ""))
        gt = state.get("emotion", "neutral")
        pred = env.get("action", "").strip().lower()
        raw = 1.0 if pred == gt else 0.0
        return max(0.001, min(0.999, raw))
