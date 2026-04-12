class AgentGrader:
    def grade(self, env, *args, **kwargs) -> float:
        from server.environment import MentalHealthEnv
        e = MentalHealthEnv()
        state = e.reset(env.get("input", ""))
        _, raw, _ = e.step({
            "task": "agent_selection",
            "type": env.get("action", "emotional").strip().lower(),
            "message": env.get("action", "emotional").strip().lower(),
        })
        return max(0.001, min(0.999, (raw + 5.0) / 20.0))
