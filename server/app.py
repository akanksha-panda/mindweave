import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openenv.core.env_server.http_server import create_app
from models import MindweaveAction, MindweaveObservation
from server.environment2 import MindweaveEnvironment
from server.grader import grade_emotion, grade_intent, grade_agent  # ← import
from typing import List, Dict
from pydantic import BaseModel

app = create_app(
    MindweaveEnvironment,
    MindweaveAction,
    MindweaveObservation,
    env_name="mindweave",
    max_concurrent_envs=1,
)

@app.get("/tasks")
def get_tasks() -> List[Dict]:
    return [
        {
            "id": "intent_detection",
            "description": "Easy task - Determines if user intent is emotional, question, or statement.",
            "grader": "grade_intent",
            "reward_range": [0.001, 0.999],
        },
        {
            "id": "emotion_classification",
            "description": "Medium task - Identifies the core emotion from user input.",
            "grader": "grade_emotion",
            "reward_range": [0.001, 0.999],
        },
        {
            "id": "agent_selection",
            "description": "Hard task - PPO-driven selection between cognitive, behavioral, or emotional agents.",
            "grader": "grade_agent",
            "reward_range": [0.001, 0.999],
        },
    ]

class GraderRequest(BaseModel):
    task_id: str = "emotion_classification"
    input: str

@app.post("/grader")
def run_grader(request: GraderRequest) -> Dict:
    import asyncio
    from server.environment import MentalHealthEnv

    if request.task_id == "emotion_classification":
        env = MentalHealthEnv()
        state = env.reset(request.input)
        action = state.get("emotion", "neutral")
        score = grade_emotion({"input": request.input, "action": action})

    elif request.task_id == "intent_detection":
        env = MentalHealthEnv()
        state = env.reset(request.input)
        action = state.get("intent", "statement")
        score = grade_intent({"input": request.input, "action": action})

    elif request.task_id == "agent_selection":
        env_instance = MindweaveEnvironment()
        env_instance.reset()
        env_instance.env.reset(request.input)
        env_instance.initial_state = env_instance.env.state.copy()
        env_instance.current_task_index = 2
        obs = asyncio.run(env_instance.step_async(
            MindweaveAction(message="emotional", task="agent_selection")
        ))
        action = obs.state.get("agent", "emotional")
        score = grade_agent({"input": request.input, "action": action})
    else:
        action = "unknown"
        score = 0.001

    return {
        "task_id": request.task_id,
        "input": request.input,
        "action": action,
        "score": round(max(0.001, min(0.999, score)), 4),
        "reward_range": [0.001, 0.999],
    }

import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
