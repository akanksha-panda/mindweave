import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openenv.core.env_server.http_server import create_app
from models import MindweaveAction, MindweaveObservation
from server.environment2 import MindweaveEnvironment
from typing import List, Dict
from pydantic import BaseModel  # ← moved to top

# =========================
# CREATE APP
# =========================
app = create_app(
    MindweaveEnvironment,
    MindweaveAction,
    MindweaveObservation,
    env_name="mindweave",
    max_concurrent_envs=1,
)

# =========================
# /tasks ENDPOINT
# =========================
@app.get("/tasks")
def get_tasks() -> List[Dict]:
    return [
        {
            "id": "emotion_classification",
            "description": "Identifies the core emotion from user input (e.g. sad, anxious, neutral).",
            "grader": "programmatic",
            "reward_range": [0.001, 0.999],
        },
        {
            "id": "intent_detection",
            "description": "Determines if user intent is emotional disclosure, a question, or a statement.",
            "grader": "programmatic",
            "reward_range": [0.001, 0.999],
        },
        {
            "id": "agent_selection",
            "description": "PPO-driven selection between cognitive, behavioral, or emotional agents.",
            "grader": "programmatic",
            "reward_range": [0.001, 0.999],
        },
    ]

class GraderRequest(BaseModel):
    task_id: str = "emotion_classification"
    input: str
    

@app.post("/grader")
def run_grader(request: GraderRequest) -> Dict:
    import asyncio
    env_instance = MindweaveEnvironment()

    # reset with real user input
    env_instance.reset()
    state = env_instance.env.reset(request.input)
    env_instance.initial_state = state.copy()

    if request.task_id == "emotion_classification":
        # action = what environment detected from input
        action = state.get("emotion", "neutral")
        gt = state.get("emotion", "neutral")
        raw = 1.0 if action == gt else 0.0
        score = env_instance.normalize_reward(raw, request.task_id)

    elif request.task_id == "intent_detection":
        # action = what environment detected from input
        action = state.get("intent", "statement")
        gt = state.get("intent", "statement")
        raw = 1.0 if action == gt else 0.0
        score = env_instance.normalize_reward(raw, request.task_id)

    elif request.task_id == "agent_selection":
        # action = what PPO selects
        env_instance.current_task_index = 2
        obs = asyncio.run(env_instance.step_async(
            MindweaveAction(
                message="emotional",  # seed message, PPO overrides
                task="agent_selection"
            )
        ))
        action = obs.state.get("agent", "emotional")
        score = obs.reward
    else:
        action = "unknown"
        score = 0.001

    return {
        "task_id": request.task_id,
        "input": request.input,
        "action": action,        # ← derived from RL/environment
        "score": round(max(0.001, min(0.999, score)), 4),
        "reward_range": [0.001, 0.999],
    }

# =========================
# OPTIONAL LOCAL RUN
# =========================
import uvicorn

def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
