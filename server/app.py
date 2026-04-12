import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openenv.core.env_server.http_server import create_app
from models import MindweaveAction, MindweaveObservation
from server.environment2 import MindweaveEnvironment
from typing import List, Dict
from pydantic import BaseModel

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
# GRADER FUNCTIONS - pointed to by openenv.yaml
# =========================
def grade_emotion(input: str, action: str) -> float:
    from server.environment import MentalHealthEnv
    env = MentalHealthEnv()
    state = env.reset(input)
    gt = state.get("emotion", "neutral")
    raw = 1.0 if action.strip().lower() == gt else 0.0
    return max(0.001, min(0.999, raw))

def grade_intent(input: str, action: str) -> float:
    from server.environment import MentalHealthEnv
    env = MentalHealthEnv()
    state = env.reset(input)
    gt = state.get("intent", "statement")
    raw = 1.0 if action.strip().lower() == gt else 0.0
    return max(0.001, min(0.999, raw))

def grade_agent(input: str, action: str) -> float:
    from server.environment import MentalHealthEnv
    env = MentalHealthEnv()
    state = env.reset(input)
    _, raw, _ = env.step({
        "task": "agent_selection",
        "type": action.strip().lower(),
        "message": action.strip().lower(),
    })
    return max(0.001, min(0.999, (raw + 5.0) / 20.0))

# =========================
# /tasks ENDPOINT
# =========================
@app.get("/tasks")
def get_tasks() -> List[Dict]:
    return [
        {
            "id": "emotion_classification",
            "description": "Identifies the core emotion from user input (e.g. sad, anxious, neutral).",
            "grader": "server.app:grade_emotion",
            "reward_range": [0.001, 0.999],
        },
        {
            "id": "intent_detection",
            "description": "Determines if user intent is emotional disclosure, a question, or a statement.",
            "grader": "server.app:grade_intent",
            "reward_range": [0.001, 0.999],
        },
        {
            "id": "agent_selection",
            "description": "PPO-driven selection between cognitive, behavioral, or emotional agents.",
            "grader": "server.app:grade_agent",
            "reward_range": [0.001, 0.999],
        },
    ]

# =========================
# GRADER REQUEST MODEL
# =========================
class GraderRequest(BaseModel):
    task_id: str = "emotion_classification"
    input: str

# =========================
# /grader ENDPOINT
# =========================
@app.post("/grader")
def run_grader(request: GraderRequest) -> Dict:
    import asyncio

    if request.task_id == "emotion_classification":
        action = grade_emotion(request.input, "")
        # re-run to get detected emotion as action
        from server.environment import MentalHealthEnv
        env = MentalHealthEnv()
        state = env.reset(request.input)
        action = state.get("emotion", "neutral")
        score = grade_emotion(request.input, action)

    elif request.task_id == "intent_detection":
        from server.environment import MentalHealthEnv
        env = MentalHealthEnv()
        state = env.reset(request.input)
        action = state.get("intent", "statement")
        score = grade_intent(request.input, action)

    elif request.task_id == "agent_selection":
        env_instance = MindweaveEnvironment()
        env_instance.reset()
        env_instance.env.reset(request.input)
        env_instance.initial_state = env_instance.env.state.copy()
        env_instance.current_task_index = 2
        obs = asyncio.run(env_instance.step_async(
            MindweaveAction(
                message="emotional",
                task="agent_selection"
            )
        ))
        action = obs.state.get("agent", "emotional")
        score = grade_agent(request.input, action)
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
