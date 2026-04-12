import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from openenv.core.env_server.http_server import create_app
from models import MindweaveAction, MindweaveObservation
from server.environment2 import MindweaveEnvironment
from typing import List, Dict

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
# /tasks ENDPOINT -
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

# =========================
# OPTIONAL LOCAL RUN
# =========================
import uvicorn
def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",  # fixed from 127.0.0.1
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
