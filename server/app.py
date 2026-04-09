# server/app.py


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from openenv.core.env_server.http_server import create_app

# clean absolute imports
from models import MindweaveAction, MindweaveObservation
from server.environment2 import MindweaveEnvironment


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
# OPTIONAL LOCAL RUN
# =========================
import uvicorn

def main():
    uvicorn.run(
        "server.app:app",   
        host="127.0.0.1",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()