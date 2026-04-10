# mindweave_env\client.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mindweave Env Environment Client."""

# client.py

from typing import Dict, Any, Optional
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import MindweaveAction, MindweaveObservation

class MindweaveEnv(EnvClient[MindweaveAction, MindweaveObservation, State]):
    """Client that communicates with the FastAPI endpoints shown in your docs."""

    def _step_payload(self, action: MindweaveAction) -> Dict[str, Any]:
        # This payload is what is sent to POST /step
        return {
            "message": action.message,
            "task": action.task,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MindweaveObservation]:
        obs_data = payload.get("observation", {}) or {}

        # . ALWAYS PRESERVE FULL STATE
        raw_state = obs_data.get("state", {}) or {}
        state = dict(raw_state)  # deep copy not needed (JSON-safe)

        # . SAFE DEFAULTS (non-destructive)
        defaults = {
            "mood": 5,
            "distortion": 5,
            "energy": 1,
            "agent": None,
            "emotion": None,
            "intent": None,
            "mood_delta": None,
            "distortion_delta": None,
        }

        for k, v in defaults.items():
            state.setdefault(k, v)

        observation = MindweaveObservation(
            input=obs_data.get("input", ""),
            task=obs_data.get("task", "intent_detection"),
            state=state,
            message_length=obs_data.get("message_length", 0),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        # Matches the GET /state response schema from your screenshot
        return State(
            episode_id=str(payload.get("episode_id", "default")),
            step_count=int(payload.get("step_count", 0)),
        )