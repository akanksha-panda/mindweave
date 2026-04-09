# mindweave_env\models.py

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Dict, Optional

class MindweaveAction(Action):
    """Matches the POST /step schema"""
    message: str = Field(..., description="The classification or agent name")
    task: str = Field(default="intent_detection", description="Current task type")

class MindweaveObservation(Observation):
    """Matches the Schema shown in your GET /schema results"""
    input: str = Field(default="")
    task: str = Field(default="intent_detection")
    # This matches the "additionalProp" / dict structure in your docs
    state: Dict = Field(default_factory=dict)
    message_length: int = Field(default=0)
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)