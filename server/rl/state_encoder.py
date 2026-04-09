import numpy as np
import torch
from server.core.model_store import get_embedding_model
print(". STATE ENCODER IMPORTED")
model = None

def encode_state(state):
    global model

    # . Lazy load (only first time)
    if model is None:
        model = get_embedding_model()

    base = [
        state.get("mood", 5) / 10.0,
        state.get("energy", 1) / 2.0,
        state.get("distortion", 5) / 10.0,
        (state.get("sentiment", 0.0) + 1.0) / 2.0
    ]

    emotion_text = state.get("emotion") or state.get("category") or "neutral"

    embedding = model.encode(
        emotion_text,
        normalize_embeddings=True
    )

    return np.concatenate([base, embedding]).astype(np.float32)