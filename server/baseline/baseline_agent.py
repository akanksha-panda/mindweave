# server\baseline\baseline_agent.py

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

async def get_baseline_action(user_input):
    prompt = f"""
You are a helpful AI assistant.

User: "{user_input}"

Respond naturally and helpfully.
"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

    response = res.json().get("response", "").strip()

    return {
        "type": "emotional",   # baseline doesn't reason
        "intensity": 2,
        "text": response
    }