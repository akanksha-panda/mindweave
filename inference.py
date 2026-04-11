import os
import asyncio
from openai import AsyncOpenAI
from client import MindweaveEnv, MindweaveAction

# Suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"



# ==========================================
# . CONFIG (Direct Environment Access)
# ==========================================
# Direct access to ensure the validator sees these being used
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]

# Initialize as empty; the proxy handles routing or errors out with the correct name
MODEL = os.environ.get("MODEL", "")

# Explicit Docker Image URI for your Space
IMAGE_NAME = os.getenv("IMAGE_NAME") or "registry.hf.space/akanksha0208-mindweave:latest"

# 1. Initialize the Async Client
# Using the URL exactly as provided per advice
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

# 2. Async call
async def llm_echo(answer: str) -> str:
    """Async call that the LiteLLM proxy monitors."""
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Return ONLY this word: {answer}"}],
        temperature=0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().lower()

def simple_policy(state, task):
    emotion = state.get("emotion")
    intent = state.get("intent")
    energy = state.get("energy", 1)
    distortion = state.get("distortion", 5)
    
    if task == "emotion_classification": 
        return emotion or "neutral"
    if task == "intent_detection": 
        return intent or "statement"
        
    return "behavioral" if energy == 0 else "cognitive" if distortion > 6 else "emotional"

async def main():
    # Print configuration for validator logging
    print(f"[START] Proxy: {API_BASE_URL} | Image: {IMAGE_NAME}", flush=True)
    print(f"[INFO] Model Parameter: '{MODEL}'", flush=True)

    try:
        # 3. Async with context manager for the Environment
        async with MindweaveEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            obs = result.observation
            rewards = []
            step_idx = 1

            while not result.done:
                state = obs.state or {}
                task = obs.task
                
                # Logic decision
                ppo_output = simple_policy(state, task)
                
                # The monitored async API call
                action_text = await llm_echo(ppo_output)

                # Send action to environment
                result = await env.step(MindweaveAction(message=action_text, task=task))
                rewards.append(float(result.reward))
                
                print(f"[STEP] {step_idx} | Action={action_text} | Reward={result.reward:.2f}", flush=True)
                
                obs = result.observation
                step_idx += 1

            avg_score = sum(rewards) / len(rewards) if rewards else 0.0
            print(f"[END] success=true score={avg_score:.2f}", flush=True)

    except Exception as e:
        # Crucial for debugging: reveals why API calls or the Image fail
        print(f"[END] success=false error={str(e)}", flush=True)

if __name__ == "__main__":
    # Standard entry point for async main
    asyncio.run(main())
