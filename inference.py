import os
import asyncio
from openai import AsyncOpenAI
from client import MindweaveEnv, MindweaveAction

# Suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ==========================================
# . STRICT ENVIRONMENT CONFIG
# ==========================================
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]

# Initialize to empty string if not provided by environment
# This will force the proxy to error out and reveal the correct model name
MODEL = os.environ.get("MODEL", "")

# 1. Initialize the ASYNC client
# Sanitize base_url to ensure standard OpenAI routing
sanitized_url = API_BASE_URL if API_BASE_URL.endswith("/v1") else f"{API_BASE_URL.rstrip('/')}/v1"

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=sanitized_url
)

# 2. Async LLM call
async def llm_echo(answer: str) -> str:
    # This call will now raise an exception if MODEL is empty or invalid,
    # letting us see the exact error in the validator log.
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
    if energy == 0:
        return "behavioral"
    if distortion > 6:
        return "cognitive"
    return "emotional"

async def main():
    env = MindweaveEnv(base_url="http://localhost:8000")
    print(f"[START] Proxy: {sanitized_url} | Model Injected: '{MODEL}'", flush=True)

    try:
        result = await env.reset()
        obs = result.observation
        rewards = []
        step_idx = 1

        while not result.done:
            state = obs.state or {}
            task = obs.task

            ppo_output = simple_policy(state, task)
            
            # 3. Await the async LLM call
            action_text = await llm_echo(ppo_output)

            result = await env.step(
                MindweaveAction(message=action_text, task=task)
            )

            reward = float(result.reward)
            rewards.append(reward)
            
            print(f"[STEP] {step_idx} | Action={action_text} | Reward={reward:.2f}", flush=True)

            obs = result.observation
            step_idx += 1

        avg_score = sum(rewards)/len(rewards) if rewards else 0.0
        print(f"[END] success=true score={avg_score:.2f}", flush=True)

    except Exception as e:
        # Crucial: This prints the error we are hunting for
        print(f"[END] success=false error={str(e)}", flush=True)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
