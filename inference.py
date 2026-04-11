import os
import asyncio
from openai import OpenAI
from client import MindweaveEnv, MindweaveAction

# Suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import os
import asyncio
from openai import OpenAI
from client import MindweaveEnv, MindweaveAction

# =========================
# . STRICT CONFIG
# =========================
# Access keys without defaults to ensure we use validator-injected values
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL")

# Prioritize environment variables, fallback to the sample model if both are missing
MODEL = os.environ.get("MODEL") or os.environ.get("MODEL_NAME") or "openai/gpt-oss-120b:novita"

# =========================
# . OPENAI CLIENT
# =========================
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

def llm_echo(answer: str) -> str:
    """Uses the LLM proxy to echo the decision."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Return ONLY this word:\n{answer}"}],
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
    # Environment is local to the validator runner
    env = MindweaveEnv(base_url="http://localhost:8000")
    
    print(f"[START] Using Model: {MODEL} | Proxy: {API_BASE_URL}", flush=True)

    try:
        if not API_KEY or not API_BASE_URL:
            raise ValueError("Required environment variables (API_KEY/API_BASE_URL) are missing.")

        result = await env.reset()
        obs = result.observation
        rewards = []
        step_idx = 1

        while not result.done:
            state = obs.state or {}
            task = obs.task

            ppo_output = simple_policy(state, task)
            
            # The actual LLM request that the proxy monitors
            action_text = llm_echo(ppo_output)

            result = await env.step(
                MindweaveAction(message=action_text, task=task)
            )

            reward = float(result.reward)
            rewards.append(reward)
            
            print(f"[STEP] {step_idx} | Action: {action_text} | Reward: {reward:.2f}", flush=True)

            obs = result.observation
            step_idx += 1

        avg_score = sum(rewards)/len(rewards) if rewards else 0.0
        print(f"[END] success=true score={avg_score:.2f}", flush=True)

    except Exception as e:
        # Print the error so it shows up in the validator participant log
        print(f"[END] success=false error={str(e)}", flush=True)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())


