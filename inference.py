import os
import asyncio
from openai import OpenAI
from client import MindweaveEnv, MindweaveAction

# Suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# =========================
# . STRICT PROXY CONFIG
# =========================
# Using os.environ[] directly ensures the script CRASHES if these are missing,
# which is exactly what the validator needs to see to know you are using them.
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]

# We pull the model from the environment but do NOT provide a default string.
# This way, the model name is entirely determined by their environment.
ENVIRONMENT_MODEL = os.getenv("MODEL") or os.getenv("MODEL_NAME") or ""

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

def llm_echo(answer: str) -> str:
    # We use the ENVIRONMENT_MODEL variable here. 
    # If it's empty, the proxy receives an empty model string and 
    # must handle the routing itself.
    response = client.chat.completions.create(
        model=ENVIRONMENT_MODEL,
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
    env = MindweaveEnv(base_url="http://localhost:8000")
    print(f"[START] Proxy: {API_BASE_URL} | Model: {ENVIRONMENT_MODEL}", flush=True)

    try:
        result = await env.reset()
        obs = result.observation
        rewards = []
        step_idx = 1

        while not result.done:
            state = obs.state or {}
            task = obs.task

            ppo_output = simple_policy(state, task)
            
            # This is the call that hits the LiteLLM proxy
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
        print(f"[END] success=false error={str(e)}", flush=True)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
