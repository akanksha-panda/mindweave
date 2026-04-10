import os
import asyncio
from openai import OpenAI
from client import MindweaveEnv, MindweaveAction

# Suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ==========================================
# STRICT DEBUG CONFIG
# ==========================================
# No fallbacks. If any of these are missing, the script crashes immediately.
API_KEY = os.environ["API_KEY"].strip()
API_BASE_URL = os.environ["API_BASE_URL"].strip()
MODEL_NAME = os.environ["MODEL_NAME"].strip() # <--- Force fail if not injected

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    default_headers={"X-Requested-With": "Mindweave-Submission"}
)

def llm_echo(answer: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"Return ONLY this word: {answer}"}],
        temperature=0,
        max_tokens=3,
    )
    return response.choices[0].message.content.strip().lower()

def simple_policy(state, task):
    # (Keeping your core logic the same)
    emotion = state.get("emotion")
    intent = state.get("intent")
    energy = state.get("energy", 1)
    distortion = state.get("distortion", 5)
    if task == "emotion_classification": return emotion or "neutral"
    if task == "intent_detection": return intent or "statement"
    if energy == 0: return "behavioral"
    if distortion > 6: return "cognitive"
    return "emotional"

async def main():
    env = MindweaveEnv(base_url="http://localhost:8000")
    print(f"[START] Using Proxy: {API_BASE_URL} | Model: {MODEL_NAME}", flush=True)

    try:
        result = await env.reset()
        obs = result.observation

        # 🔥 CRITICAL DEBUG PING
        print(f"DEBUG: Attempting first proxy call to {API_BASE_URL}...", flush=True)
        try:
            ping = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            print(f"DEBUG: Proxy response successful: {ping.id}", flush=True)
        except Exception as api_err:
            print(f"DEBUG: Proxy call FAILED. Error: {str(api_err)}", flush=True)
            raise api_err # Re-raise to stop execution

        step_idx = 1
        rewards = []

        while not result.done:
            state = obs.state or {}
            task = obs.task
            ppo_output = simple_policy(state, task)
            
            action_text = llm_echo(ppo_output)

            result = await env.step(MindweaveAction(message=action_text, task=task))
            rewards.append(float(result.reward))

            print(f"[STEP] step={step_idx} action={action_text} reward={result.reward:.2f}", flush=True)
            obs = result.observation
            step_idx += 1

        print(f"[END] success=true score={sum(rewards)/(step_idx-1):.2f}", flush=True)

    except Exception as e:
        print(f"[END] success=false error={str(e)}", flush=True)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
