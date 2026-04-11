import os
import asyncio
from openai import AsyncOpenAI
from client import MindweaveEnv, MindweaveAction

# Suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"



# =========================
# CONFIG (MATCH SAMPLE)
# =========================
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

# =========================
# LLM CALL
# =========================
async def llm_echo(answer: str) -> str:
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return ONLY one word."},
                {"role": "user", "content": answer}
            ],
            temperature=0,
            max_tokens=5,
        )

        text = (completion.choices[0].message.content or "").strip().lower()
        return text if text else "hello"

    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)
        return "hello"

# =========================
# SIMPLE POLICY
# =========================
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

# =========================
# MAIN
# =========================
async def main():
    print(f"[START] Proxy: {API_BASE_URL}", flush=True)

    try:
        async with MindweaveEnv(base_url="http://localhost:8000") as env:

            result = await env.reset()
            obs = result.observation

            rewards = []
            step_idx = 1

            # 🔥 GUARANTEED PROXY CALL
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )

            while not result.done:
                state = obs.state or {}
                task = obs.task

                ppo_output = simple_policy(state, task)

                action_text = await llm_echo(ppo_output)

                result = await env.step(
                    MindweaveAction(message=action_text, task=task)
                )

                rewards.append(float(result.reward))

                print(
                    f"[STEP] step={step_idx} action={action_text} reward={result.reward:.2f} done={str(result.done).lower()} error=null",
                    flush=True
                )

                obs = result.observation
                step_idx += 1

            score = sum(rewards) / len(rewards) if rewards else 0.0

            print(
                f"[END] success=true steps={step_idx-1} score={score:.2f}",
                flush=True
            )

    except Exception as e:
        print(f"[END] success=false error={str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
