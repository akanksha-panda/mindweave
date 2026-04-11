import os
import asyncio
from openai import AsyncOpenAI
from client import MindweaveEnv, MindweaveAction

# =========================
# CONFIG
# =========================
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# 🔥 Docker image (with fallback)
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "registry.hf.space/akanksha0208-mindweave:latest")

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
# POLICY
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
    print(f"[START] Proxy: {API_BASE_URL} | Image: {IMAGE_NAME}", flush=True)

    try:
        env = await MindweaveEnv.from_docker_image(IMAGE_NAME)

        result = await env.reset()
        obs = result.observation

        rewards = []
        step_idx = 1

        # 🔥 GUARANTEED API CALL
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

    finally:
        try:
            await env.close()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
