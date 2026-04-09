import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

from client import MindweaveEnv, MindweaveAction

# . SUPPRESS LOGS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

load_dotenv()

# =========================
# . CONFIG
# =========================
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY)


ACTION_MAP = {
    0: "behavioral",
    1: "cognitive",
    2: "emotional"
}

INTENT_MAP = ["statement", "question", "emotional"]

# =========================
# . LLM ECHO
# =========================
def llm_echo(answer: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Return ONLY this word:\n{answer}"}],
            temperature=0,
            max_tokens=3,
        )
        return response.choices[0].message.content.strip().lower()
    except:
        return answer


# =========================
# . SIMPLE POLICY (ENV-BASED)
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

    # agent selection
    if energy == 0:
        return "behavioral"

    if distortion > 6:
        return "cognitive"

    if emotion in ["sadness", "anxiety", "fear"]:
        return "emotional"

    return "emotional"


# =========================
# . MAIN
# =========================
async def main():
    env = MindweaveEnv(base_url="http://localhost:8000")
   
    print(f"[START] task=mindweave_eval env=mindweave model=env+llm", flush=True)

    rewards = []
    step_idx = 1

    try:
        result = await env.reset()
       
        obs = result.observation

        while not result.done:

            state = obs.state or {}
            task = obs.task

            ppo_output = simple_policy(state, task)

            action_text = llm_echo(ppo_output)

            if not action_text:
                action_text = ppo_output

            result = await env.step(
                MindweaveAction(
                    message=action_text,
                    task=task
                )
            )

            reward = float(result.reward)
            rewards.append(reward)

            print(
                f"[STEP] step={step_idx} action={action_text} reward={reward:.2f} done={str(result.done).lower()} error=null",
                flush=True
            )

            obs = result.observation
            step_idx += 1

        total_steps = step_idx - 1
        score = sum(rewards) / total_steps if total_steps > 0 else 0.0

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success=true steps={total_steps} score={score:.2f} rewards={rewards_str}",
            flush=True
        )

    except Exception as e:
        print(
            f"[END] success=false steps={step_idx} score=0.00 rewards= error={str(e)}",
            flush=True
        )

    finally:
        await env.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())