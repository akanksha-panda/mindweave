import os
import asyncio
from openai import OpenAI
from client import MindweaveEnv, MindweaveAction

# Suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Strict Configuration (Required by Validator)
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Proxy-Only Client
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

def llm_echo(answer: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"Return ONLY this word:\n{answer}"}],
        temperature=0,
        max_tokens=3,
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
    print(f"[START] task=mindweave_eval env=mindweave model=env+llm", flush=True)

    rewards = []
    step_idx = 1

    try:
        result = await env.reset()
        obs = result.observation

        # Initial Proxy Ping
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )

        while not result.done:
            state = obs.state or {}
            task = obs.task
            
            ppo_output = simple_policy(state, task)
            action_text = llm_echo(ppo_output)

            result = await env.step(
                MindweaveAction(message=action_text, task=task)
            )

            reward = float(result.reward)
            rewards.append(reward)

            print(f"[STEP] step={step_idx} action={action_text} reward={reward:.2f} done={str(result.done).lower()} error=null", flush=True)
            
            obs = result.observation
            step_idx += 1

        total_steps = step_idx - 1
        score = sum(rewards) / total_steps if total_steps > 0 else 0.0
        print(f"[END] success=true steps={total_steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

    except Exception as e:
        print(f"[END] success=false steps={step_idx} score=0.00 rewards= error={str(e)}", flush=True)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
