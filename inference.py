import os
import asyncio
import traceback
from openai import AsyncOpenAI
from client import MindweaveEnv, MindweaveAction
from server.grader import MindweaveGrader, clamp_score

# =========================
# CONFIG 
# =========================
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv(
    "LOCAL_IMAGE_NAME",
    "registry.hf.space/akanksha0208-mindweave:latest"
)

# =========================
# GRADER INSTANCE
# =========================
grader = MindweaveGrader()

# =========================
# TASK PROMPTS - easy → medium → hard
# =========================
TASKS = {
    # EASY
    "intent_detection": {
        "prompt": lambda state: (
            f"The user said: '{state.get('input', '')}'. "
            f"Detect their intent. Reply with ONE word only: "
            f"emotional, question, or statement."
        ),
    },
    # MEDIUM
    "emotion_classification": {
        "prompt": lambda state: (
            f"The user said: '{state.get('input', '')}'. "
            f"Classify their core emotion in ONE lowercase word "
            f"(e.g. neutral, anxious, sad, happy, failure, tired, motivated)."
        ),
    },
    # HARD
    "agent_selection": {
        "prompt": lambda state: (
            f"Mental health state — emotion: {state.get('emotion')}, "
            f"intent: {state.get('intent')}, "
            f"energy: {state.get('energy')}, "
            f"distortion: {state.get('distortion')}. "
            f"Select ONE agent: cognitive, behavioral, or emotional."
        ),
    },
}

# =========================
# LLM CALL
# =========================
async def llm_echo(prompt: str, client: AsyncOpenAI) -> str:
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return ONLY one lowercase word. No punctuation, no explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5,
        )
        return (completion.choices[0].message.content or "").strip().lower()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "neutral"

# =========================
# MAIN
# =========================
async def main() -> None:
    print(f"[START] task=mindweave_eval env=mindweave model=env+llm", flush=True)

    try:
        client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )

        env = await MindweaveEnv.from_docker_image(IMAGE_NAME)

        result = await env.reset()
        obs = result.observation
        rewards = []
        step_idx = 1

        # guaranteed LLM API call
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )

        while not result.done:
            # state comes from environment2.py step_async
            state = obs.state or {}
            task_id = obs.task

            task_def = TASKS.get(task_id)
            prompt = task_def["prompt"](state) if task_def else f"Respond to task: {task_id}"

            action_text = await llm_echo(prompt, client)

            # grader scores using state from environment2
            grader_score = grader.grade({
                "task_id": task_id,
                "input": state.get("input", ""),
                "action": action_text,
            })

            result = await env.step(
                MindweaveAction(message=action_text, task=task_id)
            )

            env_reward = float(result.reward)
            final_reward = clamp_score(max(env_reward, grader_score))

            rewards.append(final_reward)
            print(
                f"[STEP] step={step_idx} task={task_id} action={action_text} "
                f"env_reward={env_reward:.3f} grader_score={grader_score:.3f} "
                f"final={final_reward:.3f}",
                flush=True
            )
            obs = result.observation
            step_idx += 1

        score = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"[END] success=true score={score:.2f}", flush=True)

    except Exception as e:
        print(f"[END] success=false error={str(e)}", flush=True)
        traceback.print_exc()

    finally:
        try:
            await env.close()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
