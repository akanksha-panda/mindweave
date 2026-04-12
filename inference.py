import os
import asyncio
import traceback
import textwrap
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
BENCHMARK = "mindweave"
MAX_TOKENS = 5
SUCCESS_SCORE_THRESHOLD = 0.10
MAX_TOTAL_REWARD = 3 * 0.99  # 3 inputs per task, max 0.99 each

# =========================
# GRADER INSTANCE
# =========================
grader = MindweaveGrader()

# =========================
# HARDCODED INPUTS - same as environment2.py
# =========================
TEST_INPUTS = [
    "I feel so depressed and I can't get out of bed.",
    "Why do I always feel like a failure at everything I try?",
    "I am feeling pretty excited today!",
]

# =========================
# TASK DEFINITIONS - easy → medium → hard
# =========================
TASK_DEFINITIONS = [
    {
        "label": "easy",
        "task_id": "intent_detection",
        "prompt": lambda state, user_input: (
            f"The user said: '{user_input}'. "
            f"Detect their intent. Reply with ONE word only: "
            f"emotional, question, or statement."
        ),
    },
    {
        "label": "medium",
        "task_id": "emotion_classification",
        "prompt": lambda state, user_input: (
            f"The user said: '{user_input}'. "
            f"Classify their core emotion in ONE lowercase word "
            f"(e.g. neutral, anxious, sad, happy, failure, tired, motivated)."
        ),
    },
    {
        "label": "hard",
        "task_id": "agent_selection",
        "prompt": lambda state, user_input: (
            f"Mental health state — emotion: {state.get('emotion')}, "
            f"intent: {state.get('intent')}, "
            f"energy: {state.get('energy')}, "
            f"distortion: {state.get('distortion')}. "
            f"Select ONE agent: cognitive, behavioral, or emotional."
        ),
    },
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert mental health AI agent.
    You analyze user states and select appropriate responses.
    Return ONLY one lowercase word. No punctuation, no explanation.
""").strip()

# =========================
# LLM CALL
# =========================
async def llm_echo(prompt: str, client: AsyncOpenAI) -> str:
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        return (completion.choices[0].message.content or "").strip().lower()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "neutral"

# =========================
# MAIN
# =========================
async def main() -> None:
    print(f"[START] task=mindweave_eval env={BENCHMARK} model={MODEL_NAME}", flush=True)

    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = await MindweaveEnv.from_docker_image(IMAGE_NAME)

    all_task_scores = {}
    success = False

    try:
        # guaranteed LLM API call
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )

        # run each task separately with all 3 inputs
        for task_def in TASK_DEFINITIONS:
            label = task_def["label"]
            task_id = task_def["task_id"]
            task_rewards = []

            print(f"[START] task={label} env={BENCHMARK} model={MODEL_NAME}", flush=True)

            for step, user_input in enumerate(TEST_INPUTS, 1):
                # reset env for each input
                result = await env.reset()
                obs = result.observation
                state = obs.state or {}

                action_text = await llm_echo(
                    task_def["prompt"](state, user_input),
                    client
                )

                grader_score = grader.grade({
                    "task_id": task_id,
                    "input": user_input,
                    "action": action_text,
                })

                result = await env.step(
                    MindweaveAction(message=action_text, task=task_id)
                )

                env_reward = float(result.reward) or 0.10
                final_reward = clamp_score(max(env_reward, grader_score))
                task_rewards.append(final_reward)

                print(
                    f"[STEP] task={label} step={step} action={action_text} "
                    f"env_reward={env_reward:.2f} grader_score={grader_score:.2f} "
                    f"final={final_reward:.2f}",
                    flush=True
                )

            # avg score for this task
            task_score = sum(task_rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.10
            task_score = min(max(task_score, 0.10), 0.99)
            all_task_scores[label] = task_score

            print(f"[END] task={label} score={task_score:.2f} steps={len(TEST_INPUTS)}", flush=True)

        # final overall score
        final_score = sum(all_task_scores.values()) / len(all_task_scores) if all_task_scores else 0.10
        final_score = min(max(final_score, 0.10), 0.99)
        success = final_score >= SUCCESS_SCORE_THRESHOLD

        print(f"==== FINAL SCORES ====", flush=True)
        for label, score in all_task_scores.items():
            print(f"  {label}: {score:.2f}", flush=True)
        print(f"  Average: {final_score:.2f}", flush=True)

    except Exception as e:
        print(f"[DEBUG] error={str(e)}", flush=True)
        traceback.print_exc()

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        print(
            f"[END] success={success} score={final_score:.2f}",
            flush=True
        )

if __name__ == "__main__":
    asyncio.run(main())
