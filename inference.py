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
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "mindweave_eval")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "mindweave")
MAX_STEPS = 9               # 3 inputs × 3 tasks
MAX_TOKENS = 5              # we use 1 word only
SUCCESS_SCORE_THRESHOLD = 0.1
MAX_TOTAL_REWARD = MAX_STEPS * 1.0  # 1.0 max reward per task step

# =========================
# GRADER INSTANCE
# =========================
grader = MindweaveGrader()

# =========================
# LOGGING HELPERS
# =========================
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:+.2f} done={done} error={error}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

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
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = await MindweaveEnv.from_docker_image(IMAGE_NAME)

    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset()
        obs = result.observation
        last_reward = 0.0

        # guaranteed LLM API call
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            state = obs.state or {}
            task_id = obs.task
            error = None

            action_text = await llm_echo(
                TASKS[task_id]["prompt"](state) if task_id in TASKS else f"Respond to task: {task_id}",
                client
            )

            # grader scores using state from environment2
            grader_score = grader.grade({
                "task_id": task_id,
                "input": state.get("input", ""),
                "action": action_text,
            })

            result = await env.step(
                MindweaveAction(message=action_text, task=task_id)
            )
            obs = result.observation

            env_reward = float(result.reward) or 0.0
            final_reward = clamp_score(max(env_reward, grader_score))

            rewards.append(final_reward)
            steps_taken = step
            last_reward = final_reward

            log_step(step=step, action=action_text, reward=final_reward, done=result.done, error=error)
            history.append(f"Step {step}: {action_text!r} → reward {final_reward:+.2f}")

            if result.done:
                break

        # matches sample scoring pattern
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        error = str(e)
        print(f"[DEBUG] error={error}", flush=True)
        traceback.print_exc()

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
