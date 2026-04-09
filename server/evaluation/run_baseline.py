# server/evaluation/run_baseline.py

import asyncio
import json
import os

from server.environment import MentalHealthEnv
from server.baseline.baseline_agent import get_baseline_action
from server.evaluator.grader import grade_with_llm

BASE_DIR = os.path.dirname(__file__)

env = MentalHealthEnv()

user_inputs = [
    "I feel like a failure",
    "I have no energy",
    "Nothing makes sense anymore",
   
]


async def run():
    state = env.reset()
    rewards = []
    results = []

    for user_input in user_inputs:
        print(f"Processing: {user_input}")
        # . LLM baseline
        for _ in range(3):  # retry 3 times
            try:
                action = await get_baseline_action(user_input)
                break
            except Exception as e:
                print("Retrying baseline...", e)

        # . LLM grading (real evaluation)
        score = await grade_with_llm(user_input, action["text"])

        # . step env
        state, _, _ = env.step(action)

        rewards.append(score)

        results.append({
            "input": user_input,
            "response": action["text"],
            "score": score
        })

    print("Baseline Scores:", rewards)

   # . SAVE DATA
    output_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "baseline.json"), "w") as f:
        json.dump(rewards, f)

    with open(os.path.join(output_dir, "baseline_full.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run())