# server/evaluation/run_mindweave.py

import asyncio
import json
import os
import torch

from server.environment import MentalHealthEnv
from server.router import route
from server.agents.safety import safety_check
from server.evaluator.grader import grade_with_llm
# . Ensure we are using the 388-dim version of the policy
from server.rl.ppo_model import PPOPolicy 
from server.llm.llm_handler import generate_response_stream

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the root folder to find the models directory
# Adjust this join based on your exact folder structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))

async def get_full_response(action, user_input, state):
    full = ""
    async for token in generate_response_stream(action, user_input, state):
        full += token
    return full.strip()

# . DYNAMIC PATH HANDLING
MODEL_FILENAME = "ppo_mental_health_final.pt"
possible_paths = [
    os.path.join(PROJECT_ROOT, "models", MODEL_FILENAME),
    os.path.join(os.getcwd(), MODEL_FILENAME),
    os.path.join(BASE_DIR, "..", "..", "models", MODEL_FILENAME)
]

MODEL_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# . INITIALIZE MODEL (388 DIMENSIONS)
# state_dim MUST match your trainer (388)
model = PPOPolicy(state_dim=388, action_dim=3)

if MODEL_PATH:
    try:
        # map_location ensures it loads even if you trained on GPU but are eval-ing on CPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f".PPO model loaded from: {MODEL_PATH}")
    except Exception as e:
        print(f". PPO model load error: {e}")
        model = None
else:
    print(f". CRITICAL: {MODEL_FILENAME} not found. Running in fallback mode.")
    model = None

env = MentalHealthEnv()

user_inputs = [
    "I feel like a failure",
    "I have no energy",
    "Nothing makes sense anymore",
]

async def run():
    # Initial state
    state = env.reset()
    rewards = []
    results = []

    print(f". Starting evaluation on {len(user_inputs)} samples...")

    for user_input in user_inputs:
        print(f"Processing: '{user_input}'")
        
        # . UPDATE STATE BASED ON INPUT
        # This triggers the environment logic (mood drops, energy depletion)
        state = env.reset(user_input) 

        # . ROUTING (PPO + rules)
        # Note: router.py handles the encode_state() call internally via model.get_action
        action = route(state, user_input, model=model)

        # . SAFETY CHECK
        safe_action = safety_check(state, action)

        # . GENERATE REAL LLM RESPONSE
        response = await get_full_response(safe_action, user_input, state)

        # . GRADE RESPONSE
        score = await grade_with_llm(user_input, response)

        # . STEP ENV (Updates state for the next turn)
        state, _, _ = env.step(safe_action)

        rewards.append(score)
        results.append({
            "input": user_input,
            "response": response,
            "score": score,
            "agent": safe_action["type"]
        })

    print("\n. MindWeave Evaluation Complete")
    print(f"Final Scores: {rewards}")
    print(f"Average Score: {sum(rewards)/len(rewards):.2f}")

    # . SAVE DATA
    output_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "mindweave_scores.json"), "w") as f:
        json.dump(rewards, f)

    with open(os.path.join(output_dir, "mindweave_report_full.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run())