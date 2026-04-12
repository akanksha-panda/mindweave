print("... ENVIRONMENT2 FILE LOADED ...")

import torch
import os
import copy
import traceback

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

ACTION_MAP = {0: "behavioral", 1: "cognitive", 2: "emotional"}

print(". FILE IMPORT STARTED")


class MindweaveEnvironment(Environment):
    
    

    
    def __init__(self, config=None, episode_id=None, **kwargs):
        print(". INIT STARTED")

        try:
            from server.environment import MentalHealthEnv

            # =========================
            # . BASE ENV
            # =========================
            self.env = MentalHealthEnv()

            self.test_inputs = [
                "I feel so depressed and I can't get out of bed.",
                "Why do I always feel like a failure at everything I try?",
                "I am feeling pretty excited today!",
            ]

            # . keep this (needed for state structure)
            self.env.reset(self.test_inputs[0])

            # =========================
            # . TASK STATE
            # =========================
            self.input_index = 0
            self.current_task_index = 0
            self.step_count = 0

            self.tasks = [
                "emotion_classification",
                "intent_detection",
                "agent_selection",
            ]
            self.last_agent = None
            # =========================
            # . LAZY MODEL
            # =========================
            self.rl_model = None
            self.state_dim = None

            # Model path kept (used later lazily)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.normpath(
                os.path.join(current_dir, "..", "model", "ppo_mental_health_final.pt")
            )

            print(".Environment Ready (Light Init).")

        except Exception as e:
            print(". CRITICAL INIT ERROR:")
            traceback.print_exc()
            raise e

    # =========================
    # . REQUIRED BY OPENENV
    # =========================
    @property
    def state(self) -> State:
        return State(
            episode_id="default",
            step_count=self.step_count
        )

    # =========================
    # . LAZY MODEL LOADER
    # =========================
    def _ensure_model_loaded(self):
        if self.rl_model is not None:
            return

        print(". Lazy loading PPO...")

        from server.rl.state_encoder import encode_state
        from server.rl.ppo_model import PPOPolicy

        dummy_state = self.env.state
        encoded = encode_state(dummy_state)
        self.state_dim = len(encoded)

        print(f". STATE DIM = {self.state_dim}")

        self.rl_model = PPOPolicy(self.state_dim)

        if os.path.exists(self.model_path):
            try:
                self.rl_model.load_state_dict(
                    torch.load(self.model_path, map_location="cpu")
                )
                self.rl_model.eval()
                print(".PPO model loaded")
            except Exception as e:
                print(". PPO load failed (continuing without weights)")
                print(e)
        else:
            print(". Model file not found")

    # =========================
    # . NORMALIZE REWARD
    # =========================
    def normalize_reward(self, r, task):
        if task in ["emotion_classification", "intent_detection"]:
            raw = float(max(0.0, min(1.0, r)))
        else:
            raw = float(max(0.0, min(1.0, (r + 5.0) / 20.0)))
        
        # ← KEY FIX: clamp to 0.1 - 0.99 (validator requirement)
        return max(0.1, min(0.99, raw))

    # =========================
    # . STEP ASYNC
    # =========================
    async def step_async(self, action):
        from server.rl.state_encoder import encode_state
        from models import MindweaveObservation

        self.step_count += 1

        # . ensure PPO ready
        self._ensure_model_loaded()

        current_task = self.tasks[self.current_task_index]
        gt_state = copy.deepcopy(self.initial_state)
        current_state = self.env.state

        user_pred = action.message.strip().lower()
        reward = 0.0

        # =========================
        # . PPO INFERENCE
        # =========================
        s_vec = torch.tensor(
            encode_state(current_state), dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.rl_model(s_vec)
            action_idx = torch.argmax(logits, dim=1).item()

        
        # =========================
        # . TASK LOGIC
        # =========================
        if current_task == "emotion_classification":
            gt = gt_state.get("emotion", "neutral")
            reward = 1.0 if user_pred == gt else 0.0

        elif current_task == "intent_detection":
            gt = gt_state.get("intent", "statement")
            reward = 1.0 if user_pred == gt else 0.0

        elif current_task == "agent_selection":
            agent = ACTION_MAP[action_idx]

            # . STORE PRE-STATE (for reward explanation)
            prev_state = copy.deepcopy(self.env.state)

            # . RUN ENV STEP
            next_state, raw_reward, _ = self.env.step({
                "type": agent,
                "task": "agent_selection"
            })

            # . STORE PPO AGENT (SAFE — AFTER STEP)
            self.last_agent = agent

            # . COMPUTE DELTAS (for logging/debugging)
            mood_before = prev_state.get("mood", 0)
            mood_after = next_state.get("mood", 0)

            dist_before = prev_state.get("distortion", 0)
            dist_after = next_state.get("distortion", 0)

            mood_delta = mood_after - mood_before
            dist_delta = dist_before - dist_after

            # . OPTIONAL: store in state for frontend / logs
            next_state["mood_delta"] = mood_delta
            next_state["distortion_delta"] = dist_delta
            next_state["agent"] = agent  # ensure persistence

            reward = raw_reward

        norm_reward = self.normalize_reward(reward, current_task)

        # =========================
        # . PROGRESSION
        # =========================
        self.current_task_index += 1

        if self.current_task_index >= len(self.tasks):
            self.current_task_index = 0
            self.input_index += 1

            if self.input_index < len(self.test_inputs):
                new_state = self.env.reset(self.test_inputs[self.input_index])

                # . UPDATE GT STATE FOR NEW INPUT
                self.initial_state = copy.deepcopy(new_state)

                # . reset agent
                self.env.state["agent"] = None

        done = self.input_index >= len(self.test_inputs)

        state = copy.deepcopy(self.env.state)
        state["agent"] = self.last_agent

        return MindweaveObservation(
            input=self.test_inputs[min(self.input_index, len(self.test_inputs) - 1)],
            task=self.tasks[self.current_task_index] if not done else "done",
            state=state,
            reward=norm_reward,
            done=done,
        )
    # =========================
    # . STEP WRAPPER
    # =========================
    def step(self, action):
        import asyncio
        return asyncio.run(self.step_async(action))

    # =========================
    # . RESET
    # =========================
    def reset(self):
        from models import MindweaveObservation
        import copy

        # =========================
        # . RESET INDICES
        # =========================
        self.input_index = 0
        self.current_task_index = 0
        self.step_count = 0

        # =========================
        # . LOAD FIRST INPUT
        # =========================
        user_input = self.test_inputs[self.input_index]

        # Reset base env
        state = self.env.reset(user_input)

        # =========================
        # . CACHE INITIAL STATE (CRITICAL FIX)
        # =========================
        self.initial_state = copy.deepcopy(state)

        # =========================
        # . CLEAR RL OUTPUT
        # =========================
        self.env.state["agent"] = None

        # =========================
        # . RETURN OBSERVATION
        # =========================
        return MindweaveObservation(
            input=user_input,
            task=self.tasks[self.current_task_index],
            state=self.env.state,
            reward=0.0,
            done=False,
        )
