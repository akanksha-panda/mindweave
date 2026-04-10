---
title: Mindweave
emoji: 🔥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: multiagent mental help RL environment
---

## . Overview

MindWeave is a state-aware mental health support system built using a reinforcement learning (PPO-based) decision engine combined with a multi-agent architecture.
It is fully Dockerized and compliant with the OpenEnv specification.

It processes user input through a structured pipeline of:

Emotion Classification |
Intent Detection |
Agent Selection

Based on internal psychological state variables:

Mood |
Energy |
Distortion |
Sentiment

Each step updates the environment state and produces a reward, enabling adaptive and context-aware responses.

## 🧩 Key Features

. Environment-driven reasoning (state-based decisions)
🎯 Multi-step task pipeline (3 tasks per input)
🔁 9-step evaluation episodes (3 inputs × 3 tasks)
. Multi-agent system:
Emotional |
Cognitive |
Behavioral |
Adaptive
⚡ OpenEnv-compatible inference output ([START] → [STEP] → [END])
💬 Optional real-time chat UI
. Optional local LLM support via Ollama
. Achieves ~0.9+ therapeutic score out of 1 (competitive with LLM baselines)

⚙️ System Flow

1. User Input

2. Environment State Update

3. Emotion → Intent → Agent

4. Policy Decision (RL / Environment)

5. LLM (Echo only, no reasoning)

6. Environment Step + Reward

🧪 Example Output
[START]
[STEP] step=1 ...
...
[STEP] step=9 ...
[END]

---

---

## 🛠️ Setup & Installation

.Prerequisites
Python 3.10+
Git
uv
hugging Face
openenv
docker

1. Build the Image

From the root directory, run the following to create the Docker image:

docker build -t mindweave .

2. Run the Container
   To run the server locally while maintaining compatibility with evaluation scripts, use the following port mapping:

docker run -p 8000:7860 mindweave

Internal Port (7860): Required for Hugging Face Spaces compatibility.

External Port (8000): Mapping to 8000 allows the OpenEnv evaluation tools to connect seamlessly.

## 🛠️ Alternative Usage (UV)

If you prefer to run the server directly using the uv package manager without Docker, use the following commands:

1. Install Dependencies

uv pip install -e .

2. Run the Server
   To ensure the server matches the project's network configuration, explicitly define the port:

uv run server
or
uv run uvicorn server.app:app --reload

## 🛠️OpenEnv Specification

The project includes an openenv.yaml file that defines the environment metadata.

SDK: Docker

App Port: 7860

🛠️Secrets & API Keys
This app requires an OpenAI API Key for the LLM inference.

Local: Create a .env file and add OPENAI_API_KEY=your_key_here.

Production (Hugging Face): Add OPENAI_API_KEY as a Secret in the Space Settings.

.

## 📊 Evaluation & Inference

Once the server is running (via Docker or UV), you can run the inference script:

python inference.py

or
uv run python inference.py

## Expected Output:

[START]
[STEP] ...
[END]

---

## 💻 Live UI (Optional)

1. Start UI Backend
   python -m uv run uvicorn mindweave_env.server.main_ui:app --reload |
2. Open Frontend

Open:

index.html file and go live(in vs code)

Or run via Live Server:

http://127.0.0.1:5500/index.html |

. Local LLM

Install Ollama and run:

ollama pull phi3 |

Used for chat UI only — not required for evaluation

---

## . Evaluation(if want to compare baseline(only llm- llama) and Mindweave)

Results stored in:

server/evaluation/results/

---

## . Architecture

1. PPO-based decision policy
2. Environment-driven state transitions
3. Multi-agent response system
4. LLM used only as a response renderer (no reasoning)

. Notes

## If uv is not recognized, use:

python -m uv run ...
Ensure commands are run from project root
No GPU required

---

## 👩‍💻 Author

## Akanksha Panda

## Built for Meta PyTorch OpenEnv Hackathon

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
