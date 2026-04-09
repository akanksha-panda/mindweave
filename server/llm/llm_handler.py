# server/llm/llm_handler.py

import httpx
import json
from server.emotions.emotion_mapper import get_response_style
from server.emotions.emotion_mapper import get_emotional_opening
import asyncio
from server.memory.vector_store import retrieve_memory
from server.memory.vector_store import add_memory


OLLAMA_URL = "http://localhost:11434/api/generate"


# =========================
# . BASIC CALL
# =========================
async def call_llm(prompt):
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            OLLAMA_URL,
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                 "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        )
        return response.json().get("response", "").strip()


# =========================
# . STREAMING
# =========================



async def stream_llm(prompt):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": "phi3",
                    "prompt": prompt,
                    "stream": True,
                    
                }
            ) as response:

                first_token_sent = False
                start_time = asyncio.get_event_loop().time()

                async for line in response.aiter_lines():

                    # . SAFETY: if no response for too long
                    if asyncio.get_event_loop().time() - start_time > 15 and not first_token_sent:
                        yield "..."
                        first_token_sent = True

                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        if data.get("done"):
                            break

                        token = data.get("response", "")

                        # .FIRST TOKEN FIX
                        if not first_token_sent:
                            yield ""   # cleaner than " "
                            first_token_sent = True

                        if token:
                            yield token

                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        print(". LLM STREAM FAILURE:", e)
        yield "Something went wrong. Try again."

# =========================
# . RENDER RESPONSE (CLEAN)
# =========================
async def generate_response_stream(action, user_input, state, context=""):
    strategy = action.get("type")
    mode = action.get("mode", "normal")
    base_text = action.get("text", "")

    emotion = state.get("emotion_category", "neutral")
    energy = state.get("energy", 1)
    opening = get_emotional_opening(emotion)

    style = get_response_style(emotion)

    tone = style.get("tone", "balanced")
    rules = style.get("rules", [])

    rules_text = "\n".join([f"- {r}" for r in rules]) if rules else ""
    safe_input = json.dumps(user_input)
    safe_base = json.dumps(base_text)
    memories = retrieve_memory(user_input)
    if memories:
        context += "\nRelevant past:\n" + "\n".join(memories)

    # =========================
    # . LIGHT STRATEGY GUIDANCE
    # =========================
    if strategy == "behavioral":
        instruction = "Suggest simple, low-effort appropriate actions."

    elif strategy == "cognitive":
        instruction = "Address the concern. validate feelings if detected and Gently challenge the thought and include one reflective question. DON'T GIVE ACTIONS BASED SUGGESTIONS.Avoid excessive questioning. Keep it natural."
        

    elif strategy == "emotional":
        instruction = """
        Adjust emotional tone based on intensity:
        - Strong emotions → deeper empathy
        - Mild emotions → light acknowledgment
        - Positive emotions → encouraging tone
        - Neutral → do NOT force empathy

        Do NOT overuse phrases like "I'm here with you"
        Keep it natural and context-aware.
        """
    elif strategy == "adaptive" and mode == "philosophical":
        instruction = """
        Respond with depth and thoughtfulness.

        - Acknowledge the emotional weight of the question
        - Give a reflective, meaningful answer
        - Do not give shallow reassurance
        - Avoid sounding like a lecture
        - Keep it human, warm, and slightly introspective

        The goal is to make the user feel understood, not just answered.
        """


    elif strategy == "adaptive":
        instruction = """
        Respond naturally and intelligently.
        - Do NOT correct grammar or spelling in the user input.
        - acknowledge even if there is no question in the input just a statement.
        - Do NOT over-analyze the sentence
        - Focus on what the user is trying to say
        - If unclear → gently ask or continue conversation
        - Keep it natural and smooth
        - Understand the user's intent from context
        - If it's conversational → continue naturally
        - If it's emotional → be supportive but not excessive
        - If it's informational → answer clearly

        Do not sound robotic.
        Do not force empathy.
        Keep it human and relevant.
        """

    


    else:
        instruction = "stay context aware and respond appropriately."

    # =========================
    # . ENERGY CONTROL
    # =========================
    brevity = ""
    if energy <= 1:
        brevity = "suggest minimal appropriate actions that can help to elevate mood, energy or lethargy. Avoid overwhelming the user"

    if state.get("intent") == "greeting":
        instruction = """
        Respond with a simple, natural greeting.
        Keep it short and casual.
        Do not ask deep or personal questions.
        """

    # =========================
    # . FINAL PROMPT (CLEAN)
    # =========================
    prompt = f"""
You are a supportive AI assistant.
Conversation so far:
{context}
User: {safe_input}

Context:
- Emotion: {emotion}
- Tone: {tone}

Guidelines:
- {rules_text}
- {instruction}
- {brevity}

Use emotional acknowledgment ONLY if appropriate.
Do not force it for neutral or informational inputs.
"{opening}"

Base direction (internal guidance only, DO NOT repeat verbatim):
{safe_base}
Stay strictly relevant to the user's current question.
Do NOT bring unrelated past topics unless clearly connected.

DO NOT repeat it mechanically. Make it feel human.
If user is rude, respond calmly but with grounded confidence.
Do not over-explain or sound defensive.
- Be natural and human
- No pet names like sweetheart, etc.
-Do not always end with a question. Sometimes just reflect.



Use this as inspiration, but DO NOT copy or repeat it directly.

Write the final response.
"""

    # =========================
    # . STREAM OUTPUT
    # =========================
    full_response = ""

    async for token in stream_llm(prompt):
        full_response += token
        yield token


    # . ensure something is returned
    if not full_response.strip():
        yield "Tell me a bit more about what’s on your mind."

    add_memory(f"user: {user_input}")
    add_memory(f"assistant: {full_response}")