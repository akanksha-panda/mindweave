from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=API_KEY) if API_KEY else None


# =========================
# . RULE-BASED GRADER
# =========================
def grade_action(state, action):
    score = 0.5

    if action["type"] == "activity":
        if state.get("energy", 1) == 0 and action.get("intensity", 1) == 1:
            score = 0.9
        elif action.get("intensity", 1) > 2:
            score = 0.2

    if action["type"] == "reframe" and state.get("distortion", 0) > 7:
        score = 0.8

    if action["type"] == "empathy":
        score = 0.7

    return score


# =========================
# . SAFE PARSER
# =========================
def safe_parse_score(text):
    try:
        return float(text.strip())
    except:
        return 0.5


# =========================
# . OPENAI GRADER
# =========================
async def grade_with_llm(user_input, response):
    if client is None:
        return 0.5  # . fallback (no crash)

    prompt = f"""
Evaluate this therapy response:

User: {user_input}
AI: {response}

Score from 0 to 1 based on:
- empathy
- relevance
- helpfulness

Return ONLY a number between 0 and 1.
"""

    try:
        res = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = res.choices[0].message.content.strip()
        return float(text)

    except Exception as e:
        print(". OpenAI grading failed:", e)
        return 0.5