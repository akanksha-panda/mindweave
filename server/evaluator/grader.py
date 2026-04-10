import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# =========================
# . LOAD ENV (LOCAL ONLY)
# =========================
load_dotenv()  # loads MY_OPENAI_API_KEY from .env

# =========================
# . YOUR PERSONAL CONFIG
# =========================
MY_OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")
MY_BASE_URL = "https://api.openai.com/v1"
MY_MODEL_NAME = os.getenv("MY_MODEL_NAME", "gpt-4o-mini")

# =========================
# . YOUR CLIENT (SEPARATE)
# =========================
my_client = None

try:
    if MY_OPENAI_API_KEY:
        my_client = AsyncOpenAI(
            api_key=MY_OPENAI_API_KEY,
            base_url=MY_BASE_URL
        )
except Exception as e:
    print(f"[MY CLIENT ERROR] {e}")
    my_client = None


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
# . YOUR LLM GRADER (SEPARATE API)
# =========================
async def grade_with_my_llm(user_input, response):
    if my_client is None:
        return 0.5

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
        res = await my_client.chat.completions.create(
            model=MY_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = res.choices[0].message.content.strip()
        return safe_parse_score(text)

    except Exception as e:
        print(f"[MY LLM ERROR] {e}")
        return 0.5
