# server/emotions/emotion_mapper.py

from server.emotions.emotion_data import FLAT_EMOTION_MAP
NEGATIONS = ["not", "no", "never", "dont", "don't", "doesnt", "doesn't"]

ABSOLUTES = ["never", "always", "nothing", "everything"]

# =========================
# . MAP TO CATEGORY (DYNAMIC)
# =========================
def map_to_category(emotions):
    if not emotions:
        return "neutral"

    best_category = None
    best_score = -1

    for e in emotions:
        e = e.lower()

        if e.startswith("not_"):
            continue

        if e in FLAT_EMOTION_MAP:
            cat = FLAT_EMOTION_MAP[e]["category"]

            # normalize
            if cat == "sad":
                cat = "sadness"

            # . score = length of word (proxy for specificity)
            score = len(e)

            if score > best_score:
                best_score = score
                best_category = cat

    return best_category if best_category else "neutral"



# =========================
# . APPLY STATE DELTA
# =========================
def apply_emotion_to_state(state, category):
    
    from server.emotions.emotion_data import CATEGORY_STATE_HINTS

    # . HARD SAFETY
    if not category or category not in CATEGORY_STATE_HINTS:
        category = "neutral"

    updates = CATEGORY_STATE_HINTS.get(category, {})

    for key, value in updates.items():
        if key == "sentiment":
            state[key] = value
        else:
            state[key] += value

    return state


# =========================
# . RESPONSE STYLE (FULLY EXPANDED)
# =========================
def get_response_style(category):

    styles = {

        # . NEGATIVE
        "afraid": {
            "tone": "reassuring, grounding",
            "rules": ["reduce fear", "create safety"]
        },

        "aversion": {
            "tone": "accepting, non-forceful",
            "rules": ["acknowledge resistance", "do not push"]
        },

        "disquiet": {
            "tone": "calm, stabilizing",
            "rules": ["reduce overwhelm"]
        },

        "fatigue": {
            "tone": "very soft, low-energy",
            "rules": ["keep very short", "suggest minimal effort"]
        },

        "tense": {
            "tone": "slow, calming",
            "rules": ["reduce anxiety", "slow pace"]
        },

        "annoyed": {
            "tone": "validating",
            "rules": ["acknowledge frustration", "avoid escalation"]
        },

        "confused": {
            "tone": "clear, simple",
            "rules": ["simplify everything"]
        },

        "angry": {
            "tone": "validating but stabilizing",
            "rules": ["acknowledge anger", "gently redirect"]
        },

        "embarrassed": {
            "tone": "non-judgmental",
            "rules": ["normalize feeling"]
        },

        "pain": {
            "tone": "deeply empathetic",
            "rules": ["no solutions", "just support"]
        },

        "sad": {
            "tone": "soft, empathetic",
            "rules": ["validate emotions"]
        },

        "vulnerable": {
            "tone": "safe, gentle",
            "rules": ["reassure softly"]
        },

        "disconnected": {
            "tone": "gentle, inviting",
            "rules": ["encourage connection"]
        },

        "yearning": {
            "tone": "understanding",
            "rules": ["acknowledge longing"]
        },

        # . POSITIVE
        "love": {
            "tone": "warm",
            "rules": ["reinforce connection"]
        },

        "confidence": {
            "tone": "encouraging",
            "rules": ["support growth"]
        },

        "focus": {
            "tone": "clear",
            "rules": ["reinforce clarity"]
        },

        "growth": {
            "tone": "motivational",
            "rules": ["support progress"]
        },

        "engagement": {
            "tone": "active",
            "rules": ["keep interaction engaging"]
        },

        "energy": {
            "tone": "energizing",
            "rules": ["channel energy positively"]
        },

        "joy": {
            "tone": "celebratory",
            "rules": ["amplify positivity"]
        },

        "calm": {
            "tone": "peaceful",
            "rules": ["maintain stability"]
        },

        "gratitude": {
            "tone": "appreciative",
            "rules": ["deepen appreciation"]
        },

        "creativity": {
            "tone": "expressive",
            "rules": ["encourage ideas"]
        },

        "kindness": {
            "tone": "warm",
            "rules": ["reinforce kindness"]
        },

        "stability": {
            "tone": "steady",
            "rules": ["maintain balance"]
        },

        "misc_positive": {
            "tone": "light positive",
            "rules": ["support gently"]
        },

        # . PHRASES
        "sadness": {
            "tone": "deep empathy",
            "rules": ["emotional validation"]
        },

        "fatigue_phrase": {
            "tone": "very low energy",
            "rules": ["tiny suggestions"]
        },

        "anxiety": {
            "tone": "grounding",
            "rules": ["calm breathing", "reduce worry"]
        },

        "fear": {
            "tone": "reassuring",
            "rules": ["provide safety"]
        },

        "anger": {
            "tone": "validating",
            "rules": ["acknowledge then calm"]
        },

        "confusion": {
            "tone": "simple",
            "rules": ["clarify step by step"]
        },

        "positive": {
            "tone": "encouraging",
            "rules": ["reinforce positivity"]
        },

        "calm_phrase": {
            "tone": "still, peaceful",
            "rules": ["maintain calm"]
        },

        # DEFAULT
        "neutral": {
            "tone": "balanced",
            "rules": []
        }
    }

    return styles.get(category, styles["neutral"])


def get_emotional_opening(emotion):
    openings = {
        "sadness": "I'm really sorry you're feeling this way.",
        "pain": "That sounds really painful to carry.",
        "vulnerable": "That takes courage to share.",
        "yearning": "Missing something or someone can feel really heavy.",
        "confusion": "It sounds like things feel unclear right now.",
        "anger": "I can sense there's frustration there.",
        "fear": "That sounds scary and unsettling.",
        "disconnected": "Feeling disconnected can be really isolating.",

        "joy": "That's really beautiful to hear.",
        "love": "That sounds warm and meaningful.",
        "gratitude": "That’s a really grounding perspective.",
        "calm": "There’s a nice sense of steadiness in that.",
        "confidence": "That sounds strong and self-assured.",
        "growth": "That shows real progress.",

        "neutral": "I'm here with you."
    }

    return openings.get(emotion, "I'm here with you.")