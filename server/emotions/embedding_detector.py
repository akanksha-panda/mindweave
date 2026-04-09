from sentence_transformers import util
from server.core.model_store import get_embedding_model
from server.emotions.emotion_data import ALL_EMOTIONS


_embeddings = None
_words = None


def preload_embeddings():
    global _embeddings, _words

    if _embeddings is not None:
        return

    #print(". Preloading emotion embeddings...")

    model = get_embedding_model()

    _words = ALL_EMOTIONS
    _embeddings = model.encode(
        _words,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    #print(".Emotion embeddings ready")


def get_embeddings():
    global _embeddings, _words

    if _embeddings is None:
        preload_embeddings()

    return _words, _embeddings


# . FIXED DETECTOR
def detect_emotions(user_input, threshold=0.7):
    try:
        model = get_embedding_model()
        if model is None:
            return None

        query_embedding = model.encode(
            user_input.lower(),
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        words, embeddings = get_embeddings()

        scores = util.cos_sim(query_embedding, embeddings)[0]

        best_idx = scores.argmax()
        best_score = scores[best_idx].item()
        best_word = words[best_idx]

        # . DO NOT return "neutral"
        if best_score < threshold:
            return None

        return best_word

    except Exception as e:
        print(". Emotion detection error:", e)
        return None