import os
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run 'python ml/train.py' first to generate model files."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

emotion_model = None
sentiment_model = None
vectorizer = None
label_encoder = None

def init_models():
    global emotion_model, sentiment_model, vectorizer, label_encoder
    emotion_model = load_model("emotion.pkl")
    sentiment_model = load_model("sentiment.pkl")
    vectorizer = load_model("vectorizer.pkl")
    label_encoder = load_model("label_encoder.pkl")
    try:
        label_encoder_sentiment = load_model("label_encoder_sentiment.pkl")
    except FileNotFoundError:
        label_encoder_sentiment = None
    from cognitive_mirror.models.manager import ModelManager

    ModelManager._models = {
        "emotion": emotion_model,
        "sentiment": sentiment_model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "label_encoder_sentiment": label_encoder_sentiment,
    }
    ModelManager._initialized = True

def predict(text):
    """Run local inference; returns API-shaped dict for scripts and tests."""
    from cognitive_mirror.models.manager import ModelManager

    if not ModelManager.is_healthy():
        init_models()

    emotion_result = ModelManager.predict_emotion(text)
    sentiment_result = ModelManager.predict_sentiment(text)
    mind_state = ModelManager.generate_mindstate(
        emotion_result, sentiment_result, raw_text=text
    )
    return {
        "emotion": emotion_result["emotion"],
        "sentiment": sentiment_result["sentiment"],
        "confidence": emotion_result["confidence"],
        "mind_state": mind_state,
        "top_emotions": emotion_result.get("top_emotions", []),
    }