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

def predict(text):
    if vectorizer is None:
        init_models()
    features = vectorizer.transform([text])
    emotion_pred = emotion_model.predict(features)[0]
    sentiment_pred = sentiment_model.predict(features)[0]
    emotion_label = label_encoder.inverse_transform([emotion_pred])[0]
    return {
        "emotion": emotion_label,
        "sentiment": sentiment_pred,
    }