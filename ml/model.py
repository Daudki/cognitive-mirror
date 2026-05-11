import os
import pickle

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

if os.path.exists(MODEL_PATH):
    try:
        saved = pickle.load(open(MODEL_PATH, "rb"))
        if isinstance(saved, dict):
            emotion_model = saved["emotion_model"]
            sentiment_model = saved["sentiment_model"]
            confidence_model = saved["confidence_model"]
            mind_state_model = saved["mind_state_model"]
            vectorizer = saved["vectorizer"]
        else:
            raise ValueError("model.pkl format not recognized")
    except (ValueError, KeyError, TypeError):
        emotion_model = pickle.load(open(os.path.join(MODEL_DIR, "emotion.pkl"), "rb"))
        sentiment_model = pickle.load(open(os.path.join(MODEL_DIR, "sentiment.pkl"), "rb"))
        confidence_model = pickle.load(open(os.path.join(MODEL_DIR, "confidence.pkl"), "rb"))
        mind_state_model = pickle.load(open(os.path.join(MODEL_DIR, "mind_state.pkl"), "rb"))
        vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb"))
else:
    emotion_model = pickle.load(open(os.path.join(MODEL_DIR, "emotion.pkl"), "rb"))
    sentiment_model = pickle.load(open(os.path.join(MODEL_DIR, "sentiment.pkl"), "rb"))
    confidence_model = pickle.load(open(os.path.join(MODEL_DIR, "confidence.pkl"), "rb"))
    mind_state_model = pickle.load(open(os.path.join(MODEL_DIR, "mind_state.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb"))

def predict(text):
    vec = vectorizer.transform([text])

    emotion = emotion_model.predict(vec)[0]
    sentiment = sentiment_model.predict(vec)[0]
    confidence = confidence_model.predict(vec)[0]
    mind_state = mind_state_model.predict(vec)[0]

    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "confidence": round(float(confidence), 2),
        "mind_state": mind_state
    }
