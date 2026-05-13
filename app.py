#!/usr/bin/env python

from flask import Flask, request, jsonify, render_template
from pathlib import Path
import re

from cognitive_mirror.services.predictor import PredictorService
from cognitive_mirror.services.cache import CacheService
from cognitive_mirror.services.review import submit_case, list_pending, approve_case, list_approved
from cognitive_mirror.models.manager import ModelManager

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


from cognitive_mirror.preprocessing import clean_text


# Initialize ModelManager from existing pickle artifacts if available (backcompat)
def _bootstrap_models_from_pickles():
    import pickle
    try:
        emotion_model = pickle.load(open(MODEL_DIR / "emotion.pkl", "rb"))
        sentiment_model = pickle.load(open(MODEL_DIR / "sentiment.pkl", "rb"))
        vectorizer = pickle.load(open(MODEL_DIR / "vectorizer.pkl", "rb"))
        label_encoder = pickle.load(open(MODEL_DIR / "label_encoder.pkl", "rb"))
        # optional sentiment label encoder (keeps mapping consistent)
        try:
            label_encoder_sentiment = pickle.load(open(MODEL_DIR / "label_encoder_sentiment.pkl", "rb"))
        except Exception:
            label_encoder_sentiment = None
        # Populate ModelManager internal structures for compatibility
        ModelManager._models = {
            "emotion": emotion_model,
            "sentiment": sentiment_model,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "label_encoder_sentiment": label_encoder_sentiment,
        }
        ModelManager._metadata = {"version": "bootstrap-pickles", "loaded_at": "unknown"}
        ModelManager._initialized = True
    except Exception:
        # Leave ModelManager uninitialized; health checks will report it
        pass


_bootstrap_models_from_pickles()



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    text = (data or {}).get("text", "")
    consent = (data or {}).get("consent", False)

    if not text or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        service = PredictorService(cache_service=CacheService())
        result = service.predict(text)

        # If user opted in to data collection, submit to review queue
        if consent:
            submit_case({
                "text": text,
                "emotion": result.emotion,
                "sentiment": result.sentiment,
                "mind_state": result.mind_state,
                "consent": True,
            })

        return jsonify({
            "emotion": result.emotion.get("emotion"),
            "sentiment": result.sentiment.get("sentiment"),
            "confidence": result.emotion.get("confidence"),
            "mind_state": result.mind_state,
            "top_emotions": result.emotion.get("top_emotions", []),
        })

    except Exception as e:
        return jsonify({"error": f"Unable to analyze: {str(e)}"}), 500


@app.route("/review/pending", methods=["GET"])
def review_pending():
    items = list_pending()
    # Return minimal view for review
    return jsonify({"pending": items}), 200


@app.route("/review/approved", methods=["GET"])
def review_approved():
    items = list_approved()
    return jsonify({"approved": items}), 200


@app.route("/review/approve", methods=["POST"])
def review_approve():
    data = request.get_json() or {}
    idx = data.get("index")
    if idx is None:
        return jsonify({"error": "index is required"}), 400
    try:
        idx = int(idx)
    except Exception:
        return jsonify({"error": "index must be an integer"}), 400

    approved = approve_case(idx)
    if not approved:
        return jsonify({"error": "invalid index"}), 404
    return jsonify({"approved": approved}), 200


if __name__ == "__main__":
    app.run(debug=True)