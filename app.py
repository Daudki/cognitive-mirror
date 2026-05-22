#!/usr/bin/env python
"""Cognitive Mirror web app — local ML first, optional LLM for mind-state prose."""

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from cognitive_mirror.models.manager import ModelManager
from cognitive_mirror.services.cache import CacheService
from cognitive_mirror.services.predictor import PredictorService
from cognitive_mirror.services.review import approve_case, list_approved, list_pending, submit_case
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def _bootstrap_models() -> None:
    """Load trained artifacts; required for /predict to work without OpenAI."""
    checkpoint = MODEL_DIR / "model.pkl"
    if checkpoint.exists():
        ModelManager.initialize(str(checkpoint))
        return
    ModelManager.initialize(str(MODEL_DIR))


_bootstrap_models()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    consent = bool(data.get("consent", False))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not ModelManager.is_healthy():
        return jsonify({
            "error": "Local models are not loaded. Run: python -m ml.train",
        }), 503

    try:
        service = PredictorService(cache_service=CacheService())
        result = service.predict(text)

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
    return jsonify({"pending": list_pending()}), 200


@app.route("/review/approved", methods=["GET"])
def review_approved():
    return jsonify({"approved": list_approved()}), 200


@app.route("/review/approve", methods=["POST"])
def review_approve():
    data = request.get_json(silent=True) or {}
    idx = data.get("index")
    if idx is None:
        return jsonify({"error": "index is required"}), 400
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return jsonify({"error": "index must be an integer"}), 400

    approved = approve_case(idx)
    if not approved:
        return jsonify({"error": "invalid index"}), 404
    return jsonify({"approved": approved}), 200


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1")
