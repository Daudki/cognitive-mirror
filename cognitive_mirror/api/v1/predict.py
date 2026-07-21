"""Stateless prediction endpoint — no auth, no persistence.

Useful for quick/anonymous testing and for the public landing page demo.
Authenticated users who want their entry saved and run through the Mirror's
distortion detection should use POST /api/v1/entries instead.
"""

from flask import Blueprint, jsonify, request
from marshmallow import ValidationError

from cognitive_mirror.schema.request import PredictRequestSchema
from cognitive_mirror.services.cache import CacheService
from cognitive_mirror.services.predictor import (
    ModelNotReadyError,
    PredictionError,
    PredictorService,
    TextTooLongError,
)

bp = Blueprint("predict", __name__)

_request_schema = PredictRequestSchema()


@bp.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        validated = _request_schema.load(payload)
    except ValidationError as err:
        return jsonify({"error": "validation_error", "detail": err.messages}), 400

    try:
        service = PredictorService(cache_service=CacheService())
        result = service.predict(
            validated["text"],
            include_explanation=validated.get("include_explanation", False),
        )
    except TextTooLongError as e:
        return jsonify({"error": "text_too_long", "detail": str(e)}), 400
    except ModelNotReadyError as e:
        return jsonify({"error": "model_not_ready", "detail": str(e)}), 503
    except PredictionError as e:
        return jsonify({"error": "prediction_failed", "detail": str(e)}), 500

    return jsonify({
        "request_id": result.request_id,
        "text": result.text,
        "emotion": result.emotion,
        "sentiment": result.sentiment,
        "mind_state": result.mind_state,
        "processing_time_ms": result.processing_time_ms,
        "model_version": result.model_version,
        "from_cache": result.from_cache,
    }), 200
