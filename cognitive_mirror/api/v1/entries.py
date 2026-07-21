"""Entry routes — submit a journal entry (Mirror runs here), list history."""

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required

from cognitive_mirror.extensions import db
from cognitive_mirror.domain.entry import Entry
from cognitive_mirror.services.cache import CacheService
from cognitive_mirror.services.predictor import PredictorService, PredictionError
from cognitive_mirror.services.distortion import DistortionDetector

bp = Blueprint("entries", __name__)

MAX_TEXT_LENGTH = 1000


@bp.route("/entries", methods=["POST"])
@login_required
def submit_entry():
    """Submit a journal entry. Runs the Mirror (sentiment/emotion + cognitive
    distortion detection) synchronously and stores the result.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Entry text is required"}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text exceeds maximum length ({MAX_TEXT_LENGTH})"}), 400

    try:
        predictor = PredictorService(cache_service=CacheService())
        prediction = predictor.predict(text)
    except PredictionError as e:
        return jsonify({"error": str(e)}), 503

    distortions = []
    try:
        detector = DistortionDetector()
        distortions = detector.analyze(text)
    except Exception:
        # Distortion detection is a bonus layer on top of the core prediction —
        # don't fail the whole submission if the LLM classification pass errors.
        distortions = []

    entry = Entry(
        user_id=current_user.id,
        text=text,
        emotion=prediction.emotion.get("emotion"),
        sentiment=prediction.sentiment.get("sentiment"),
        confidence=prediction.emotion.get("confidence"),
        mind_state=prediction.mind_state,
        distortions=distortions,
    )
    db.session.add(entry)
    db.session.commit()

    return jsonify({"entry": entry.to_dict()}), 201


@bp.route("/entries", methods=["GET"])
@login_required
def list_entries():
    """List the current user's entries, newest first, paginated."""
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)

    query = Entry.query.filter_by(user_id=current_user.id).order_by(Entry.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        "entries": [e.to_dict() for e in pagination.items],
        "page": page,
        "per_page": per_page,
        "total": pagination.total,
    }), 200


@bp.route("/entries/<int:entry_id>", methods=["GET"])
@login_required
def get_entry(entry_id: int):
    entry = Entry.query.filter_by(id=entry_id, user_id=current_user.id).first()
    if not entry:
        return jsonify({"error": "Entry not found"}), 404
    return jsonify({"entry": entry.to_dict()}), 200


@bp.route("/entries/<int:entry_id>", methods=["DELETE"])
@login_required
def delete_entry(entry_id: int):
    entry = Entry.query.filter_by(id=entry_id, user_id=current_user.id).first()
    if not entry:
        return jsonify({"error": "Entry not found"}), 404
    db.session.delete(entry)
    db.session.commit()
    return jsonify({"ok": True}), 200
