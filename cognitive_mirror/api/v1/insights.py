"""Sherlock Lens routes — cross-entry deductions."""

from flask import Blueprint, jsonify
from flask_login import current_user, login_required

from cognitive_mirror.extensions import db
from cognitive_mirror.domain.insight import SherlockInsight
from cognitive_mirror.services.sherlock import generate_insights_for_user

bp = Blueprint("insights", __name__)

MIN_ENTRIES_REQUIRED = 4


@bp.route("/insights/generate", methods=["POST"])
@login_required
def generate_insights():
    """Run the Sherlock Lens now (on-demand). In production this would also
    run as a scheduled job, but an explicit trigger keeps this usable without
    a task queue set up yet.
    """
    raw_insights = generate_insights_for_user(current_user.id)

    saved = []
    for item in raw_insights:
        insight = SherlockInsight(
            user_id=current_user.id,
            insight_type=item["insight_type"],
            deduction=item["deduction"],
            evidence=item["evidence"],
            confidence=item["confidence"],
        )
        db.session.add(insight)
        saved.append(insight)
    db.session.commit()

    return jsonify({"insights": [i.to_dict() for i in saved]}), 201


@bp.route("/insights", methods=["GET"])
@login_required
def list_insights():
    insights = (
        SherlockInsight.query.filter_by(user_id=current_user.id)
        .order_by(SherlockInsight.generated_at.desc())
        .limit(50)
        .all()
    )
    return jsonify({"insights": [i.to_dict() for i in insights]}), 200
