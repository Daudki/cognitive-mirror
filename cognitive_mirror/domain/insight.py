"""Sherlock Lens insight model — deductions drawn across multiple entries."""

from datetime import datetime, timezone

from cognitive_mirror.extensions import db


class SherlockInsight(db.Model):
    """A single cross-entry deduction produced by the Sherlock Lens.

    Unlike Entry.distortions (single-entry, generated at submit time),
    these are produced by a periodic batch job that looks across a
    user's entry history for avoidance, contradiction, and trend patterns.
    """

    __tablename__ = "sherlock_insights"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    insight_type = db.Column(db.String(30), nullable=False)  # avoidance | contradiction | trend
    deduction = db.Column(db.Text, nullable=False)  # human-readable inference
    evidence = db.Column(db.JSON, default=list)  # [{entry_id, excerpt}]
    confidence = db.Column(db.Float, default=0.5)

    generated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "insight_type": self.insight_type,
            "deduction": self.deduction,
            "evidence": self.evidence or [],
            "confidence": self.confidence,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }
