"""Journal entry model — a single 'Mirror' reading."""

from datetime import datetime, timezone

from cognitive_mirror.extensions import db


class Entry(db.Model):
    """A single journal entry and its in-the-moment (Mirror) analysis.

    Sherlock insights are computed separately, across many entries, and
    live in SherlockInsight rather than on the entry itself.
    """

    __tablename__ = "entries"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    text = db.Column(db.Text, nullable=False)

    # Mirror output (from PredictorService + DistortionDetector)
    emotion = db.Column(db.String(50))
    sentiment = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    mind_state = db.Column(db.Text)
    distortions = db.Column(db.JSON, default=list)  # list of {type, sentence, confidence, explanation}

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "emotion": self.emotion,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "mind_state": self.mind_state,
            "distortions": self.distortions or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
