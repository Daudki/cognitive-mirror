from datetime import datetime, timezone
from types import SimpleNamespace


def test_generate_insights_for_user_returns_empty_when_not_enough_entries(monkeypatch):
    entries = [
        SimpleNamespace(text="I feel sad today", created_at=datetime.now(timezone.utc))
        for _ in range(3)
    ]

    class DummyQuery:
        def filter_by(self, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def all(self):
            return entries

    monkeypatch.setattr("cognitive_mirror.domain.entry.Entry.query", DummyQuery())

    from cognitive_mirror.services.sherlock import generate_insights_for_user

    insights = generate_insights_for_user(1)

    assert insights == []
