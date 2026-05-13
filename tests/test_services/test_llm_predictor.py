import pytest

from cognitive_mirror.services.predictor import PredictorService
from cognitive_mirror import models as models_pkg
from cognitive_mirror.models import manager as manager_module
from cognitive_mirror.llm.adapters import DummyAdapter


def test_predictor_uses_llm_adapter(monkeypatch):
    # Ensure ModelManager reports healthy
    monkeypatch.setattr(manager_module.ModelManager, "is_healthy", lambda: True)

    # Patch emotion and sentiment predictors to return deterministic values
    monkeypatch.setattr(
        manager_module.ModelManager,
        "predict_emotion",
        lambda text: {"emotion": "joy", "confidence": 0.95, "top_emotions": []},
    )
    monkeypatch.setattr(
        manager_module.ModelManager,
        "predict_sentiment",
        lambda text: {"sentiment": "positive", "confidence": 0.9},
    )

    # Patch DummyAdapter.generate to return a fixed, testable string
    monkeypatch.setattr(
        DummyAdapter,
        "generate",
        lambda self, prompt, max_tokens=256, **kwargs: {"text": "LLM GENERATED TEXT", "model": "dummy", "tokens": 5},
    )

    svc = PredictorService()
    result = svc.predict("I feel great today")

    assert result.mind_state == "LLM GENERATED TEXT"
    assert result.emotion["emotion"] == "joy"
    assert result.sentiment["sentiment"] == "positive"
