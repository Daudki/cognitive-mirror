"""Load sklearn artifacts and run local emotion/sentiment inference."""

from __future__ import annotations

import os
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from cognitive_mirror.preprocessing import clean_text
from cognitive_mirror.llm.prompting import build_prompt
from cognitive_mirror.llm.adapters import BaseAdapter, DummyAdapter, OpenAIAdapter, LocalAdapter

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DATA_PATH = BASE_DIR / "ml" / "data.csv"

# Fallback mind-state lines when no CSV match exists
MIND_STATE_FALLBACKS: Dict[Tuple[str, str], List[str]] = {
    ("joy", "positive"): [
        "They seem uplifted and optimistic, with energy directed toward what matters to them.",
        "A positive, open mood suggests satisfaction and forward-looking hope.",
    ],
    ("sadness", "negative"): [
        "Sadness and heaviness appear to weigh on their thoughts and motivation.",
        "They may be grieving, disappointed, or feeling emotionally drained.",
    ],
    ("anger", "negative"): [
        "Frustration or anger seems present, possibly tied to perceived unfairness or blocked goals.",
        "Their tone suggests irritation that may need acknowledgment before it escalates.",
    ],
    ("fear", "negative"): [
        "Anxiety or worry appears to dominate, with attention on uncertain or threatening outcomes.",
        "They may feel on edge, seeking safety or clarity about what happens next.",
    ],
    ("surprise", "positive"): [
        "Unexpected positive news or events seem to have caught them off guard in a good way.",
    ],
    ("surprise", "neutral"): [
        "Something unexpected has disrupted their expectations; they may still be processing it.",
    ],
    ("disgust", "negative"): [
        "Strong aversion or repulsion colors their reaction to the situation.",
    ],
    ("neutral", "neutral"): [
        "They appear emotionally steady, neither strongly positive nor negative.",
        "A calm, matter-of-fact stance suggests routine processing rather than intense feeling.",
    ],
    ("neutral", "positive"): [
        "Quiet contentment or mild satisfaction seems present beneath a low-key tone.",
    ],
    ("joy", "neutral"): [
        "Light positivity shows through without intense excitement.",
    ],
}


def _get_llm_adapter() -> BaseAdapter:
    provider = os.environ.get("LLM_PROVIDER", "dummy").lower().strip()
    if provider in ("api", "openai") and os.environ.get("OPENAI_API_KEY"):
        return OpenAIAdapter()
    if provider == "local":
        try:
            return LocalAdapter()
        except Exception:
            pass
    return DummyAdapter()


def _load_mind_state_templates() -> Dict[Tuple[str, str], List[str]]:
    """Build (emotion, sentiment) -> mind_state snippets from training CSV."""
    templates: Dict[Tuple[str, str], List[str]] = {}
    if not DATA_PATH.exists():
        return templates
    try:
        import pandas as pd
        from ml.labels import map_emotion_label, map_sentiment_label

        df = pd.read_csv(DATA_PATH)
        for _, row in df.iterrows():
            emotion = map_emotion_label(str(row.get("emotion", "neutral")))
            sentiment = map_sentiment_label(str(row.get("sentiment", "neutral")))
            mind = str(row.get("mind_state", "")).strip()
            if not mind:
                continue
            key = (emotion, sentiment)
            templates.setdefault(key, [])
            if mind not in templates[key]:
                templates[key].append(mind)
    except Exception:
        pass
    return templates


class ModelManager:
    """Singleton-style access to trained classifiers and mind-state generation."""

    _models: Dict[str, Any] = {}
    _metadata: Dict[str, Any] = {}
    _initialized: bool = False
    _mind_templates: Dict[Tuple[str, str], List[str]] = {}
    _llm_adapter: Optional[BaseAdapter] = None

    @classmethod
    def initialize(cls, model_path: Optional[str] = None) -> None:
        path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
        if path.is_file():
            cls._load_checkpoint(path)
            return
        cls._load_legacy_pickles(path.parent)

    @classmethod
    def _load_checkpoint(cls, path: Path) -> None:
        checkpoint = joblib.load(path)
        cls._models = {
            "emotion": checkpoint["emotion_model"],
            "sentiment": checkpoint["sentiment_model"],
            "vectorizer": checkpoint["vectorizer"],
            "label_encoder": checkpoint["label_encoder"],
            "label_encoder_sentiment": checkpoint.get("label_encoder_sentiment"),
        }
        cls._metadata = {
            "version": checkpoint.get("version", "unknown"),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "emotion_classes": checkpoint.get("emotion_classes", []),
            "sentiment_classes": checkpoint.get("sentiment_classes", []),
        }
        cls._initialized = True
        cls._mind_templates = _load_mind_state_templates()

    @classmethod
    def _load_legacy_pickles(cls, model_dir: Path) -> None:
        try:
            with open(model_dir / "emotion.pkl", "rb") as f:
                emotion_model = pickle.load(f)
            with open(model_dir / "sentiment.pkl", "rb") as f:
                sentiment_model = pickle.load(f)
            with open(model_dir / "vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            with open(model_dir / "label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
            try:
                with open(model_dir / "label_encoder_sentiment.pkl", "rb") as f:
                    label_encoder_sentiment = pickle.load(f)
            except FileNotFoundError:
                label_encoder_sentiment = None
            cls._models = {
                "emotion": emotion_model,
                "sentiment": sentiment_model,
                "vectorizer": vectorizer,
                "label_encoder": label_encoder,
                "label_encoder_sentiment": label_encoder_sentiment,
            }
            cls._metadata = {
                "version": "legacy-pickles",
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }
            cls._initialized = True
            cls._mind_templates = _load_mind_state_templates()
        except FileNotFoundError:
            cls._initialized = False

    @classmethod
    def is_healthy(cls) -> bool:
        return cls._initialized and bool(cls._models)

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        return dict(cls._metadata)

    @classmethod
    def _vectorize(cls, text: str):
        cleaned = clean_text(text)
        return cls._models["vectorizer"].transform([cleaned]), cleaned

    @classmethod
    def predict_emotion(cls, text: str) -> Dict[str, Any]:
        features, _ = cls._vectorize(text)
        model = cls._models["emotion"]
        encoder = cls._models["label_encoder"]
        pred_idx = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        classes = encoder.classes_
        emotion = encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])
        top_indices = proba.argsort()[-3:][::-1]
        top_emotions = [
            {"emotion": classes[i], "probability": round(float(proba[i]), 4)}
            for i in top_indices
        ]
        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "top_emotions": top_emotions,
        }

    @classmethod
    def predict_sentiment(cls, text: str) -> Dict[str, Any]:
        features, _ = cls._vectorize(text)
        model = cls._models["sentiment"]
        pred_idx = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        sentiment_encoder = cls._models.get("label_encoder_sentiment")
        if sentiment_encoder is not None:
            sentiment = sentiment_encoder.inverse_transform([pred_idx])[0]
            classes = sentiment_encoder.classes_
        else:
            sentiment = str(pred_idx)
            classes = getattr(model, "classes_", [pred_idx])
        confidence = float(proba[pred_idx])
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
        }

    @classmethod
    def _template_mind_state(
        cls,
        emotion: str,
        sentiment: str,
        confidence: float,
    ) -> str:
        key = (emotion, sentiment)
        pool = cls._mind_templates.get(key) or MIND_STATE_FALLBACKS.get(key)
        if not pool:
            pool = MIND_STATE_FALLBACKS.get((emotion, "neutral")) or MIND_STATE_FALLBACKS.get(
                ("neutral", sentiment)
            ) or [
                "They appear to be processing their experience with mixed or subtle emotional cues."
            ]
        text = random.choice(pool)
        if confidence < 0.45:
            text = f"Tentatively: {text}"
        return text

    @classmethod
    def generate_mindstate(
        cls,
        emotion_result: Dict[str, Any],
        sentiment_result: Dict[str, Any],
        raw_text: Optional[str] = None,
    ) -> str:
        emotion = emotion_result.get("emotion", "neutral")
        sentiment = sentiment_result.get("sentiment", "neutral")
        confidence = emotion_result.get("confidence", 0.5)

        provider = os.environ.get("LLM_PROVIDER", "dummy").lower().strip()
        use_llm = provider not in ("", "none", "disabled")

        if provider in ("api", "openai"):
            use_llm = bool(os.environ.get("OPENAI_API_KEY"))
        elif provider == "local":
            use_llm = bool(os.environ.get("LLM_LOCAL_MODEL"))

        if not use_llm:
            return cls._template_mind_state(emotion, sentiment, confidence)

        if cls._llm_adapter is None:
            cls._llm_adapter = _get_llm_adapter()

        prompt = build_prompt(raw_text or "", emotion_result, sentiment_result)
        try:
            out = cls._llm_adapter.generate(
                prompt,
                max_tokens=120,
                emotion=emotion,
                sentiment=sentiment,
                confidence=confidence,
            )
            text = (out.get("text") or "").strip()
            if text:
                return text
        except Exception:
            pass
        return cls._template_mind_state(emotion, sentiment, confidence)
