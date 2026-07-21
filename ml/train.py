#!/usr/bin/env python3
"""
Cognitive Mirror — Reproducible Training Pipeline (v2.0)

Architecture:
    TF-IDF + Logistic Regression (multi-class, calibrated)

Dataset:
    Synthetic + curated examples covering 7 emotion classes
    with linguistic diversity, negations, mixed emotions, and neutral statements.

Pipeline:
    1. Generate balanced synthetic dataset
    2. Load supplementary real datasets (optional)
    3. Clean and validate labels
    4. Preprocess text with IDENTICAL pipeline as inference
    5. TF-IDF vectorization
    6. Train Logistic Regression with class balancing
    7. Calibrate probabilities with Platt scaling
    8. Evaluate on held-out test set
    9. Serialize all artifacts for production

Usage:
    python ml/train.py
"""

import os
import sys
import pickle
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cognitive_mirror.preprocessing import clean_text
from ml.labels import (
    map_emotion_label,
    map_sentiment_label,
    emotion_to_sentiment,
    ALL_CANONICAL_EMOTIONS,
)
from ml.dataset_builder import TRAINING_DATA

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset() -> Tuple[List[str], List[str], List[str]]:
    """Build the training dataset from synthetic + optional external data.

    Returns:
        texts, emotion_labels, sentiment_labels
    """
    texts, emotions, sentiments = [], [], []

    # 1. Load synthetic training data
    for text, emotion, sentiment in TRAINING_DATA:
        texts.append(text)
        emotions.append(emotion)
        sentiments.append(sentiment)

    synthetic_count = len(texts)
    print(f"Synthetic training data: {synthetic_count} samples")

    # 2. Try loading supplementary real datasets (if available)
    hf_count = 0
    if os.environ.get("TRAIN_USE_HF", "").lower() in ("1", "true", "yes"):
        hf_texts, hf_emotions, hf_sentiments = _load_huggingface_data()
        texts.extend(hf_texts)
        emotions.extend(hf_emotions)
        sentiments.extend(hf_sentiments)
        hf_count = len(hf_texts)
        print(f"Supplementary HuggingFace data: {hf_count} samples")

    # 3. Preprocess and filter
    clean_texts, clean_emotions, clean_sentiments = [], [], []
    seen = set()

    for text, emotion, sentiment in zip(texts, emotions, sentiments):
        cleaned = clean_text(text)
        if not cleaned or len(cleaned.split()) < 1:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        clean_texts.append(cleaned)
        clean_emotions.append(emotion)
        clean_sentiments.append(sentiment)

    # 4. Report class distribution
    from collections import Counter
    emotion_counts = Counter(clean_emotions)
    print(f"\nAfter dedup: {len(clean_texts)} samples")
    print("Emotion distribution:")
    for cls in ALL_CANONICAL_EMOTIONS:
        print(f"  {cls:12s}: {emotion_counts.get(cls, 0):4d}")

    return clean_texts, clean_emotions, clean_sentiments


def _load_huggingface_data() -> Tuple[List[str], List[str], List[str]]:
    """Load supplementary datasets from HuggingFace if available."""
    texts, emotions, sentiments = [], [], []
    try:
        from datasets import load_dataset

        # dair-ai/emotion dataset
        try:
            ds = load_dataset("dair-ai/emotion", "split", split="train", trust_remote_code=True)
            label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
            for item in ds:
                t = str(item.get("text", "")).strip()
                if not t:
                    continue
                emotion = label_map.get(item.get("label", -1), "neutral")
                if emotion == "love":
                    emotion = "joy"
                sentiment = emotion_to_sentiment(emotion)
                texts.append(t)
                emotions.append(emotion)
                sentiments.append(sentiment)
            print(f"  Loaded {len(texts)} from dair-ai/emotion")
        except Exception as e:
            print(f"  dair-ai/emotion skipped: {e}")
    except ImportError:
        print("  datasets library not available — skipping HF data")
    except Exception as e:
        print(f"  HF data loading failed: {e}")

    return texts, emotions, sentiments


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    texts: List[str],
    emotions: List[str],
    sentiments: List[str],
    model_dir: Path = MODEL_DIR,
):
    """Train emotion and sentiment classifiers.

    Architecture:
        - TF-IDF vectorizer (n-grams 1-2, max 8000 features)
        - Logistic Regression with class_weight='balanced'
        - CalibratedClassifierCV for reliable probability estimates
    """
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)

    # --- Train/validation/test split (70/15/15) ---
    # Use stratified split to preserve class distribution
    X_temp, X_test, y_e_temp, y_e_test, y_s_temp, y_s_test = train_test_split(
        texts, emotions, sentiments, test_size=0.15, random_state=42, stratify=emotions
    )
    X_train, X_val, y_e_train, y_e_val, y_s_train, y_s_val = train_test_split(
        X_temp, y_e_temp, y_s_temp, test_size=0.1765, random_state=42, stratify=y_e_temp
    )  # 0.1765 * 0.85 ≈ 0.15

    print(f"\nTrain:   {len(X_train)} samples")
    print(f"Val:     {len(X_val)} samples")
    print(f"Test:    {len(X_test)} samples")

    # --- TF-IDF Vectorization ---
    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # --- Label Encoding ---
    emotion_encoder = LabelEncoder()
    sentiment_encoder = LabelEncoder()

    y_e_train_enc = emotion_encoder.fit_transform(y_e_train)
    y_e_val_enc = emotion_encoder.transform(y_e_val)
    y_e_test_enc = emotion_encoder.transform(y_e_test)

    y_s_train_enc = sentiment_encoder.fit_transform(y_s_train)
    y_s_val_enc = sentiment_encoder.transform(y_s_val)
    y_s_test_enc = sentiment_encoder.transform(y_s_test)

    print(f"Emotion classes: {emotion_encoder.classes_.tolist()}")
    print(f"Sentiment classes: {sentiment_encoder.classes_.tolist()}")

    # --- Train Emotion Classifier ---
    print("\nTraining emotion classifier...")
    emotion_model = LogisticRegression(
        max_iter=5000,
        C=0.5,
        solver="saga",
        class_weight="balanced",
        random_state=42,
    )

    # Calibrate for meaningful confidence scores
    emotion_calibrated = CalibratedClassifierCV(
        estimator=emotion_model,
        method="sigmoid",  # Platt scaling
        cv=3,
    )
    emotion_calibrated.fit(X_train_vec, y_e_train_enc)

    # --- Train Sentiment Classifier ---
    print("Training sentiment classifier...")
    sentiment_model = LogisticRegression(
        max_iter=5000,
        C=0.5,
        solver="saga",
        class_weight="balanced",
        random_state=42,
    )
    sentiment_calibrated = CalibratedClassifierCV(
        estimator=sentiment_model,
        method="sigmoid",
        cv=3,
    )
    sentiment_calibrated.fit(X_train_vec, y_s_train_enc)

    # --- Evaluation ---
    print("\n" + "-" * 40)
    print("EVALUATION (Validation Set)")
    print("-" * 40)

    # Emotion metrics
    y_e_val_pred = emotion_calibrated.predict(X_val_vec)
    e_acc = accuracy_score(y_e_val_enc, y_e_val_pred)
    e_p, e_r, e_f1, _ = precision_recall_fscore_support(
        y_e_val_enc, y_e_val_pred, average="macro", zero_division=0
    )
    print(f"Emotion — Accuracy: {e_acc:.2%}  Precision: {e_p:.2%}  Recall: {e_r:.2%}  F1: {e_f1:.2%}")

    # Sentiment metrics
    y_s_val_pred = sentiment_calibrated.predict(X_val_vec)
    s_acc = accuracy_score(y_s_val_enc, y_s_val_pred)
    s_p, s_r, s_f1, _ = precision_recall_fscore_support(
        y_s_val_enc, y_s_val_pred, average="macro", zero_division=0
    )
    print(f"Sentiment — Accuracy: {s_acc:.2%}  Precision: {s_p:.2%}  Recall: {s_r:.2%}  F1: {s_f1:.2%}")

    # Per-class emotion report
    print("\nPer-class emotion metrics (validation):")
    val_report = classification_report(
        y_e_val_enc, y_e_val_pred,
        target_names=emotion_encoder.classes_,
        zero_division=0,
    )
    print(val_report)

    # Cross-validation
    print("\nCross-validation (3-fold)...")
    X_all_vec = vectorizer.transform(texts)
    y_e_all_enc = emotion_encoder.transform(emotions)
    cv_scores = cross_val_score(
        emotion_calibrated, X_all_vec, y_e_all_enc, cv=3, scoring="accuracy"
    )
    print(f"CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    # --- Test set evaluation ---
    print("\n" + "-" * 40)
    print("FINAL EVALUATION (Test Set)")
    print("-" * 40)
    y_e_test_pred = emotion_calibrated.predict(X_test_vec)
    test_acc = accuracy_score(y_e_test_enc, y_e_test_pred)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(
        y_e_test_enc, y_e_test_pred, average="macro", zero_division=0
    )
    print(f"Emotion — Accuracy: {test_acc:.2%}  Precision: {test_p:.2%}  Recall: {test_r:.2%}  F1: {test_f1:.2%}")

    print("\nTest confusion matrix:")
    cm = confusion_matrix(y_e_test_enc, y_e_test_pred)
    labels = emotion_encoder.classes_.tolist()
    header = " " * 12 + "".join(f"{l:>8s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>12s}" + "".join(f"{cm[i][j]:8d}" for j in range(len(labels)))
        print(row)

    # --- Serialize ---
    print("\nSaving model artifacts...")
    timestamp = datetime.now(timezone.utc).isoformat()

    # Single checkpoint (recommended)
    checkpoint = {
        "version": "2.0.0",
        "trained_at": timestamp,
        "emotion_model": emotion_calibrated,
        "sentiment_model": sentiment_calibrated,
        "vectorizer": vectorizer,
        "label_encoder": emotion_encoder,
        "label_encoder_sentiment": sentiment_encoder,
        "emotion_classes": emotion_encoder.classes_.tolist(),
        "sentiment_classes": sentiment_encoder.classes_.tolist(),
        "metrics": {
            "val_emotion_accuracy": e_acc,
            "val_emotion_f1_macro": e_f1,
            "test_emotion_accuracy": test_acc,
            "test_emotion_f1_macro": test_f1,
            "n_samples": len(texts),
        },
    }

    joblib.dump(checkpoint, model_dir / "model.pkl")
    print(f"  Saved: {model_dir / 'model.pkl'}")

    # Legacy pickle files for backward compatibility
    with open(model_dir / "emotion.pkl", "wb") as f:
        pickle.dump(emotion_calibrated, f)
    with open(model_dir / "sentiment.pkl", "wb") as f:
        pickle.dump(sentiment_calibrated, f)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(emotion_encoder, f)
    with open(model_dir / "label_encoder_sentiment.pkl", "wb") as f:
        pickle.dump(sentiment_encoder, f)

    # Training metadata (human-readable)
    metadata = {
        "version": "2.0.0",
        "trained_at": timestamp,
        "emotion_classes": emotion_encoder.classes_.tolist(),
        "sentiment_classes": sentiment_encoder.classes_.tolist(),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "total_samples": len(texts),
        "class_distribution": {str(k): int(v) for k, v in zip(
            *np.unique(emotions, return_counts=True)
        )},
        "metrics": {
            "val_emotion_accuracy": float(e_acc),
            "val_emotion_f1_macro": float(e_f1),
            "test_emotion_accuracy": float(test_acc),
            "test_emotion_f1_macro": float(test_f1),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
        },
        "config": {
            "tfidf_max_features": 8000,
            "tfidf_ngram_range": [1, 2],
            "model_type": "LogisticRegression",
            "calibration": "Platt scaling (sigmoid)",
            "class_weight": "balanced",
        },
    }
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {model_dir / 'training_metadata.json'}")
    print("\nTraining complete.")

    # --- Quick smoke test ---
    print("\n" + "-" * 40)
    print("SMOKE TEST")
    print("-" * 40)
    test_inputs = [
        "I am very happy today",
        "I am not happy today",
        "I feel sad and lonely",
        "I am not sad at all",
        "I am furious about this",
        "I am terrified of what might happen",
        "What time is the meeting",
        "Everything is fine",
        "I am fine, everything is fine",
        "Oh great another problem",
        "I hate this so much",
        "Wow I did not expect that",
        "This is absolutely disgusting",
        "I am happy but feel empty inside",
        "Help me please",
    ]

    for t in test_inputs:
        cleaned = clean_text(t)
        features = vectorizer.transform([cleaned])

        e_pred = emotion_calibrated.predict(features)[0]
        e_proba = emotion_calibrated.predict_proba(features)[0]
        e_label = emotion_encoder.inverse_transform([e_pred])[0]
        e_conf = float(e_proba.max())

        s_pred = sentiment_calibrated.predict(features)[0]
        s_label = sentiment_encoder.inverse_transform([s_pred])[0]

        print(f"  '{t}' -> emotion={e_label}, sentiment={s_label}, confidence={e_conf:.2f}")

    return checkpoint


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("COGNITIVE MIRROR — TRAINING PIPELINE v2.0")
    print("=" * 60)

    texts, emotions, sentiments = build_dataset()
    train_model(texts, emotions, sentiments)


if __name__ == "__main__":
    main()
