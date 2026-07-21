#!/usr/bin/env python3
"""Comprehensive model evaluation suite for Cognitive Mirror.

Evaluates the trained model on a curated test set covering:
- Basic emotions
- Negations
- Sarcasm
- Ambiguity
- Mixed emotions
- Edge cases
- Neutral/ambiguous statements

Usage:
    python ml/evaluate.py
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Test cases: (text, expected_emotion, expected_sentiment, category)
TEST_CASES = [
    # === Basic emotions ===
    ("I am very happy today", "joy", "positive", "basic"),
    ("I feel amazing and wonderful", "joy", "positive", "basic"),
    ("This is the best day of my life", "joy", "positive", "basic"),
    ("I am so grateful for everything", "joy", "positive", "basic"),
    ("I feel sad and lonely", "sadness", "negative", "basic"),
    ("Everything feels hopeless right now", "sadness", "negative", "basic"),
    ("I am heartbroken and devastated", "sadness", "negative", "basic"),
    ("I miss the people I love", "sadness", "negative", "basic"),
    ("I am furious about what happened", "anger", "negative", "basic"),
    ("This is completely unacceptable", "anger", "negative", "basic"),
    ("I hate how they treated me", "anger", "negative", "basic"),
    ("I am so frustrated right now", "anger", "negative", "basic"),
    ("I am terrified of what might happen", "fear", "negative", "basic"),
    ("I feel so anxious about tomorrow", "fear", "negative", "basic"),
    ("I am worried something bad will happen", "fear", "negative", "basic"),
    ("I feel panicked and overwhelmed", "fear", "negative", "basic"),
    ("That was completely unexpected", "surprise", "neutral", "basic"),
    ("I did not see that coming at all", "surprise", "neutral", "basic"),
    ("Wow, what a shock", "surprise", "neutral", "basic"),
    ("This is absolutely disgusting", "disgust", "negative", "basic"),
    ("I feel sick to my stomach", "disgust", "negative", "basic"),
    ("That is revolting", "disgust", "negative", "basic"),
    ("I feel calm and steady today", "neutral", "neutral", "basic"),
    ("Just another ordinary day", "neutral", "neutral", "basic"),
    ("I am going about my routine", "neutral", "neutral", "basic"),
    ("Nothing particularly notable happened", "neutral", "neutral", "basic"),

    # === NEGATIONS (critical test!) ===
    ("I am not happy today", "sadness", "negative", "negation"),
    ("I am not sad at all", "joy", "positive", "negation"),
    ("This is not good", "sadness", "negative", "negation"),
    ("I do not feel great", "sadness", "negative", "negation"),
    ("I am not angry, just disappointed", "sadness", "negative", "negation"),
    ("Not a bad day actually", "joy", "positive", "negation"),
    ("I cannot say I am happy", "sadness", "negative", "negation"),
    ("I do not hate this", "neutral", "neutral", "negation"),
    ("I am not afraid anymore", "joy", "positive", "negation"),
    ("Nothing feels right today", "sadness", "negative", "negation"),
    ("I never expected this to happen", "surprise", "neutral", "negation"),
    ("No one understands me", "sadness", "negative", "negation"),

    # === MIXED / CONTRADICTORY ===
    ("I am happy but I feel empty inside", "sadness", "negative", "mixed"),
    ("I succeeded but I feel nothing", "sadness", "negative", "mixed"),
    ("I am excited yet terrified", "fear", "negative", "mixed"),
    ("I love you but I hate how you make me feel", "anger", "negative", "mixed"),
    ("Everything is fine but something feels wrong", "fear", "negative", "mixed"),
    ("I should be happy but I am not", "sadness", "negative", "mixed"),
    ("Good things are happening but I feel numb", "sadness", "negative", "mixed"),
    ("I am grateful but also deeply sad", "sadness", "negative", "mixed"),

    # === AMBIGUOUS / NEUTRAL ===
    ("What time is the meeting", "neutral", "neutral", "ambiguous"),
    ("The weather is cloudy today", "neutral", "neutral", "ambiguous"),
    ("I ate lunch at noon", "neutral", "neutral", "ambiguous"),
    ("The train arrives at 3pm", "neutral", "neutral", "ambiguous"),
    ("I need to buy groceries", "neutral", "neutral", "ambiguous"),
    ("My computer is updating", "neutral", "neutral", "ambiguous"),
    ("I have a meeting tomorrow", "neutral", "neutral", "ambiguous"),
    ("The document is on the desk", "neutral", "neutral", "ambiguous"),

    # === SARCASTIC / INDIRECT ===
    ("Oh great, another problem to deal with", "anger", "negative", "sarcasm"),
    ("Yeah, that is exactly what I needed today", "anger", "negative", "sarcasm"),
    ("Wonderful, just wonderful", "anger", "negative", "sarcasm"),
    ("Thanks for nothing", "anger", "negative", "sarcasm"),
    ("I am fine, everything is fine", "sadness", "negative", "sarcasm"),
    ("Perfect, another deadline missed", "anger", "negative", "sarcasm"),

    # === SHORT SENTENCES ===
    ("Help", "fear", "negative", "short"),
    ("No", "anger", "negative", "short"),
    ("Yes", "joy", "positive", "short"),
    ("Why", "sadness", "negative", "short"),
    ("Stop", "anger", "negative", "short"),
    ("Wow", "surprise", "neutral", "short"),

    # === LONG SENTENCES ===
    (
        "I have been thinking about everything that happened over the past few months "
        "and I realize that despite all the challenges and difficulties, I have grown "
        "so much as a person and I am genuinely proud of how far I have come",
        "joy", "positive", "long"
    ),
    (
        "Every day feels like a struggle and I keep wondering if things will ever get "
        "better or if I am just stuck in this endless cycle of disappointment and failure",
        "sadness", "negative", "long"
    ),
    (
        "The way they handled the entire situation was completely unprofessional and "
        "disrespectful and I am tired of being treated like my concerns do not matter",
        "anger", "negative", "long"
    ),

    # === INFORMAL / SLANG ===
    ("feeling kinda meh today", "sadness", "negative", "slang"),
    ("this is lit", "joy", "positive", "slang"),
    ("ugh can't deal with this rn", "anger", "negative", "slang"),
    ("so done with everything", "sadness", "negative", "slang"),
    ("feeling blessed fr", "joy", "positive", "slang"),
    ("lowkey stressed but we move", "fear", "negative", "slang"),
    ("no cap this is amazing", "joy", "positive", "slang"),

    # === EDGE CASES ===
    ("I feel nothing", "neutral", "neutral", "edge"),
    ("I am beyond words", "surprise", "neutral", "edge"),
    ("It is what it is", "neutral", "neutral", "edge"),
    ("I do not know how I feel", "neutral", "neutral", "edge"),
    ("Emotions are complicated", "neutral", "neutral", "edge"),
    ("I want to scream but I also want to cry", "sadness", "negative", "edge"),
    ("Today was a day", "neutral", "neutral", "edge"),
    ("I think therefore I am", "neutral", "neutral", "edge"),
    ("Everything is fine", "neutral", "neutral", "edge"),  # Could be denial
    ("Everything is fine 🙂", "neutral", "neutral", "edge"),  # Emoji stripped
    ("I finally achieved my dream", "joy", "positive", "edge"),
    ("I can't do this anymore", "sadness", "negative", "edge"),
]


def evaluate_model():
    """Run the full evaluation suite."""
    from cognitive_mirror.preprocessing import clean_text
    from cognitive_mirror.models.manager import ModelManager

    # Initialize model if needed
    if not ModelManager.is_healthy():
        try:
            ModelManager.initialize()
        except Exception as e:
            print(f"ERROR: Could not load model: {e}")
            print("Run 'python ml/train.py' first to train the model.")
            sys.exit(1)

    print("=" * 70)
    print("COGNITIVE MIRROR - MODEL EVALUATION REPORT")
    print("=" * 70)
    print(f"Model version: {ModelManager.get_metadata().get('version', 'unknown')}")
    print(f"Emotion classes: {ModelManager.get_metadata().get('emotion_classes', [])}")
    print(f"Sentiment classes: {ModelManager.get_metadata().get('sentiment_classes', [])}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    results = []
    failures = []
    category_results = {}

    for text, expected_emotion, expected_sentiment, category in TEST_CASES:
        try:
            emotion_result = ModelManager.predict_emotion(text)
            sentiment_result = ModelManager.predict_sentiment(text)
        except Exception as e:
            results.append({
                "text": text,
                "expected_emotion": expected_emotion,
                "predicted_emotion": "ERROR",
                "expected_sentiment": expected_sentiment,
                "predicted_sentiment": "ERROR",
                "confidence": 0.0,
                "category": category,
                "error": str(e),
            })
            failures.append((text, category, str(e)))
            continue

        predicted_emotion = emotion_result["emotion"]
        predicted_sentiment = sentiment_result["sentiment"]
        confidence = emotion_result["confidence"]

        emotion_correct = predicted_emotion == expected_emotion
        sentiment_correct = predicted_sentiment == expected_sentiment

        results.append({
            "text": text,
            "expected_emotion": expected_emotion,
            "predicted_emotion": predicted_emotion,
            "expected_sentiment": expected_sentiment,
            "predicted_sentiment": predicted_sentiment,
            "confidence": confidence,
            "category": category,
            "emotion_correct": emotion_correct,
            "sentiment_correct": sentiment_correct,
        })

        if category not in category_results:
            category_results[category] = {"emotion": [], "sentiment": []}
        category_results[category]["emotion"].append(emotion_correct)
        category_results[category]["sentiment"].append(sentiment_correct)

    # --- Compute overall metrics ---
    y_true_emotion = [r["expected_emotion"] for r in results if "error" not in r]
    y_pred_emotion = [r["predicted_emotion"] for r in results if "error" not in r]
    y_true_sentiment = [r["expected_sentiment"] for r in results if "error" not in r]
    y_pred_sentiment = [r["predicted_sentiment"] for r in results if "error" not in r]

    # --- Per-category breakdown ---
    print("-" * 70)
    print("PER-CATEGORY BREAKDOWN")
    print("-" * 70)
    for category in sorted(category_results.keys()):
        emo_acc = np.mean(category_results[category]["emotion"]) if category_results[category]["emotion"] else 0
        sent_acc = np.mean(category_results[category]["sentiment"]) if category_results[category]["sentiment"] else 0
        n = len(category_results[category]["emotion"])
        print(f"  {category:15s} | n={n:2d} | Emotion acc: {emo_acc:.1%} | Sentiment acc: {sent_acc:.1%}")

    # --- Emotion metrics ---
    print()
    print("-" * 70)
    print("EMOTION CLASSIFICATION METRICS")
    print("-" * 70)
    emotion_acc = accuracy_score(y_true_emotion, y_pred_emotion)
    emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(
        y_true_emotion, y_pred_emotion, average="macro", zero_division=0
    )
    print(f"  Accuracy:  {emotion_acc:.2%}")
    print(f"  Precision: {emotion_precision:.2%}")
    print(f"  Recall:    {emotion_recall:.2%}")
    print(f"  F1-score:  {emotion_f1:.2%}")

    # --- Per-class emotion metrics ---
    print()
    print("  Per-class emotion metrics:")
    class_report = classification_report(
        y_true_emotion, y_pred_emotion, zero_division=0, output_dict=True
    )
    for label in sorted(set(y_true_emotion + y_pred_emotion)):
        if label in class_report:
            m = class_report[label]
            print(f"    {label:12s} | P:{m['precision']:.2f} R:{m['recall']:.2f} F1:{m['f1-score']:.2f} (n={m['support']:.0f})")

    # --- Confusion matrix ---
    print()
    print("-" * 70)
    print("EMOTION CONFUSION MATRIX")
    print("-" * 70)
    labels = sorted(set(y_true_emotion + y_pred_emotion))
    cm = confusion_matrix(y_true_emotion, y_pred_emotion, labels=labels)
    
    # Print header
    header = " " * 12 + "".join(f"{l:>8s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>12s}" + "".join(f"{cm[i][j]:8d}" for j in range(len(labels)))
        print(row)

    # --- Sentiment metrics ---
    print()
    print("-" * 70)
    print("SENTIMENT CLASSIFICATION METRICS")
    print("-" * 70)
    sentiment_acc = accuracy_score(y_true_sentiment, y_pred_sentiment)
    sentiment_precision, sentiment_recall, sentiment_f1, _ = precision_recall_fscore_support(
        y_true_sentiment, y_pred_sentiment, average="macro", zero_division=0
    )
    print(f"  Accuracy:  {sentiment_acc:.2%}")
    print(f"  Precision: {sentiment_precision:.2%}")
    print(f"  Recall:    {sentiment_recall:.2%}")
    print(f"  F1-score:  {sentiment_f1:.2%}")

    # --- Print failures in detail ---
    print()
    print("-" * 70)
    print("DETAILED FAILURES (emotion misclassifications)")
    print("-" * 70)
    emotion_failures = [r for r in results if "error" not in r and not r.get("emotion_correct", True)]
    for r in emotion_failures:
        clean = clean_text(r["text"])
        print(f"  TEXT:     \"{r['text'][:80]}\"")
        print(f"  CLEANED:  \"{clean}\"")
        print(f"  EXPECTED: {r['expected_emotion']} | GOT: {r['predicted_emotion']} | CONF: {r['confidence']:.2f}")
        print(f"  CATEGORY: {r['category']}")
        print()

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total test cases:         {len(TEST_CASES)}")
    print(f"  Emotion accuracy:         {emotion_acc:.2%}")
    print(f"  Sentiment accuracy:       {sentiment_acc:.2%}")
    print(f"  Emotion misclassifications: {len(emotion_failures)}")
    print(f"  Errors:                   {len(failures)}")
    print()

    # Save results to JSON for comparison
    output_path = project_root / "ml" / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "emotion_accuracy": emotion_acc,
            "emotion_precision_macro": emotion_precision,
            "emotion_recall_macro": emotion_recall,
            "emotion_f1_macro": emotion_f1,
            "sentiment_accuracy": sentiment_acc,
            "sentiment_precision_macro": sentiment_precision,
            "sentiment_recall_macro": sentiment_recall,
            "sentiment_f1_macro": sentiment_f1,
            "category_results": {k: {
                "emotion_accuracy": float(np.mean(v["emotion"])) if v["emotion"] else 0,
                "sentiment_accuracy": float(np.mean(v["sentiment"])) if v["sentiment"] else 0,
                "n": len(v["emotion"]),
            } for k, v in category_results.items()},
            "class_report": class_report,
            "confusion_matrix": cm.tolist(),
            "labels": labels,
            "failures": emotion_failures,
            "errors": [{"text": t, "category": c, "error": e} for t, c, e in failures],
        }, f, indent=2, default=str)

    print(f"Detailed results saved to: {output_path}")
    return results


if __name__ == "__main__":
    evaluate_model()
