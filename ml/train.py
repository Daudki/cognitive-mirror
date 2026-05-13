import os
import pandas as pd
import numpy as np
import pickle
import joblib
import re
import string
from pathlib import Path
import sys

# Ensure project root is on sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local imports that require project root on sys.path
from cognitive_mirror.preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = Path(__file__).resolve().parent / "data.csv"
APPROVED_PATH = BASE_DIR / "data" / "interaction_cases" / "approved.jsonl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

import argparse
import json


def load_approved(path: Path):
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                continue
    return items


parser = argparse.ArgumentParser(description="Train emotion and sentiment classifiers")
parser.add_argument("--use-approved", action="store_true", help="Include approved review cases from data/interaction_cases/approved.jsonl")
args = parser.parse_args()


df = pd.read_csv(DATA_PATH)
if args.use_approved:
    approved = load_approved(APPROVED_PATH)
    if approved:
        # Build a small DataFrame with required columns and append
        rows = []
        for a in approved:
            text = a.get("text")
            emotion = None
            sentiment = None
            # approved cases may include emotion/sentiment structures
            if isinstance(a.get("emotion"), dict):
                emotion = a.get("emotion", {}).get("emotion")
            else:
                emotion = a.get("emotion")
            if isinstance(a.get("sentiment"), dict):
                sentiment = a.get("sentiment", {}).get("sentiment")
            else:
                sentiment = a.get("sentiment")

            if text and emotion and sentiment:
                rows.append({"text": text, "emotion": emotion, "sentiment": sentiment})

        if rows:
            df_extra = pd.DataFrame(rows)
            print(f"Loaded {len(df_extra)} approved cases for training")
            df = pd.concat([df, df_extra], ignore_index=True)

df["text"] = df["text"].astype(str).str.strip().str.strip('"').str.strip("'")
df["emotion"] = df["emotion"].astype(str).str.strip().str.strip('"').str.strip("'")
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.strip('"').str.strip("'")

# Use shared preprocessing.clean_text

# Consolidate fine-grained emotion labels into a smaller canonical set
EMOTION_GROUPS = {
    "joy": {"happy", "joy", "glad", "amazing", "proud", "secure", "content", "satisfied", "relieved", "excited"},
    "anger": {"angry", "angryness", "furious", "irritated", "annoyed", "rage"},
    "sadness": {"sad", "unhappy", "depressed", "miserable", "lonely", "down"},
    "fear": {"afraid", "scared", "fearful", "anxious", "nervous", "worried"},
    "surprise": {"surprised", "shocked", "astonished"},
    "disgust": {"disgust", "disgusted", "repulsed"},
    "neutral": {"neutral", "stoic", "calm", "resolute", "indifferent"},
}


def map_emotion_label(label: str) -> str:
    if not isinstance(label, str) or not label:
        return "neutral"
    s = label.lower().strip()
    # exact match
    for canon, words in EMOTION_GROUPS.items():
        if s in words:
            return canon
    # substring match fallback
    for canon, words in EMOTION_GROUPS.items():
        for w in words:
            if w in s:
                return canon
    # default fallback
    return "neutral"


df["emotion"] = df["emotion"].apply(map_emotion_label)


df["clean_text"] = df["text"].apply(clean_text)

print(f"Loaded {len(df)} samples")
print(f"Emotions: {df['emotion'].value_counts().to_dict()}")
print(f"Sentiments: {df['sentiment'].value_counts().to_dict()}")

X = df["clean_text"].tolist()
y_emotion = df["emotion"].tolist()
y_sentiment = df["sentiment"].tolist()

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True,
)

label_encoder_emotion = LabelEncoder()
label_encoder_sentiment = LabelEncoder()

X_vec = vectorizer.fit_transform(X)
y_emotion_enc = label_encoder_emotion.fit_transform(y_emotion)
y_sentiment_enc = label_encoder_sentiment.fit_transform(y_sentiment)

emotion_model = LogisticRegression(
    max_iter=3000,
    C=0.5,
    solver='saga',
    class_weight='balanced',
)
emotion_model.fit(X_vec, y_emotion_enc)

sentiment_model = LogisticRegression(
    max_iter=3000,
    C=0.5,
    solver='saga',
    class_weight='balanced',
)
sentiment_model.fit(X_vec, y_sentiment_enc)

emotion_scores = cross_val_score(emotion_model, X_vec, y_emotion_enc, cv=3)
sentiment_scores = cross_val_score(sentiment_model, X_vec, y_sentiment_enc, cv=3)

print(f"\nEmotion CV accuracy: {emotion_scores.mean():.2%} (+/- {emotion_scores.std():.2%})")
print(f"Sentiment CV accuracy: {sentiment_scores.mean():.2%} (+/- {sentiment_scores.std():.2%})")

checkpoint = {
    "version": "2.0.0",
    "emotion_model": emotion_model,
    "sentiment_model": sentiment_model,
    "vectorizer": vectorizer,
    "label_encoder": label_encoder_emotion,
    "label_encoder_sentiment": label_encoder_sentiment,
    "emotion_classes": label_encoder_emotion.classes_.tolist(),
    "sentiment_classes": label_encoder_sentiment.classes_.tolist(),
}

joblib.dump(checkpoint, MODEL_DIR / "model.pkl")

with open(MODEL_DIR / "emotion.pkl", "wb") as f:
    pickle.dump(emotion_model, f)
with open(MODEL_DIR / "sentiment.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)
with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder_emotion, f)
with open(MODEL_DIR / "label_encoder_sentiment.pkl", "wb") as f:
    pickle.dump(label_encoder_sentiment, f)

print(f"\nModels saved to {MODEL_DIR}")
print(f"Emotion classes: {label_encoder_emotion.classes_.tolist()}")
print(f"Sentiment classes: {label_encoder_sentiment.classes_.tolist()}")
os.system("cls" if os.name == "nt" else "clear")

print("\nTest predictions:")
test_texts = [
    "i am glad it finally worked",
    "i am not happy",
    "i feel amazing today",
    "everything is terrible",
    "i am so angry right now",
]

for t in test_texts:
    clean = clean_text(t)
    features = vectorizer.transform([clean])
    ep = emotion_model.predict(features)[0]
    sp = sentiment_model.predict(features)[0]
    el = label_encoder_emotion.inverse_transform([ep])[0]
    sl = label_encoder_sentiment.inverse_transform([sp])[0]
    proba = emotion_model.predict_proba(features)[0].max()
    print(f"  '{t}' -> emotion={el}, sentiment={sl}, confidence={proba:.2%}")