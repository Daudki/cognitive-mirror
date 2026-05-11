import os
import pandas as pd
import numpy as np
import pickle
import joblib
import re
import string
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = Path(__file__).resolve().parent / "data.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["text"] = df["text"].astype(str).str.strip().str.strip('"').str.strip("'")
df["emotion"] = df["emotion"].astype(str).str.strip().str.strip('"').str.strip("'")
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.strip('"').str.strip("'")

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
              "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
              "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
              "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
              "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
              "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
              "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
              "at", "by", "for", "with", "about", "between", "into", "through", "during",
              "before", "after", "above", "below", "to", "from", "in", "out", "on", "off",
              "over", "under", "again", "further", "then", "once", "here", "there", "when",
              "where", "why", "how", "all", "both", "each", "few", "more", "most", "other",
              "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "can", "will", "just", "should", "now"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(words)


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