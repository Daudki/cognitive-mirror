import os
import pandas as pd
import numpy as np
import pickle
import joblib
import re
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cognitive_mirror.preprocessing import clean_text
from ml.labels import map_emotion_label, map_sentiment_label

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = Path(__file__).resolve().parent / "data.csv"


def load_local_data(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip().str.strip('"').str.strip("'")
    df["emotion"] = df["emotion"].astype(str).str.strip().str.strip('"').str.strip("'")
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.strip('"').str.strip("'")
    df["emotion"] = df["emotion"].apply(map_emotion_label)
    df["sentiment"] = df["sentiment"].apply(map_sentiment_label)
    df["clean_text"] = df["text"].apply(clean_text)
    return df[["clean_text", "emotion", "sentiment"]]


def load_huggingface_data():
    frames = []
    try:
        from datasets import load_dataset

        go_emotions = load_dataset("go_emotions", "simplified", split="train")
        go_df = pd.DataFrame(go_emotions)
        go_df = go_df.rename(columns={"text": "text"})
        go_df["emotion"] = go_df["labels"].apply(
            lambda x: map_emotion_label(go_emotions.features["labels"].feature.int2str(x[0]) if isinstance(x, list) and x else "neutral")
        )
        go_df["sentiment"] = go_df["emotion"].apply(
            lambda e: "positive" if e in {"joy", "surprise"} else ("negative" if e in {"anger", "sadness", "fear", "disgust"} else "neutral")
        )
        go_df["clean_text"] = go_df["text"].apply(clean_text)
        frames.append(go_df[["clean_text", "emotion", "sentiment"]])
        print(f"Loaded {len(go_df)} samples from go_emotions")
    except Exception as e:
        print(f"go_emotions unavailable: {e}")

    try:
        emotion_dataset = load_dataset("dair-ai/emotion", "split", split="train")
        emo_df = pd.DataFrame(emotion_dataset)
        emo_df = emo_df.rename(columns={"text": "text", "label": "emotion"})
        label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
        emo_df["emotion"] = emo_df["emotion"].apply(lambda x: map_emotion_label(label_map.get(x, "neutral")))
        emo_df["sentiment"] = emo_df["emotion"].apply(
            lambda e: "positive" if e in {"joy", "love", "surprise"} else ("negative" if e in {"anger", "sadness", "fear"} else "neutral")
        )
        emo_df["clean_text"] = emo_df["text"].apply(clean_text)
        frames.append(emo_df[["clean_text", "emotion", "sentiment"]])
        print(f"Loaded {len(emo_df)} samples from dair-ai/emotion")
    except Exception as e:
        print(f"dair-ai/emotion unavailable: {e}")

    try:
        tweet_eval = load_dataset("tweet_eval", "sentiment", split="train")
        tw_df = pd.DataFrame(tweet_eval)
        tw_df = tw_df.rename(columns={"text": "text", "label": "sentiment_raw"})
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        tw_df["sentiment"] = tw_df["sentiment_raw"].apply(lambda x: sentiment_map.get(x, "neutral"))
        tw_df["emotion"] = tw_df["sentiment"].apply(
            lambda s: "joy" if s == "positive" else ("sadness" if s == "negative" else "neutral")
        )
        tw_df["clean_text"] = tw_df["text"].apply(clean_text)
        frames.append(tw_df[["clean_text", "emotion", "sentiment"]])
        print(f"Loaded {len(tw_df)} samples from tweet_eval sentiment")
    except Exception as e:
        print(f"tweet_eval unavailable: {e}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    local_df = load_local_data(DATA_PATH)
    local_weight = max(1, int(os.environ.get("LOCAL_SAMPLE_WEIGHT", "15")))
    if local_weight > 1:
        local_df = pd.concat([local_df] * local_weight, ignore_index=True)
    print(f"Local data: {len(local_df)} samples (weight={local_weight})")

    skip_hf = os.environ.get("TRAIN_SKIP_HF", "").lower() in ("1", "true", "yes")
    hf_df = None if skip_hf else load_huggingface_data()

    if hf_df is not None and len(hf_df) > 0:
        df = pd.concat([local_df, hf_df], ignore_index=True)
        print(f"Combined data: {len(df)} samples (local + HuggingFace)")
    else:
        df = local_df
        print("No internet datasets available, using local data only")

    df = df.drop_duplicates(subset=["clean_text"])
    df = df[df["clean_text"].str.strip() != ""]
    print(f"After dedup and cleaning: {len(df)} samples")

    local_count = len(local_df)
    max_rows = int(os.environ.get("TRAIN_MAX_ROWS", "20000"))
    if len(df) > max_rows:
        local_part = df.iloc[:local_count]
        extra = df.iloc[local_count:]
        need = max(0, max_rows - len(local_part))
        if need > 0 and len(extra) > 0:
            extra = extra.sample(n=min(need, len(extra)), random_state=42)
            df = pd.concat([local_part, extra], ignore_index=True)
        else:
            df = local_part
        print(f"Downsampled to {len(df)} samples ({local_count} local + {len(df) - local_count} external)")

    X = df["clean_text"].tolist()
    y_emotion = df["emotion"].tolist()
    y_sentiment = df["sentiment"].tolist()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
    )

    label_encoder_emotion = LabelEncoder()
    label_encoder_sentiment = LabelEncoder()

    X_vec = vectorizer.fit_transform(X)
    y_emotion_enc = label_encoder_emotion.fit_transform(y_emotion)
    y_sentiment_enc = label_encoder_sentiment.fit_transform(y_sentiment)

    emotion_model = LogisticRegression(max_iter=10000, C=0.5, solver="saga", class_weight="balanced")
    emotion_model.fit(X_vec, y_emotion_enc)

    sentiment_model = LogisticRegression(max_iter=10000, C=0.5, solver="saga", class_weight="balanced")
    sentiment_model.fit(X_vec, y_sentiment_enc)

    emotion_scores = cross_val_score(emotion_model, X_vec, y_emotion_enc, cv=3)
    sentiment_scores = cross_val_score(sentiment_model, X_vec, y_sentiment_enc, cv=3)

    print(f"\nEmotion CV accuracy: {emotion_scores.mean():.2%} (+/- {emotion_scores.std():.2%})")
    print(f"Sentiment CV accuracy: {sentiment_scores.mean():.2%} (+/- {sentiment_scores.std():.2%})")

    checkpoint = {
        "version": "3.0.0",
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

    test_texts = [
        "i am glad it finally worked",
        "i am not happy",
        "i feel amazing today",
        "everything is terrible",
        "i am so angry right now",
        "i am happy",
        "today was a good day",
        "i hate this so much",
    ]

    print("\nTest predictions:")
    for t in test_texts:
        clean = clean_text(t)
        features = vectorizer.transform([clean])
        ep = emotion_model.predict(features)[0]
        sp = sentiment_model.predict(features)[0]
        el = label_encoder_emotion.inverse_transform([ep])[0]
        sl = label_encoder_sentiment.inverse_transform([sp])[0]
        proba = emotion_model.predict_proba(features)[0].max()
        print(f"  '{t}' -> emotion={el}, sentiment={sl}, confidence={proba:.2%}")


if __name__ == "__main__":
    main()