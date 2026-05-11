import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

data = pd.read_csv("data/data.csv")
X = data["text"]
y_emotion = data["emotion"]
y_sentiment = data["sentiment"]
y_confidence = data["confidence"]
y_mind_state = data["mind_state"]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_vec = vectorizer.fit_transform(X)

emotion_model = LogisticRegression(max_iter=1000)
sentiment_model = LogisticRegression(max_iter=1000)
confidence_model = RandomForestRegressor(n_estimators=100)
mind_state_model = LogisticRegression(max_iter=1000)

emotion_model.fit(X_vec, y_emotion)
sentiment_model.fit(X_vec, y_sentiment)
confidence_model.fit(X_vec, y_confidence)
mind_state_model.fit(X_vec, y_mind_state)

pickle.dump(emotion_model, open(os.path.join(MODEL_DIR, "emotion.pkl"), "wb"))
pickle.dump(sentiment_model, open(os.path.join(MODEL_DIR, "sentiment.pkl"), "wb"))
pickle.dump(confidence_model, open(os.path.join(MODEL_DIR, "confidence.pkl"), "wb"))
pickle.dump(mind_state_model, open(os.path.join(MODEL_DIR, "mind_state.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb"))

combined = {
    "vectorizer": vectorizer,
    "emotion_model": emotion_model,
    "sentiment_model": sentiment_model,
    "confidence_model": confidence_model,
    "mind_state_model": mind_state_model,
}
pickle.dump(combined, open(os.path.join(MODEL_DIR, "model.pkl"), "wb"))

print("Training complete ? Cognitive layer built")
