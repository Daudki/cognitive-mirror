import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
data = pd.read_csv("data.csv")

X = data["text"]

# Targets
y_emotion = data["emotion"]
y_sentiment = data["sentiment"]
y_confidence = data["confidence"]

# Vectorizer
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Models
emotion_model = LogisticRegression()
sentiment_model = LogisticRegression()
confidence_model = RandomForestRegressor()

# Train
emotion_model.fit(X_vec, y_emotion)
sentiment_model.fit(X_vec, y_sentiment)
confidence_model.fit(X_vec, y_confidence)

# Save everything
pickle.dump(emotion_model, open("emotion.pkl", "wb"))
pickle.dump(sentiment_model, open("sentiment.pkl", "wb"))
pickle.dump(confidence_model, open("confidence.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Multi-model training complete")