import pickle

emotion_model = pickle.load(open("emotion.pkl", "rb"))
sentiment_model = pickle.load(open("sentiment.pkl", "rb"))
confidence_model = pickle.load(open("confidence.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(text):
    vec = vectorizer.transform([text])

    emotion = emotion_model.predict(vec)[0]
    sentiment = sentiment_model.predict(vec)[0]
    confidence = confidence_model.predict(vec)[0]

    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "confidence": round(float(confidence), 2)
    }