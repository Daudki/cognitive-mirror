from flask import Flask, request, jsonify, render_template
from pathlib import Path
import pickle
import re

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

emotion_model = pickle.load(open(MODEL_DIR / "emotion.pkl", "rb"))
sentiment_model = pickle.load(open(MODEL_DIR / "sentiment.pkl", "rb"))
vectorizer = pickle.load(open(MODEL_DIR / "vectorizer.pkl", "rb"))
label_encoder_emotion = pickle.load(open(MODEL_DIR / "label_encoder.pkl", "rb"))
label_encoder_sentiment = pickle.load(open(MODEL_DIR / "label_encoder_sentiment.pkl", "rb"))

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


TEMPLATES = {
    "happy": {
        "positive": "The person is beaming with joy and optimism about life.",
        "negative": "A forced smile masks deeper pain beneath the surface.",
        "neutral": "A gentle sense of contentment colors their outlook.",
    },
    "sad": {
        "negative": "Deep sorrow and emotional heaviness weigh on them.",
        "positive": "Bittersweet melancholy carries echoes of something meaningful.",
        "neutral": "A quiet sadness lingers beneath their composed exterior.",
    },
    "fear": {
        "negative": "Anxiety and dread are overwhelming their thoughts.",
        "positive": "Cautious hope wrestles with nervous anticipation.",
        "neutral": "Subtle worry underlies their otherwise steady demeanor.",
    },
    "angry": {
        "negative": "Frustration and resentment are building toward a breaking point.",
        "positive": "Righteous anger fuels their determination for change.",
        "neutral": "Irritation simmers just beneath the surface.",
    },
    "neutral": {
        "positive": "Calm contentment with a steady emotional baseline.",
        "negative": "Emotionally flat and disconnected, merely going through motions.",
        "neutral": "A balanced and uneventful state of mind prevails.",
    },
    "excited": {
        "positive": "Enthusiasm and eager anticipation fill their mind.",
        "negative": "Nervous excitement borders on overwhelming anxiety.",
        "neutral": "A spark of anticipation adds energy to their thoughts.",
    },
    "frustrated": {
        "negative": "Thwarted efforts and mounting irritation dominate their focus.",
        "positive": "Frustration is channeled into productive determination.",
        "neutral": "Mild annoyance lingers from unresolved obstacles.",
    },
    "proud": {
        "positive": "Accomplishment and self-satisfaction give them a sense of victory.",
        "negative": "Pride is tainted by awareness of what could have gone better.",
        "neutral": "Quiet satisfaction comes from a job well done.",
    },
    "anxious": {
        "negative": "Worry and dread about what might come next.",
        "positive": "Nervous energy is tempered by hope and preparation.",
        "neutral": "Restless thoughts bounce between concern and calm.",
    },
    "confident": {
        "positive": "Self-assured certainty and unwavering belief in themselves.",
        "negative": "Overconfidence borders on arrogance and denial of risk.",
        "neutral": "Steady self-trust guides their decisions without fanfare.",
    },
    "loving": {
        "positive": "Deep affection and emotional warmth radiate from within.",
        "negative": "Love is tinged with fear of loss or unworthiness.",
        "neutral": "Genuine care and connection ground their emotional state.",
    },
    "hopeless": {
        "negative": "Loss of faith and deep discouragement cloud their outlook.",
        "positive": "Surrender brings an unexpected peace in letting go.",
        "neutral": "Resignation settles in as expectations fade away.",
    },
    "lonely": {
        "negative": "Deep isolation and yearning for meaningful connection.",
        "positive": "Solitude brings clarity and self-discovery.",
        "neutral": "Time alone feels neither good nor bad, simply empty.",
    },
    "grieving": {
        "negative": "Profound sorrow and loss weigh heavily on their spirit.",
        "positive": "Grief honors something deeply loved and now missed.",
        "neutral": "The weight of loss sits quietly in their daily awareness.",
    },
    "determined": {
        "positive": "Purposeful drive and commitment to forge ahead.",
        "negative": "Stubborn persistence masks fear of failure.",
        "neutral": "Steady resolve keeps them moving forward.",
    },
    "curious": {
        "positive": "Eager anticipation of discovery and new understanding.",
        "negative": "Restless questioning stems from dissatisfaction.",
        "neutral": "Open interest in what might unfold next.",
    },
    "guilty": {
        "negative": "Self-condemnation and remorse eat at them.",
        "positive": "Guilt motivates genuine efforts to make things right.",
        "neutral": "A nagging sense of having fallen short persists.",
    },
    "stoic": {
        "negative": "Emotional restraint masks deeper unresolved pain.",
        "positive": "Quiet strength and philosophical resolve carry them forward.",
        "neutral": "Calm acceptance with composed emotional restraint.",
    },
    "overwhelmed": {
        "negative": "Emotional exhaustion from unrelenting demands.",
        "positive": "Being needed by many brings a sense of purpose.",
        "neutral": "Busyness fills their days without clear direction.",
    },
    "inspired": {
        "positive": "Creative energy and motivation flow freely.",
        "negative": "Inspiration strikes but feels impossible to act upon.",
        "neutral": "New ideas spark interest without urgency.",
    },
    "content": {
        "positive": "Quiet satisfaction and appreciation for what is.",
        "negative": "Contentment masks a fear of wanting more.",
        "neutral": "A steady acceptance of the present moment.",
    },
    "empty": {
        "negative": "Emotional numbness and depletion of feeling.",
        "positive": "Emptiness creates space for something new to emerge.",
        "neutral": "A hollow quiet where emotions used to be.",
    },
}


def generate_mind_state(emotion, sentiment):
    if emotion in TEMPLATES:
        if sentiment in TEMPLATES[emotion]:
            return TEMPLATES[emotion][sentiment]
        return TEMPLATES[emotion].get("neutral", f"The person is feeling {emotion} with a {sentiment} outlook.")
    return f"The person is feeling {emotion} with a {sentiment} outlook."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    text = data.get("text", "").strip().strip('"').strip("'")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        clean = clean_text(text)

        if not clean:
            return jsonify({
                "emotion": "neutral",
                "sentiment": "neutral",
                "confidence": 0.0,
                "mind_state": "Unable to extract meaningful content from the input.",
                "top_emotions": [],
            })

        features = vectorizer.transform([clean])

        emotion_pred = emotion_model.predict(features)[0]
        sentiment_pred = sentiment_model.predict(features)[0]

        emotion_label = str(label_encoder_emotion.inverse_transform([emotion_pred])[0])
        sentiment_label = str(label_encoder_sentiment.inverse_transform([sentiment_pred])[0])

        emotion_proba = emotion_model.predict_proba(features)[0]
        confidence = float(emotion_proba.max())

        top_indices = emotion_proba.argsort()[-3:][::-1]
        top_emotions = [
            {
                "emotion": str(label_encoder_emotion.inverse_transform([i])[0]),
                "probability": round(float(emotion_proba[i]), 4),
            }
            for i in top_indices
        ]

        mind_state = generate_mind_state(emotion_label, sentiment_label)

        result = {
            "emotion": emotion_label,
            "sentiment": sentiment_label,
            "confidence": round(confidence, 4),
            "mind_state": mind_state,
            "top_emotions": top_emotions,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Unable to analyze: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)