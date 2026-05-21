"""Canonical emotion/sentiment label mapping shared by training and inference."""

EMOTION_GROUPS = {
    "joy": {
        "happy", "joy", "glad", "amazing", "proud", "secure", "content", "satisfied",
        "relieved", "excited", "love", "loving", "wonderful", "great", "good", "fantastic",
        "blessed", "grateful", "thankful", "cheerful", "delighted", "elated", "enthusiastic",
        "optimistic", "playful", "pleased",
    },
    "anger": {
        "angry", "furious", "irritated", "annoyed", "rage", "frustrated", "mad", "outraged",
        "bitter", "resentful", "hostile", "aggravated", "livid",
    },
    "sadness": {
        "sad", "unhappy", "depressed", "miserable", "lonely", "down", "hopeless", "grief",
        "grieving", "sorrow", "heartbroken", "disappointed", "hurt", "despair", "devastated",
        "melancholy", "gloomy", "mournful",
    },
    "fear": {
        "afraid", "scared", "fearful", "anxious", "nervous", "worried", "terrified", "panic",
        "dread", "frightened", "uneasy", "apprehensive", "stressed", "overwhelmed",
    },
    "surprise": {
        "surprised", "shocked", "astonished", "amazed", "stunned", "startled", "bewildered",
        "speechless",
    },
    "disgust": {"disgust", "disgusted", "repulsed", "revolted", "sickened"},
    "neutral": {
        "neutral", "stoic", "calm", "resolute", "indifferent", "okay", "fine", "alright",
        "normal", "balanced", "steady",
    },
}


def map_emotion_label(label):
    if not isinstance(label, str) or not label:
        return "neutral"
    s = label.lower().strip()
    for canon, words in EMOTION_GROUPS.items():
        if s in words:
            return canon
    for canon, words in EMOTION_GROUPS.items():
        for w in words:
            if w in s:
                return canon
    return "neutral"


def map_sentiment_label(label):
    if not isinstance(label, str) or not label:
        return "neutral"
    s = label.lower().strip()
    if s in {"positive", "pos", "1", "joy", "love", "happiness"}:
        return "positive"
    if s in {"negative", "neg", "0", "sadness", "anger", "fear", "hate"}:
        return "negative"
    if s in {"neutral", "neu", "mixed", "2"}:
        return "neutral"
    return "neutral"
