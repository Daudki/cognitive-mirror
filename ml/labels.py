"""Canonical emotion/sentiment label mapping — redesigned for accuracy.

Key fixes from the broken v1:
1. Canonical names ARE in their own synonym sets (so "fear" maps to "fear")
2. Negation words preserved (NOT removed as stopwords)
3. Clear 7-class taxonomy with meaningful separations
4. Proper mapping for all emotion labels to 7 canonical classes

Taxonomy:
    joy      — happiness, contentment, excitement, gratitude, love, pride
    sadness  — grief, loneliness, disappointment, despair, melancholy
    anger    — frustration, rage, irritation, resentment, indignation
    fear     — anxiety, worry, dread, panic, nervousness, stress
    surprise — shock, amazement, astonishment, bewilderment
    disgust  — revulsion, aversion, repulsion, contempt
    neutral  — calm, balanced, stoic, indifferent, factual statements
"""

EMOTION_GROUPS: dict[str, set[str]] = {
    "joy": {
        # Canonical
        "joy", "joyful", "joyous",
        # Happiness
        "happy", "happiness", "glad", "delighted", "cheerful", "pleased",
        "thrilled", "elated", "overjoyed", "ecstatic", "euphoric",
        # Contentment
        "content", "satisfied", "fulfilled", "peaceful", "serene",
        # Excitement
        "excited", "enthusiastic", "eager", "energetic",
        # Gratitude
        "grateful", "thankful", "blessed", "appreciative",
        # Love
        "love", "loving", "affectionate", "caring", "compassionate",
        # Pride
        "proud", "accomplished", "confident", "empowered", "capable",
        # Optimism
        "optimistic", "hopeful", "positive", "upbeat",
        # Other positive
        "wonderful", "great", "fantastic", "amazing", "excellent",
        "good", "terrific", "fabulous", "marvelous", "splendid",
        "inspired", "motivated", "determined", "resilient",
    },
    "sadness": {
        # Canonical
        "sadness", "sad",
        # Grief
        "grief", "grieving", "mourning", "sorrow", "heartbroken",
        "devastated", "bereaved",
        # Loneliness
        "lonely", "alone", "isolated", "abandoned", "forgotten",
        # Disappointment
        "disappointed", "let down", "disheartened", "discouraged",
        # Despair
        "despair", "hopeless", "helpless", "defeated", "worthless",
        "empty", "numb", "hollow",
        # Melancholy
        "melancholy", "gloomy", "blue", "down", "low",
        # Pain
        "hurt", "wounded", "broken", "aching", "suffering",
        # Negative self
        "miserable", "unhappy", "depressed", "upset", "distraught",
        "wretched", "dismal", "dejected", "despondent",
        # Common expressions
        "crying", "tears", "weep", "sobbing",
    },
    "anger": {
        # Canonical
        "anger", "angry",
        # Frustration
        "frustrated", "frustration", "annoyed", "irritated", "aggravated",
        # Rage
        "rage", "furious", "enraged", "livid", "outraged",
        "seething", "boiling", "fuming",
        # Resentment
        "resentful", "bitter", "indignant", "offended",
        # Hostility
        "hostile", "aggressive", "mad", "pissed",
        # Dislike
        "hate", "hatred", "despise", "loathe", "detest",
        # Expressions
        "vengeful", "spiteful", "vindictive",
    },
    "fear": {
        # Canonical
        "fear", "fearful",
        # Anxiety
        "anxious", "anxiety", "nervous", "nervousness", "uneasy",
        "apprehensive", "on edge", "jittery",
        # Worry
        "worried", "worry", "concerned", "troubled",
        # Dread/Panic
        "dread", "panic", "panicked", "panicking",
        # Terror
        "terrified", "terrifying", "terrifiedly",
        "scared", "frightened", "terrified", "horrified",
        "petrified", "spooked",
        # Stress
        "stressed", "stress", "overwhelmed", "overwhelming",
        "pressured", "strained",
        # Insecurity
        "insecure", "vulnerable", "exposed", "threatened",
        # Common expressions
        "afraid", "fright", "alarmed", "paranoid",
    },
    "surprise": {
        # Canonical
        "surprise", "surprised",
        # Shock
        "shocked", "shock", "startled", "stunned", "thunderstruck",
        "flabbergasted", "taken aback",
        # Amazement
        "amazed", "astonished", "astounded", "awe", "awed",
        "dumbfounded", "speechless",
        # Bewilderment
        "bewildered", "perplexed", "puzzled", "confounded",
        # Unexpected positive/negative
        "unexpected", "unforeseen", "sudden",
        # Common expressions
        "wow", "whoa", "unbelievable", "incredible",
        "staggered", "jarred",
    },
    "disgust": {
        # Canonical
        "disgust", "disgusted",
        # Revulsion
        "revulsion", "repulsed", "repulsive", "revolted", "revolting",
        "nauseated", "nauseous", "sickened", "sickening",
        # Aversion
        "aversion", "distaste", "loathing", "abhorrence",
        # Contempt
        "contempt", "scorn", "disdain",
        # Common expressions
        "gross", "grossed out", "ew", "yuck",
        "vile", "filthy", "repugnant",
        "despise" "detest",
    },
    "neutral": {
        # Canonical
        "neutral",
        # Calm/steady
        "calm", "steady", "balanced", "centered", "grounded",
        "composed", "collected",
        # Stoic/indifferent
        "stoic", "indifferent", "unmoved", "unaffected", "detached",
        "apathetic", "dispassionate",
        # Okay/fine
        "okay", "ok", "fine", "alright", "all right",
        "normal", "usual", "average", "ordinary",
        # Factual
        "factual", "objective", "matter of fact",
        # Expressions
        "meh", "whatever", "so so",
        "nothing special", "no feelings",
    },
}


def map_emotion_label(label: str) -> str:
    """Map a raw emotion label to one of 7 canonical classes.

    Returns the canonical emotion name or 'neutral' if unrecognized.

    Args:
        label: Raw emotion label string (e.g., 'happy', 'fear', 'angry')

    Returns:
        Canonical emotion name: 'joy', 'sadness', 'anger', 'fear',
        'surprise', 'disgust', or 'neutral'
    """
    if not isinstance(label, str) or not label:
        return "neutral"

    s = label.lower().strip()

    # 1. Exact match against all synonym sets
    for canon, words in EMOTION_GROUPS.items():
        if s in words:
            return canon

    # 2. Substring match (e.g., "very_happy" contains "happy")
    for canon, words in EMOTION_GROUPS.items():
        for w in words:
            if len(w) >= 3 and w in s:
                return canon

    return "neutral"


ALL_CANONICAL_EMOTIONS = list(EMOTION_GROUPS.keys())
# ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']


def map_sentiment_label(label: str) -> str:
    """Map a raw sentiment label to positive/negative/neutral.

    Args:
        label: Raw sentiment label

    Returns:
        'positive', 'negative', or 'neutral'
    """
    if not isinstance(label, str) or not label:
        return "neutral"

    s = label.lower().strip()

    positive_words = {
        "positive", "pos", "good", "great", "1",
        "joy", "happy", "love", "happiness", "excited",
        "optimistic", "hopeful",
    }
    negative_words = {
        "negative", "neg", "bad", "poor", "0",
        "sadness", "sad", "anger", "angry", "fear",
        "hate", "terrible", "awful", "miserable",
        "depressed", "anxious", "worried",
    }
    neutral_words = {
        "neutral", "neu", "mixed", "2", "okay", "fine",
        "balanced", "calm",
    }

    if s in positive_words:
        return "positive"
    if s in negative_words:
        return "negative"
    if s in neutral_words:
        return "neutral"

    # Substring fallback
    for w in positive_words:
        if w in s:
            return "positive"
    for w in negative_words:
        if w in s:
            return "negative"
    return "neutral"


def emotion_to_sentiment(emotion: str) -> str:
    """Derive sentiment label from canonical emotion.

    Args:
        emotion: Canonical emotion name

    Returns:
        'positive', 'negative', or 'neutral'
    """
    positive_emotions = {"joy"}
    negative_emotions = {"sadness", "anger", "fear", "disgust"}
    neutral_emotions = {"neutral", "surprise"}

    if emotion in positive_emotions:
        return "positive"
    if emotion in negative_emotions:
        return "negative"
    return "neutral"
