"""Text preprocessing for Cognitive Mirror emotion analysis.

CRITICAL DESIGN DECISIONS:
1. Negation words (not, no, never, nothing) are PRESERVED — removing them
   destroys the model's ability to distinguish "I am happy" from "I am not happy"
2. Contractions are normalized (don't → do not) so negation is always explicit
3. Punctuation is removed but not whitespace structure
4. Lowercasing only — no lemmatization (preserves "sad" vs "sadness" distinctions)
5. Stopwords REMOVED only for connecting words that carry no emotional signal

This module MUST produce IDENTICAL output during training AND inference.
"""

import re

# Negation words that MUST be preserved
NEGATION_WORDS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor"}

# Intensifiers — carry emotional signal, MUST be preserved
INTENSIFIERS = {
    "very", "really", "so", "too", "absolutely", "completely", "totally",
    "extremely", "utterly", "highly", "deeply", "incredibly", "quite",
    "somewhat", "slightly", "barely", "hardly", "almost",
}

# Pronoun words (self-referential) — moderate signal, KEEP
PRONOUNS_TO_KEEP = {"i", "me", "my", "myself", "we", "us", "our", "ourselves"}

# Stopwords that can safely be removed (no emotional signal)
# Conservative stopword list — only remove pure function words.
# Content words (feel, know, think, love, hate, etc.) MUST be preserved
# because they carry emotional signal.
REMOVABLE_STOPWORDS = {
    # Articles
    "the", "a", "an",
    # Auxiliary verbs ("be" forms)
    "am", "is", "are", "was", "were", "be", "been", "being",
    # Auxiliary verbs ("have" forms)
    "have", "has", "had", "having",
    # Auxiliary verbs ("do" forms)
    "do", "does", "did", "doing",
    # Modal verbs (but NOT "will" as noun, "can" as ability — these stay)
    "would", "shall", "could", "may", "might", "must", "should",
    # Prepositions
    "of", "in", "to", "for", "with", "on", "at", "by", "from",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "about",
    # Common conjunctions
    "and", "but", "or", "if", "while", "until", "because", "as",
    # Generic function words
    "that", "these", "those", "which", "who", "whom",
    "then", "than", "each", "every", "both", "few",
    "some", "such", "other", "most", "all", "more",
    "here", "there", "again", "further", "once",
    "just", "also", "same", "own", "now",
    "ago", "already", "always", "often", "usually",
    "perhaps", "maybe", "though", "although", "however",
    "still", "yet", "very", "much", "many",
}

# Contraction mapping
CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    "ain't": "am not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
}


def clean_text(text: str) -> str:
    """Clean text for ML model input.

    Preserves negation signals and emotional vocabulary while removing
    noise that would dilute TF-IDF features.

    Uses IDENTICAL logic during training and inference.

    Args:
        text: Raw input text

    Returns:
        Cleaned text suitable for vectorization
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Expand contractions so negation is explicit
    words = text.split()
    words = [CONTRACTIONS.get(w, w) for w in words]
    text = " ".join(words)

    # Remove punctuation but keep whitespace structure
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Filter — KEEP negation words, intensifiers, and self-referential pronouns
    kept_words = []
    for w in words:
        if len(w) <= 1:
            continue
        if w in NEGATION_WORDS:
            kept_words.append(w)
            continue
        if w in INTENSIFIERS:
            kept_words.append(w)
            continue
        if w in PRONOUNS_TO_KEEP:
            kept_words.append(w)
            continue
        if w not in REMOVABLE_STOPWORDS:
            kept_words.append(w)

    result = " ".join(kept_words)
    return result if result else text


def clean_text_light(text: str) -> str:
    """Lightweight cleaning — preserves more signal for Sherlock Lens analysis.

    Used for feature extraction in the Sherlock Lens, not for ML model input.
    Only lowercases and strips punctuation, keeps all words.

    Args:
        text: Raw input text

    Returns:
        Lightly cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
