"""Sherlock Lens — Evidence-based analysis layer for Cognitive Mirror.

Two modes:
1. Single-entry analysis: deep linguistic + emotional analysis of one text
2. Cross-entry insights: pattern detection across a user's entry history

Design principles:
- Every claim is grounded in the actual input text
- Observations are clearly separated from inferences
- Speculation is explicitly labeled
- Confidence is honestly communicated
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from cognitive_mirror.preprocessing import clean_text_light

# ============================================================================
# Emotional vocabulary with intensity estimates
# ============================================================================

EMOTIONAL_LEXICON: Dict[str, Tuple[str, float]] = {
    # joy
    "happy": ("joy", 0.7), "joy": ("joy", 0.8), "wonderful": ("joy", 0.7),
    "amazing": ("joy", 0.7), "love": ("joy", 0.8), "grateful": ("joy", 0.6),
    "blessed": ("joy", 0.6), "excited": ("joy", 0.7), "thrilled": ("joy", 0.8),
    "proud": ("joy", 0.6), "delighted": ("joy", 0.7), "fantastic": ("joy", 0.7),
    "beautiful": ("joy", 0.6), "glorious": ("joy", 0.7), "cheerful": ("joy", 0.6),
    "ecstatic": ("joy", 0.9), "elated": ("joy", 0.8), "jubilant": ("joy", 0.8),
    "content": ("joy", 0.5), "peaceful": ("joy", 0.5), "satisfied": ("joy", 0.5),
    "accomplished": ("joy", 0.6), "triumphant": ("joy", 0.8),
    # sadness
    "sad": ("sadness", 0.7), "sadness": ("sadness", 0.8), "lonely": ("sadness", 0.7),
    "grief": ("sadness", 0.9), "heartbroken": ("sadness", 0.9), "crying": ("sadness", 0.8),
    "hopeless": ("sadness", 0.8), "empty": ("sadness", 0.6), "miserable": ("sadness", 0.8),
    "devastated": ("sadness", 0.9), "sorrow": ("sadness", 0.8), "melancholy": ("sadness", 0.7),
    "despair": ("sadness", 0.9), "worthless": ("sadness", 0.8), "numb": ("sadness", 0.5),
    "disappointed": ("sadness", 0.6), "regret": ("sadness", 0.6), "hurt": ("sadness", 0.7),
    "abandoned": ("sadness", 0.8), "isolated": ("sadness", 0.7),
    # anger
    "angry": ("anger", 0.8), "furious": ("anger", 0.9), "rage": ("anger", 0.9),
    "hate": ("anger", 0.8), "frustrated": ("anger", 0.7), "irritated": ("anger", 0.6),
    "outraged": ("anger", 0.9), "resentful": ("anger", 0.7), "livid": ("anger", 0.9),
    "infuriated": ("anger", 0.9), "mad": ("anger", 0.7), "seething": ("anger", 0.9),
    "bitter": ("anger", 0.6), "indignant": ("anger", 0.7),
    # fear
    "afraid": ("fear", 0.7), "scared": ("fear", 0.7), "terrified": ("fear", 0.9),
    "anxious": ("fear", 0.7), "worried": ("fear", 0.6), "nervous": ("fear", 0.6),
    "panic": ("fear", 0.9), "dread": ("fear", 0.8), "stressed": ("fear", 0.6),
    "overwhelmed": ("fear", 0.7), "frightened": ("fear", 0.7), "paranoid": ("fear", 0.8),
    "vulnerable": ("fear", 0.6), "insecure": ("fear", 0.5),
    # surprise
    "surprised": ("surprise", 0.7), "shocked": ("surprise", 0.8), "stunned": ("surprise", 0.8),
    "amazed": ("surprise", 0.7), "astonished": ("surprise", 0.8), "speechless": ("surprise", 0.7),
    "bewildered": ("surprise", 0.7), "startled": ("surprise", 0.6), "unexpected": ("surprise", 0.6),
    "wow": ("surprise", 0.6),
    # disgust
    "disgusted": ("disgust", 0.8), "disgusting": ("disgust", 0.8), "revolting": ("disgust", 0.8),
    "gross": ("disgust", 0.7), "vile": ("disgust", 0.8), "repulsive": ("disgust", 0.8),
    "nauseating": ("disgust", 0.8), "sickening": ("disgust", 0.8), "filthy": ("disgust", 0.7),
    "foul": ("disgust", 0.7),
    # neutral
    "calm": ("neutral", 0.5), "steady": ("neutral", 0.4), "balanced": ("neutral", 0.4),
    "okay": ("neutral", 0.3), "fine": ("neutral", 0.3), "normal": ("neutral", 0.3),
    "stoic": ("neutral", 0.4), "indifferent": ("neutral", 0.4),
}

# Negation patterns
NEGATION_PATTERNS = [
    r"\bnot\s+(\w+)", r"\bno\s+(\w+)", r"\bnever\s+(\w+)",
    r"\bcan'?t\s+(\w+)", r"\bdon'?t\s+(\w+)", r"\bwon'?t\s+(\w+)",
    r"\bdoesn'?t\s+(\w+)", r"\bhaven'?t\s+(\w+)",
]

# Intensifier patterns
INTENSIFIERS = [
    "very", "really", "so", "extremely", "absolutely", "completely",
    "totally", "utterly", "deeply", "profoundly", "incredibly",
    "beyond", "immensely", "tremendously",
]

# Uncertainty markers
UNCERTAINTY_MARKERS = [
    "maybe", "perhaps", "possibly", "might", "could be", "not sure",
    "i think", "i guess", "i suppose", "kind of", "sort of",
    "a bit", "somewhat", "seems", "appears", "i wonder",
]

# Contradiction connectors
CONTRADICTION_CONNECTORS = ["but", "yet", "however", "although", "though", "while", "despite"]

# Stopwords for cross-entry word counting
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "of", "in", "to", "for", "with", "on", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "than", "that", "this", "these", "those", "which", "who", "whom",
    "what", "just", "also", "only", "own", "same", "now", "then",
    "if", "or", "and", "but", "because", "while", "until", "about",
    "am", "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "not", "no", "so", "very", "really", "too",
}


# ============================================================================
# SINGLE-ENTRY ANALYSIS LENS
# ============================================================================

class SherlockLens:
    """Evidence-based analysis layer for single text entries."""

    def __init__(self):
        pass

    def analyze(
        self,
        text: str,
        emotion_result: Dict[str, Any],
        sentiment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform deep analysis of a single text entry."""
        cleaned = clean_text_light(text)
        words = cleaned.split()

        linguistic = self._linguistic_analysis(text, words)
        evidence = self._extract_evidence(words, emotion_result)
        contradictions = self._detect_contradictions(text, words)
        intensity = self._estimate_intensity(words, emotion_result)
        reasoning = self._generate_reasoning(
            text, emotion_result, sentiment_result, evidence,
            contradictions, linguistic,
        )

        return {
            "primary_emotion": {
                "emotion": emotion_result["emotion"],
                "confidence": emotion_result["confidence"],
                "intensity": intensity["level"],
                "intensity_score": intensity["score"],
            },
            "primary_sentiment": {
                "sentiment": sentiment_result["sentiment"],
                "confidence": sentiment_result["confidence"],
            },
            "top_emotions": emotion_result.get("top_emotions", []),
            "supporting_evidence": evidence["signals"],
            "contradictions": contradictions,
            "linguistic_clues": linguistic,
            "reasoning": reasoning,
            "uncertainty": {
                "level": self._uncertainty_level(emotion_result["confidence"]),
                "message": self._uncertainty_message(emotion_result["confidence"]),
            },
        }

    def _linguistic_analysis(self, text: str, words: List[str]) -> Dict[str, Any]:
        return {
            "word_count": len(words),
            "has_negation": any(w in {"not", "no", "never", "nothing", "nobody"} for w in words),
            "has_intensifier": any(w in INTENSIFIERS for w in words),
            "has_uncertainty": any(marker in text.lower() for marker in UNCERTAINTY_MARKERS),
            "has_contradiction_connector": any(conn in words for conn in CONTRADICTION_CONNECTORS),
            "has_first_person": any(w in {"i", "me", "my", "myself"} for w in words),
            "has_question": "?" in text or any(
                text.lower().startswith(w) for w in ["what", "why", "how", "who", "where", "when"]
            ),
            "detected_emotion_words": [
                w for w in words if w.lower() in EMOTIONAL_LEXICON
            ],
        }

    def _extract_evidence(self, words: List[str], emotion_result: Dict[str, Any]) -> Dict[str, Any]:
        signals = []
        primary_emotion = emotion_result["emotion"]

        for word in words:
            w = word.lower()
            if w in EMOTIONAL_LEXICON:
                emotion, intensity = EMOTIONAL_LEXICON[w]
                signals.append({
                    "word": word,
                    "emotion": emotion,
                    "intensity": intensity,
                    "supports_prediction": emotion == primary_emotion,
                })

        for pattern in NEGATION_PATTERNS:
            matches = re.findall(pattern, " ".join(words).lower())
            for match in matches:
                if match in EMOTIONAL_LEXICON:
                    orig_emotion, _ = EMOTIONAL_LEXICON[match]
                    signals.append({
                        "word": f"not {match}",
                        "emotion": orig_emotion,
                        "intensity": 0.5,
                        "negated": True,
                        "supports_prediction": False,
                        "note": f"Negated emotional word — may indicate opposite of {orig_emotion}",
                    })

        return {
            "signals": signals,
            "count": len(signals),
            "has_supporting_evidence": any(s.get("supports_prediction") for s in signals),
        }

    def _detect_contradictions(self, text: str, words: List[str]) -> List[Dict[str, Any]]:
        contradictions = []
        text_lower = text.lower()

        for connector in CONTRADICTION_CONNECTORS:
            parts = text_lower.split(f" {connector} ")
            if len(parts) >= 2:
                left_emotions = self._emotions_in_text(parts[0])
                right_emotions = self._emotions_in_text(parts[1])
                if left_emotions and right_emotions:
                    left_vals = [self._emotion_valence(e) for e in left_emotions]
                    right_vals = [self._emotion_valence(e) for e in right_emotions]
                    left_dominant = max(set(left_vals), key=left_vals.count) if left_vals else "neutral"
                    right_dominant = max(set(right_vals), key=right_vals.count) if right_vals else "neutral"

                    if left_dominant != right_dominant:
                        contradictions.append({
                            "type": "emotional_conflict",
                            "left_emotions": left_emotions,
                            "right_emotions": right_emotions,
                            "connector": connector,
                            "description": (
                                f"The text expresses {', '.join(left_emotions)} on one side but "
                                f"{', '.join(right_emotions)} on the other, suggesting mixed or "
                                f"conflicting feelings."
                            ),
                        })
                        break
        return contradictions

    def _emotions_in_text(self, text: str) -> List[str]:
        emotions = []
        for word in text.split():
            w = word.strip(".,!?;:\"'").lower()
            if w in EMOTIONAL_LEXICON:
                emotions.append(EMOTIONAL_LEXICON[w][0])
        return list(set(emotions))

    def _emotion_valence(self, emotion: str) -> str:
        if emotion == "joy":
            return "positive"
        if emotion in {"sadness", "anger", "fear", "disgust"}:
            return "negative"
        return "neutral"

    def _estimate_intensity(self, words: List[str], emotion_result: Dict[str, Any]) -> Dict[str, Any]:
        intensities = []
        for word in words:
            w = word.lower()
            if w in EMOTIONAL_LEXICON:
                _, intensity = EMOTIONAL_LEXICON[w]
                intensities.append(intensity)

        intensifier_count = sum(1 for w in words if w in INTENSIFIERS)
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0.0
        boosted = avg_intensity * (1 + 0.1 * intensifier_count)
        boosted = boosted * emotion_result.get("confidence", 0.5)
        boosted = min(1.0, max(0.0, boosted))

        if boosted >= 0.7:
            level = "strong"
        elif boosted >= 0.35:
            level = "moderate"
        else:
            level = "subtle"
        return {"score": round(boosted, 2), "level": level}

    def _generate_reasoning(
        self, text: str, emotion_result: Dict[str, Any], sentiment_result: Dict[str, Any],
        evidence: Dict[str, Any], contradictions: List[Dict[str, Any]],
        linguistic: Dict[str, Any],
    ) -> str:
        parts = []
        emotion = emotion_result["emotion"]
        confidence = emotion_result["confidence"]

        if confidence >= 0.6:
            parts.append(f"The primary emotional signal detected is {emotion}. ")
        elif confidence >= 0.35:
            parts.append(
                f"The text shows indications of {emotion}, though the signal "
                f"is not strong enough to be certain. "
            )
        else:
            parts.append(
                f"The emotional tone is difficult to determine with confidence. "
                f"The model's best guess is {emotion}, but this should be treated "
                f"as tentative. "
            )

        signals = evidence.get("signals", [])
        supporting = [s for s in signals if s.get("supports_prediction")]
        if supporting:
            words_list = [f'"{s["word"]}"' for s in supporting[:3]]
            parts.append(f"The key words that contributed to this reading are: {', '.join(words_list)}. ")

        negated = [s for s in signals if s.get("negated")]
        if negated:
            neg_words = ", ".join(f'"{s["word"]}"' for s in negated[:2])
            parts.append(f"Note: the text contains negated emotional language ({neg_words}) which may alter the intended meaning. ")

        if contradictions:
            for c in contradictions[:1]:
                parts.append(c["description"] + " ")

        if linguistic.get("has_uncertainty"):
            parts.append(
                "The text contains uncertainty markers (e.g. 'maybe', 'I think'), "
                "suggesting the person may not be fully certain of their own emotional state. "
            )

        if linguistic.get("has_intensifier"):
            parts.append(
                "Intensifying language (e.g. 'very', 'extremely') suggests "
                "the emotion is felt strongly. "
            )

        if confidence < 0.45:
            parts.append(
                "Due to low confidence, this analysis should be interpreted "
                "cautiously. The text may be emotionally ambiguous or may not "
                "contain enough linguistic signals for a reliable reading."
            )

        return " ".join(parts).strip()

    def _uncertainty_level(self, confidence: float) -> str:
        if confidence >= 0.7:
            return "low"
        elif confidence >= 0.4:
            return "medium"
        return "high"

    def _uncertainty_message(self, confidence: float) -> str:
        if confidence >= 0.7:
            return "The system has relatively high confidence in this analysis."
        elif confidence >= 0.4:
            return "The system has moderate confidence. There is some ambiguity in the text."
        return "The system has low confidence. The text is ambiguous or contains limited emotional signals."


def analyze_single_entry(
    text: str,
    emotion_result: Dict[str, Any],
    sentiment_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Run full Sherlock Lens analysis on a single entry."""
    lens = SherlockLens()
    return lens.analyze(text, emotion_result, sentiment_result)


# ============================================================================
# CROSS-ENTRY INSIGHTS (for /api/v1/insights)
# ============================================================================

MIN_ENTRIES_FOR_TREND = 4


def generate_insights_for_user(user_id: int) -> List[Dict[str, Any]]:
    """Generate cross-entry insights from a user's entry history.

    Looks for:
    - Emotional trends (emotion becoming more frequent)
    - Topic avoidance (topics that appeared then disappeared)

    Returns list of insight dicts with: insight_type, deduction, evidence, confidence
    """
    try:
        from cognitive_mirror.domain.entry import Entry

        entries = Entry.query.filter_by(user_id=user_id).order_by(Entry.created_at).all()
    except Exception:
        return []

    if len(entries) < MIN_ENTRIES_FOR_TREND:
        return []

    insights: List[Dict[str, Any]] = []

    # --- Trend: is an emotion becoming more frequent? ---
    insights.extend(_detect_emotion_trend(entries))

    # --- Avoidance: topics that appeared early then stopped ---
    insights.extend(_detect_topic_avoidance(entries))

    return insights


def _detect_emotion_trend(entries) -> List[Dict[str, Any]]:
    """Detect if an emotion is trending upward across entries."""
    if len(entries) < MIN_ENTRIES_FOR_TREND:
        return []

    midpoint = len(entries) // 2
    earlier = entries[:midpoint]
    later = entries[midpoint:]

    earlier_counts = Counter(e.emotion for e in earlier if e.emotion)
    later_counts = Counter(e.emotion for e in later if e.emotion)

    insights = []
    all_emotions = set(earlier_counts) | set(later_counts)
    for emotion in all_emotions:
        before = earlier_counts.get(emotion, 0)
        after = later_counts.get(emotion, 0)
        if after >= before + 2 and after >= 3:
            evidence_entries = [e for e in later if e.emotion == emotion][:3]
            insights.append({
                "insight_type": "trend",
                "deduction": (
                    f"'{emotion}' appeared {before} time(s) in your earlier entries "
                    f"and {after} time(s) more recently — worth noticing if that is a "
                    f"pattern picking up."
                ),
                "evidence": [
                    {"entry_id": e.id, "excerpt": (e.text or "")[:140]}
                    for e in evidence_entries
                ],
                "confidence": min(0.9, 0.4 + 0.1 * after),
            })
    return insights


def _detect_topic_avoidance(
    entries,
    min_mentions: int = 3,
    silence_window: int = 5,
) -> List[Dict[str, Any]]:
    """Flag topics mentioned early on, then absent from recent entries."""
    if len(entries) < min_mentions + silence_window:
        return []

    recent = entries[-silence_window:]
    older = entries[:-silence_window]

    older_words = _word_counts(older)
    recent_words = set(_word_counts(recent).keys())

    insights = []
    for word, count in older_words.items():
        if count >= min_mentions and word not in recent_words:
            evidence_entries = [e for e in older if word in (e.text or "").lower()][:3]
            insights.append({
                "insight_type": "avoidance",
                "deduction": (
                    f'You mentioned "{word}" {count} times earlier on, but it has not '
                    f"come up in your last {silence_window} entries. Not necessarily "
                    f"meaningful on its own, but worth asking yourself why."
                ),
                "evidence": [
                    {"entry_id": e.id, "excerpt": (e.text or "")[:140]}
                    for e in evidence_entries
                ],
                "confidence": 0.5,
            })
    return insights[:5]


def _word_counts(entries, min_len: int = 5) -> Dict[str, int]:
    """Simple content-word frequency count, stopword-filtered."""
    counter: Counter = Counter()
    for entry in entries:
        text = (entry.text or "").lower()
        words = [
            w.strip(".,!?;:\"'")
            for w in text.split()
            if len(w) >= min_len and w.lower() not in _STOPWORDS
        ]
        for w in set(words):
            counter[w] += 1
    return dict(counter)
