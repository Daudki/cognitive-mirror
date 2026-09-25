"""Idiom and euphemism overrides for the emotion classifier.

The trained sklearn model generalizes reasonably well within the vocabulary
it has seen, but ~625 hand-written training examples can't cover common
idioms, euphemisms, and set phrases that don't share surface-level words
with anything in the training set — e.g. "passed away" (sadness), "can't
stand it" (anger), "burnt out" (sadness), "gone off" of food (disgust).
These get randomly misclassified not because the phrase is ambiguous, but
because the classifier has simply never seen it.

This module is a small, high-precision phrase lexicon used as a targeted
override ONLY when a phrase unambiguously signals one emotion — it is not
a general sentiment lexicon and deliberately does not include single common
words (those are exactly what the trained model already handles).
"""

import re
from typing import Dict, List, Optional, Tuple

# Each phrase should be unambiguous enough that seeing it is a strong signal
# on its own — avoid single emotionally-loaded words here; the classifier
# already covers those. Order doesn't matter; matching is phrase-based.
IDIOM_LEXICON: Dict[str, List[str]] = {
    "sadness": [
        "passed away", "passed on", "lost my", "lost her", "lost him",
        "no longer with us", "funeral", "burnt out", "burned out",
        "running on empty", "completely drained", "emotionally drained",
        "empty inside", "going through the motions", "at my lowest",
        "hit rock bottom", "can't shake this feeling", "heavy heart",
        "miss them so much", "miss him so much", "miss her so much",
    ],
    "anger": [
        "can't stand it", "cannot stand it", "sick of this", "sick and tired",
        "fed up with", "had enough of", "boiling with rage", "seeing red",
        "makes my blood boil", "at the end of my rope", "losing my patience",
        "done with this", "over this", "so unfair",
    ],
    "disgust": [
        "gone off", "gone bad", "smells rotten", "smells rancid",
        "covered in mold", "covered in mould", "made me gag", "made me sick",
        "turns my stomach", "want to throw up", "wanted to throw up",
        "so filthy", "absolutely filthy", "rancid smell",
    ],
    "fear": [
        "can't stop shaking", "cannot stop shaking", "heart is racing",
        "my heart raced", "frozen with fear", "scared out of my mind",
        "dread the thought", "keeps me up at night", "on edge about",
        "worst case scenario",
    ],
    "surprise": [
        "didn't see that coming", "did not see that coming",
        "out of nowhere", "caught off guard", "never in a million years",
        "couldn't believe my eyes", "could not believe my eyes",
        "took me by surprise",
    ],
    "joy": [
        "over the moon", "on top of the world", "walking on air",
        "couldn't stop smiling", "could not stop smiling",
        "best day ever", "dream come true", "so proud of myself",
        "everything fell into place",
    ],
}

# Compile once: (compiled_pattern, emotion) pairs, longest phrases first so a
# more specific match wins if two overlap.
_ALL_PHRASES: List[Tuple[re.Pattern, str]] = sorted(
    (
        (re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), emotion)
        for emotion, phrases in IDIOM_LEXICON.items()
        for phrase in phrases
    ),
    key=lambda pair: len(pair[0].pattern),
    reverse=True,
)

OVERRIDE_CONFIDENCE = 0.65  # honest, fixed value — this isn't a real model probability


def match_idiom(text: str) -> Optional[Dict[str, object]]:
    """Return an override dict if a lexicon phrase matches, else None.

    Only ever returns a single best (longest) match — if a text somehow
    contains phrases for two different emotions, longest-match-wins avoids
    a coin-flip between them.
    """
    for pattern, emotion in _ALL_PHRASES:
        match = pattern.search(text)
        if match:
            return {
                "emotion": emotion,
                "confidence": OVERRIDE_CONFIDENCE,
                "matched_phrase": match.group(0),
                "source": "lexicon_override",
            }
    return None
