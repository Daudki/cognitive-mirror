"""Cognitive distortion detection — the 'Mirror' lens.

Two-pass pipeline:
  1. Regex candidate flagging (cheap, catches linguistic markers)
  2. LLM classification on flagged sentences only (accurate, contextual,
     cheap because only flagged snippets are sent — not the whole entry)

Output is framed as pattern-noticing, never as a diagnostic claim. Callers
(API layer / UI) are responsible for keeping that framing intact.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cognitive_mirror.llm.adapters import DummyAdapter
from cognitive_mirror.models.manager import _get_llm_adapter


# --- Distortion taxonomy -----------------------------------------------

DISTORTION_MARKERS: Dict[str, str] = {
    "all_or_nothing": r"\b(always|never|every time|everyone|no one|nobody|completely|totally|entirely)\b",
    "should_statements": r"\b(should|shouldn't|must|have to|ought to|need to)\b",
    "catastrophizing": r"\b(ruined|disaster|the end|can't handle|terrible|worst|falling apart|unbearable)\b",
    "labeling": r"\bI(?:'m| am) (?:a |an )?(failure|loser|idiot|worthless|stupid|useless|broken)\b",
    "mind_reading": r"\bthey (?:think|believe|know) I\b|\bhe (?:thinks|knows) I\b|\bshe (?:thinks|knows) I\b",
    "fortune_telling": r"\b(it('s| is) (going to|gonna) (fail|go wrong)|nothing will (ever )?(work|change)|I('ll| will) never)\b",
    "emotional_reasoning": r"\bI feel (like )?.+,? so (it|that) must be\b",
    "mental_filtering": r"\b(only thing that matters|can't stop thinking about (how|the) (bad|wrong))\b",
    "personalization": r"\b(it('s| is) my fault|because of me|I made (this|it) happen)\b",
}

DISTORTION_LABELS: Dict[str, str] = {
    "all_or_nothing": "All-or-nothing thinking",
    "should_statements": "Should statements",
    "catastrophizing": "Catastrophizing",
    "labeling": "Labeling",
    "mind_reading": "Mind reading",
    "fortune_telling": "Fortune telling",
    "emotional_reasoning": "Emotional reasoning",
    "mental_filtering": "Mental filtering",
    "personalization": "Personalization",
}

CONFIDENCE_THRESHOLD = 0.6

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class DistortionFlag:
    sentence: str
    distortion_type: str
    label: str
    confidence: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "type": self.distortion_type,
            "label": self.label,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


class DistortionDetector:
    """Detects CBT-style cognitive distortions in journal text."""

    def __init__(self, adapter=None, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        # Reuses the same adapter selection as mind-state generation
        # (OpenAI -> local HF model -> deterministic dummy fallback).
        self.adapter = adapter or _get_llm_adapter()
        self.confidence_threshold = confidence_threshold

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """Run the full two-pass pipeline and return confirmed distortions."""
        candidates = self._flag_candidates(text)
        confirmed: List[DistortionFlag] = []

        for sentence, distortion_type in candidates:
            result = self._classify(sentence, distortion_type)
            if result and result["confidence"] >= self.confidence_threshold:
                confirmed.append(
                    DistortionFlag(
                        sentence=sentence,
                        distortion_type=distortion_type,
                        label=DISTORTION_LABELS[distortion_type],
                        confidence=result["confidence"],
                        explanation=result["explanation"],
                    )
                )

        return [f.to_dict() for f in confirmed]

    # --- Pass 1: regex candidate flagging -------------------------------

    def _flag_candidates(self, text: str) -> List[tuple]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        candidates = []
        for sentence in sentences:
            for distortion_type, pattern in DISTORTION_MARKERS.items():
                if re.search(pattern, sentence, flags=re.IGNORECASE):
                    candidates.append((sentence, distortion_type))
        return candidates

    # --- Pass 2: LLM classification on flagged sentences only ----------

    def _classify(self, sentence: str, distortion_type: str) -> Optional[Dict[str, Any]]:
        label = DISTORTION_LABELS[distortion_type]
        prompt = (
            "You are helping a journaling app detect cognitive distortions "
            "(CBT terminology). Given a single sentence from someone's private "
            f"journal, decide whether it exhibits the distortion \"{label}\".\n\n"
            f'Sentence: "{sentence}"\n\n'
            "Respond with ONLY a JSON object, no other text:\n"
            '{"is_distortion": true or false, "confidence": 0.0-1.0, '
            '"explanation": "one short, gentle sentence, no diagnosis language"}'
        )

        try:
            raw = self.adapter.generate(prompt, max_tokens=120)
            parsed = self._parse_json_response(raw.get("text", ""))
        except Exception:
            parsed = None

        if parsed is not None:
            if not parsed.get("is_distortion"):
                return None
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            return {
                "confidence": max(0.0, min(1.0, confidence)),
                "explanation": str(parsed.get("explanation", "")).strip(),
            }

        # No structured LLM confirmation available (e.g. DummyAdapter in dev,
        # or a real adapter that returned unparseable output). Rather than
        # silently dropping every candidate, fall back to the regex match
        # itself as a lower-confidence, clearly-labeled heuristic flag.
        if isinstance(self.adapter, DummyAdapter):
            return {
                "confidence": self.confidence_threshold,
                "explanation": (
                    "Flagged by language pattern only (no LLM confirmation configured)."
                ),
            }
        return None

    @staticmethod
    def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        # Adapters (especially DummyAdapter/LocalAdapter) may wrap or fail to
        # return clean JSON — extract the first {...} block defensively.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
