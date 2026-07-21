"""Sherlock Lens — deduction from evidence across a user's entry history.

Unlike the Mirror (per-entry, real-time), this runs as a batch job over a
window of entries and looks for patterns a single entry can't show:
avoidance (topics that vanished), contradiction (stated values vs. actual
mentions), and trend (a distortion type becoming more frequent over time).

Deductions are evidence-cited by design — every insight points back to the
specific entries that support it, so this stays "here's what the data
shows" rather than an unfounded claim about the person.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from cognitive_mirror.domain.entry import Entry

MIN_ENTRIES_FOR_TREND = 4
TREND_WINDOW_SPLIT = 2  # compares first half vs second half of the window


class SherlockLens:
    """Runs deduction passes over a user's entry history."""

    def __init__(self, entries: List[Entry]):
        # Oldest -> newest, since trend/avoidance logic depends on order
        self.entries = sorted(entries, key=lambda e: e.created_at)

    def run(self) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []
        insights.extend(self._detect_distortion_trend())
        insights.extend(self._detect_topic_avoidance())
        return insights

    # --- Trend: is a distortion type becoming more frequent? -----------

    def _detect_distortion_trend(self) -> List[Dict[str, Any]]:
        if len(self.entries) < MIN_ENTRIES_FOR_TREND:
            return []

        midpoint = len(self.entries) // 2
        earlier, later = self.entries[:midpoint], self.entries[midpoint:]

        earlier_counts = self._count_distortions(earlier)
        later_counts = self._count_distortions(later)

        insights = []
        all_types = set(earlier_counts) | set(later_counts)
        for distortion_type in all_types:
            before = earlier_counts.get(distortion_type, 0)
            after = later_counts.get(distortion_type, 0)
            # Require a real jump, not noise from 1 -> 2
            if after >= before + 2 and after >= 3:
                evidence_entries = [
                    e for e in later
                    if any(d.get("type") == distortion_type for d in (e.distortions or []))
                ][:3]
                insights.append({
                    "insight_type": "trend",
                    "deduction": (
                        f"'{distortion_type.replace('_', ' ')}' language showed up "
                        f"{before} time(s) in your earlier entries and {after} time(s) "
                        f"more recently — worth noticing if that's a pattern picking up."
                    ),
                    "evidence": [
                        {"entry_id": e.id, "excerpt": e.text[:140]} for e in evidence_entries
                    ],
                    "confidence": min(0.9, 0.4 + 0.1 * after),
                })
        return insights

    @staticmethod
    def _count_distortions(entries: List[Entry]) -> Dict[str, int]:
        counter: Counter = Counter()
        for entry in entries:
            for d in (entry.distortions or []):
                counter[d.get("type", "unknown")] += 1
        return counter

    # --- Avoidance: topics that appeared, then stopped appearing -------

    def _detect_topic_avoidance(self, min_mentions: int = 3, silence_window: int = 5) -> List[Dict[str, Any]]:
        """Flags simple keyword topics mentioned repeatedly early on, then
        absent for the most recent `silence_window` entries.

        This is intentionally a simple keyword-frequency heuristic, not a
        topic model — cheap, explainable, and evidence-citable, which
        matters more here than sophistication.
        """
        if len(self.entries) < min_mentions + silence_window:
            return []

        recent = self.entries[-silence_window:]
        older = self.entries[:-silence_window]

        older_words = self._word_counts(older)
        recent_words = set(self._word_counts(recent).keys())

        insights = []
        for word, count in older_words.items():
            if count >= min_mentions and word not in recent_words:
                evidence_entries = [e for e in older if word in e.text.lower()][:3]
                insights.append({
                    "insight_type": "avoidance",
                    "deduction": (
                        f"You mentioned \"{word}\" {count} times earlier on, but it hasn't "
                        f"come up in your last {silence_window} entries. Not necessarily "
                        f"meaningful on its own, but worth asking yourself why."
                    ),
                    "evidence": [
                        {"entry_id": e.id, "excerpt": e.text[:140]} for e in evidence_entries
                    ],
                    "confidence": 0.5,
                })
        return insights[:5]  # cap noise

    @staticmethod
    def _word_counts(entries: List[Entry], min_len: int = 5) -> Dict[str, int]:
        """Very simple content-word frequency count, stopword-free-ish via length filter."""
        STOPWORDS = {
            "about", "there", "which", "would", "could", "should", "their",
            "these", "those", "because", "before", "after", "really", "always",
        }
        counter: Counter = Counter()
        for entry in entries:
            words = {
                w.strip(".,!?;:\"'").lower()
                for w in entry.text.split()
                if len(w) >= min_len
            }
            for w in words - STOPWORDS:
                counter[w] += 1
        return counter


def generate_insights_for_user(user_id: int) -> List[Dict[str, Any]]:
    """Convenience entry point for the batch job / on-demand route."""
    entries = Entry.query.filter_by(user_id=user_id).order_by(Entry.created_at).all()
    lens = SherlockLens(entries)
    return lens.run()
