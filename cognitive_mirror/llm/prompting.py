"""Prompt builder and persona/safety glue for LLM prompts.

This module centralizes persona text and safe prompt construction so prompts
are consistent and maintainable. Prompts are *not* fixed response templates;
they provide contextual information to the LLM about the task and constraints.
"""
from typing import Dict, Any


DEFAULT_PERSONA = (
    "You are a compassionate psychology expert. Provide empathetic, evidence-based, "
    "brief but insightful observations about the person's mental state. Avoid giving medical diagnoses or legal advice."
)


def build_prompt(text: str, emotion_result: Dict[str, Any], sentiment_result: Dict[str, Any], persona: str = DEFAULT_PERSONA) -> str:
    """Compose a structured prompt for the LLM.

    The prompt includes the raw text, classifier signals (emotion/sentiment),
    and simple instructions for style and safety. This keeps generation flexible
    while providing useful structure for learning and review.
    """
    emotion = emotion_result.get("emotion", "neutral")
    emotion_conf = emotion_result.get("confidence", 0.0)
    sentiment = sentiment_result.get("sentiment", "neutral")
    sentiment_conf = sentiment_result.get("confidence", 0.0)

    prompt = (
        f"{persona}\n\n"
        f"Input text:\n" 
        f"""{text}""" + "\n\n"
        f"Classifier signals:\n"
        f"- emotion: {emotion} (confidence: {emotion_conf})\n"
        f"- sentiment: {sentiment} (confidence: {sentiment_conf})\n\n"
        "Task: Provide a concise, human-readable summary of the person's likely mental state,"
        " focusing on observable cues, possible needs, and gentle, actionable suggestions."
        " Keep tone empathetic and avoid deterministic claims."
    )

    return prompt
