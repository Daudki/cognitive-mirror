"""Pluggable LLM adapter interfaces and simple implementations.

Adapters provide a `generate` method that accepts a prompt and returns generated text
and optional metadata. Implementations can be local model wrappers or remote API callers.
"""
from typing import Dict, Any, Optional
import os
import random


class BaseAdapter:
    """Base adapter interface for LLM providers."""

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()


class DummyAdapter(BaseAdapter):
    """Lightweight generator used when no real LLM backend is configured.

    This avoids hard templates by composing varied, contextualized sentences
    using the provided prompt.
    """

    VARIATIONS = [
        "The person appears to be {emotion} with a {sentiment} tone. {detail}",
        "Reading between the lines, they seem {emotion} and {sentiment}: {detail}",
        "There is a sense of {emotion} coupled with {sentiment}. {detail}",
    ]

    DETAILS = [
        "Their attention is drawn to personal relationships and inner states.",
        "Subtle cues point to internal conflict and layered feelings.",
        "Behavioral patterns suggest coping strategies and latent needs.",
        "They may benefit from reflection and supportive dialogue.",
    ]

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> Dict[str, Any]:
        # Try to extract short signals from prompt when possible
        emotion = kwargs.get("emotion", "neutral")
        sentiment = kwargs.get("sentiment", "neutral")
        confidence = kwargs.get("confidence", 0.5)

        template = random.choice(self.VARIATIONS)
        detail = random.choice(self.DETAILS)

        text = template.format(emotion=emotion, sentiment=sentiment, detail=detail)

        return {
            "text": text,
            "model": "dummy-adapter",
            "tokens": len(text.split()),
        }


class APIAdapter(BaseAdapter):
    """Adapter skeleton for managed API providers.

    This is a minimal, safe scaffold — real implementations should handle
    authentication, rate limits, retry/backoff, and error handling.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY")

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> Dict[str, Any]:
        # Placeholder: real call out to provider goes here.
        # For now, return a structured acknowledgement so the system can be
        # wired without external dependencies.
        return {
            "text": f"[APIAdapter stub] Generated response for prompt: {prompt[:120]}",
            "model": "api-adapter-stub",
            "tokens": 0,
        }


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI's API using the `openai` Python package.

    Environment variables:
      - OPENAI_API_KEY: your API key
      - OPENAI_MODEL: model name (default: gpt-3.5-turbo)
      - OPENAI_TIMEOUT: request timeout seconds
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        try:
            self.timeout = int(os.environ.get("OPENAI_TIMEOUT", str(timeout)))
        except Exception:
            self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> Dict[str, Any]:
        try:
            import openai
        except Exception as e:
            raise RuntimeError("OpenAI package not installed. Install with `pip install openai`") from e

        if self.api_key:
            openai.api_key = self.api_key

        # Build messages for ChatCompletion
        messages = [
            {"role": "system", "content": "You are a compassionate psychology expert. Provide empathetic, evidence-based, concise observations."},
            {"role": "user", "content": prompt},
        ]

        # Basic retry/backoff
        import time

        attempts = 3
        backoff = 1.0
        last_err = None
        for attempt in range(attempts):
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    timeout=self.timeout,
                )
                # extract text
                if resp and getattr(resp, "choices", None):
                    choice = resp["choices"][0]
                    text = choice.get("message", {}).get("content") or choice.get("text") or ""
                else:
                    text = resp.get("choices", [])[0].get("message", {}).get("content", "") if resp else ""

                usage = resp.get("usage") if isinstance(resp, dict) else None
                tokens = usage.get("total_tokens") if usage and isinstance(usage, dict) else (len(text.split()) if text else 0)

                return {"text": text, "model": self.model, "tokens": tokens, "usage": usage}

            except Exception as e:
                last_err = e
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"OpenAI request failed after retries: {last_err}")


class LocalAdapter(BaseAdapter):
    """Adapter for local models using Hugging Face `transformers` pipeline.

    Environment variables:
      - LLM_LOCAL_MODEL: model identifier (default: distilgpt2)
      - LLM_LOCAL_DEVICE: 'cpu' or 'cuda' (auto-detected by default)
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or os.environ.get("LLM_LOCAL_MODEL", "distilgpt2")
        self.device = device or os.environ.get("LLM_LOCAL_DEVICE")
        self._pipeline = None

    def _init_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch

            # Decide device
            if not self.device:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            device_arg = 0 if self.device == "cuda" else -1

            # Initialize generation pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=device_arg,
            )
        except Exception as e:
            # Leave _pipeline as None to signal failure
            self._pipeline = None
            raise RuntimeError(f"Failed to initialize local model pipeline: {e}")

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> Dict[str, Any]:
        try:
            self._init_pipeline()
            if not self._pipeline:
                raise RuntimeError("Pipeline not initialized")

            # transformers pipeline accepts `max_new_tokens` in newer versions
            gen = self._pipeline(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
            if isinstance(gen, list) and gen:
                text = gen[0].get("generated_text", str(gen[0]))
            else:
                text = str(gen)

            tokens = len(text.split())
            return {"text": text, "model": self.model_name, "tokens": tokens}
        except Exception as e:
            # Bubble up as runtime error for caller to handle
            raise
