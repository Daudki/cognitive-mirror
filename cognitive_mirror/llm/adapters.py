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
    """NO-OP adapter — should never produce mind state text.

    The Sherlock Lens handles all evidence-based reasoning generation.
    This adapter exists only as a stub for the distortion detection
    LLM pipeline. For mind state generation, the ModelManager skips
    the LLM path entirely and uses Sherlock Lens reasoning.
    """

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> Dict[str, Any]:
        return {
            "text": "",
            "model": "dummy-adapter",
            "tokens": 0,
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
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a compassionate psychology expert. Provide empathetic, "
                    "evidence-based, concise observations."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        import time

        attempts = 3
        backoff = 1.0
        last_err = None
        for _ in range(attempts):
            try:
                text, usage = self._chat_completion(messages, max_tokens)
                tokens = 0
                if usage and isinstance(usage, dict):
                    tokens = usage.get("total_tokens", 0)
                if not tokens and text:
                    tokens = len(text.split())
                return {"text": text, "model": self.model, "tokens": tokens, "usage": usage}
            except Exception as e:
                last_err = e
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"OpenAI request failed after retries: {last_err}")

    def _chat_completion(self, messages, max_tokens: int):
        """Support both OpenAI Python SDK v1+ and legacy v0.28."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            if usage is not None and hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            elif usage is not None:
                usage = dict(usage) if not isinstance(usage, dict) else usage
            return text, usage
        except ImportError:
            pass

        import openai

        openai.api_key = self.api_key
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            request_timeout=self.timeout,
        )
        choice = resp["choices"][0] if isinstance(resp, dict) else resp.choices[0]
        if isinstance(choice, dict):
            text = (choice.get("message", {}) or {}).get("content") or choice.get("text", "")
        else:
            text = getattr(getattr(choice, "message", None), "content", None) or getattr(choice, "text", "")
        usage = resp.get("usage") if isinstance(resp, dict) else None
        return (text or "").strip(), usage


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
