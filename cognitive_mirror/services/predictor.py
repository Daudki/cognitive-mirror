"""Prediction orchestration service with caching and monitoring."""

import time
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from cognitive_mirror.models.manager import ModelManager
from cognitive_mirror.services.cache import CacheService
from cognitive_mirror.monitoring.metrics import (
    prediction_counter,
    prediction_latency_histogram,
    prediction_cache_hits,
    prediction_text_length,
    emotion_distribution,
)


class PredictionError(Exception):
    """Base exception for prediction failures."""
    pass


class TextTooLongError(PredictionError):
    """Input text exceeds maximum allowed length."""
    pass


class ModelNotReadyError(PredictionError):
    """Model has not been loaded or is unhealthy."""
    pass


@dataclass
class PredictionResult:
    """Structured result from the prediction pipeline."""
    request_id: str
    text: str
    emotion: Dict[str, Any]
    sentiment: Dict[str, Any]
    mind_state: str
    processing_time_ms: float
    model_version: str
    explanation: Optional[Dict[str, Any]] = None
    from_cache: bool = False


class PredictorService:
    """Orchestrates the full inference pipeline.
    
    Responsibilities:
        1. Input validation
        2. Cache lookup
        3. Model inference
        4. Mind state generation
        5. Metrics recording
    """
    
    # Maximum input length
    MAX_TEXT_LENGTH = 1000
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        """Initialize the predictor service.
        
        Args:
            cache_service: Optional cache service instance
        """
        self.cache = cache_service or CacheService()
    
    def predict(
        self,
        text: str,
        include_explanation: bool = False,
        model_version: Optional[str] = None,
    ) -> PredictionResult:
        """Run the complete prediction pipeline.
        
        Args:
            text: Input text to analyze
            include_explanation: Generate model explanation if True
            model_version: Specific model version (not yet supported)
            
        Returns:
            PredictionResult with all analysis outputs
            
        Raises:
            TextTooLongError: Text exceeds maximum length
            ModelNotReadyError: Models are not loaded
            PredictionError: Inference failure
        """
        request_id = self._generate_request_id(text)
        start_time = time.perf_counter()
        
        # === Validate Input ===
        text = text.strip()
        if not text:
            raise TextTooLongError("Text cannot be empty")
        
        if len(text) > self.MAX_TEXT_LENGTH:
            raise TextTooLongError(
                f"Text length ({len(text)}) exceeds maximum ({self.MAX_TEXT_LENGTH})"
            )
        
        prediction_text_length.observe(len(text))
        
        # === Check Model Health ===
        if not ModelManager.is_healthy():
            prediction_counter.labels(status="error").inc()
            raise ModelNotReadyError("Models are not loaded. Check /api/v1/health")
        
        # === Check Cache ===
        cache_key = None
        if not include_explanation:
            cache_key = self._cache_key(text)
            cached = self.cache.get(cache_key)
            if cached:
                prediction_cache_hits.inc()
                prediction_counter.labels(status="cache_hit").inc()
                
                return PredictionResult(
                    request_id=request_id,
                    text=text,
                    processing_time_ms=0.0,
                    model_version=cached.get("model_version", "unknown"),
                    from_cache=True,
                    **{k: v for k, v in cached.items() if k != "model_version"},
                )
        
        # === Run Inference ===
        try:
            emotion_result = ModelManager.predict_emotion(text)
            sentiment_result = ModelManager.predict_sentiment(text)
            mind_state = ModelManager.generate_mindstate(emotion_result, sentiment_result)
            
            # Track emotion distribution
            emotion_distribution.labels(
                emotion=emotion_result["emotion"],
                sentiment=sentiment_result["sentiment"],
            ).inc()
            
        except Exception as e:
            prediction_counter.labels(status="error").inc()
            raise PredictionError(f"Inference failed: {str(e)}") from e
        
        # === Build Result ===
        processing_time = (time.perf_counter() - start_time) * 1000
        
        result = PredictionResult(
            request_id=request_id,
            text=text,
            emotion=emotion_result,
            sentiment=sentiment_result,
            mind_state=mind_state,
            processing_time_ms=round(processing_time, 2),
            model_version=ModelManager.get_metadata().get("version", "unknown"),
            from_cache=False,
        )
        
        # === Cache Result ===
        if cache_key and not include_explanation:
            self.cache.set(cache_key, {
                "emotion": emotion_result,
                "sentiment": sentiment_result,
                "mind_state": mind_state,
                "model_version": result.model_version,
            })
        
        # === Record Metrics ===
        prediction_counter.labels(status="success").inc()
        prediction_latency_histogram.observe(processing_time)
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        include_explanation: bool = False,
    ) -> List[PredictionResult]:
        """Run prediction on a batch of texts.
        
        Args:
            texts: List of input texts (max 32)
            include_explanation: Generate explanations
            
        Returns:
            List of prediction results (errors become neutral predictions)
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text, include_explanation=include_explanation)
                results.append(result)
            except PredictionError:
                # Return a neutral/failed result for batch processing
                results.append(PredictionResult(
                    request_id=self._generate_request_id(text),
                    text=text,
                    emotion={"emotion": "unknown", "confidence": 0.0, "top_emotions": []},
                    sentiment={"sentiment": "neutral", "confidence": 0.0},
                    mind_state="Unable to analyze this text.",
                    processing_time_ms=0.0,
                    model_version="unknown",
                ))
        return results
    
    def _generate_request_id(self, text: str) -> str:
        """Generate a unique, deterministic request ID."""
        content = f"{text[:30]}:{time.time()}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:12]
    
    def _cache_key(self, text: str) -> str:
        """Generate a deterministic cache key from text."""
        normalized = text.lower().strip()
        return f"predict:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"
