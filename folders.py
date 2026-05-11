#!/usr/bin/env python3
"""
create_structure.py
Creates the production-grade Cognitive Mirror directory structure.
Safely creates new directories and files WITHOUT overwriting existing ones.

Usage:
    python create_structure.py
    python create_structure.py --dry-run    # Preview without creating
    python create_structure.py --verbose    # Show all operations
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# Configuration
# ============================================================================

DIRECTORIES = [
    "cognitive_mirror",
    "cognitive_mirror/api",
    "cognitive_mirror/api/v1",
    "cognitive_mirror/models",
    "cognitive_mirror/schema",
    "cognitive_mirror/services",
    "cognitive_mirror/monitoring",
    "cognitive_mirror/errors",
    "tests",
    "tests/test_api",
    "tests/test_models",
    "tests/test_services",
    "scripts",
]

# Files to create with their content
FILES = {
    # ===== Package init files =====
    "cognitive_mirror/__init__.py": '''"""Cognitive Mirror - Production ML Inference Service.

A Flask-based mindstate inference tool that predicts emotion, sentiment,
and confidence from text input with production-grade reliability.
"""

__version__ = "2.0.0"
__author__ = "Daudki"
__description__ = "Production-grade mindstate inference API"
''',

    "cognitive_mirror/api/__init__.py": '"""API Blueprints package."""\n',
    "cognitive_mirror/api/v1/__init__.py": '"""API v1 Blueprints."""\n',
    "cognitive_mirror/models/__init__.py": '"""Model wrappers and lifecycle management."""\n',
    "cognitive_mirror/schema/__init__.py": '"""Request/Response validation schemas."""\n',
    "cognitive_mirror/services/__init__.py": '"""Business logic services layer."""\n',
    "cognitive_mirror/monitoring/__init__.py": '"""Observability: logging, metrics, tracing."""\n',
    "cognitive_mirror/errors/__init__.py": '"""Custom exception classes."""\n',

    # ===== Core Application Files =====
    "config.py": '''"""Application configuration with environment-based profiles."""

import os
from typing import Dict


class Config:
    """Base configuration shared across all environments."""
    
    # Security
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "change-me-in-production")
    
    # Model
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "models/model.pkl")
    MODEL_VERSION: str = os.environ.get("MODEL_VERSION", "1.0.0")
    
    # Redis Cache
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL: int = int(os.environ.get("CACHE_TTL", "300"))
    
    # API Settings
    API_TITLE: str = "Cognitive Mirror API"
    API_VERSION: str = "v1"
    MAX_TEXT_LENGTH: int = int(os.environ.get("MAX_TEXT_LENGTH", "1000"))
    
    # Rate Limiting
    RATELIMIT_ENABLED: bool = os.environ.get("RATELIMIT_ENABLED", "true").lower() == "true"
    RATELIMIT_DEFAULT: str = os.environ.get("RATELIMIT_DEFAULT", "100/hour")
    
    # Monitoring
    PROMETHEUS_METRICS: bool = True
    STRUCTURED_LOGGING: bool = True
    
    # Inference
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))
    INFERENCE_TIMEOUT: int = int(os.environ.get("INFERENCE_TIMEOUT", "5"))


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG: bool = True
    CACHE_TTL: int = 60


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG: bool = False
    CACHE_TTL: int = 300


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING: bool = True
    CACHE_TTL: int = 1


# Environment mapping
config_by_name: Dict[str, type] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
''',

    "cognitive_mirror/extensions.py": '''"""Flask extensions initialization."""

from typing import Optional
from redis import Redis

# Global extension instances
redis_client: Optional[Redis] = None


def init_extensions(app) -> None:
    """Initialize all Flask extensions with the application context."""
    global redis_client
    
    redis_url = app.config.get("REDIS_URL")
    if redis_url:
        try:
            redis_client = Redis.from_url(
                redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            redis_client.ping()
            app.logger.info(f"Redis connected: {redis_url}")
        except Exception as e:
            redis_client = None
            app.logger.warning(f"Redis unavailable, using fallback: {e}")
''',

    "cognitive_mirror/factory.py": '''"""Application factory pattern for creating Flask instances."""

from flask import Flask
from flask_cors import CORS
import structlog

from cognitive_mirror.config import config_by_name
from cognitive_mirror.extensions import init_extensions
from cognitive_mirror.errors.handlers import register_error_handlers
from cognitive_mirror.api.middleware import register_middleware
from cognitive_mirror.monitoring.logging import configure_logging


def create_app(config_name: str = "development") -> Flask:
    """Create and configure the Flask application.
    
    Args:
        config_name: Environment name ('development', 'production', 'testing')
    
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    config_class = config_by_name.get(config_name, config_by_name["development"])
    app.config.from_object(config_class)
    
    # Configure structured logging
    configure_logging(app)
    logger = structlog.get_logger(__name__)
    
    # Initialize extensions
    init_extensions(app)
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Rate limiting (if Redis is available)
    if app.config.get("RATELIMIT_ENABLED") and app.config.get("REDIS_URL"):
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address
            Limiter(
                app=app,
                key_func=get_remote_address,
                default_limits=[app.config["RATELIMIT_DEFAULT"]],
                storage_uri=app.config["REDIS_URL"],
            )
        except ImportError:
            app.logger.warning("flask-limiter not installed, rate limiting disabled")
    
    # Register middleware
    register_middleware(app)
    
    # Register API blueprints
    from cognitive_mirror.api.v1.predict import bp as predict_bp
    from cognitive_mirror.api.v1.health import bp as health_bp
    from cognitive_mirror.api.v1.metrics import bp as metrics_bp
    
    app.register_blueprint(predict_bp, url_prefix="/api/v1")
    app.register_blueprint(health_bp, url_prefix="/api/v1")
    app.register_blueprint(metrics_bp, url_prefix="/api/v1")
    
    # Register error handlers
    register_error_handlers(app)
    
    # Initialize model manager on startup
    with app.app_context():
        try:
            from cognitive_mirror.models.manager import ModelManager
            ModelManager.initialize(app.config["MODEL_PATH"])
            logger.info(
                "Application initialized",
                config=config_name,
                model_version=app.config["MODEL_VERSION"],
            )
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Don't crash - allow health check to report unhealthy
    
    return app
''',

    # ===== API Middleware =====
    "cognitive_mirror/api/middleware.py": '''"""Request/Response middleware for logging and tracing."""

import uuid
import time
from flask import Flask, request, g, current_app


def register_middleware(app: Flask) -> None:
    """Register before/after request handlers."""
    
    @app.before_request
    def before_request():
        """Set up per-request context."""
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        g.start_time = time.perf_counter()
    
    @app.after_request
    def after_request(response):
        """Add tracing headers and log request completion."""
        elapsed_ms = (time.perf_counter() - g.get("start_time", time.perf_counter())) * 1000
        
        response.headers["X-Request-ID"] = g.get("request_id", "unknown")
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-API-Version"] = current_app.config.get("API_VERSION", "v1")
        
        # Don't add headers for static files or metrics
        if not request.path.startswith("/static"):
            current_app.logger.debug(
                f"{request.method} {request.path} -> {response.status_code} "
                f"({elapsed_ms:.1f}ms) [req={g.get('request_id', '?')}]"
            )
        
        return response
''',

    # ===== API v1 Blueprints =====
    "cognitive_mirror/api/v1/health.py": '''"""Health check and Kubernetes probe endpoints."""

from flask import Blueprint, jsonify, current_app
from cognitive_mirror.models.manager import ModelManager
from cognitive_mirror.extensions import redis_client

bp = Blueprint("health", __name__)


@bp.route("/health", methods=["GET"])
def health():
    """Comprehensive health check.
    
    Returns:
        200 if all systems healthy, 503 if degraded.
    """
    model_healthy = ModelManager.is_healthy()
    
    # Check Redis
    redis_healthy = False
    if redis_client:
        try:
            redis_healthy = redis_client.ping()
        except Exception:
            redis_healthy = False
    
    status = {
        "status": "healthy" if model_healthy else "unhealthy",
        "version": ModelManager.get_metadata().get("version", "unknown"),
        "timestamp": ModelManager.get_metadata().get("loaded_at"),
        "checks": {
            "model_loaded": model_healthy,
            "redis_connected": redis_healthy,
        },
    }
    
    return jsonify(status), 200 if model_healthy else 503


@bp.route("/ready", methods=["GET"])
def ready():
    """Kubernetes readiness probe.
    
    Returns 200 when the application is ready to serve traffic.
    """
    if ModelManager.is_healthy():
        return jsonify({"ready": True}), 200
    return jsonify({"ready": False, "reason": "Model not loaded"}), 503


@bp.route("/live", methods=["GET"])
def live():
    """Kubernetes liveness probe.
    
    Returns 200 as long as the process is alive.
    """
    return jsonify({"alive": True}), 200
''',

    "cognitive_mirror/api/v1/metrics.py": '''"""Prometheus metrics endpoint."""

from flask import Blueprint, Response
from prometheus_client import generate_latest, REGISTRY, CollectorRegistry

bp = Blueprint("metrics", __name__)


@bp.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics.
    
    Returns:
        Plain text Prometheus metrics format.
    """
    return Response(
        generate_latest(REGISTRY),
        mimetype="text/plain; version=0.0.4",
    )
''',

    # ===== Schema Validation =====
    "cognitive_mirror/schema/request.py": '''"""Request validation schemas using Marshmallow."""

from marshmallow import Schema, fields, validate, ValidationError


class PredictRequestSchema(Schema):
    """Validation for single prediction requests."""
    
    text = fields.String(
        required=True,
        validate=validate.Length(min=1, max=1000),
        metadata={"description": "Text input for emotion/sentiment analysis"},
        error_messages={
            "required": "The 'text' field is required.",
            "validator_failed": "Text must be between 1 and 1000 characters.",
        },
    )
    
    include_explanation = fields.Boolean(
        missing=False,
        metadata={"description": "Whether to generate LIME/SHAP explanation"},
    )
    
    model_version = fields.String(
        missing=None,
        validate=validate.Length(max=50),
        metadata={"description": "Specific model version tag (optional)"},
    )


class BatchPredictRequestSchema(Schema):
    """Validation for batch prediction requests."""
    
    texts = fields.List(
        fields.String(validate=validate.Length(min=1, max=1000)),
        required=True,
        validate=validate.Length(min=1, max=32),
        metadata={"description": "List of texts (max 32) for batch inference"},
        error_messages={
            "required": "The 'texts' field is required.",
            "validator_failed": "Batch must contain 1-32 items.",
        },
    )
    
    include_explanation = fields.Boolean(
        missing=False,
        metadata={"description": "Generate explanations for each prediction"},
    )
''',

    "cognitive_mirror/schema/response.py": '''"""Response serialization schemas."""

from marshmallow import Schema, fields


class TopEmotionSchema(Schema):
    """Top predicted emotion with probability."""
    emotion = fields.String(description="Emotion label")
    probability = fields.Float(description="Prediction probability")


class EmotionResultSchema(Schema):
    """Complete emotion prediction result."""
    emotion = fields.String(description="Primary predicted emotion")
    confidence = fields.Float(description="Confidence score for primary emotion")
    top_emotions = fields.List(
        fields.Nested(TopEmotionSchema),
        description="Top 3 emotions with probabilities",
    )


class SentimentProbabilitiesSchema(Schema):
    """Sentiment probability distribution."""
    negative = fields.Float()
    neutral = fields.Float()
    positive = fields.Float()


class SentimentResultSchema(Schema):
    """Complete sentiment prediction result."""
    sentiment = fields.String(description="Predicted sentiment label")
    confidence = fields.Float(description="Confidence score")
    probabilities = fields.Nested(
        SentimentProbabilitiesSchema,
        allow_none=True,
        description="Probability distribution (if available)",
    )


class ExplanationSchema(Schema):
    """Model explanation result."""
    top_features = fields.List(fields.Dict(), description="Top contributing features")
    method = fields.String(description="Explanation method used")


class PredictResponseSchema(Schema):
    """Complete prediction API response."""
    request_id = fields.String(description="Unique request identifier")
    text = fields.String(description="Original input text")
    emotion = fields.Nested(EmotionResultSchema, description="Emotion prediction")
    sentiment = fields.Nested(SentimentResultSchema, description="Sentiment prediction")
    mind_state = fields.String(description="Human-readable mind state summary")
    processing_time_ms = fields.Float(description="Inference time in milliseconds")
    model_version = fields.String(description="Model version used")
    explanation = fields.Nested(
        ExplanationSchema,
        allow_none=True,
        description="Model explanation (if requested)",
    )


class ErrorResponseSchema(Schema):
    """Standard error response."""
    error = fields.String(description="Error type")
    detail = fields.String(description="Error details")
    request_id = fields.String(allow_none=True, description="Request ID if available")
    timestamp = fields.String(description="ISO 8601 timestamp")
''',

    # ===== Monitoring =====
    "cognitive_mirror/monitoring/__init__.py": '"""Observability: structured logging, Prometheus metrics, and tracing."""\n',

    "cognitive_mirror/monitoring/metrics.py": '''"""Prometheus metrics definitions for the Cognitive Mirror API."""

from prometheus_client import Counter, Histogram, Gauge, Info


# === Prediction Metrics ===

prediction_counter = Counter(
    "cognitive_mirror_predictions_total",
    "Total number of predictions processed",
    ["status"],  # success, error, cache_hit
    namespace="cognitive_mirror",
)

prediction_latency_histogram = Histogram(
    "cognitive_mirror_prediction_latency_ms",
    "End-to-end prediction latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
    namespace="cognitive_mirror",
)

prediction_cache_hits = Counter(
    "cognitive_mirror_cache_hits_total",
    "Total number of cache hits for predictions",
    namespace="cognitive_mirror",
)

prediction_text_length = Histogram(
    "cognitive_mirror_input_text_length",
    "Input text character length distribution",
    buckets=[10, 25, 50, 100, 250, 500, 1000],
    namespace="cognitive_mirror",
)


# === Model Metrics ===

model_health_gauge = Gauge(
    "cognitive_mirror_model_healthy",
    "Model health status (1=healthy, 0=unhealthy)",
    namespace="cognitive_mirror",
)

model_info = Info(
    "cognitive_mirror_model",
    "Model version and metadata",
    namespace="cognitive_mirror",
)


# === Request Metrics ===

request_count = Counter(
    "cognitive_mirror_http_requests_total",
    "Total HTTP requests by endpoint",
    ["endpoint", "method", "status_code"],
    namespace="cognitive_mirror",
)

active_requests = Gauge(
    "cognitive_mirror_active_requests",
    "Number of requests currently being processed",
    namespace="cognitive_mirror",
)

request_size_bytes = Histogram(
    "cognitive_mirror_request_size_bytes",
    "HTTP request body size in bytes",
    buckets=[100, 500, 1000, 5000, 10000],
    namespace="cognitive_mirror",
)


# === Emotion Distribution ===

emotion_distribution = Counter(
    "cognitive_mirror_emotion_predictions_total",
    "Distribution of predicted emotions",
    ["emotion", "sentiment"],
    namespace="cognitive_mirror",
)
''',

    "cognitive_mirror/monitoring/logging.py": '''"""Structured logging configuration using structlog."""

import logging
import sys
from flask import Flask, has_request_context, g
import structlog
import uuid


def configure_logging(app: Flask) -> None:
    """Configure structured JSON logging for the application.
    
    In development: pretty-printed console logs.
    In production: JSON-formatted logs for log aggregation systems.
    """
    
    # Set log level
    log_level = logging.DEBUG if app.debug else logging.INFO
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if app.debug:
        # Development: colorful console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(serializer=structlog.processors.json.dumps),
        ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set levels for noisy libraries
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)


def get_logger(name: str = None):
    """Get a structured logger instance."""
    return structlog.get_logger(name or __name__)
''',

    # ===== Error Handling =====
    "cognitive_mirror/errors/handlers.py": '''"""Global error handlers for the Flask application."""

import time
from flask import Flask, jsonify, request, g


def register_error_handlers(app: Flask) -> None:
    """Register application-wide error handlers."""

    def _error_response(error_type: str, detail: str, status_code: int) -> tuple:
        """Build a standardized error response."""
        return jsonify({
            "error": error_type,
            "detail": detail,
            "request_id": g.get("request_id", "unknown"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status_code": status_code,
        }), status_code
    
    @app.errorhandler(400)
    def bad_request(error):
        return _error_response("Bad Request", str(error.description or "Invalid request"), 400)
    
    @app.errorhandler(404)
    def not_found(error):
        return _error_response("Not Found", f"Route not found: {request.path}", 404)
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return _error_response(
            "Method Not Allowed",
            f"Method {request.method} not allowed for {request.path}",
            405,
        )
    
    @app.errorhandler(413)
    def payload_too_large(error):
        return _error_response("Payload Too Large", "Request body exceeds maximum size", 413)
    
    @app.errorhandler(429)
    def too_many_requests(error):
        return _error_response(
            "Too Many Requests",
            f"Rate limit exceeded. Retry after: {error.description or '60 seconds'}",
            429,
        )
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal server error: {error}", exc_info=True)
        return _error_response("Internal Server Error", "An unexpected error occurred", 500)
    
    @app.errorhandler(503)
    def service_unavailable(error):
        return _error_response(
            "Service Unavailable",
            "The service is temporarily unavailable. Please try again later.",
            503,
        )
''',

    # ===== Services =====
    "cognitive_mirror/services/cache.py": '''"""Redis-based prediction caching with in-memory fallback."""

import pickle
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheService:
    """Two-tier cache: Redis primary with in-memory dictionary fallback.
    
    Features:
        - Automatic fallback if Redis is unavailable
        - Configurable TTL
        - Pickle serialization for complex objects
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 300):
        """Initialize cache service.
        
        Args:
            redis_url: Redis connection URL
            ttl: Default time-to-live in seconds
        """
        self.ttl = ttl
        self._redis = None
        self._fallback: dict = {}
        self._stats = {"hits": 0, "misses": 0, "sets": 0}
        
        try:
            import redis
            self._redis = redis.from_url(
                redis_url,
                decode_responses=False,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            self._redis.ping()
            logger.info(f"Redis cache connected successfully")
        except ImportError:
            logger.warning("redis-py not installed, using in-memory cache only")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), using in-memory fallback")
            self._redis = None
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not key:
            return None
        
        # Try Redis first
        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    self._stats["hits"] += 1
                    return pickle.loads(data)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")
        
        # Fallback to in-memory
        if key in self._fallback:
            self._stats["hits"] += 1
            return self._fallback[key]
        
        self._stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be picklable)
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if stored successfully
        """
        if not key:
            return False
        
        ttl = ttl or self.ttl
        
        if self._redis:
            try:
                data = pickle.dumps(value)
                self._redis.setex(key, ttl, data)
                self._stats["sets"] += 1
                return True
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")
        
        # Fallback to in-memory (no TTL in simple dict - consider using lru_cache)
        self._fallback[key] = value
        self._stats["sets"] += 1
        return True
    
    def delete(self, key: str) -> bool:
        """Remove a key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted
        """
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        
        self._fallback.pop(key, None)
        return True
    
    def clear(self) -> None:
        """Clear all cached data."""
        if self._redis:
            try:
                self._redis.flushdb()
            except Exception:
                pass
        self._fallback.clear()
        self._stats = {"hits": 0, "misses": 0, "sets": 0}
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self._stats,
            "fallback_size": len(self._fallback),
            "redis_available": self._redis is not None,
        }
''',

    "cognitive_mirror/services/predictor.py": '''"""Prediction orchestration service with caching and monitoring."""

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
''',

    # ===== Model Manager =====
    "cognitive_mirror/models/manager.py": '''"""Model lifecycle manager with health checks and inference."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import logging

import joblib

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when model file is missing or invalid."""
    pass


class ModelManager:
    """Singleton manager for ML model lifecycle.
    
    Handles:
        - Loading models from disk
        - Health checks
        - Version tracking
        - Inference orchestration
    """
    
    _models: Dict[str, Any] = {}
    _metadata: Dict[str, Any] = {}
    _initialized: bool = False
    
    @classmethod
    def initialize(cls, model_path: str) -> None:
        """Load and validate models from a checkpoint file.
        
        Args:
            model_path: Path to the joblib checkpoint file
            
        Raises:
            ModelNotFoundError: If file doesn't exist or is invalid
        """
        path = Path(model_path)
        
        if not path.exists():
            raise ModelNotFoundError(
                f"Model file not found: {model_path}. "
                f"Run 'python ml/train.py' first."
            )
        
        try:
            checkpoint = joblib.load(path)
            logger.info(f"Loaded checkpoint from {path} ({path.stat().st_size} bytes)")
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model file: {e}")
        
        # Validate required components
        required = ["emotion_model", "sentiment_model", "vectorizer", "label_encoder"]
        missing = [k for k in required if k not in checkpoint]
        if missing:
            raise ModelNotFoundError(
                f"Checkpoint missing required components: {missing}"
            )
        
        cls._models = {
            "emotion": checkpoint["emotion_model"],
            "sentiment": checkpoint["sentiment_model"],
            "vectorizer": checkpoint["vectorizer"],
            "label_encoder": checkpoint["label_encoder"],
        }
        
        cls._metadata = {
            "path": str(path),
            "checksum": cls._compute_checksum(path),
            "version": checkpoint.get("version", "unknown"),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "model_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }
        
        cls._initialized = True
        logger.info(f"Models loaded successfully. Version: {cls._metadata['version']}")
    
    @classmethod
    def is_healthy(cls) -> bool:
        """Check if all required models are loaded and ready."""
        required = {"emotion", "sentiment", "vectorizer", "label_encoder"}
        return cls._initialized and required.issubset(cls._models.keys())
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get model metadata dictionary."""
        return cls._metadata.copy()
    
    @classmethod
    def predict_emotion(cls, text: str) -> Dict[str, Any]:
        """Predict emotion from text input.
        
        Args:
            text: Input text
            
        Returns:
            Dict with emotion, confidence, and top_emotions
        """
        model = cls._models["emotion"]
        vectorizer = cls._models["vectorizer"]
        label_encoder = cls._models["label_encoder"]
        
        features = vectorizer.transform([text])
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities.max())
            prediction_idx = probabilities.argmax()
        else:
            prediction_idx = model.predict(features)[0]
            probabilities = None
            confidence = 1.0
        
        prediction = str(label_encoder.inverse_transform([prediction_idx])[0])
        
        # Get top 3 emotions
        top_emotions = []
        if probabilities is not None:
            top_indices = probabilities.argsort()[-3:][::-1]
            top_emotions = [
                {
                    "emotion": str(label_encoder.inverse_transform([i])[0]),
                    "probability": round(float(probabilities[i]), 4),
                }
                for i in top_indices
            ]
        
        return {
            "emotion": prediction,
            "confidence": round(confidence, 4),
            "top_emotions": top_emotions,
        }
    
    @classmethod
    def predict_sentiment(cls, text: str) -> Dict[str, Any]:
        """Predict sentiment from text input.
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment, confidence, and probability distribution
        """
        model = cls._models["sentiment"]
        vectorizer = cls._models["vectorizer"]
        
        features = vectorizer.transform([text])
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            prediction_idx = probabilities.argmax()
            confidence = float(probabilities.max())
        else:
            prediction_idx = model.predict(features)[0]
            probabilities = None
            confidence = 1.0
        
        # Map to sentiment labels
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = label_map.get(int(prediction_idx), "neutral")
        
        result = {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
        }
        
        if probabilities is not None:
            result["probabilities"] = {
                "negative": round(float(probabilities[0]), 4) if len(probabilities) > 0 else 0.0,
                "neutral": round(float(probabilities[1]), 4) if len(probabilities) > 1 else 0.0,
                "positive": round(float(probabilities[2]), 4) if len(probabilities) > 2 else 0.0,
            }
        else:
            result["probabilities"] = None
        
        return result
    
    @classmethod
    def generate_mindstate(cls, emotion_result: Dict, sentiment_result: Dict) -> str:
        """Generate a human-readable mind state description.
        
        Args:
            emotion_result: Emotion prediction output
            sentiment_result: Sentiment prediction output
            
        Returns:
            Natural language description of the inferred mind state
        """
        emotion = emotion_result["emotion"]
        sentiment = sentiment_result["sentiment"]
        confidence = emotion_result["confidence"]
        
        # Template-based generation
        templates = {
            ("joy", "positive"): "The person is cheerful, optimistic, and experiencing genuine happiness.",
            ("joy", "negative"): "The person may be masking pain behind a smile, showing bittersweet emotions.",
            ("sadness", "negative"): "The person is feeling down, disappointed, or experiencing loss.",
            ("sadness", "positive"): "The person is reflecting on something meaningful with a touch of melancholy.",
            ("fear", "negative"): "The person is anxious, uneasy, and sensing uncertainty.",
            ("fear", "positive"): "The person is cautiously optimistic, nervous but hopeful about what's ahead.",
            ("anger", "negative"): "The person is frustrated, irritated, or feeling wronged.",
            ("anger", "positive"): "The person is channeling frustration into motivation for positive change.",
            ("surprise", "positive"): "The person is experiencing unexpected joy or delightful wonder.",
            ("surprise", "negative"): "The person is shocked or caught off guard by something unpleasant.",
            ("disgust", "negative"): "The person is repulsed or strongly disapproving of something.",
            ("love", "positive"): "The person is feeling deep affection, connection, and emotional warmth.",
        }
        
        key = (emotion, sentiment)
        base = templates.get(
            key,
            f"The person is feeling {emotion} with a {sentiment} emotional outlook.",
        )
        
        # Add confidence qualifier
        if confidence < 0.5:
            base += " This assessment is uncertain."
        elif confidence > 0.9:
            base += " This assessment has very high confidence."
        
        return base
    
    @staticmethod
    def _compute_checksum(path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
''',

    # ===== New app.py entry point =====
    "app.py": '''#!/usr/bin/env python3
"""Cognitive Mirror - Production Inference API.

Usage:
    python app.py                  # Run development server
    FLASK_ENV=production python app.py  # Run in production mode
    gunicorn app:app               # Production WSGI server
    
Environment Variables:
    FLASK_ENV: Environment (development | production | testing)
    MODEL_PATH: Path to model checkpoint (default: models/model.pkl)
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
    PORT: Server port (default: 5000)
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_mirror.factory import create_app

# Get environment from FLASK_ENV or default to development
config_name = os.environ.get("FLASK_ENV", "development")

# Create the application
app = create_app(config_name)


def main():
    """Run the development server."""
    port = int(os.environ.get("PORT", 5000))
    debug = config_name == "development"
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║           Cognitive Mirror API v2.0.0                ║
╠══════════════════════════════════════════════════════╣
║  Environment: {config_name:<38}║
║  URL:         http://localhost:{port:<21}║
║  Health:      http://localhost:{port}/api/v1/health ║
║  Metrics:     http://localhost:{port}/api/v1/metrics║
║  API Docs:    http://localhost:{port}/api/v1/predict║
╚══════════════════════════════════════════════════════╝
    """)
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
    )


if __name__ == "__main__":
    main()
''',

    # ===== Configuration Files =====
    "requirements.txt": '''# Cognitive Mirror - Production Dependencies
# Core Framework
flask>=3.0.0,<4.0.0
flask-cors>=4.0.0
flask-limiter>=3.5.0

# Data Validation
marshmallow>=3.20.0

# Model Dependencies
joblib>=1.3.0
scikit-learn>=1.3.0
numpy>=1.24.0,<2.0.0
scipy>=1.11.0

# Caching & Storage
redis>=5.0.0

# Monitoring & Observability
prometheus-client>=0.19.0
structlog>=23.0.0

# Production Server
gunicorn>=21.2.0

# Optional: Enhanced Performance
# ujson>=5.8.0
# python-dotenv>=1.0.0

# Optional: Model Explainability
# lime>=0.2.0.1
# shap>=0.42.0
''',

    "Dockerfile": '''# Multi-stage build for Cognitive Mirror
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production image
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r cognitive && useradd -r -g cognitive cognitive

# Copy installed packages from builder
COPY --from=builder /root/.local /home/cognitive/.local
ENV PATH=/home/cognitive/.local/bin:$PATH

# Copy application code
COPY --chown=cognitive:cognitive . .

# Switch to non-root user
USER cognitive

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/v1/health')"

EXPOSE 5000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \\
     "--workers", "4", \\
     "--timeout", "30", \\
     "--access-logfile", "-", \\
     "--error-logfile", "-", \\
     "app:app"]
''',

    "docker-compose.yml": '''version: "3.8"

services:
  redis:
    image: redis:7-alpine
    container_name: cognitive-mirror-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  app:
    build: .
    container_name: cognitive-mirror-app
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
      - REDIS_URL=redis://redis:6379/0
      - MODEL_PATH=models/model.pkl
      - SECRET_KEY=dev-secret-change-in-production
    volumes:
      - ./models:/app/models:ro
    depends_on:
      redis:
        condition: service_healthy
    command: >
      gunicorn --bind 0.0.0.0:5000
      --workers 2
      --reload
      --timeout 30
      --access-logfile -
      app:app
    restart: unless-stopped

volumes:
  redis_data:
''',

    ".gitignore": '''# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
venv/
env/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# Model files (large)
models/*.pkl
models/*.joblib

# Redis
dump.rdb

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
''',

    "Makefile": '''.PHONY: help install train run test clean docker-build docker-up docker-down

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\\n", $$1, $$2}'

install:  ## Install Python dependencies
	pip install -r requirements.txt

train:  ## Train the ML models
	python ml/train.py

run:  ## Run development server
	python app.py

test:  ## Run test suite
	python -m pytest tests/ -v

lint:  ## Check code style
	python -m ruff check cognitive_mirror/

format:  ## Format code with ruff
	python -m ruff format cognitive_mirror/

clean:  ## Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker-build:  ## Build Docker image
	docker build -t cognitive-mirror:latest .

docker-up:  ## Start with Docker Compose
	docker-compose up -d

docker-down:  ## Stop Docker Compose
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f app

health:  ## Check API health
	@curl -s http://localhost:5000/api/v1/health | python -m json.tool

predict:  ## Test prediction endpoint
	@curl -s -X POST http://localhost:5000/api/v1/predict \\
		-H "Content-Type: application/json" \\
		-d '{"text": "I am feeling very curious and excited about this project!"}' \\
		| python -m json.tool

benchmark:  ## Run load test (requires Apache Bench)
	@echo '{"text": "test benchmark"}' > /tmp/bench.json
	ab -n 1000 -c 10 -p /tmp/bench.json -T "application/json" http://localhost:5000/api/v1/predict
	@rm /tmp/bench.json
''',

    "scripts/load_test.py": '''#!/usr/bin/env python3
"""Simple load testing script for the Cognitive Mirror API."""

import time
import json
import urllib.request
import concurrent.futures
from typing import List, Dict


def make_prediction(text: str, url: str = "http://localhost:5000/api/v1/predict") -> Dict:
    """Make a single prediction request."""
    data = json.dumps({"text": text}).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as response:
        result = json.loads(response.read().decode())
    
    elapsed = (time.perf_counter() - start) * 1000
    result["_client_time_ms"] = round(elapsed, 2)
    return result


def run_load_test(num_requests: int = 100, concurrency: int = 5):
    """Run a concurrent load test."""
    test_texts = [
        "I am feeling very happy today!",
        "This is making me quite anxious and worried.",
        "I'm not sure how to feel about this situation.",
        "What a wonderful surprise!",
        "I am so frustrated with everything right now.",
    ]
    
    print(f"Starting load test: {num_requests} requests, {concurrency} concurrent")
    print("-" * 60)
    
    start_time = time.time()
    successful = 0
    failed = 0
    times = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            text = test_texts[i % len(test_texts)]
            futures.append(executor.submit(make_prediction, text))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                successful += 1
                times.append(result.get("processing_time_ms", 0))
            except Exception as e:
                failed += 1
                print(f"  Error: {e}")
    
    total_time = time.time() - start_time
    
    # Statistics
    print(f"\\nResults:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Successful:     {successful}")
    print(f"  Failed:         {failed}")
    print(f"  Throughput:     {successful / total_time:.1f} req/s")
    
    if times:
        times.sort()
        print(f"  Server latency (mean): {sum(times) / len(times):.1f}ms")
        print(f"  Server latency (p50):  {times[len(times)//2]:.1f}ms")
        print(f"  Server latency (p95):  {times[int(len(times)*0.95)]:.1f}ms")
        print(f"  Server latency (p99):  {times[int(len(times)*0.99)]:.1f}ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test Cognitive Mirror API")
    parser.add_argument("-n", "--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("-c", "--concurrency", type=int, default=5, help="Concurrent workers")
    
    args = parser.parse_args()
    run_load_test(args.requests, args.concurrency)
''',
}

# Files that must NOT be overwritten (your existing files)
PROTECTED_FILES = {
    "ml/train.py",       # Your current training script
    "templates/",        # Your HTML templates
    "static/",           # Your CSS/JS files
    ".git/",             # Git history
}


# ============================================================================
# Core Logic
# ============================================================================

class StructureCreator:
    """Creates the project directory structure safely."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = {
            "dirs_created": 0,
            "dirs_existing": 0,
            "files_created": 0,
            "files_skipped": 0,
            "files_protected": 0,
        }
    
    def log(self, message: str, level: str = "info"):
        """Print formatted log message."""
        prefixes = {
            "info": "  →",
            "create": "  ✅",
            "skip": "  ⏭️",
            "protect": "  🔒",
            "warn": "  ⚠️",
            "error": "  ❌",
        }
        prefix = prefixes.get(level, "  •")
        print(f"{prefix} {message}")
    
    def is_protected(self, path: str) -> bool:
        """Check if a file/directory is in the protected list."""
        path_obj = Path(path)
        for protected in PROTECTED_FILES:
            if path_obj.match(protected) or protected in str(path_obj):
                return True
        return False
    
    def create_directories(self):
        """Create all required directories."""
        print("\n📁 Creating directories...")
        print("-" * 50)
        
        for dir_path in DIRECTORIES:
            path = Path(dir_path)
            
            if self.is_protected(dir_path):
                self.stats["files_protected"] += 1
                if self.verbose:
                    self.log(f"{dir_path} (PROTECTED - skipped)", "protect")
                continue
            
            if path.exists():
                self.stats["dirs_existing"] += 1
                if self.verbose:
                    self.log(f"{dir_path} (already exists)", "skip")
            else:
                self.stats["dirs_created"] += 1
                self.log(f"Creating: {dir_path}", "create")
                
                if not self.dry_run:
                    path.mkdir(parents=True, exist_ok=True)
    
    def create_files(self):
        """Create all required files."""
        print(f"\n📄 Creating files...")
        print("-" * 50)
        
        for file_path, content in FILES.items():
            path = Path(file_path)
            
            # Check protection
            if self.is_protected(file_path):
                self.stats["files_protected"] += 1
                if self.verbose:
                    self.log(f"{file_path} (PROTECTED)", "protect")
                continue
            
            # Check if file exists
            if path.exists():
                self.stats["files_skipped"] += 1
                if self.verbose:
                    self.log(f"{file_path} (already exists - skipped)", "skip")
            else:
                # Ensure parent directory exists
                if not self.dry_run:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write file
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                
                self.stats["files_created"] += 1
                self.log(f"Created: {file_path}", "create")
    
    def print_summary(self):
        """Print creation summary."""
        print(f"\n{'=' * 50}")
        print("📊 Summary")
        print("=" * 50)
        print(f"  Directories created:  {self.stats['dirs_created']}")
        print(f"  Directories existing: {self.stats['dirs_existing']}")
        print(f"  Files created:        {self.stats['files_created']}")
        print(f"  Files skipped:        {self.stats['files_skipped']}")
        print(f"  Files protected:      {self.stats['files_protected']}")
        
        total_new = self.stats["dirs_created"] + self.stats["files_created"]
        total_skipped = self.stats["dirs_existing"] + self.stats["files_skipped"] + self.stats["files_protected"]
        
        print(f"\n  ✅ {total_new} new items created")
        print(f"  ⏭️  {total_skipped} items skipped/preserved")
        
        if self.dry_run:
            print("\n  ⚠️  DRY RUN - No files were actually created")
        else:
            print("\n  ✅ Done! Your project structure is ready.")
            print("\n  Next steps:")
            print("    1. pip install -r requirements.txt")
            print("    2. python ml/train.py   # If you haven't trained yet")
            print("    3. python app.py")
    
    def run(self):
        """Execute the full creation process."""
        print("\n" + "=" * 50)
        print("🚀 Cognitive Mirror - Structure Creator")
        print("=" * 50)
        
        if self.dry_run:
            print("⚠️  DRY RUN MODE - No changes will be made")
        
        self.create_directories()
        self.create_files()
        self.print_summary()


def main():
    """Parse arguments and run the structure creator."""
    parser = argparse.ArgumentParser(
        description="Create Cognitive Mirror project structure without overwriting existing files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_structure.py              # Create all files and directories
  python create_structure.py --dry-run    # Preview without creating
  python create_structure.py --verbose    # Show all operations including skips
        """,
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without creating files",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all operations including existing/skipped files",
    )
    
    args = parser.parse_args()
    
    creator = StructureCreator(
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    creator.run()


if __name__ == "__main__":
    main()