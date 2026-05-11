"""Prometheus metrics definitions for the Cognitive Mirror API."""

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
