"""Health check and Kubernetes probe endpoints."""

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
