"""Prometheus metrics endpoint."""

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
