"""Request/Response middleware for logging and tracing."""

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
