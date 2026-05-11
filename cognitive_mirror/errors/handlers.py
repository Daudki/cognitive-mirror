"""Global error handlers for the Flask application."""

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
