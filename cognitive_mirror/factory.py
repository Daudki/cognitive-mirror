"""Application factory pattern for creating Flask instances."""

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
