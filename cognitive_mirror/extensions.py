"""Flask extensions initialization."""

from typing import Optional

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from redis import Redis

# Global extension instances
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()

redis_client: Optional[Redis] = None


def init_extensions(app) -> None:
    """Initialize all Flask extensions with the application context."""
    global redis_client

    # --- Database ---
    db.init_app(app)
    migrate.init_app(app, db)

    # --- Auth ---
    login_manager.init_app(app)
    login_manager.login_view = None  # API-only; unauthenticated hits return 401 JSON, not a redirect

    @login_manager.user_loader
    def load_user(user_id: str):
        from cognitive_mirror.domain.user import User
        return db.session.get(User, int(user_id))

    @login_manager.unauthorized_handler
    def unauthorized():
        from flask import jsonify
        return jsonify({"error": "Authentication required"}), 401

    # --- Redis (cache; optional, falls back gracefully) ---
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
        except Exception:
            redis_client = None
            app.logger.info("Redis unavailable, using in-memory fallback")
