"""Flask extensions initialization."""

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
