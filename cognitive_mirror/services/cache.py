"""Redis-based prediction caching with in-memory fallback."""

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
