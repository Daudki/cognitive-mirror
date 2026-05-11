"""Application configuration with environment-based profiles."""

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
