"""
Redis Client Setup for Railway

Production-ready Redis client with connection pooling optimized for Railway infrastructure.
Based on K-31 research: https://docs.railway.app/reference/redis

Enhanced version with connection pooling, serialization, and health monitoring.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from redis import Redis, ConnectionPool, BlockingConnectionPool

logger = logging.getLogger(__name__)

_client: Optional[Redis] = None
_pool: Optional[ConnectionPool] = None


def get_redis_url() -> str:
    """Get Redis URL from environment with validation."""
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError(
            "REDIS_URL environment variable is required. "
            "Set it to your Railway Redis connection string."
        )
    return url


def create_connection_pool() -> ConnectionPool:
    """
    Create optimized connection pool for Railway.

    K-31 Recommendation:
    - Max connections: 30 (2-4 workers × 10-20 threads + buffer)
    - BlockingConnectionPool: Prevent pool exhaustion under load
    - Health check: Every 30 seconds
    - Socket timeouts: 5 seconds (Railway latency: 0.5-3ms private, 2-5ms TCP proxy)
    """
    url = get_redis_url()

    # Determine if this is external connection (needs TLS)
    use_tls = "rediss://" in url or "railway.app" in url

    pool_kwargs = {
        "max_connections": int(os.environ.get("REDIS_POOL_SIZE", "30")),
        "socket_timeout": 5.0,
        "socket_connect_timeout": 5.0,
        "socket_keepalive": True,
        "retry_on_timeout": True,
        "health_check_interval": 30,
        "decode_responses": False,  # We handle decoding manually for msgpack
    }

    if use_tls:
        import ssl
        pool_kwargs["ssl"] = True
        pool_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED

    return BlockingConnectionPool.from_url(url, **pool_kwargs)


def get_redis() -> Redis:
    """
    Get shared Redis client with connection pooling.

    Usage:
        redis = get_redis()
        redis.set('key', 'value')
        value = redis.get('key')
    """
    global _client, _pool

    if _client is None:
        _pool = create_connection_pool()
        _client = Redis(connection_pool=_pool)
        logger.info("Redis client initialized with connection pool")

    return _client


def close_redis():
    """
    Close Redis connection pool. Call on application shutdown.

    Usage in FastAPI:
        @app.on_event("shutdown")
        def shutdown():
            close_redis()
    """
    global _client, _pool

    if _pool:
        _pool.disconnect()
        _pool = None
        _client = None
        logger.info("Redis connection pool closed")


# ============================================================================
# LEGACY CODE PRESERVED - NamespacedCache (existing pattern)
# ============================================================================

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _redis_lib = None  # type: ignore[assignment]
    _REDIS_AVAILABLE = False


class NamespacedCache:
    """Redis cache with enforced key namespace prefix.

    Usage:
        fantasy_cache.set("ros:2026-04-04", json.dumps(data), ex=43200)
        edge_cache.get("bdl:rate:games")

    All keys are stored as '<prefix>:<k>' -- edge: and fantasy: keys
    can never collide regardless of the key string passed by callers.
    """

    def __init__(self, prefix: str, client=None) -> None:
        """
        Args:
            prefix: Namespace prefix, e.g. 'edge' or 'fantasy'.
            client: Optional injected Redis client for testing.
                    If None, uses get_redis() at call time.
        """
        self._prefix = prefix
        self._client = client

    def _r(self):
        return self._client if self._client is not None else get_redis()

    def key(self, k: str) -> str:
        return f"{self._prefix}:{k}"

    def get(self, k: str):
        return self._r().get(self.key(k))

    def set(self, k: str, v, ex: Optional[int] = None) -> None:
        self._r().set(self.key(k), v, ex=ex)

    def delete(self, k: str) -> None:
        self._r().delete(self.key(k))

    def exists(self, k: str) -> bool:
        return bool(self._r().exists(self.key(k)))


# Module-level singletons used by both services.
# Import these directly: from backend.redis_client import edge_cache, fantasy_cache
edge_cache = NamespacedCache("edge")
fantasy_cache = NamespacedCache("fantasy")
