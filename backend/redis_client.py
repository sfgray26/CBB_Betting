"""
backend/redis_client.py -- Shared Redis connection with namespace enforcement.

Both edge_app.py and fantasy_app.py import this module.
Use edge_cache for edge: keys, fantasy_cache for fantasy: keys.
Neither service can pollute the other's keyspace.
"""
from __future__ import annotations

import os
from typing import Optional

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _redis_lib = None  # type: ignore[assignment]
    _REDIS_AVAILABLE = False

_client: Optional[object] = None


def get_redis():
    """Return the shared Redis client, initialised lazily from REDIS_URL.

    Raises RuntimeError if REDIS_URL is not set -- forces explicit
    configuration rather than silent fallback.
    """
    global _client
    if _client is not None:
        return _client
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError(
            "REDIS_URL environment variable is required for Redis operations. "
            "Set it to your Railway Redis connection string."
        )
    if not _REDIS_AVAILABLE:
        raise RuntimeError(
            "redis-py is not installed. Add 'redis>=5.0' to requirements.txt."
        )
    _client = _redis_lib.from_url(url, decode_responses=True)
    return _client


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
