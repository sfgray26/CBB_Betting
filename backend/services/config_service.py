"""
Config service — thread-safe in-memory cache for threshold_config table.

Usage:
    from backend.services.config_service import get_threshold, is_flag_enabled

    z_cap = get_threshold("scoring.z_cap", default=3.0)
    if is_flag_enabled("opportunity_enabled"):
        ...

Cache TTL: 60 seconds.
Returns the default when the DB is unavailable or the key is not found.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from backend.models import SessionLocal
from sqlalchemy import text

logger = logging.getLogger(__name__)

_CACHE_TTL = 60.0

_threshold_cache: dict[str, Any] = {}
_flag_cache: dict[str, bool] = {}
_cache_expiry: float = 0.0
_lock = threading.Lock()


def _refresh_cache() -> None:
    """Pull all threshold_config and feature_flags rows into module-level dicts."""
    global _threshold_cache, _flag_cache, _cache_expiry
    db = SessionLocal()
    try:
        rows = db.execute(
            text("SELECT config_key, config_value FROM threshold_config WHERE scope = 'global'")
        ).fetchall()
        new_thresholds: dict[str, Any] = {r[0]: r[1] for r in rows}

        flags = db.execute(
            text("SELECT flag_name, enabled FROM feature_flags")
        ).fetchall()
        new_flags: dict[str, bool] = {r[0]: bool(r[1]) for r in flags}

        _threshold_cache = new_thresholds
        _flag_cache = new_flags
        _cache_expiry = time.monotonic() + _CACHE_TTL
    except Exception:
        logger.exception("config_service: failed to refresh cache; keeping stale values")
    finally:
        db.close()


def _ensure_cache() -> None:
    if time.monotonic() <= _cache_expiry:
        return
    with _lock:
        if time.monotonic() <= _cache_expiry:  # double-checked locking
            return
        _refresh_cache()


def get_threshold(key: str, default: Any = None, scope: str = "global") -> Any:
    """
    Return a runtime config value from threshold_config.

    Falls back to `default` when the DB is unavailable or the key is not present.
    The `scope` parameter is reserved for future per-league overrides.
    """
    try:
        _ensure_cache()
    except Exception:
        logger.exception("config_service.get_threshold: cache refresh failed")
        return default
    return _threshold_cache.get(key, default)


def is_flag_enabled(flag_name: str) -> bool:
    """Return True if the named feature flag is enabled, False otherwise."""
    try:
        _ensure_cache()
    except Exception:
        logger.exception("config_service.is_flag_enabled: cache refresh failed")
        return False
    return _flag_cache.get(flag_name, False)


def invalidate_cache() -> None:
    """Force the next call to re-query the DB. Used in tests and after config updates."""
    global _cache_expiry
    with _lock:
        _cache_expiry = 0.0
