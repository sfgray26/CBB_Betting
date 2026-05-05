"""
Tests for backend/services/config_service.py.

Uses unittest.mock to isolate from the real database.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import backend.services.config_service as cs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(threshold_rows=None, flag_rows=None):
    """Return a mock SessionLocal() context that returns predetermined rows."""
    threshold_rows = threshold_rows or []
    flag_rows = flag_rows or []

    mock_db = MagicMock()
    execute = mock_db.execute.return_value

    def side_effect(stmt):
        sql_text = str(stmt)
        if "threshold_config" in sql_text:
            execute.fetchall.return_value = threshold_rows
        else:
            execute.fetchall.return_value = flag_rows
        return execute

    mock_db.execute.side_effect = side_effect
    return mock_db


def _row(key, value):
    return (key, value)


def _flag(name, enabled):
    return (name, enabled)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level cache before every test."""
    cs._threshold_cache = {}
    cs._flag_cache = {}
    cs._cache_expiry = 0.0
    yield
    cs._threshold_cache = {}
    cs._flag_cache = {}
    cs._cache_expiry = 0.0


# ---------------------------------------------------------------------------
# get_threshold
# ---------------------------------------------------------------------------

def test_get_threshold_returns_default_when_db_empty():
    mock_db = _make_db(threshold_rows=[], flag_rows=[])
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db):
        result = cs.get_threshold("nonexistent.key", default=42)
    assert result == 42


def test_get_threshold_returns_db_value():
    mock_db = _make_db(
        threshold_rows=[_row("scoring.z_cap", 2.5)],
        flag_rows=[],
    )
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db):
        result = cs.get_threshold("scoring.z_cap", default=3.0)
    assert result == 2.5


def test_get_threshold_returns_dict_value():
    weights = {"hr": 1.0, "rbi": 0.9}
    mock_db = _make_db(
        threshold_rows=[_row("scoring.category_weights", weights)],
        flag_rows=[],
    )
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db):
        result = cs.get_threshold("scoring.category_weights", default={})
    assert result == weights


def test_get_threshold_returns_default_on_db_error():
    with patch("backend.services.config_service.SessionLocal", side_effect=Exception("DB down")):
        result = cs.get_threshold("any.key", default=99)
    assert result == 99


def test_cache_hit_avoids_db_call():
    """After first load, subsequent calls within TTL must not hit DB."""
    mock_db = _make_db(
        threshold_rows=[_row("scoring.z_cap", 3.0)],
        flag_rows=[],
    )
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db) as mock_cls:
        cs.get_threshold("scoring.z_cap", default=3.0)
        cs.get_threshold("scoring.z_cap", default=3.0)
        cs.get_threshold("scoring.z_cap", default=3.0)

    # SessionLocal() should have been called exactly once
    assert mock_cls.call_count == 1


def test_cache_refresh_after_ttl():
    """After TTL expires, the next call should re-query the DB."""
    mock_db = _make_db(
        threshold_rows=[_row("scoring.z_cap", 3.0)],
        flag_rows=[],
    )
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db) as mock_cls:
        with patch("backend.services.config_service._CACHE_TTL", -1.0):
            # TTL is negative → every call re-queries
            cs.get_threshold("scoring.z_cap")
            cs.get_threshold("scoring.z_cap")

    assert mock_cls.call_count == 2


# ---------------------------------------------------------------------------
# is_flag_enabled
# ---------------------------------------------------------------------------

def test_is_flag_enabled_true():
    mock_db = _make_db(
        threshold_rows=[],
        flag_rows=[_flag("opportunity_enabled", True)],
    )
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db):
        assert cs.is_flag_enabled("opportunity_enabled") is True


def test_is_flag_enabled_false():
    mock_db = _make_db(
        threshold_rows=[],
        flag_rows=[_flag("opportunity_enabled", False)],
    )
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db):
        assert cs.is_flag_enabled("opportunity_enabled") is False


def test_is_flag_enabled_missing_flag_returns_false():
    mock_db = _make_db(threshold_rows=[], flag_rows=[])
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db):
        assert cs.is_flag_enabled("nonexistent_flag") is False


def test_is_flag_enabled_returns_false_on_db_error():
    with patch("backend.services.config_service.SessionLocal", side_effect=Exception("DB down")):
        assert cs.is_flag_enabled("opportunity_enabled") is False


# ---------------------------------------------------------------------------
# invalidate_cache
# ---------------------------------------------------------------------------

def test_invalidate_cache_forces_refresh():
    """After invalidate_cache(), the next call must re-query."""
    mock_db = _make_db(threshold_rows=[], flag_rows=[])
    with patch("backend.services.config_service.SessionLocal", return_value=mock_db) as mock_cls:
        cs.get_threshold("any.key")
        cs.invalidate_cache()
        cs.get_threshold("any.key")

    assert mock_cls.call_count == 2
