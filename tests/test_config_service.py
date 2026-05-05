"""
Tests for backend/services/config_service.py.

Uses unittest.mock to isolate from the real database.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import backend.services.config_service as cs

_PATCH_TARGET = "backend.services.config_service.SessionLocal"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(threshold_rows=None, flag_rows=None):
    """Return a MagicMock DB session that returns predetermined query results."""
    threshold_rows = threshold_rows or []
    flag_rows = flag_rows or []

    session = MagicMock()

    def execute_side_effect(stmt):
        result = MagicMock()
        sql_text = str(stmt)
        if "threshold_config" in sql_text:
            result.fetchall.return_value = threshold_rows
        else:
            result.fetchall.return_value = flag_rows
        return result

    session.execute.side_effect = execute_side_effect
    return session


def _make_factory(threshold_rows=None, flag_rows=None):
    """Return a callable that acts as SessionLocal() → session."""
    session = _make_session(threshold_rows=threshold_rows, flag_rows=flag_rows)
    factory = MagicMock(return_value=session)
    return factory, session


def _row(key, value):
    return (key, value)


def _flag(name, enabled):
    return (name, enabled)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset module-level cache state before every test."""
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
    factory, _ = _make_factory()
    with patch(_PATCH_TARGET, factory):
        result = cs.get_threshold("nonexistent.key", default=42)
    assert result == 42


def test_get_threshold_returns_db_value():
    factory, _ = _make_factory(threshold_rows=[_row("scoring.z_cap", 2.5)])
    with patch(_PATCH_TARGET, factory):
        result = cs.get_threshold("scoring.z_cap", default=3.0)
    assert result == 2.5


def test_get_threshold_returns_dict_value():
    weights = {"hr": 1.0, "rbi": 0.9}
    factory, _ = _make_factory(threshold_rows=[_row("scoring.category_weights", weights)])
    with patch(_PATCH_TARGET, factory):
        result = cs.get_threshold("scoring.category_weights", default={})
    assert result == weights


def test_get_threshold_returns_default_on_db_error():
    factory = MagicMock(side_effect=Exception("DB down"))
    with patch(_PATCH_TARGET, factory):
        result = cs.get_threshold("any.key", default=99)
    assert result == 99


def test_cache_hit_avoids_db_call():
    """After the first successful load, subsequent calls within TTL skip the DB."""
    factory, _ = _make_factory(threshold_rows=[_row("scoring.z_cap", 3.0)])
    with patch(_PATCH_TARGET, factory):
        cs.get_threshold("scoring.z_cap", default=3.0)
        cs.get_threshold("scoring.z_cap", default=3.0)
        cs.get_threshold("scoring.z_cap", default=3.0)
    # SessionLocal() called once, not three times
    assert factory.call_count == 1


def test_cache_refresh_after_invalidate():
    """After invalidate_cache(), the next call re-queries the DB."""
    factory, _ = _make_factory(threshold_rows=[_row("scoring.z_cap", 3.0)])
    with patch(_PATCH_TARGET, factory):
        cs.get_threshold("scoring.z_cap")
        cs.invalidate_cache()
        cs.get_threshold("scoring.z_cap")
    assert factory.call_count == 2


# ---------------------------------------------------------------------------
# is_flag_enabled
# ---------------------------------------------------------------------------

def test_is_flag_enabled_true():
    factory, _ = _make_factory(flag_rows=[_flag("opportunity_enabled", True)])
    with patch(_PATCH_TARGET, factory):
        assert cs.is_flag_enabled("opportunity_enabled") is True


def test_is_flag_enabled_false():
    factory, _ = _make_factory(flag_rows=[_flag("opportunity_enabled", False)])
    with patch(_PATCH_TARGET, factory):
        assert cs.is_flag_enabled("opportunity_enabled") is False


def test_is_flag_enabled_missing_flag_returns_false():
    factory, _ = _make_factory()
    with patch(_PATCH_TARGET, factory):
        assert cs.is_flag_enabled("nonexistent_flag") is False


def test_is_flag_enabled_returns_false_on_db_error():
    factory = MagicMock(side_effect=Exception("DB down"))
    with patch(_PATCH_TARGET, factory):
        assert cs.is_flag_enabled("opportunity_enabled") is False


# ---------------------------------------------------------------------------
# invalidate_cache
# ---------------------------------------------------------------------------

def test_invalidate_cache_forces_refresh():
    """After invalidate_cache(), the next call must re-query."""
    factory, _ = _make_factory()
    with patch(_PATCH_TARGET, factory):
        cs.get_threshold("any.key")
        cs.invalidate_cache()
        cs.get_threshold("any.key")
    assert factory.call_count == 2
