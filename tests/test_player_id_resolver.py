"""
Tests for PlayerIDResolver service.

All tests mock the DB session and pybaseball -- no live DB or API calls made.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the service under test
# ---------------------------------------------------------------------------

from backend.services.player_id_resolver import PlayerIDResolver, _normalize_name


# ===========================================================================
# Unit tests for _normalize_name helper
# ===========================================================================

def test_normalize_name_lowercases():
    assert _normalize_name("Freddie Freeman") == "freddie freeman"


def test_normalize_name_strips_accents():
    # Unicode accent stripping
    assert _normalize_name("Yordan Alvarez") == "yordan alvarez"


def test_normalize_name_handles_empty():
    assert _normalize_name("") == ""


# ===========================================================================
# Test 1: Cache hit returns mlbam_id without calling pybaseball
# ===========================================================================

def test_cache_hit_returns_mlbam_without_pybaseball():
    """When a cache row exists, resolve() must return mlbam_id immediately."""
    db_mock = MagicMock()

    # Simulate PlayerIDMapping row returned from cache
    cache_row = MagicMock()
    cache_row.mlbam_id = 660271  # Shohei Ohtani's mlbam_id

    # db.query(...).filter(...).order_by(...).first() -> cache_row
    db_mock.query.return_value.filter.return_value.order_by.return_value.first.return_value = cache_row

    resolver = PlayerIDResolver(db_session=db_mock)

    with patch("backend.services.player_id_resolver.PlayerIDResolver._pybaseball_lookup") as pb_mock:
        result = resolver.resolve(bdl_player_id=12345, full_name="Shohei Ohtani")

    assert result == 660271
    pb_mock.assert_not_called(), "pybaseball must NOT be called on a cache hit"


# ===========================================================================
# Test 2: Cache miss calls pybaseball and persists the result
# ===========================================================================

def test_cache_miss_calls_pybaseball_and_persists():
    """When cache returns None, pybaseball is called and result is cached."""
    db_mock = MagicMock()

    # _cache_lookup returns None (no row in DB)
    db_mock.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    resolver = PlayerIDResolver(db_session=db_mock)

    with patch.object(resolver, "_cache_lookup", return_value=None), \
         patch.object(resolver, "_pybaseball_lookup", return_value=660271) as pb_mock, \
         patch.object(resolver, "_persist_to_cache") as persist_mock:
        result = resolver.resolve(bdl_player_id=12345, full_name="Shohei Ohtani")

    assert result == 660271
    pb_mock.assert_called_once_with("Shohei Ohtani")
    persist_mock.assert_called_once_with(12345, "Shohei Ohtani", 660271)


# ===========================================================================
# Test 3: pybaseball returning no rows returns None
# ===========================================================================

def test_pybaseball_no_results_returns_none():
    """When pybaseball finds no rows, resolve() must return None."""
    import pandas as pd

    resolver = PlayerIDResolver(db_session=MagicMock())

    # Simulate an empty DataFrame from pybaseball
    empty_df = MagicMock()
    empty_df.empty = True
    empty_df.__bool__ = lambda self: False

    with patch.object(resolver, "_cache_lookup", return_value=None), \
         patch("pybaseball.playerid_lookup", return_value=empty_df):
        result = resolver._pybaseball_lookup("NonExistentPlayer Zzz")

    assert result is None


# ===========================================================================
# Test 4: Manual override (source='manual') takes precedence
# ===========================================================================

def test_manual_override_takes_precedence():
    """
    A row with source='manual' must be returned first by _cache_lookup,
    even if a pybaseball row with different mlbam_id also exists.
    """
    db_mock = MagicMock()

    manual_row = MagicMock()
    manual_row.mlbam_id = 999_001    # manual override value
    manual_row.source = "manual"

    # Simulate query returning the manual row (ORDER BY puts manual first)
    db_mock.query.return_value.filter.return_value.order_by.return_value.first.return_value = manual_row

    resolver = PlayerIDResolver(db_session=db_mock)

    with patch.object(resolver, "_pybaseball_lookup") as pb_mock:
        result = resolver.resolve(bdl_player_id=99999, full_name="Test Player")

    assert result == 999_001
    pb_mock.assert_not_called()


# ===========================================================================
# Test 5: pybaseball result with multiple rows -- sorts by mlb_played_last
# ===========================================================================

def test_pybaseball_multiple_results_takes_last_valid():
    """
    When pybaseball returns multiple rows, _pybaseball_lookup must:
    - Filter to rows where key_mlbam is not null
    - Sort by mlb_played_last descending and take the most recently active player
    """
    resolver = PlayerIDResolver(db_session=MagicMock())

    # Build a minimal pandas DataFrame-like mock
    import pandas as pd

    rows = pd.DataFrame([
        {"name_last": "Garcia", "name_first": "Adolis", "key_mlbam": None, "mlb_played_last": None},
        {"name_last": "Garcia", "name_first": "Adolis", "key_mlbam": 671221, "mlb_played_last": 2018},
        {"name_last": "Garcia", "name_first": "Adolis", "key_mlbam": 690013, "mlb_played_last": 2024},
    ])

    with patch("pybaseball.playerid_lookup", return_value=rows):
        result = resolver._pybaseball_lookup("Adolis Garcia")

    # Should take the row with the highest mlb_played_last: key_mlbam=690013
    assert result == 690013


# ===========================================================================
# Test 6: _persist_to_cache skips manual override rows
# ===========================================================================

def test_persist_skips_manual_override():
    """_persist_to_cache must not overwrite source='manual' rows."""
    db_mock = MagicMock()

    manual_row = MagicMock()
    manual_row.source = "manual"

    # First filter().filter().first() returns a manual row
    db_mock.query.return_value.filter.return_value.filter.return_value.first.return_value = manual_row

    resolver = PlayerIDResolver(db_session=db_mock)
    resolver._persist_to_cache(
        bdl_player_id=12345,
        full_name="Freddie Freeman",
        mlbam_id=518692,
    )

    # db.add() must never be called -- manual row is sacred
    db_mock.add.assert_not_called()
    db_mock.commit.assert_not_called()


# ===========================================================================
# Test 7: resolve returns None when both cache and pybaseball fail
# ===========================================================================

def test_resolve_returns_none_when_all_lookups_fail():
    """If both cache and pybaseball return None, resolve() returns None."""
    resolver = PlayerIDResolver(db_session=MagicMock())

    with patch.object(resolver, "_cache_lookup", return_value=None), \
         patch.object(resolver, "_pybaseball_lookup", return_value=None), \
         patch.object(resolver, "_persist_to_cache") as persist_mock:
        result = resolver.resolve(bdl_player_id=99999, full_name="Unknown Player")

    assert result is None
    persist_mock.assert_not_called()
