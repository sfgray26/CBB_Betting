"""
Tests for PlayerAutoHealService.

All tests mock DB session and BDL client — no live DB or API calls.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from backend.services.player_autoheal import (
    AUTO_HEAL_TTL_DAYS,
    CONFIDENCE_THRESHOLD,
    PlayerAutoHealService,
    _name_confidence,
    _normalize,
    _is_fresh,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_bdl_player(bdl_id: int, full_name: str) -> MagicMock:
    p = MagicMock()
    p.id = bdl_id
    p.full_name = full_name
    return p


def _make_db(existing_row=None):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = existing_row
    return db


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

def test_normalize_lowercases_and_strips_accents():
    assert _normalize("Yordan Alvarez") == "yordan alvarez"
    assert _normalize("Adolis García") == "adolis garcia"


def test_name_confidence_exact():
    assert _name_confidence("jackson chourio", "jackson chourio") == 1.0


def test_name_confidence_substring():
    assert _name_confidence("chourio", "jackson chourio") == 0.9


def test_name_confidence_no_match():
    assert _name_confidence("freddie freeman", "shohei ohtani") == 0.0


def test_is_fresh_within_ttl():
    recent = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=3)
    assert _is_fresh(recent) is True


def test_is_fresh_beyond_ttl():
    old = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=AUTO_HEAL_TTL_DAYS + 1)
    assert _is_fresh(old) is False


def test_is_fresh_naive_datetime():
    recent = datetime.now() - timedelta(days=1)  # naive
    assert _is_fresh(recent) is True


# ---------------------------------------------------------------------------
# Test 1: New call-up — no existing row — BDL finds a match → healed
# ---------------------------------------------------------------------------

def test_heal_new_callup_inserts_row():
    """New player with no existing mapping heals successfully."""
    db = MagicMock()
    # No existing row for this yahoo_key, no bdl_id conflict
    db.query.return_value.filter.return_value.first.return_value = None

    bdl_player = _make_bdl_player(99001, "Jackson Chourio")
    bdl = MagicMock()
    bdl.search_mlb_players.return_value = [bdl_player]

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    result = svc.heal_player(
        yahoo_id="99001",
        yahoo_key="422.p.99001",
        name="Jackson Chourio",
    )

    assert result is True
    db.add.assert_called_once()
    db.commit.assert_called_once()
    added_row = db.add.call_args[0][0]
    assert added_row.bdl_id == 99001
    assert added_row.source == "bdl_search"
    assert added_row.yahoo_key == "422.p.99001"


# ---------------------------------------------------------------------------
# Test 2: Persisted mapping — auto-healed mapping has correct source
# ---------------------------------------------------------------------------

def test_heal_sets_source_bdl_search():
    """Inserted row must have source='bdl_search'."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    bdl = MagicMock()
    bdl.search_mlb_players.return_value = [_make_bdl_player(50001, "Test Player")]

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    svc.heal_player(yahoo_id="50001", yahoo_key="422.p.50001", name="Test Player")

    added = db.add.call_args[0][0]
    assert added.source == "bdl_search"
    assert added.resolution_confidence == 1.0


# ---------------------------------------------------------------------------
# Test 3: Manual override row — must never be overwritten
# ---------------------------------------------------------------------------

def test_heal_skips_manual_override():
    """heal_player must return False and not modify source='manual' rows."""
    manual_row = MagicMock()
    manual_row.source = "manual"
    manual_row.bdl_id = 12345

    db = _make_db(existing_row=manual_row)
    bdl = MagicMock()

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    result = svc.heal_player(
        yahoo_id="12345",
        yahoo_key="422.p.12345",
        name="Freddie Freeman",
    )

    assert result is False
    bdl.search_mlb_players.assert_not_called()
    db.add.assert_not_called()
    db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Fresh bdl_search row — TTL skip
# ---------------------------------------------------------------------------

def test_heal_skips_fresh_bdl_search_row():
    """A fresh bdl_search row must be skipped without calling BDL API."""
    fresh_row = MagicMock()
    fresh_row.source = "bdl_search"
    fresh_row.bdl_id = 77001
    fresh_row.updated_at = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=2)

    db = _make_db(existing_row=fresh_row)
    bdl = MagicMock()

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    result = svc.heal_player(
        yahoo_id="77001",
        yahoo_key="422.p.77001",
        name="Some Player",
    )

    assert result is False
    bdl.search_mlb_players.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: Stale bdl_search row — refresh after TTL
# ---------------------------------------------------------------------------

def test_heal_refreshes_stale_bdl_search_row():
    """A stale bdl_search row (beyond TTL) is refreshed with a new BDL search."""
    stale_row = MagicMock()
    stale_row.source = "bdl_search"
    stale_row.bdl_id = 77002
    stale_row.updated_at = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=AUTO_HEAL_TTL_DAYS + 2)

    db = _make_db(existing_row=stale_row)
    bdl = MagicMock()
    bdl.search_mlb_players.return_value = [_make_bdl_player(77002, "Stale Player")]

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    result = svc.heal_player(
        yahoo_id="77002",
        yahoo_key="422.p.77002",
        name="Stale Player",
    )

    assert result is True
    bdl.search_mlb_players.assert_called_once_with("Stale Player")
    db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Test 6: BDL returns no results → False, no DB write
# ---------------------------------------------------------------------------

def test_heal_returns_false_when_no_bdl_results():
    """If BDL finds no players, heal_player returns False without DB writes."""
    db = _make_db(existing_row=None)
    bdl = MagicMock()
    bdl.search_mlb_players.return_value = []

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    result = svc.heal_player(
        yahoo_id="88001",
        yahoo_key="422.p.88001",
        name="Unknown Callup",
    )

    assert result is False
    db.add.assert_not_called()
    db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Test 7: BDL search raises exception → False, no crash
# ---------------------------------------------------------------------------

def test_heal_handles_bdl_exception_gracefully():
    """BDL search failure must not raise — return False."""
    db = _make_db(existing_row=None)
    bdl = MagicMock()
    bdl.search_mlb_players.side_effect = RuntimeError("BDL timeout")

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    result = svc.heal_player(
        yahoo_id="11111",
        yahoo_key="422.p.11111",
        name="Any Player",
    )

    assert result is False
    db.add.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: batch_heal — mixed list → correct counts
# ---------------------------------------------------------------------------

def test_batch_heal_counts():
    """batch_heal returns correct healed/skipped/failed counts."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    bdl = MagicMock()
    bdl.search_mlb_players.side_effect = lambda name: (
        [_make_bdl_player(30001, name)] if name == "Healable Player" else []
    )

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)

    unmatched = [
        {"name": "Healable Player", "yahoo_id": "30001", "yahoo_key": "422.p.30001"},
        {"name": "No BDL Match", "yahoo_id": "30002", "yahoo_key": "422.p.30002"},
        {"name": "Collision", "yahoo_id": "30003", "yahoo_key": "422.p.30003", "reason": "bdl_name_collision"},
        {"name": "Conflict", "yahoo_id": "30004", "yahoo_key": "422.p.30004", "reason": "bdl_id_conflict"},
        {"name": "", "yahoo_id": "30005", "yahoo_key": "422.p.30005"},  # missing name → skip
    ]

    result = svc.batch_heal(unmatched)

    assert result["healed"] == 1   # Healable Player
    assert result["failed"] == 1   # No BDL Match
    assert result["skipped"] == 3  # collision + conflict + missing name


# ---------------------------------------------------------------------------
# Test 9: batch_heal — missing yahoo_key → skip
# ---------------------------------------------------------------------------

def test_batch_heal_skips_missing_yahoo_key():
    """Players without yahoo_key are skipped (can't insert without it)."""
    db = MagicMock()
    bdl = MagicMock()

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    result = svc.batch_heal([{"name": "Player", "yahoo_id": "999"}])

    assert result["skipped"] == 1
    assert result["healed"] == 0
    bdl.search_mlb_players.assert_not_called()


# ---------------------------------------------------------------------------
# Test 10: bdl_id conflict on insert → False
# ---------------------------------------------------------------------------

def test_heal_skips_bdl_id_conflict_on_insert():
    """If BDL player's ID is already used by another mapping row, skip."""
    db = MagicMock()
    # No existing row for this yahoo_key
    # But conflict check (second .first() call) returns a row
    existing_conflict = MagicMock()
    existing_conflict.id = 5555
    # first call (yahoo_key lookup) → None; second call (bdl_id conflict) → existing_conflict
    db.query.return_value.filter.return_value.first.side_effect = [None, existing_conflict]

    bdl = MagicMock()
    bdl.search_mlb_players.return_value = [_make_bdl_player(5555, "Conflicting Player")]

    svc = PlayerAutoHealService(db_session=db, bdl_client=bdl)
    result = svc.heal_player(
        yahoo_id="40001",
        yahoo_key="422.p.40001",
        name="Conflicting Player",
    )

    assert result is False
    db.add.assert_not_called()
