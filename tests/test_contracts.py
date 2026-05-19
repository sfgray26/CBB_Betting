"""Tests for backend/contracts.py — CanonicalPlayerRow field validators."""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from backend.contracts import CanonicalPlayerRow, FreshnessMetadata


def _make_freshness():
    return FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=None,
        computed_at=datetime.now(ZoneInfo("America/New_York")),
        staleness_threshold_minutes=60,
        is_stale=False,
    )


def _make_row(**overrides):
    base = dict(
        player_name="Garrett Crochet",
        team="CWS",
        eligible_positions=["SP"],
        status="IL",
        freshness=_make_freshness(),
    )
    base.update(overrides)
    return CanonicalPlayerRow(**base)


# ---------------------------------------------------------------------------
# injury_status coercion
# ---------------------------------------------------------------------------

def test_injury_status_bool_true_coerced_to_string():
    row = _make_row(injury_status=True)
    assert isinstance(row.injury_status, str), f"Expected str, got {type(row.injury_status)}"
    assert row.injury_status in ("IL", "Active", "Inactive", "true", "True"), \
        f"Unexpected coercion: {row.injury_status!r}"
    # Key assertion: it must NOT be a bool
    assert not isinstance(row.injury_status, bool)


def test_injury_status_bool_false_coerced_to_string():
    row = _make_row(injury_status=False)
    assert isinstance(row.injury_status, str)
    assert not isinstance(row.injury_status, bool)


def test_injury_status_string_passes_through():
    row = _make_row(injury_status="IL")
    assert row.injury_status == "IL"


def test_injury_status_none_stays_none():
    row = _make_row(injury_status=None)
    assert row.injury_status is None


# ---------------------------------------------------------------------------
# injury_return_timeline coercion (same validator covers both fields)
# ---------------------------------------------------------------------------

def test_injury_return_timeline_bool_true_coerced_to_string():
    row = _make_row(injury_return_timeline=True)
    assert isinstance(row.injury_return_timeline, str)
    assert not isinstance(row.injury_return_timeline, bool)


def test_injury_return_timeline_bool_false_coerced_to_string():
    row = _make_row(injury_return_timeline=False)
    assert isinstance(row.injury_return_timeline, str)
    assert not isinstance(row.injury_return_timeline, bool)


def test_injury_return_timeline_string_passes_through():
    row = _make_row(injury_return_timeline="7-10 days")
    assert row.injury_return_timeline == "7-10 days"


def test_injury_return_timeline_none_stays_none():
    row = _make_row(injury_return_timeline=None)
    assert row.injury_return_timeline is None
