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

def test_injury_status_bool_true_coerced_to_il():
    """Boolean True from Yahoo means player is on IL."""
    row = _make_row(injury_status=True)
    assert row.injury_status == "IL", f"Expected 'IL', got {row.injury_status!r}"


def test_injury_status_bool_false_coerced_to_none():
    """Boolean False from Yahoo means no injury — must serialize as null, not a string."""
    row = _make_row(injury_status=False)
    assert row.injury_status is None


def test_injury_status_string_passes_through():
    row = _make_row(injury_status="IL")
    assert row.injury_status == "IL"


def test_injury_status_none_stays_none():
    row = _make_row(injury_status=None)
    assert row.injury_status is None


# ---------------------------------------------------------------------------
# injury_return_timeline coercion (same validator covers both fields)
# ---------------------------------------------------------------------------

def test_injury_return_timeline_bool_true_coerced_to_il():
    """Boolean True for return timeline coerces to 'IL' (same validator as injury_status)."""
    row = _make_row(injury_return_timeline=True)
    assert row.injury_return_timeline == "IL"


def test_injury_return_timeline_bool_false_coerced_to_none():
    """Boolean False for return timeline means no timeline — must be null."""
    row = _make_row(injury_return_timeline=False)
    assert row.injury_return_timeline is None


def test_injury_return_timeline_string_passes_through():
    row = _make_row(injury_return_timeline="7-10 days")
    assert row.injury_return_timeline == "7-10 days"


def test_injury_return_timeline_none_stays_none():
    row = _make_row(injury_return_timeline=None)
    assert row.injury_return_timeline is None
