"""
Unit tests for injury_status boolean coercion across all three schema layers.

Spec: reports/2026-05-19-spec-injury-status-api-contract-fix.md
Rule: Yahoo boolean True  → "IL"   (player is on injured list)
      Yahoo boolean False → None   (no injury; frontend uses null, not "Active")
"""
import json
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.contracts import CanonicalPlayerRow, FreshnessMetadata
from backend.schemas import LineupPlayerOut, RosterPlayerOut


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freshness():
    return FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=None,
        computed_at=datetime.now(ZoneInfo("America/New_York")),
        staleness_threshold_minutes=60,
        is_stale=False,
    )


def _canonical_row(**kwargs):
    defaults = dict(
        player_name="Garrett Crochet",
        team="BOS",
        eligible_positions=["SP"],
        status="IL",
        freshness=_freshness(),
    )
    defaults.update(kwargs)
    return CanonicalPlayerRow(**defaults)


def _lineup_player(**kwargs):
    defaults = dict(
        player_id="mlb.p.12345",
        name="Garrett Crochet",
        team="BOS",
        position="SP",
    )
    defaults.update(kwargs)
    return LineupPlayerOut(**defaults)


def _roster_player(**kwargs):
    defaults = dict(
        yahoo_player_key="mlb.p.12345",
        player_name="Garrett Crochet",
    )
    defaults.update(kwargs)
    return RosterPlayerOut(**defaults)


# ---------------------------------------------------------------------------
# CanonicalPlayerRow (contracts.py) — primary canonical schema
# ---------------------------------------------------------------------------

class TestCanonicalPlayerRowBoolCoercion:
    def test_bool_true_becomes_il(self):
        """Yahoo raw boolean True must coerce to 'IL'."""
        row = _canonical_row(injury_status=True)
        assert row.injury_status == "IL", f"Expected 'IL', got {row.injury_status!r}"

    def test_bool_false_becomes_none(self):
        """Yahoo raw boolean False means no injury — must serialize as null."""
        row = _canonical_row(injury_status=False)
        assert row.injury_status is None, f"Expected None, got {row.injury_status!r}"

    def test_string_il_passes_through(self):
        row = _canonical_row(injury_status="IL")
        assert row.injury_status == "IL"

    def test_string_dtd_passes_through(self):
        row = _canonical_row(injury_status="DTD")
        assert row.injury_status == "DTD"

    def test_none_stays_none(self):
        row = _canonical_row(injury_status=None)
        assert row.injury_status is None

    def test_json_never_emits_true_boolean(self):
        """JSON serialization must never emit `true` for injury_status."""
        row = _canonical_row(injury_status=True)
        payload = json.loads(row.model_dump_json())
        assert payload["injury_status"] == "IL"
        assert payload["injury_status"] is not True

    def test_json_never_emits_false_boolean(self):
        """JSON serialization must never emit `false` for injury_status."""
        row = _canonical_row(injury_status=False)
        payload = json.loads(row.model_dump_json())
        assert payload["injury_status"] is None
        assert payload["injury_status"] is not False

    def test_injury_return_timeline_bool_true_coerces(self):
        """Same validator covers injury_return_timeline."""
        row = _canonical_row(injury_return_timeline=True)
        assert row.injury_return_timeline == "IL"

    def test_injury_return_timeline_bool_false_coerces(self):
        row = _canonical_row(injury_return_timeline=False)
        assert row.injury_return_timeline is None


# ---------------------------------------------------------------------------
# LineupPlayerOut (schemas.py) — lineup recommendation shape
# ---------------------------------------------------------------------------

class TestLineupPlayerOutBoolCoercion:
    def test_bool_true_becomes_il(self):
        player = _lineup_player(injury_status=True)
        assert player.injury_status == "IL", f"Expected 'IL', got {player.injury_status!r}"

    def test_bool_false_becomes_none(self):
        player = _lineup_player(injury_status=False)
        assert player.injury_status is None

    def test_string_passes_through(self):
        player = _lineup_player(injury_status="DTD")
        assert player.injury_status == "DTD"

    def test_none_stays_none(self):
        player = _lineup_player(injury_status=None)
        assert player.injury_status is None

    def test_json_never_emits_boolean(self):
        player = _lineup_player(injury_status=True)
        payload = json.loads(player.model_dump_json())
        assert payload["injury_status"] == "IL"
        assert payload["injury_status"] is not True


# ---------------------------------------------------------------------------
# RosterPlayerOut (schemas.py) — legacy roster shape
# ---------------------------------------------------------------------------

class TestRosterPlayerOutBoolCoercion:
    def test_bool_true_becomes_il(self):
        player = _roster_player(injury_status=True)
        assert player.injury_status == "IL", f"Expected 'IL', got {player.injury_status!r}"

    def test_bool_false_becomes_none(self):
        player = _roster_player(injury_status=False)
        assert player.injury_status is None

    def test_status_bool_true_becomes_il(self):
        """status field shares the same validator as injury_status in RosterPlayerOut."""
        player = _roster_player(status=True)
        assert player.status == "IL"

    def test_status_bool_false_becomes_none(self):
        player = _roster_player(status=False)
        assert player.status is None

    def test_string_passes_through(self):
        player = _roster_player(injury_status="IL")
        assert player.injury_status == "IL"

    def test_json_never_emits_boolean(self):
        player = _roster_player(injury_status=True)
        payload = json.loads(player.model_dump_json())
        assert payload["injury_status"] == "IL"
        assert payload["injury_status"] is not True


# ---------------------------------------------------------------------------
# Cross-layer consistency: all three layers must agree on semantics
# ---------------------------------------------------------------------------

class TestCrossLayerConsistency:
    """Ensure all three schema layers use identical boolean → string semantics."""

    def test_all_layers_true_maps_to_il(self):
        canonical = _canonical_row(injury_status=True)
        lineup = _lineup_player(injury_status=True)
        roster = _roster_player(injury_status=True)
        assert canonical.injury_status == lineup.injury_status == roster.injury_status == "IL"

    def test_all_layers_false_maps_to_none(self):
        canonical = _canonical_row(injury_status=False)
        lineup = _lineup_player(injury_status=False)
        roster = _roster_player(injury_status=False)
        assert canonical.injury_status is None
        assert lineup.injury_status is None
        assert roster.injury_status is None
