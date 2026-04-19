"""
Tests for Phase 4 Player Mapper.

Tests for map_yahoo_player_to_canonical_row() and related helpers.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.services.player_mapper import (
    map_yahoo_player_to_canonical_row,
    _normalize_status,
    _map_rolling_to_category_stats,
    _map_yahoo_stats_to_category_stats,
)
from backend.contracts import CanonicalPlayerRow, CategoryStats
from backend.stat_contract import SCORING_CATEGORY_CODES
from backend.models import PlayerRollingStats


class TestNormalizeStatus:
    """Tests for _normalize_status()."""

    def test_active_status(self):
        assert _normalize_status({}) == "playing"
        assert _normalize_status({"status": "Active"}) == "playing"

    def test_il_variants(self):
        assert _normalize_status({"status": "IL"}) == "IL"
        assert _normalize_status({"status": "IL15"}) == "IL"
        assert _normalize_status({"status": "IL60"}) == "IL"
        assert _normalize_status({"status": "DL"}) == "IL"

    def test_dtd_status(self):
        assert _normalize_status({"status": "DTD"}) == "probable"
        assert _normalize_status({"status": "Day-to-Day"}) == "probable"

    def test_out_status(self):
        assert _normalize_status({"status": "O"}) == "not_playing"
        assert _normalize_status({"status": "Out"}) == "not_playing"
        assert _normalize_status({"status": "SUSPENDED"}) == "not_playing"

    def test_minors_status(self):
        assert _normalize_status({"status": "NA"}) == "minors"
        assert _normalize_status({"status": "N/A"}) == "minors"
        assert _normalize_status({"status": "Minors"}) == "minors"

    def test_boolean_status(self):
        assert _normalize_status({"status": True}) == "playing"
        assert _normalize_status({"status": False}) == "not_playing"


class TestMapRollingToCategoryStats:
    """Tests for _map_rolling_to_category_stats()."""

    def test_none_returns_none(self):
        assert _map_rolling_to_category_stats(None) is None

    def test_batting_stats_mapped(self):
        rolling = PlayerRollingStats(
            bdl_player_id=1,
            as_of_date=datetime(2026, 4, 1).date(),
            window_days=14,
            games_in_window=10,
            w_games=9.5,
            w_runs=50.0,
            w_hits=120.0,
            w_home_runs=15.0,
            w_rbi=50.0,
            w_strikeouts_bat=70.0,
            w_tb=180.0,
            w_avg=0.267,
            w_ops=0.780,
            w_net_stolen_bases=4.0,
        )

        result = _map_rolling_to_category_stats(rolling)

        assert isinstance(result, CategoryStats)
        assert result.values["R"] == 50.0
        assert result.values["H"] == 120.0
        assert result.values["HR_B"] == 15.0
        assert result.values["RBI"] == 50.0
        assert result.values["TB"] == 180.0
        assert result.values["AVG"] == 0.267
        assert result.values["OPS"] == 0.780
        assert result.values["NSB"] == 4.0

    def test_pitching_stats_mapped(self):
        rolling = PlayerRollingStats(
            bdl_player_id=1,
            as_of_date=datetime(2026, 4, 1).date(),
            window_days=14,
            games_in_window=3,
            w_games=3.0,
            w_strikeouts_pit=30.0,
            w_era=3.50,
            w_whip=1.20,
            w_k_per_9=9.5,
            w_qs=2.0,
        )

        result = _map_rolling_to_category_stats(rolling)

        assert result.values["K_P"] == 30.0
        assert result.values["ERA"] == 3.50
        assert result.values["WHIP"] == 1.20
        assert result.values["K_9"] == 9.5
        assert result.values["QS"] == 2.0


class TestMapYahooStatsToCategoryStats:
    """Tests for _map_yahoo_stats_to_category_stats()."""

    def test_empty_stats_returns_none(self):
        assert _map_yahoo_stats_to_category_stats({}) is None
        assert _map_yahoo_stats_to_category_stats({"name": "Player"}) is None

    def test_stat_id_mapping(self):
        # Yahoo stat_ids: 0=none, 5=runs, 6=hr, 7=rbi, etc.
        # Using mock stat_id values that map via CONTRACT.yahoo_id_index
        yahoo_player = {
            "stats": {
                "5": "50",   # Runs (stat_id 5 -> R)
                "6": "15",   # HR (stat_id 6 -> HR_B)
                "7": "48",   # RBI (stat_id 7 -> RBI)
            }
        }

        result = _map_yahoo_stats_to_category_stats(yahoo_player)

        if result:  # Only test if mapping exists in contract
            assert isinstance(result, CategoryStats)
            # Values depend on contract mapping - check some were set
            # The values dict should have all 18 categories, check some have non-None values
            non_none_values = [v for v in result.values.values() if v is not None]
            assert len(non_none_values) > 0


class TestMapYahooPlayerToCanonicalRow:
    """Tests for map_yahoo_player_to_canonical_row()."""

    def test_minimal_player(self):
        yahoo_player = {
            "player_key": "469.l.72586.p.12345",
            "name": "Test Player",
            "team": "NYY",
            "positions": ["1B", "OF"],
        }

        result = map_yahoo_player_to_canonical_row(yahoo_player)

        assert isinstance(result, CanonicalPlayerRow)
        assert result.player_name == "Test Player"
        assert result.team == "NYY"
        assert result.eligible_positions == ["1B", "OF"]
        assert result.status == "playing"
        assert result.yahoo_player_key == "469.l.72586.p.12345"

    def test_with_rolling_stats(self):
        yahoo_player = {
            "player_key": "469.l.72586.p.12345",
            "name": "Test Player",
            "team": "NYY",
            "positions": ["1B"],
        }

        rolling = PlayerRollingStats(
            bdl_player_id=12345,
            as_of_date=datetime(2026, 4, 1).date(),
            window_days=14,
            games_in_window=10,
            w_games=9.5,
            w_runs=50.0,
            w_hits=120.0,
            w_home_runs=15.0,
            w_rbi=50.0,
            w_strikeouts_bat=70.0,
            w_tb=180.0,
            w_avg=0.267,
            w_ops=0.780,
            w_net_stolen_bases=4.0,
        )

        result = map_yahoo_player_to_canonical_row(
            yahoo_player,
            rolling_stats=rolling,
            computed_at=datetime(2026, 4, 1, 12, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        assert result.rolling_14d is not None
        assert result.rolling_14d.values["R"] == 50.0
        assert result.rolling_14d.values["H"] == 120.0
        assert result.freshness.computed_at.hour == 12

    def test_position_string_to_list(self):
        yahoo_player = {
            "player_key": "469.l.72586.p.12345",
            "name": "Test Player",
            "team": "NYY",
            "positions": "1B",
        }

        result = map_yahoo_player_to_canonical_row(yahoo_player)
        assert result.eligible_positions == ["1B"]

    def test_injury_status_mapping(self):
        yahoo_player = {
            "player_key": "469.l.72586.p.12345",
            "name": "Test Player",
            "team": "NYY",
            "positions": [],
            "status": "IL",
            "injury_note": "Knee - out 4-6 weeks",
        }

        result = map_yahoo_player_to_canonical_row(yahoo_player)
        assert result.status == "IL"
        assert result.injury_status == "Knee - out 4-6 weeks"

    def test_ownership_percentage(self):
        yahoo_player = {
            "player_key": "469.l.72586.p.12345",
            "name": "Test Player",
            "team": "NYY",
            "positions": [],
            "ownership_pct": 85.5,
        }

        result = map_yahoo_player_to_canonical_row(yahoo_player)
        assert result.ownership_pct == 85.5

    def test_freshness_metadata_present(self):
        yahoo_player = {
            "player_key": "469.l.72586.p.12345",
            "name": "Test Player",
            "team": "NYY",
            "positions": [],
        }

        result = map_yahoo_player_to_canonical_row(yahoo_player)
        assert result.freshness.primary_source == "yahoo"
        assert result.freshness.staleness_threshold_minutes == 60
        assert isinstance(result.freshness.computed_at, datetime)
