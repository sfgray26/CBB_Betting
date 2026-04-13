"""
Tests for MLB stats data validation (Task 20).

Tests the _validate_mlb_stats function that prevents bad data from being
inserted during ingestion.
"""

import pytest
from backend.services.daily_ingestion import _validate_mlb_stats
from backend.data_contracts.mlb_player_stats import MLBPlayerStats
from backend.data_contracts.mlb_player import MLBPlayer


def create_mock_player(player_id: int) -> MLBPlayer:
    """Create a valid MLBPlayer instance for testing."""
    return MLBPlayer(
        id=player_id,
        first_name="Test",
        last_name="Player",
        full_name="Test Player",
        position="P",
        active=True
    )


class TestMLBStatsValidation:
    """Test suite for _validate_mlb_stats function."""

    def test_valid_stat_row_passes_validation(self):
        """Test that a valid stat row passes validation."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            era=3.50,
            avg=0.295,
            ip="6.2"
        )
        assert _validate_mlb_stats(stat) is True

    def test_era_rejects_negative_value(self):
        """Test that negative ERA is rejected."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            era=-1.5
        )
        assert _validate_mlb_stats(stat) is False

    def test_era_rejects_excessive_value(self):
        """Test that ERA > 100 is rejected."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            era=150.0
        )
        assert _validate_mlb_stats(stat) is False

    def test_era_accepts_zero(self):
        """Test that ERA = 0 is accepted (perfect game possible)."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            era=0.0
        )
        assert _validate_mlb_stats(stat) is True

    def test_era_accepts_boundary_value(self):
        """Test that ERA = 100 is accepted (upper boundary)."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            era=100.0
        )
        assert _validate_mlb_stats(stat) is True

    def test_avg_rejects_negative_value(self):
        """Test that negative AVG is rejected."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            avg=-0.1
        )
        assert _validate_mlb_stats(stat) is False

    def test_avg_rejects_excessive_value(self):
        """Test that AVG > 1.0 is rejected."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            avg=1.5
        )
        assert _validate_mlb_stats(stat) is False

    def test_avg_accepts_zero(self):
        """Test that AVG = 0 is accepted (hitless game possible)."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            avg=0.0
        )
        assert _validate_mlb_stats(stat) is True

    def test_avg_accepts_boundary_value(self):
        """Test that AVG = 1.0 is accepted (upper boundary)."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            avg=1.0
        )
        assert _validate_mlb_stats(stat) is True

    def test_ip_rejects_invalid_format(self):
        """Test that invalid IP format is rejected."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            ip="invalid"
        )
        assert _validate_mlb_stats(stat) is False

    def test_ip_accepts_standard_format(self):
        """Test that standard IP format (e.g., '6.2') is accepted."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            ip="6.2"
        )
        assert _validate_mlb_stats(stat) is True

    def test_ip_accepts_integer(self):
        """Test that integer IP is accepted."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            ip=7
        )
        assert _validate_mlb_stats(stat) is True

    def test_ip_accepts_float(self):
        """Test that float IP is accepted."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            ip=7.0
        )
        assert _validate_mlb_stats(stat) is True

    def test_ip_accepts_partial_inning(self):
        """Test that partial inning (e.g., '0.2') is accepted."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            ip="0.2"
        )
        assert _validate_mlb_stats(stat) is True

    def test_none_values_pass_validation(self):
        """Test that None values pass validation (optional fields)."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890
        )
        assert _validate_mlb_stats(stat) is True

    def test_multiple_validation_errors_all_logged(self):
        """Test that multiple validation errors are all detected."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            era=150.0,  # Invalid
            avg=1.5,    # Invalid
            ip="invalid"  # Invalid
        )
        assert _validate_mlb_stats(stat) is False

    def test_realistic_pitching_line_passes(self):
        """Test a realistic pitching line passes validation."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            era=2.85,
            ip="7.0",
            h_allowed=5,
            r_allowed=2,
            er=2,
            bb_allowed=1,
            k=8
        )
        assert _validate_mlb_stats(stat) is True

    def test_realistic_batting_line_passes(self):
        """Test a realistic batting line passes validation."""
        stat = MLBPlayerStats(
            player=create_mock_player(12345),
            game_id=67890,
            avg=0.333,
            ab=3,
            h=1,
            double=1,
            hr=0,
            rbi=2
        )
        assert _validate_mlb_stats(stat) is True


class TestMLBStatsOBPValidation:
    """Tests for OBP range validation added per PR #91 recommendations."""

    def test_obp_rejects_negative_value(self):
        """Negative OBP is impossible and must be rejected."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, obp=-0.01)
        assert _validate_mlb_stats(stat) is False

    def test_obp_rejects_value_above_one(self):
        """OBP > 1.0 is impossible and must be rejected."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, obp=1.001)
        assert _validate_mlb_stats(stat) is False

    def test_obp_accepts_zero(self):
        """OBP = 0.0 (no times on base) is valid."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, obp=0.0)
        assert _validate_mlb_stats(stat) is True

    def test_obp_accepts_one(self):
        """OBP = 1.0 (reached base every PA via walk/HBP) is valid."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, obp=1.0)
        assert _validate_mlb_stats(stat) is True

    def test_obp_accepts_typical_value(self):
        """Typical OBP around 0.350 must pass."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, obp=0.350)
        assert _validate_mlb_stats(stat) is True

    def test_obp_none_passes(self):
        """None OBP (hitter not in box, or pitcher row) must not be rejected."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, obp=None)
        assert _validate_mlb_stats(stat) is True


class TestMLBStatsWHIPValidation:
    """Tests for WHIP range validation added per PR #91 recommendations."""

    def test_whip_rejects_negative_value(self):
        """Negative WHIP is physically impossible and must be rejected."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, whip=-0.5)
        assert _validate_mlb_stats(stat) is False

    def test_whip_accepts_zero(self):
        """WHIP = 0.0 (perfect game with no baserunners) is valid."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, whip=0.0)
        assert _validate_mlb_stats(stat) is True

    def test_whip_accepts_typical_value(self):
        """Typical WHIP around 1.20 must pass."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, whip=1.20)
        assert _validate_mlb_stats(stat) is True

    def test_whip_accepts_high_but_possible_value(self):
        """Very high WHIP (e.g. 8.0 for a rough outing) is still valid."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, whip=8.0)
        assert _validate_mlb_stats(stat) is True

    def test_whip_none_passes(self):
        """None WHIP (batter row) must not be rejected."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, whip=None)
        assert _validate_mlb_stats(stat) is True


class TestMLBStatsNegativeCountingStats:
    """Tests for negative counting-stat rejection added per PR #91 recommendations."""

    def test_negative_ab_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, ab=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_hits_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, h=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_runs_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, r=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_hr_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, hr=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_rbi_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, rbi=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_bb_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, bb=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_so_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, so=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_sb_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, sb=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_k_pitcher_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, k=-1)
        assert _validate_mlb_stats(stat) is False

    def test_negative_h_allowed_rejected(self):
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, h_allowed=-1)
        assert _validate_mlb_stats(stat) is False

    def test_zero_counting_stats_pass(self):
        """Zero values (e.g., 0 AB for a pinch runner) must be accepted."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1,
                              ab=0, h=0, r=0, hr=0, rbi=0, bb=0, so=0)
        assert _validate_mlb_stats(stat) is True


class TestMLBStatsLogicalConsistency:
    """Tests for logical consistency checks added per PR #91 recommendations."""

    def test_hits_exceeding_ab_rejected(self):
        """More hits than at-bats is physically impossible."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, ab=3, h=4)
        assert _validate_mlb_stats(stat) is False

    def test_hits_equal_ab_passes(self):
        """Hits == AB (hit every at-bat) is valid."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, ab=3, h=3)
        assert _validate_mlb_stats(stat) is True

    def test_hits_less_than_ab_passes(self):
        """Normal case: fewer hits than at-bats."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, ab=4, h=1)
        assert _validate_mlb_stats(stat) is True

    def test_hits_none_ab_set_passes(self):
        """Hits=None with AB set should not raise a consistency error."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, ab=4, h=None)
        assert _validate_mlb_stats(stat) is True

    def test_ab_none_hits_set_passes(self):
        """AB=None with hits set should not raise a consistency error."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, ab=None, h=2)
        assert _validate_mlb_stats(stat) is True

    def test_zero_ab_zero_hits_passes(self):
        """0 AB and 0 hits (pinch runner who never batted) is valid."""
        stat = MLBPlayerStats(player=create_mock_player(1), game_id=1, ab=0, h=0)
        assert _validate_mlb_stats(stat) is True
