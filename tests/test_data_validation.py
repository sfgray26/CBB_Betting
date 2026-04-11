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
