"""
Regression test for CRITICAL 1: Waiver Wire 503 crash caused by boolean status values.

This test ensures that all player schemas (WaiverPlayerOut, LineupPlayerOut,
StartingPitcherOut, DropPlayerOut, RosterPlayerOut) properly coerce boolean
status values to strings, preventing 503 errors when Yahoo returns boolean flags.

Root cause: Yahoo API sometimes returns status=True (boolean) instead of
status="Active" (string) for player status. Without the validator, Pydantic
raises a validation error and the endpoint returns 503.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from backend.schemas import (
    WaiverPlayerOut,
    LineupPlayerOut,
    StartingPitcherOut,
    DropPlayerOut,
    RosterPlayerOut,
)


class TestStatusFieldBooleanHandling:
    """Test that all player schemas handle boolean status values gracefully."""

    def test_waiver_player_out_boolean_status_true(self):
        """WaiverPlayerOut should convert boolean True to 'IL' string."""
        player = WaiverPlayerOut(
            player_id="bdl.12345",
            name="Test Player",
            team="NYY",
            position="1B",
            status=True,  # Boolean should be coerced to "IL"
        )
        assert player.status == "IL", "Boolean True should convert to 'IL'"

    def test_waiver_player_out_boolean_status_false(self):
        """WaiverPlayerOut should convert boolean False to None."""
        player = WaiverPlayerOut(
            player_id="bdl.12345",
            name="Test Player",
            team="NYY",
            position="1B",
            status=False,  # Boolean should be coerced to None
        )
        assert player.status is None, "Boolean False should convert to None"

    def test_waiver_player_out_string_status_passthrough(self):
        """WaiverPlayerOut should pass through string status unchanged."""
        player = WaiverPlayerOut(
            player_id="bdl.12345",
            name="Test Player",
            team="NYY",
            position="1B",
            status="Active",
        )
        assert player.status == "Active", "String status should pass through unchanged"

    def test_waiver_player_out_injury_status_boolean_true(self):
        """WaiverPlayerOut injury_status should convert boolean True to 'IL'."""
        player = WaiverPlayerOut(
            player_id="bdl.12345",
            name="Test Player",
            team="NYY",
            position="1B",
            injury_status=True,
        )
        assert player.injury_status == "IL", "Boolean injury_status=True should convert to 'IL'"

    def test_lineup_player_out_boolean_status_true(self):
        """LineupPlayerOut should convert boolean True to 'IL' string."""
        player = LineupPlayerOut(
            player_id="bdl.12345",
            name="Test Player",
            team="NYY",
            position="1B",
            status=True,
        )
        assert player.status == "IL", "Boolean True should convert to 'IL'"

    def test_lineup_player_out_injury_status_boolean(self):
        """LineupPlayerOut injury_status should handle boolean values."""
        player = LineupPlayerOut(
            player_id="bdl.12345",
            name="Test Player",
            team="NYY",
            position="1B",
            injury_status=True,
        )
        assert player.injury_status == "IL", "Boolean injury_status=True should convert to 'IL'"

    def test_starting_pitcher_out_boolean_status_true(self):
        """StartingPitcherOut should convert boolean True to 'IL' string."""
        pitcher = StartingPitcherOut(
            player_id="bdl.12345",
            name="Test Pitcher",
            team="NYY",
            pitcher_type="SP",
            status=True,
        )
        assert pitcher.status == "IL", "Boolean True should convert to 'IL'"

    def test_starting_pitcher_out_injury_status_boolean(self):
        """StartingPitcherOut injury_status should handle boolean values."""
        pitcher = StartingPitcherOut(
            player_id="bdl.12345",
            name="Test Pitcher",
            team="NYY",
            pitcher_type="SP",
            injury_status=True,
        )
        assert pitcher.injury_status == "IL", "Boolean injury_status=True should convert to 'IL'"

    def test_drop_player_out_boolean_status_true(self):
        """DropPlayerOut should convert boolean True to 'IL' string."""
        player = DropPlayerOut(
            player_id="469.p.12345",
            name="Test Player",
            position="1B",
            positions=["1B", "3B"],
            z_score=1.5,
            cat_scores={"HR": 1.0},
            tier=1,
            adp=50.0,
            percent_owned=80.0,
            status=True,
        )
        assert player.status == "IL", "Boolean True should convert to 'IL'"

    def test_roster_player_out_boolean_status_true(self):
        """RosterPlayerOut should convert boolean True to 'IL' string."""
        player = RosterPlayerOut(
            player_key="469.p.12345",
            name="Test Player",
            team="NYY",
            positions=["1B", "3B"],
            status=True,
        )
        assert player.status == "IL", "Boolean True should convert to 'IL'"

    def test_waiver_player_out_from_dict_with_boolean(self):
        """Test parsing from raw dict with boolean status (simulating API response)."""
        raw_data = {
            "player_id": "bdl.12345",
            "name": "Test Player",
            "team": "NYY",
            "position": "1B",
            "status": True,  # Boolean from Yahoo API
            "injury_status": False,
        }
        player = WaiverPlayerOut(**raw_data)
        assert player.status == "IL"
        assert player.injury_status is None

    def test_lineup_player_out_from_dict_with_boolean(self):
        """Test LineupPlayerOut parsing from raw dict with boolean status."""
        raw_data = {
            "player_id": "bdl.12345",
            "name": "Test Player",
            "team": "NYY",
            "position": "1B",
            "status": True,
            "injury_status": True,
        }
        player = LineupPlayerOut(**raw_data)
        assert player.status == "IL"
        assert player.injury_status == "IL"

    def test_starting_pitcher_out_from_dict_with_boolean(self):
        """Test StartingPitcherOut parsing from raw dict with boolean status."""
        raw_data = {
            "player_id": "bdl.12345",
            "name": "Test Pitcher",
            "team": "NYY",
            "pitcher_type": "SP",
            "status": True,
            "injury_status": False,
        }
        pitcher = StartingPitcherOut(**raw_data)
        assert pitcher.status == "IL"
        assert pitcher.injury_status is None

    def test_drop_player_out_from_dict_with_boolean(self):
        """Test DropPlayerOut parsing from raw dict with boolean status."""
        raw_data = {
            "player_id": "469.p.12345",
            "name": "Test Player",
            "position": "1B",
            "positions": ["1B", "3B"],
            "z_score": 1.5,
            "cat_scores": {"HR": 1.0},
            "tier": 1,
            "adp": 50.0,
            "percent_owned": 80.0,
            "status": True,
        }
        player = DropPlayerOut(**raw_data)
        assert player.status == "IL"


class TestStatusFieldNoneHandling:
    """Test that None values are handled correctly for optional status fields."""

    def test_waiver_player_out_none_status(self):
        """WaiverPlayerOut should accept None for optional status."""
        player = WaiverPlayerOut(
            player_id="bdl.12345",
            name="Test Player",
            team="NYY",
            position="1B",
            status=None,
        )
        assert player.status is None

    def test_lineup_player_out_default_status(self):
        """LineupPlayerOut should use 'UNKNOWN' as default status."""
        player = LineupPlayerOut(
            player_id="bdl.12345",
            name="Test Player",
            team="NYY",
            position="1B",
        )
        assert player.status == "UNKNOWN"

    def test_starting_pitcher_out_default_status(self):
        """StartingPitcherOut should use 'UNKNOWN' as default status."""
        pitcher = StartingPitcherOut(
            player_id="bdl.12345",
            name="Test Pitcher",
            team="NYY",
            pitcher_type="SP",
        )
        assert pitcher.status == "UNKNOWN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
