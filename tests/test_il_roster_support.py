"""
Tests for IL Roster Support (EMAC-080 P0)

Validates that:
1. get_roster() extracts selected_position from Yahoo API
2. IL players are excluded from waiver drop candidates
3. Roster endpoint returns selected_position field
"""
import pytest
from unittest.mock import MagicMock, patch

from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
from backend.services.waiver_edge_detector import WaiverEdgeDetector, _INACTIVE_STATUSES


class TestSelectedPositionExtraction:
    """Test extraction of selected_position from Yahoo roster data."""

    def test_extract_selected_position_from_il_player(self):
        """Yahoo returns IL status in selected_position field."""
        # Yahoo returns player as list: [metadata, selected_position]
        player_data = [
            [
                {"player_key": "469.p.12345", "full_name": "Test Player"},
            ],
            {"selected_position": {"position": "IL"}}
        ]
        
        result = YahooFantasyClient._extract_selected_position(player_data)
        assert result == "IL"

    def test_extract_selected_position_from_bench_player(self):
        """Bench players have selected_position = 'BN'."""
        player_data = [
            [{"player_key": "469.p.12345"}],
            {"selected_position": {"position": "BN"}}
        ]
        
        result = YahooFantasyClient._extract_selected_position(player_data)
        assert result == "BN"

    def test_extract_selected_position_from_active_player(self):
        """Active players have position like 'C', '1B', 'OF'."""
        player_data = [
            [{"player_key": "469.p.12345"}],
            {"selected_position": {"position": "C"}}
        ]
        
        result = YahooFantasyClient._extract_selected_position(player_data)
        assert result == "C"

    def test_extract_selected_position_missing(self):
        """Returns None if selected_position not present."""
        player_data = [{"player_key": "469.p.12345"}]
        
        result = YahooFantasyClient._extract_selected_position(player_data)
        assert result is None


class TestInactiveStatuses:
    """Test IL status constants."""

    def test_inactive_statuses_contains_il_variants(self):
        """All IL statuses should be in the frozenset."""
        assert "IL" in _INACTIVE_STATUSES
        assert "IL10" in _INACTIVE_STATUSES
        assert "IL60" in _INACTIVE_STATUSES
        assert "NA" in _INACTIVE_STATUSES
        assert "OUT" in _INACTIVE_STATUSES


class TestWaiverEdgeDetectorExcludesIL:
    """Test that waiver logic excludes IL players."""

    def test_count_position_coverage_excludes_il(self):
        """IL players should not count as position coverage."""
        detector = WaiverEdgeDetector()
        
        roster = [
            {"name": "Active 1B", "positions": ["1B"], "selected_position": "1B", "is_undroppable": False},
            {"name": "IL 1B", "positions": ["1B"], "selected_position": "IL", "is_undroppable": False},
            {"name": "IL10 1B", "positions": ["1B"], "selected_position": "IL10", "is_undroppable": False},
        ]
        
        # Should only count the active 1B
        coverage = detector._count_position_coverage(roster, ["1B"])
        assert coverage == 1

    def test_weakest_droppable_excludes_il(self):
        """IL players should not be suggested as drops."""
        detector = WaiverEdgeDetector()
        
        roster = [
            {"name": "Active Weak", "positions": ["1B"], "selected_position": "1B", 
             "is_undroppable": False, "cat_scores": {"r": 0.1}},
            {"name": "IL Player", "positions": ["1B"], "selected_position": "IL",
             "is_undroppable": False, "cat_scores": {"r": -2.0}},  # Lower z-score
        ]
        
        # Should pick the active weak player, not the IL player
        result = detector._weakest_droppable(roster)
        assert result["name"] == "Active Weak"

    def test_weakest_droppable_returns_none_when_all_il(self):
        """If all players are IL, should return None (not suggest drops)."""
        detector = WaiverEdgeDetector()
        
        roster = [
            {"name": "IL1", "positions": ["1B"], "selected_position": "IL", 
             "is_undroppable": False, "cat_scores": {"r": -1.0}},
            {"name": "IL2", "positions": ["1B"], "selected_position": "IL60",
             "is_undroppable": False, "cat_scores": {"r": -2.0}},
        ]
        
        result = detector._weakest_droppable(roster)
        assert result is None

    def test_weakest_droppable_at_protects_single_coverage(self):
        """When only one non-IL player covers a position, protect them."""
        detector = WaiverEdgeDetector()
        
        roster = [
            {"name": "Only Active C", "positions": ["C"], "selected_position": "C",
             "is_undroppable": False, "cat_scores": {"r": 0.5}},
            {"name": "IL C", "positions": ["C"], "selected_position": "IL",
             "is_undroppable": False, "cat_scores": {"r": -1.0}},
        ]
        
        # Should return None (protect the only active catcher)
        result = detector._weakest_droppable_at(roster, ["C"])
        assert result is None


class TestRosterEndpoint:
    """Test the roster endpoint includes selected_position."""

    def test_roster_player_out_schema_includes_selected_position(self):
        """RosterPlayerOut schema should accept selected_position field."""
        from backend.schemas import RosterPlayerOut
        
        # Should not raise validation error
        player = RosterPlayerOut(
            player_key="469.p.12345",
            name="Test Player",
            positions=["C"],
            selected_position="IL",
            status="IL10",
        )
        
        assert player.selected_position == "IL"
        assert player.name == "Test Player"
        
    def test_roster_player_out_without_selected_position(self):
        """RosterPlayerOut should work without selected_position (backward compat)."""
        from backend.schemas import RosterPlayerOut
        
        # Should work with optional field omitted
        player = RosterPlayerOut(
            player_key="469.p.12345",
            name="Test Player",
            positions=["C"],
        )
        
        assert player.selected_position is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
