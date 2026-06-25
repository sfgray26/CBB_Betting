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
        assert "IL15" in _INACTIVE_STATUSES
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


class TestILSlotAccounting:
    """Test IL slot counting and capacity logic."""

    def test_il_slot_positions_includes_il15(self):
        """IL15 should be recognized as a valid IL slot position."""
        from backend.services.waiver_edge_detector import _IL_SLOT_POSITIONS

        assert "IL15" in _IL_SLOT_POSITIONS
        assert "IL" in _IL_SLOT_POSITIONS
        assert "IL10" in _IL_SLOT_POSITIONS
        assert "IL60" in _IL_SLOT_POSITIONS

    def test_il_capacity_info_full_il_slots(self):
        """Test IL capacity info with 3/3 IL slots filled."""
        from backend.services.waiver_edge_detector import il_capacity_info

        roster = [
            {"name": "IL Player 1", "selected_position": "IL"},
            {"name": "IL Player 2", "selected_position": "IL10"},
            {"name": "IL Player 3", "selected_position": "IL60"},
            {"name": "Active Player 1", "selected_position": "C"},
            {"name": "Active Player 2", "selected_position": "1B"},
        ]

        result = il_capacity_info(roster)

        assert result["used"] == 3
        assert result["total"] == 3
        assert result["available"] == 0

    def test_il_capacity_info_with_il15_slot(self):
        """Test that IL15 players are counted in IL slot usage."""
        from backend.services.waiver_edge_detector import il_capacity_info

        roster = [
            {"name": "IL Player 1", "selected_position": "IL"},
            {"name": "IL Player 2", "selected_position": "IL15"},
            {"name": "Active Player 1", "selected_position": "C"},
        ]

        result = il_capacity_info(roster)

        assert result["used"] == 2
        assert result["available"] == 1  # 3 total - 2 used

    def test_il_capacity_info_overcount_edge_case(self):
        """Test edge case: 5 injured players, 3 IL slots, 2 NA status."""
        from backend.services.waiver_edge_detector import il_capacity_info

        # 3 players in IL slots
        # 2 players with NA status (not in IL slots, but marked NA)
        # Total roster has 5 injured players but only 3 IL slots used
        roster = [
            {"name": "IL Murakami", "selected_position": "IL", "status": "IL"},
            {"name": "IL Crochet", "selected_position": "IL10", "status": "IL"},
            {"name": "IL Díaz", "selected_position": "IL60", "status": "IL"},
            {"name": "Soroka 15-day", "selected_position": "IL15", "status": "IL"},
            {"name": "Max Meyer NA", "selected_position": "NA", "status": "NA"},
        ]

        result = il_capacity_info(roster)

        # Should count 4 IL slot positions (IL, IL10, IL60, IL15)
        # NA is separate from IL slots
        assert result["used"] == 4
        assert result["total"] == 3  # Default IL slots
        assert result["available"] == 0  # Over capacity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
