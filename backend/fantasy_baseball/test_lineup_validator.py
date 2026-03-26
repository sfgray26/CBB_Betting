"""
Tests for game-aware lineup validator.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backend.fantasy_baseball.lineup_validator import (
    ScheduleFetcher,
    LineupValidator,
    OptimizedSlot,
    GameStatus,
    PlayerGameInfo,
    format_lineup_report,
)


class TestScheduleFetcher:
    """Test schedule fetching functionality."""
    
    @pytest.fixture
    def fetcher(self):
        return ScheduleFetcher(cache_ttl_minutes=5)
    
    def test_cache_miss_triggers_fetch(self, fetcher):
        """When cache is empty, should fetch new data."""
        mock_schedule = {"NYY": {"game_id": 1, "game_time": datetime.now()}}
        
        with patch.object(fetcher, '_fetch_mlb_schedule', return_value=mock_schedule):
            result = fetcher.get_todays_schedule()
        
        assert result == mock_schedule
        assert fetcher._cache is not None
    
    def test_cache_hit_returns_cached_data(self, fetcher):
        """When cache is fresh, should return cached data."""
        cached_schedule = {"LAD": {"game_id": 2}}
        fetcher._cache = (datetime.now(), cached_schedule)
        
        result = fetcher.get_todays_schedule()
        
        assert result == cached_schedule
    
    def test_cache_expiry_triggers_refetch(self, fetcher):
        """When cache is stale, should fetch new data."""
        old_schedule = {"OLD": {"game_id": 99}}
        new_schedule = {"NYY": {"game_id": 1}}
        
        # Set expired cache
        fetcher._cache = (datetime.now() - timedelta(hours=1), old_schedule)
        
        with patch.object(fetcher, '_fetch_mlb_schedule', return_value=new_schedule):
            result = fetcher.get_todays_schedule()
        
        assert result == new_schedule
        assert "OLD" not in result


class TestLineupValidator:
    """Test lineup validation logic."""
    
    @pytest.fixture
    def validator(self):
        mock_schedule = Mock()
        mock_schedule.get_todays_schedule.return_value = {
            "NYY": {
                "game_id": 1,
                "game_time": datetime.now() + timedelta(hours=2),
                "status": "Scheduled",
                "is_home": True,
                "opponent": "BOS"
            },
            "BOS": {
                "game_id": 1,
                "game_time": datetime.now() + timedelta(hours=2),
                "status": "Scheduled",
                "is_home": False,
                "opponent": "NYY"
            },
            "LAD": {
                "game_id": 2,
                "game_time": datetime.now() + timedelta(hours=4),
                "status": "Scheduled",
                "is_home": True,
                "opponent": "SF"
            }
        }
        return LineupValidator(schedule_fetcher=mock_schedule)
    
    def test_valid_lineup_all_players_have_games(self, validator):
        """Lineup with all players having games should be valid."""
        slots = [
            OptimizedSlot("s1", "2B", "p1", "Player 1"),
            OptimizedSlot("s2", "3B", "p2", "Player 2"),
        ]
        
        roster = [
            {"player_id": "p1", "name": "Player 1", "team": "NYY", "positions": ["2B"]},
            {"player_id": "p2", "name": "Player 2", "team": "LAD", "positions": ["3B"]},
        ]
        
        result = validator.validate_lineup(slots, roster)
        
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.invalid_players) == 0
    
    def test_invalid_player_no_game_detected(self, validator):
        """Player with no game should be flagged."""
        slots = [
            OptimizedSlot("s1", "1B", "p1", "Has Game"),
            OptimizedSlot("s2", "2B", "p2", "No Game"),
        ]
        
        roster = [
            {"player_id": "p1", "name": "Has Game", "team": "NYY", "positions": ["1B"]},
            {"player_id": "p2", "name": "No Game", "team": "DET", "positions": ["2B"]},
        ]
        
        result = validator.validate_lineup(slots, roster, strict=False)
        
        assert len(result.invalid_players) == 1
        assert result.invalid_players[0].player_name == "No Game"
        assert result.invalid_players[0].status == GameStatus.NO_GAME
    
    def test_strict_mode_fails_on_invalid(self, validator):
        """Strict mode should fail if any player has no game."""
        slots = [
            OptimizedSlot("s1", "1B", "p1", "Has Game"),
            OptimizedSlot("s2", "2B", "p2", "No Game"),
        ]
        
        roster = [
            {"player_id": "p1", "name": "Has Game", "team": "NYY", "positions": ["1B"]},
            {"player_id": "p2", "name": "No Game", "team": "DET", "positions": ["2B"]},
        ]
        
        result = validator.validate_lineup(slots, roster, strict=True)
        
        assert result.valid is False
    
    def test_auto_correct_replaces_invalid_players(self, validator):
        """Auto-correct should swap players with no games."""
        slots = [
            OptimizedSlot("s1", "2B", "p1", "Invalid Starter"),
        ]
        
        roster = [
            {"player_id": "p1", "name": "Invalid Starter", "team": "DET", "positions": ["2B"]},
            {"player_id": "p2", "name": "Bench Player", "team": "NYY", "positions": ["2B"]},
        ]
        
        result = validator.auto_correct_lineup(slots, roster)
        
        # Should replace p1 with p2
        assert result.assignments["s1"] == "p2"
        assert len(result.changes_made) == 1
        assert "Invalid Starter" in result.changes_made[0]
        assert "Bench Player" in result.changes_made[0]
    
    def test_position_eligibility_checked(self, validator):
        """Replacement must be position-eligible."""
        slots = [
            OptimizedSlot("s1", "C", "p1", "Invalid Catcher"),
        ]
        
        roster = [
            {"player_id": "p1", "name": "Invalid Catcher", "team": "DET", "positions": ["C"]},
            {"player_id": "p2", "name": "Not a Catcher", "team": "NYY", "positions": ["1B"]},
        ]
        
        result = validator.auto_correct_lineup(slots, roster)
        
        # p2 is not eligible at Catcher, so no replacement
        assert "s1" not in result.assignments or result.assignments.get("s1") != "p2"


class TestYahooToMlbMapping:
    """Test team abbreviation mapping."""
    
    def test_direct_mappings(self):
        """Common teams should map directly."""
        from backend.fantasy_baseball.lineup_validator import LineupValidator
        
        assert LineupValidator.YAHOO_TO_MLB["NYY"] == "NYY"
        assert LineupValidator.YAHOO_TO_MLB["LAD"] == "LAD"
        assert LineupValidator.YAHOO_TO_MLB["SFG"] == "SF"
        assert LineupValidator.YAHOO_TO_MLB["TBR"] == "TB"
    
    def test_sf_giants_mapping(self):
        """San Francisco Giants special case."""
        from backend.fantasy_baseball.lineup_validator import LineupValidator
        
        # Yahoo uses SFG, MLB API uses SF
        assert LineupValidator.YAHOO_TO_MLB["SFG"] == "SF"


class TestFormatLineupReport:
    """Test report formatting."""
    
    def test_report_with_changes(self):
        """Report should show changes made."""
        from backend.fantasy_baseball.lineup_validator import LineupSubmission, LineupValidation
        
        submission = LineupSubmission(
            assignments={"s1": "p2"},
            bench_assignments={"p1": "p1"},
            changes_made=["2B: Old Player → New Player (vs BOS, 19:05)"],
            validation=LineupValidation(valid=True)
        )
        
        report = format_lineup_report(submission)
        
        assert "AUTOMATIC CORRECTIONS MADE" in report
        assert "Old Player" in report
        assert "New Player" in report
    
    def test_report_with_warnings(self):
        """Report should show warnings."""
        from backend.fantasy_baseball.lineup_validator import LineupSubmission, LineupValidation
        
        validation = LineupValidation(
            valid=True,
            warnings=["Player A: Game already started"]
        )
        
        submission = LineupSubmission(
            assignments={},
            bench_assignments={},
            changes_made=[],
            validation=validation
        )
        
        report = format_lineup_report(submission)
        
        assert "WARNINGS" in report
        assert "Game already started" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
