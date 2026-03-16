"""
Tests for fatigue service.
"""

import pytest
from datetime import datetime, timedelta

from backend.services.fatigue import (
    calculate_fatigue,
    calculate_game_fatigue,
    get_fatigue_margin_adjustment,
    get_fatigue_service,
    FatigueAdjustment,
    GameRecord,
    _travel_penalty,
    _timezone_penalty,
    _altitude_penalty,
    _cumulative_load_penalty,
    ARENA_DATA,
)


class TestTravelPenalty:
    """Test travel distance penalty calculations."""
    
    def test_no_penalty_short_trip(self):
        assert _travel_penalty(50) == 0.0
        assert _travel_penalty(100) == 0.0
    
    def test_mild_penalty_regional(self):
        penalty = _travel_penalty(250)
        assert 0.1 < penalty < 0.2
    
    def test_moderate_penalty_cross_state(self):
        penalty = _travel_penalty(750)
        assert 0.3 < penalty < 0.4
    
    def test_significant_penalty_cross_country(self):
        penalty = _travel_penalty(2000)
        assert 0.8 < penalty <= 1.0
    
    def test_max_penalty_very_long(self):
        penalty = _travel_penalty(3000)
        assert penalty == 1.0  # Capped


class TestTimezonePenalty:
    """Test timezone shift penalty calculations."""
    
    def test_no_shift_no_penalty(self):
        assert _timezone_penalty(0, 1) == 0.0
        assert _timezone_penalty(1, 1) == 0.0
    
    def test_single_zone_no_penalty(self):
        assert _timezone_penalty(1, 0) == 0.0  # Already acclimated
    
    def test_eastward_worse_than_westward(self):
        # Eastward (negative shift) is worse
        east = _timezone_penalty(-3, 0)  # LA to NY
        west = _timezone_penalty(3, 0)   # NY to LA
        assert east > west
    
    def test_acclimated_after_two_days(self):
        penalty = _timezone_penalty(-3, 2)  # 3 zones, 2 days rest
        assert penalty == 0.0
    
    def test_capped_at_1_5(self):
        penalty = _timezone_penalty(-10, 0)  # Extreme shift
        assert penalty <= 1.5


class TestAltitudePenalty:
    """Test altitude change penalty calculations."""
    
    def test_sea_level_no_effect(self):
        assert _altitude_penalty(0, False, 1) == 0.0
    
    def test_home_team_altitude_advantage(self):
        # Home team at altitude gets bonus (negative penalty)
        penalty = _altitude_penalty(5000, True, 1)
        assert penalty < 0  # Negative = advantage
    
    def test_visitor_at_altitude_suffers(self):
        penalty = _altitude_penalty(5000, False, 0)  # No rest
        assert penalty > 0
    
    def test_partial_acclimation(self):
        fresh = _altitude_penalty(5000, False, 0)
        acclimated = _altitude_penalty(5000, False, 2)
        assert acclimated < fresh


class TestCumulativeLoadPenalty:
    """Test cumulative game load penalty calculations."""
    
    def test_light_schedule_no_penalty(self):
        assert _cumulative_load_penalty(2, 4) == 0.0
    
    def test_heavy_7_day_schedule(self):
        penalty = _cumulative_load_penalty(4, 6)
        assert penalty > 0
    
    def test_extreme_schedule(self):
        penalty = _cumulative_load_penalty(5, 8)
        assert penalty > 0.5


class TestCalculateFatigue:
    """Test main fatigue calculation function."""
    
    def test_fully_rested_team(self):
        adj = calculate_fatigue(
            team="Duke",
            last_game_date=datetime.now() - timedelta(days=5),
            current_game_date=datetime.now(),
            is_home=True,
        )
        assert adj.rest_days == 5
        assert adj.rest_penalty == 0.0
        assert adj.total_penalty == 0.0
    
    def test_back_to_back(self):
        adj = calculate_fatigue(
            team="Duke",
            last_game_date=datetime.now() - timedelta(hours=20),
            current_game_date=datetime.now(),
            is_home=False,
            travel_distance_miles=350,
        )
        assert adj.rest_days == 0
        assert adj.rest_penalty > 1.0  # Significant B2B penalty
        assert adj.travel_penalty > 0
        assert "BACK-TO-BACK" in " ".join(adj.notes)
    
    def test_one_day_rest(self):
        adj = calculate_fatigue(
            team="Duke",
            last_game_date=datetime.now() - timedelta(days=1),
            current_game_date=datetime.now(),
            is_home=True,
        )
        assert adj.rest_days == 1
        assert 0.5 < adj.rest_penalty < 1.0
    
    def test_cross_country_travel(self):
        adj = calculate_fatigue(
            team="Duke",
            last_game_date=datetime.now() - timedelta(days=2),
            current_game_date=datetime.now(),
            is_home=False,
            travel_distance_miles=2500,
        )
        assert adj.travel_penalty > 0.5
    
    def test_altitude_home_advantage(self):
        adj = calculate_fatigue(
            team="New Mexico",
            last_game_date=datetime.now() - timedelta(days=3),
            current_game_date=datetime.now(),
            is_home=True,
            home_arena_altitude_ft=5312,
            game_arena_altitude_ft=5312,
        )
        assert adj.altitude_penalty < 0  # Negative = advantage
        assert "Altitude advantage" in " ".join(adj.notes)
    
    def test_cumulative_load_tracked(self):
        recent = [
            GameRecord(datetime.now() - timedelta(days=i), True, f"Opp{i}")
            for i in range(1, 8)  # 7 games in 7 days
        ]
        adj = calculate_fatigue(
            team="Duke",
            last_game_date=datetime.now() - timedelta(days=1),
            current_game_date=datetime.now(),
            is_home=True,
            recent_games=recent,
        )
        assert adj.cumulative_load_penalty > 0


class TestGameFatigue:
    """Test game-level fatigue calculations."""
    
    def test_home_team_advantage(self):
        home_adj, away_adj = calculate_game_fatigue(
            home_team="Duke",
            away_team="UNC",
            game_date=datetime.now(),
            home_last_game=datetime.now() - timedelta(days=3),
            away_last_game=datetime.now() - timedelta(hours=20),  # B2B
            travel_distance_miles=250,  # Regional trip for UNC
        )

        # Home team should be less fatigued
        assert home_adj.rest_penalty < away_adj.rest_penalty
        assert home_adj.travel_penalty == 0.0
        assert away_adj.travel_penalty > 0
    
    def test_margin_adjustment_calculation(self):
        home_adj, away_adj = calculate_game_fatigue(
            home_team="Duke",
            away_team="UNC",
            game_date=datetime.now(),
            home_last_game=datetime.now() - timedelta(days=3),
            away_last_game=datetime.now() - timedelta(hours=20),
        )
        
        margin_adj, meta = get_fatigue_margin_adjustment(home_adj, away_adj)
        
        # Away team more tired = positive margin adjustment for home
        assert margin_adj > 0
        assert "home_rest_days" in meta
        assert "away_rest_days" in meta


class TestArenaData:
    """Test arena database lookups."""
    
    def test_known_altitude_venue(self):
        data = ARENA_DATA.get("New Mexico")
        assert data["altitude_ft"] > 5000
    
    def test_known_timezone(self):
        data = ARENA_DATA.get("Gonzaga")
        assert data["timezone"] == -3  # Pacific


class TestFatigueService:
    """Test FatigueService class."""
    
    def test_singleton_pattern(self):
        svc1 = get_fatigue_service()
        svc2 = get_fatigue_service()
        assert svc1 is svc2
    
    def test_service_basic_calculation(self):
        svc = get_fatigue_service()
        home_adj, away_adj = svc.get_game_adjustments(
            home_team="Duke",
            away_team="UNC",
            game_date=datetime.now(),
        )
        
        assert home_adj.team == "Duke"
        assert away_adj.team == "UNC"
        
        margin_adj, meta = svc.get_margin_adjustment(home_adj, away_adj)
        assert isinstance(margin_adj, float)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_last_game_opener(self):
        adj = calculate_fatigue(
            team="Duke",
            last_game_date=None,
            current_game_date=datetime.now(),
            is_home=True,
        )
        assert adj.rest_days == 7  # Opening night assumption
        assert "Opening night" in " ".join(adj.notes)
    
    def test_invalid_date_sequence(self):
        adj = calculate_fatigue(
            team="Duke",
            last_game_date=datetime.now() + timedelta(days=1),  # Future date
            current_game_date=datetime.now(),
            is_home=True,
        )
        assert adj.rest_days == 7  # Fallback to rested
        assert "Invalid date" in " ".join(adj.notes)
    
    def test_negative_margin_adjustment(self):
        # Home team more tired than away
        home_adj = FatigueAdjustment(team="Home", total_penalty=2.0)
        away_adj = FatigueAdjustment(team="Away", total_penalty=0.0)
        
        margin_adj, meta = get_fatigue_margin_adjustment(home_adj, away_adj)
        assert margin_adj < 0  # Negative = away team benefits
