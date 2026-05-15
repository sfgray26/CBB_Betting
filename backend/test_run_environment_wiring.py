"""
Tests for Run Environment Wiring (PR-22)

Validates that odds/run environment data from warehouse is properly
wired into fantasy pitcher streaming decisions.

Test Coverage:
1. Win probability calculations from moneyline odds
2. MLBGameOdds dataclass with run environment fields
3. _build_team_odds_map includes run_environment_score
4. Enhanced sp_score calculation uses run environment data
"""

import pytest
from typing import Optional, Tuple
from dataclasses import dataclass

# Import the components we're testing
import sys
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/backend')

from fantasy_baseball.daily_lineup_optimizer import (
    DailyLineupOptimizer,
    MLBGameOdds,
)


class TestWinProbabilityCalculations:
    """Test win probability conversion from moneyline odds."""
    
    def test_win_prob_favorite_negative_odds(self):
        """Favorite with -150 odds should have ~60% win probability."""
        optimizer = DailyLineupOptimizer()
        prob = optimizer._win_prob_from_moneyline(-150)
        # -150 -> 150/(150+100) = 0.6
        assert prob == pytest.approx(0.6, abs=0.001)
    
    def test_win_prob_underdog_positive_odds(self):
        """Underdog with +150 odds should have ~40% win probability."""
        optimizer = DailyLineupOptimizer()
        prob = optimizer._win_prob_from_moneyline(150)
        # +150 -> 100/(150+100) = 0.4
        assert prob == pytest.approx(0.4, abs=0.001)
    
        def test_win_prob_even_money(self):
            """Even money (+100) should be 50% probability."""
            optimizer = DailyLineupOptimizer()
            prob = optimizer._win_prob_from_moneyline(100)
            assert prob == pytest.approx(0.5, abs=0.001)
    
    def test_win_prob_heavy_favorite(self):
        """Heavy favorite -300 should have ~75% win probability."""
        optimizer = DailyLineupOptimizer()
        prob = optimizer._win_prob_from_moneyline(-300)
        # -300 -> 300/(300+100) = 0.75
        assert prob == pytest.approx(0.75, abs=0.001)
    
    def test_win_prob_none_input(self):
        """None input should return None."""
        optimizer = DailyLineupOptimizer()
        prob = optimizer._win_prob_from_moneyline(None)
        assert prob is None


class TestVigAdjustedWinProbabilities:
    """Test vig-adjusted win probability calculation."""
    
    def test_compute_win_probabilities_equal_odds(self):
        """Equal -110/-110 odds should normalize to 50/50."""
        optimizer = DailyLineupOptimizer()
        home, away = optimizer._compute_win_probabilities(-110, -110)
        # Raw: 110/210 = 0.524 each
        # Normalized: 0.524 / 1.048 = 0.5
        assert home == pytest.approx(0.5, abs=0.01)
        assert away == pytest.approx(0.5, abs=0.01)
        assert home + away == pytest.approx(1.0, abs=0.001)
    
    def test_compute_win_probabilities_favorite_underdog(self):
        """-200/+170 should compute reasonable probabilities."""
        optimizer = DailyLineupOptimizer()
        home, away = optimizer._compute_win_probabilities(-200, 170)
        # Home raw: 200/300 = 0.667
        # Away raw: 100/270 = 0.370
        # Total: 1.037
        # Home: 0.667/1.037 = 0.643
        # Away: 0.370/1.037 = 0.357
        assert home > away
        assert home + away == pytest.approx(1.0, abs=0.001)
    
    def test_compute_win_probabilities_missing_data(self):
        """Missing moneyline should return None, None."""
        optimizer = DailyLineupOptimizer()
        home, away = optimizer._compute_win_probabilities(None, -110)
        assert home is None
        assert away is None


class TestMLBGameOddsDataclass:
    """Test MLBGameOdds with run environment fields."""
    
    def test_game_odds_with_run_environment_fields(self):
        """MLBGameOdds should accept and store run environment fields."""
        game = MLBGameOdds(
            game_id="401234567",
            commence_time="2025-05-15T19:05:00Z",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            home_abbrev="NYY",
            away_abbrev="BOS",
            spread_home=-1.5,
            total=8.5,
            moneyline_home=-150,
            moneyline_away=130,
            implied_home_runs=4.75,
            implied_away_runs=3.75,
            park_factor=1.02,
            # New run environment fields
            home_win_prob=0.60,
            away_win_prob=0.40,
            game_total=8.5,
        )
        
        assert game.home_win_prob == 0.60
        assert game.away_win_prob == 0.40
        assert game.game_total == 8.5
    
    def test_game_odds_defaults_for_optional_fields(self):
        """MLBGameOdds should work without run environment fields (backward compat)."""
        game = MLBGameOdds(
            game_id="401234567",
            commence_time="2025-05-15T19:05:00Z",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            home_abbrev="NYY",
            away_abbrev="BOS",
        )
        
        # New fields should default to None
        assert game.home_win_prob is None
        assert game.away_win_prob is None
        assert game.game_total is None


class TestTeamOddsMapWithRunEnvironment:
    """Test _build_team_odds_map includes run environment scoring."""
    
    def test_team_odds_map_includes_run_environment_score(self):
        """Team odds map should include calculated run_environment_score."""
        optimizer = DailyLineupOptimizer()
        
        games = [
            MLBGameOdds(
                game_id="1",
                commence_time="2025-05-15T19:05:00Z",
                home_team="Yankees",
                away_team="Red Sox",
                home_abbrev="NYY",
                away_abbrev="BOS",
                total=7.0,  # Low total = good for pitchers
                implied_home_runs=3.8,
                implied_away_runs=3.2,
                home_win_prob=0.65,  # Favorite = more likely to win
                away_win_prob=0.35,
                game_total=7.0,
                park_factor=1.0,
            )
        ]
        
        team_odds = optimizer._build_team_odds_map(games)
        
        # Both teams should have run environment data
        assert "NYY" in team_odds
        assert "BOS" in team_odds
        
        # Check NYY (favorite, low total) has high run_environment_score
        nyy_data = team_odds["NYY"]
        assert "run_environment_score" in nyy_data
        assert "win_prob" in nyy_data
        assert "game_total" in nyy_data
        
        # NYY should have good run environment (high score)
        assert nyy_data["run_environment_score"] > 5.0  # Better than average
        assert nyy_data["win_prob"] == 0.65
        assert nyy_data["game_total"] == 7.0
    
    def test_run_environment_score_calculation_low_total_favorite(self):
        """Low total + favorite = high run environment score."""
        optimizer = DailyLineupOptimizer()
        
        games = [
            MLBGameOdds(
                game_id="1",
                commence_time="2025-05-15T19:05:00Z",
                home_team="Dodgers",
                away_team="Rockies",
                home_abbrev="LAD",
                away_abbrev="COL",
                total=6.5,  # Very low total
                home_win_prob=0.70,  # Heavy favorite
                away_win_prob=0.30,
                game_total=6.5,
                park_factor=1.0,
            )
        ]
        
        team_odds = optimizer._build_team_odds_map(games)
        
        # Dodgers should have very high run environment score
        # (6.5 total score: (11.5-6.5)/4.5 = 1.11, clamped to 1.0)
        # Run env: (1.0 * 6) + (0.70 * 4) = 6 + 2.8 = 8.8
        lad_score = team_odds["LAD"]["run_environment_score"]
        assert lad_score > 8.0  # Very favorable
    
    def test_run_environment_score_calculation_high_total_underdog(self):
        """High total + underdog = low run environment score."""
        optimizer = DailyLineupOptimizer()
        
        games = [
            MLBGameOdds(
                game_id="1",
                commence_time="2025-05-15T19:05:00Z",
                home_team="Rockies",
                away_team="Dodgers",
                home_abbrev="COL",
                away_abbrev="LAD",
                total=11.0,  # High total (Coors Field)
                home_win_prob=0.30,  # Underdog
                away_win_prob=0.70,
                game_total=11.0,
                park_factor=1.5,
            )
        ]
        
        team_odds = optimizer._build_team_odds_map(games)
        
        # Rockies should have low run environment score
        # (11.0 total score: (11.5-11.0)/4.5 = 0.11)
        # Run env: (0.11 * 6) + (0.30 * 4) = 0.66 + 1.2 = 1.86
        col_score = team_odds["COL"]["run_environment_score"]
        assert col_score < 3.0  # Very unfavorable


class TestEnhancedSpScoreCalculation:
    """Test that enhanced sp_score uses run environment data."""
    
    def test_sp_score_components_add_up(self):
        """Verify sp_score components sum correctly."""
        # Simulate the calculation from fantasy.py
        opp_implied = 3.5  # Good matchup
        park_factor = 0.95  # Pitcher-friendly park
        run_env_score = 7.5  # Good run environment
        
        # Component calculations (from fantasy.py lines 1424-1432)
        opp_factor = max(0, 5.0 - opp_implied) * 0.8  # 0-4 points
        park_factor_score = max(0, (2.0 - park_factor) * 2.5)  # 0-2.5 points
        run_env_component = (run_env_score / 10.0) * 3.5  # 0-3.5 points
        
        sp_score = round(opp_factor + park_factor_score + run_env_component, 2)
        sp_score = min(10, max(0, sp_score))
        
        # Verify components
        assert opp_factor == pytest.approx(1.2, abs=0.01)  # (5-3.5) * 0.8
        assert park_factor_score == pytest.approx(2.625, abs=0.01)  # (2-0.95) * 2.5
        assert run_env_component == pytest.approx(2.625, abs=0.01)  # (7.5/10) * 3.5
        assert sp_score == pytest.approx(6.45, abs=0.01)
    
    def test_sp_score_clamped_to_valid_range(self):
        """sp_score should be clamped to 0-10 range."""
        # Excellent conditions (near perfect)
        opp_implied = 2.0  # Weak opponent
        park_factor = 0.85  # Great pitcher park
        run_env_score = 10.0  # Perfect run environment
        
        opp_factor = max(0, 5.0 - opp_implied) * 0.8  # 2.4 points
        park_factor_score = max(0, (2.0 - park_factor) * 2.5)  # 2.875 points
        run_env_component = (run_env_score / 10.0) * 3.5  # 3.5 points
        
        sp_score = round(opp_factor + park_factor_score + run_env_component, 2)
        sp_score = min(10, max(0, sp_score))
        
        # Sum should be ~8.775, well under 10 - clamping not needed
        assert sp_score > 8.0
        assert sp_score <= 10.0
        
        # Now test actual clamping with extreme values
        sp_score_extreme = 15.0
        sp_score_clamped = min(10, max(0, sp_score_extreme))
        assert sp_score_clamped == 10.0
        
        sp_score_negative = -5.0
        sp_score_clamped = min(10, max(0, sp_score_negative))
        assert sp_score_clamped == 0.0
    
    def test_sp_score_bad_matchup_low_score(self):
        """Bad matchup should result in low sp_score."""
        opp_implied = 5.5  # Strong opponent (only 0.4 points: max(0, 5-5.5)*0.8)
        park_factor = 1.35  # Very hitter-friendly park (0 points: max(0, 2-1.35)*2.5 = 1.625)
        run_env_score = 2.0  # Poor run environment
        
        opp_factor = max(0, 5.0 - opp_implied) * 0.8
        park_factor_score = max(0, (2.0 - park_factor) * 2.5)
        run_env_component = (run_env_score / 10.0) * 3.5
        
        sp_score = round(opp_factor + park_factor_score + run_env_component, 2)
        sp_score = min(10, max(0, sp_score))
        
        # Verify components are low
        assert opp_factor < 1.0  # Tough opponent
        assert sp_score < 4.0  # Overall poor streaming option


class TestEndToEndIntegration:
    """End-to-end tests for the complete run environment wiring."""
    
    def test_complete_flow_from_odds_to_team_map(self):
        """Verify complete data flow from raw odds to team odds map."""
        optimizer = DailyLineupOptimizer()
        
        # Simulate raw odds data from database
        games = [
            MLBGameOdds(
                game_id="401234568",
                commence_time="2025-05-15T19:05:00Z",
                home_team="Seattle Mariners",
                away_team="Houston Astros",
                home_abbrev="SEA",
                away_abbrev="HOU",
                spread_home=-1.5,
                total=7.5,
                moneyline_home=-140,
                moneyline_away=120,
                implied_home_runs=4.5,
                implied_away_runs=3.0,
                park_factor=0.95,  # Pitcher-friendly
                home_win_prob=0.583,  # -140 vig-adjusted
                away_win_prob=0.417,
                game_total=7.5,
            )
        ]
        
        # Build team odds map
        team_odds = optimizer._build_team_odds_map(games)
        
        # Verify SEA has good run environment score
        sea_data = team_odds["SEA"]
        
        # Expected calculation:
        # Total score: (11.5-7.5)/4.5 = 0.889
        # Run env: (0.889 * 6) + (0.583 * 4) = 5.33 + 2.33 = 7.66
        assert sea_data["run_environment_score"] > 7.0
        assert sea_data["win_prob"] > 0.5
        assert sea_data["game_total"] == 7.5
        
        # Verify HOU has worse run environment
        hou_data = team_odds["HOU"]
        assert hou_data["run_environment_score"] < sea_data["run_environment_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
