"""
Tests for recency_weight.py — P3 Late-Season Recency Weighting
"""

import pytest
from datetime import date, timedelta

from backend.services.recency_weight import (
    is_late_season,
    is_tournament_mode,
    get_recency_weight,
    compute_weighted_rating,
    get_tournament_adjustments,
    RecencyWeightEngine,
    DEFAULT_RECENCY_WEIGHTS,
    REGULAR_SEASON_WEIGHT,
)


class TestLateSeasonDetection:
    """Test late season date detection."""
    
    def test_march_is_late_season(self):
        """March should be late season."""
        march_date = date(2026, 3, 1)
        assert is_late_season(march_date) is True
    
    def test_february_not_late_season(self):
        """February should not be late season."""
        feb_date = date(2026, 2, 28)
        assert is_late_season(feb_date) is False
    
    def test_april_is_late_season(self):
        """April should be late season."""
        april_date = date(2026, 4, 1)
        assert is_late_season(april_date) is True
    
    def test_january_not_late_season(self):
        """January should not be late season."""
        jan_date = date(2026, 1, 15)
        assert is_late_season(jan_date) is False


class TestTournamentMode:
    """Test tournament mode detection."""
    
    def test_march_15_is_tournament_mode(self):
        """March 15 should trigger tournament mode."""
        march_15 = date(2026, 3, 15)
        assert is_tournament_mode(march_15) is True
    
    def test_march_10_not_tournament_mode(self):
        """March 10 should not be tournament mode."""
        march_10 = date(2026, 3, 10)
        assert is_tournament_mode(march_10) is False
    
    def test_april_is_tournament_mode(self):
        """April should always be tournament mode."""
        april_1 = date(2026, 4, 1)
        assert is_tournament_mode(april_1) is True
    
    def test_custom_threshold(self):
        """Custom threshold should work."""
        march_12 = date(2026, 3, 12)
        assert is_tournament_mode(march_12, threshold_day=10) is True
        assert is_tournament_mode(march_12, threshold_day=15) is False


class TestRecencyWeights:
    """Test recency weight calculations."""
    
    def test_regular_season_flat_weight(self):
        """Regular season should have flat weights."""
        feb_date = date(2026, 2, 15)
        
        for days in [0, 5, 10, 20, 30]:
            weight = get_recency_weight(days, is_late_season_flag=False)
            assert weight == REGULAR_SEASON_WEIGHT
    
    def test_late_season_recent_games_boosted(self):
        """Recent games in late season should have higher weight."""
        # Today (0 days ago)
        assert get_recency_weight(0, is_late_season_flag=True) == 2.0
        
        # Yesterday
        assert get_recency_weight(1, is_late_season_flag=True) == 2.0
        
        # Last week
        assert get_recency_weight(7, is_late_season_flag=True) == 1.6
    
    def test_late_season_old_games_normalized(self):
        """Old games in late season should approach normal weight."""
        weight = get_recency_weight(30, is_late_season_flag=True)
        assert weight == 1.0  # Default for old games
    
    def test_weight_monotonically_decreases(self):
        """Weights should generally decrease as days increase."""
        weights = [
            get_recency_weight(d, is_late_season_flag=True)
            for d in range(0, 22)
        ]
        
        # Check general trend (allowing for plateaus)
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1] - 0.01  # Small tolerance


class TestWeightedRatingComputation:
    """Test weighted rating computation."""
    
    def test_basic_weighted_average(self):
        """Should compute weighted average correctly."""
        today = date.today()
        
        games = [
            {"game_date": today, "adj_em": 20.0},  # Today, weight 2.0
            {"game_date": today - timedelta(days=1), "adj_em": 18.0},  # Yesterday, weight 2.0
            {"game_date": today - timedelta(days=10), "adj_em": 15.0},  # 10 days ago, weight 1.4
        ]
        
        rating, meta = compute_weighted_rating(
            games,
            is_late_season_flag=True,
        )
        
        # Expected: (20*2.0 + 18*2.0 + 15*1.4) / (2.0 + 2.0 + 1.4)
        # = (40 + 36 + 21) / 5.4 = 97 / 5.4 = ~17.96
        assert rating == pytest.approx(17.96, 0.1)
        assert meta["valid_games"] == 3
    
    def test_empty_games(self):
        """Should handle empty games list."""
        rating, meta = compute_weighted_rating([])
        assert rating == 0.0
        assert "error" in meta
    
    def test_missing_data_skipped(self):
        """Should skip games with missing data."""
        games = [
            {"game_date": date.today(), "adj_em": 20.0},
            {"game_date": date.today(), "adj_em": None},  # Missing rating
            {"adj_em": 18.0},  # Missing date
        ]
        
        rating, meta = compute_weighted_rating(games, is_late_season_flag=True)
        
        assert meta["valid_games"] == 1
        assert rating == 20.0  # Only one valid game


class TestTournamentAdjustments:
    """Test tournament mode adjustments."""
    
    def test_regular_season_no_adjustments(self):
        """Regular season should have minimal adjustments."""
        feb_date = date(2026, 2, 15)
        adj = get_tournament_adjustments(feb_date)
        
        assert adj["neutral_site_override"] is False
        assert adj["margin_se_inflation"] == 0.0
        assert adj["recency_weight_active"] is False
    
    def test_tournament_mode_adjustments(self):
        """Tournament mode should have all adjustments."""
        march_20 = date(2026, 3, 20)
        adj = get_tournament_adjustments(march_20)
        
        assert adj["neutral_site_override"] is True
        assert adj["margin_se_inflation"] == 0.20
        assert adj["recency_weight_active"] is True
        assert adj["form_window_days"] == 14
    
    def test_neutral_site_inflation(self):
        """Neutral site should add variance even in regular season."""
        feb_date = date(2026, 2, 15)
        adj = get_tournament_adjustments(feb_date, is_neutral=True)
        
        assert adj["margin_se_inflation"] >= 0.15
    
    def test_neutral_plus_tournament(self):
        """Neutral site in tournament should use higher inflation."""
        march_20 = date(2026, 3, 20)
        adj = get_tournament_adjustments(march_20, is_neutral=True)
        
        # Should use tournament value (0.20) since it's higher
        assert adj["margin_se_inflation"] == 0.20


class TestRecencyWeightEngine:
    """Test RecencyWeightEngine class."""
    
    def test_engine_is_active_in_march(self):
        """Engine should be active in March."""
        engine = RecencyWeightEngine()
        
        # Mock the date check by testing the underlying function
        assert engine.is_active(date(2026, 3, 15)) is True
        assert engine.is_active(date(2026, 2, 15)) is False
    
    def test_engine_get_weight(self):
        """Engine should return correct weights."""
        engine = RecencyWeightEngine()
        
        # Set late season
        engine._test_date = date(2026, 3, 15)
        
        weight = engine.get_weight(5)
        assert weight > 1.0  # Should be boosted in late season
    
    def test_engine_apply_to_ratings(self):
        """Engine should apply weights to team ratings."""
        engine = RecencyWeightEngine()
        today = date.today()
        
        games = [
            {"game_date": today, "adj_em": 25.0},
            {"game_date": today - timedelta(days=5), "adj_em": 20.0},
        ]
        
        rating, meta = engine.apply_to_team_ratings(games)
        
        # Rating should be between 20 and 25 (weighted toward recent)
        assert 20 < rating < 25
        assert meta["games_count"] == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_future_date_handling(self):
        """Should handle future dates gracefully."""
        today = date.today()
        future_game = {"game_date": today + timedelta(days=1), "adj_em": 20.0}
        
        rating, meta = compute_weighted_rating([future_game])
        
        # Future game should be treated as today
        assert meta["valid_games"] == 1
    
    def test_negative_days_ago(self):
        """Should handle negative days (future) in weight lookup."""
        weight = get_recency_weight(-1, is_late_season_flag=True)
        assert weight == 2.0  # Should use day 0 weight
    
    def test_very_old_games(self):
        """Should handle very old games."""
        weight = get_recency_weight(365, is_late_season_flag=True)
        assert weight == 1.0  # Default weight
    
    def test_string_date_parsing(self):
        """Should parse ISO date strings."""
        today = date.today()
        games = [
            {"game_date": today.isoformat(), "adj_em": 20.0},
        ]
        
        rating, meta = compute_weighted_rating(games)
        assert meta["valid_games"] == 1
