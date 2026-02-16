"""
Tests for Version 7 betting model
Run with: pytest tests/test_betting_model.py -v
"""

import pytest
import numpy as np
from backend.betting_model import CBBEdgeModel


class TestMonteCarloCI:
    """Test Monte Carlo confidence interval calculation"""
    
    def test_returns_three_values(self):
        model = CBBEdgeModel()
        point, lower, upper = model.monte_carlo_prob_ci(5.0, 11.0)
        
        assert isinstance(point, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
    
    def test_ci_bounds_are_valid(self):
        model = CBBEdgeModel()
        point, lower, upper = model.monte_carlo_prob_ci(5.0, 11.0)
        
        # Lower < point < upper
        assert lower < point < upper
        
        # All between 0 and 1
        assert 0 <= lower <= 1
        assert 0 <= point <= 1
        assert 0 <= upper <= 1
    
    def test_positive_margin_gives_above_50_percent(self):
        model = CBBEdgeModel()
        point, lower, upper = model.monte_carlo_prob_ci(5.0, 11.0)
        
        # 5 point favorite should have >50% chance
        assert point > 0.5
    
    def test_large_margin_gives_high_probability(self):
        model = CBBEdgeModel()
        point, lower, upper = model.monte_carlo_prob_ci(20.0, 11.0)
        
        # 20 point favorite should have >80% chance
        assert point > 0.80


class TestKellyFraction:
    """Test Kelly Criterion calculation"""
    
    def test_positive_edge_returns_positive_kelly(self):
        model = CBBEdgeModel()
        
        # 55% prob at even odds (decimal 2.0) = 10% edge
        kelly = model.kelly_fraction(0.55, 2.0)
        
        assert kelly > 0
    
    def test_no_edge_returns_zero(self):
        model = CBBEdgeModel()
        
        # 50% prob at even odds = no edge
        kelly = model.kelly_fraction(0.50, 2.0)
        
        assert kelly == 0
    
    def test_negative_edge_returns_zero(self):
        model = CBBEdgeModel()
        
        # 45% prob at even odds = negative edge
        kelly = model.kelly_fraction(0.45, 2.0)
        
        assert kelly == 0
    
    def test_kelly_is_capped_at_max(self):
        model = CBBEdgeModel(max_kelly=0.20)
        
        # Huge edge (90% at 2.0 odds) should be capped
        kelly = model.kelly_fraction(0.90, 2.0)
        
        assert kelly <= 0.20
    
    def test_edge_case_invalid_odds(self):
        model = CBBEdgeModel()
        
        # Odds <= 1.0 are invalid
        kelly = model.kelly_fraction(0.60, 0.95)
        assert kelly == 0
        
        kelly = model.kelly_fraction(0.60, 1.0)
        assert kelly == 0


class TestVigRemoval:
    """Test vig removal from American odds"""
    
    def test_symmetric_odds(self):
        model = CBBEdgeModel()
        
        fav, dog = model.remove_vig_american(-110, 110)
        
        # Should sum to 1.0 (100%)
        assert abs((fav + dog) - 1.0) < 0.0001
        
        # Should be roughly equal
        assert abs(fav - dog) < 0.01
    
    def test_symmetric_odds(self):
        model = CBBEdgeModel()
        
        # USE -110 for both to represent a true Pick'em with vig
        fav, dog = model.remove_vig_american(-110, -110)
        
        # Should sum to 1.0 (100%)
        assert abs((fav + dog) - 1.0) < 0.0001
        
        # This will now pass as both fav and dog will be exactly 0.5
        assert abs(fav - dog) < 0.01


class TestAdjustedSD:
    """Test SD adjustment with penalty budget"""
    
    def test_no_penalties_returns_base_sd(self):
        model = CBBEdgeModel(base_sd=11.0)
        
        adj_sd = model.adjusted_sd({})
        
        assert adj_sd == 11.0
    
    def test_small_penalties_increase_sd(self):
        model = CBBEdgeModel(base_sd=11.0)
        
        penalties = {'stale': 1.0, 'injury': 1.5}
        adj_sd = model.adjusted_sd(penalties)
        
        assert adj_sd > 11.0
        assert adj_sd < 15.0
    
    def test_large_penalties_are_capped(self):
        model = CBBEdgeModel(base_sd=11.0)
        
        # Extreme penalties
        penalties = {'stale': 5.0, 'injury': 5.0, 'gap': 5.0}
        adj_sd = model.adjusted_sd(penalties)
        
        # Should be capped at 15.5
        assert adj_sd <= 15.5
    
    def test_penalty_budget_uses_sqrt_sum(self):
        model = CBBEdgeModel(base_sd=11.0)
        
        # Two penalties of 3 each
        # Sqrt(9 + 9) = 4.24, not 6
        penalties = {'a': 3.0, 'b': 3.0}
        adj_sd = model.adjusted_sd(penalties)
        
        # Should be less than if penalties were additive
        expected_additive = 11.0 * (1 + 6.0 / 15)
        
        assert adj_sd < expected_additive


class TestGameAnalysis:
    """Test full game analysis"""
    
    def test_returns_game_analysis_object(self):
        model = CBBEdgeModel()
        
        game = {'home_team': 'Duke', 'away_team': 'UNC', 'is_neutral': False}
        ratings = {
            'kenpom': {'home': 25.0, 'away': 20.0},
            'barttorvik': {'home': 24.5, 'away': 19.8},
            'evanmiya': {'home': 25.2, 'away': 20.5},
        }
        odds = {'spread': -4.5, 'spread_odds': -110}
        
        result = model.analyze_game(game, odds, ratings)
        
        assert result.verdict is not None
        assert result.projected_margin > 0  # Duke favored
    
    def test_missing_kenpom_returns_pass(self):
        model = CBBEdgeModel()
        
        game = {'home_team': 'Duke', 'away_team': 'UNC', 'is_neutral': False}
        ratings = {
            'kenpom': {},  # Missing
            'barttorvik': {'home': 24.5, 'away': 19.8},
            'evanmiya': {'home': 25.2, 'away': 20.5},
        }
        odds = {'spread': -4.5, 'spread_odds': -110}
        
        result = model.analyze_game(game, odds, ratings)
        
        assert result.verdict == "PASS"
        assert "Missing KenPom" in result.pass_reason
    
    def test_stale_lines_returns_pass(self):
        model = CBBEdgeModel()
        
        game = {'home_team': 'Duke', 'away_team': 'UNC', 'is_neutral': False}
        ratings = {
            'kenpom': {'home': 25.0, 'away': 20.0},
            'barttorvik': {'home': 24.5, 'away': 19.8},
            'evanmiya': {'home': 25.2, 'away': 20.5},
        }
        odds = {'spread': -4.5, 'spread_odds': -110}
        freshness = {'lines_age_min': 45, 'ratings_age_hours': 2}  # Too old!
        
        result = model.analyze_game(game, odds, ratings, data_freshness=freshness)
        
        assert result.verdict == "PASS"
        assert "Tier 3" in result.pass_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
