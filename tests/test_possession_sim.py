"""
Tests for possession-based Markov simulation engine
Run with: pytest tests/test_possession_sim.py -v
"""

import pytest
import numpy as np
from backend.services.possession_sim import (
    PossessionSimulator,
    TeamSimProfile,
    SimulationResult,
    calculate_four_factors,
    estimate_possessions,
    build_sim_profile,
)


class TestPossessionSimBasics:
    """Test that simulations produce valid output"""

    def test_simulation_runs(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="Duke")
        away = TeamSimProfile(team="UNC")

        result = sim.simulate_game(home, away, n_sims=100, seed=42)

        assert result.n_sims == 100
        assert len(result.home_scores) == 100
        assert len(result.away_scores) == 100

    def test_scores_are_realistic(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="Duke")
        away = TeamSimProfile(team="UNC")

        result = sim.simulate_game(home, away, n_sims=1000, seed=42)

        # Average college basketball score is roughly 65-80 per team
        avg_home = np.mean(result.home_scores)
        avg_away = np.mean(result.away_scores)

        assert 45 < avg_home < 100, f"Home avg {avg_home} out of realistic range"
        assert 45 < avg_away < 100, f"Away avg {avg_away} out of realistic range"

    def test_win_probabilities_sum_close_to_one(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="A")
        away = TeamSimProfile(team="B")

        result = sim.simulate_game(home, away, n_sims=5000, seed=42)

        # Home wins + away wins + pushes should equal 1
        total = result.home_win_prob + result.away_win_prob + result.push_prob
        assert abs(total - 1.0) < 0.001

    def test_first_half_scores_generated(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="A")
        away = TeamSimProfile(team="B")

        result = sim.simulate_game(home, away, n_sims=100, seed=42)

        assert result.home_1h_scores is not None
        assert len(result.home_1h_scores) == 100
        # 1H scores should be less than full-game scores
        assert np.mean(result.home_1h_scores) < np.mean(result.home_scores)


class TestStyleEmergentVariance:
    """Test that style differences produce different distributions"""

    def test_fast_teams_score_more(self):
        sim = PossessionSimulator()
        fast = TeamSimProfile(team="Fast", pace=78.0)
        slow = TeamSimProfile(team="Slow", pace=60.0)
        default = TeamSimProfile(team="Default", pace=68.0)

        fast_result = sim.simulate_game(fast, fast, n_sims=2000, seed=42, is_neutral=True)
        slow_result = sim.simulate_game(slow, slow, n_sims=2000, seed=42, is_neutral=True)

        fast_total = np.mean(fast_result.home_scores + fast_result.away_scores)
        slow_total = np.mean(slow_result.home_scores + slow_result.away_scores)

        assert fast_total > slow_total

    def test_three_heavy_teams_have_more_variance(self):
        sim = PossessionSimulator()
        shooter = TeamSimProfile(team="Shooter", three_rate=0.50, three_fg_pct=0.36)
        painter = TeamSimProfile(team="Painter", three_rate=0.20, rim_rate=0.50, rim_fg_pct=0.65)

        shoot_result = sim.simulate_game(shooter, shooter, n_sims=3000, seed=42, is_neutral=True)
        paint_result = sim.simulate_game(painter, painter, n_sims=3000, seed=42, is_neutral=True)

        # 3-point shooting should create wider margin distributions
        assert shoot_result.margin_sd > paint_result.margin_sd * 0.8  # Roughly wider

    def test_good_team_beats_bad_team(self):
        sim = PossessionSimulator()
        good = TeamSimProfile(team="Good", efg_pct=0.56, to_pct=0.14, orb_pct=0.32)
        bad = TeamSimProfile(team="Bad", efg_pct=0.44, to_pct=0.22, orb_pct=0.22)

        result = sim.simulate_game(good, bad, n_sims=3000, seed=42, is_neutral=True)

        assert result.home_win_prob > 0.65


class TestHomeAdvantage:
    """Test home court advantage modeling"""

    def test_home_team_gets_boost(self):
        sim = PossessionSimulator(home_advantage_pts=3.0)
        team = TeamSimProfile(team="Equal")

        home_result = sim.simulate_game(team, team, n_sims=5000, seed=42, is_neutral=False)
        neutral_result = sim.simulate_game(team, team, n_sims=5000, seed=42, is_neutral=True)

        # Home team should win more at home vs neutral
        assert home_result.home_win_prob > neutral_result.home_win_prob

    def test_neutral_site_is_roughly_even(self):
        sim = PossessionSimulator()
        team = TeamSimProfile(team="Equal")

        result = sim.simulate_game(team, team, n_sims=10000, seed=42, is_neutral=True)

        # Should be roughly 50-50 on neutral court
        assert abs(result.home_win_prob - 0.5) < 0.05


class TestDerivativeMarkets:
    """Test derivative market probability calculations"""

    def test_spread_cover_prob(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="Fav", efg_pct=0.54)
        away = TeamSimProfile(team="Dog", efg_pct=0.46)

        result = sim.simulate_game(home, away, n_sims=5000, seed=42)

        # Favorite should cover -3.5 more than 50% of the time
        cover = result.spread_cover_prob(-3.5)
        assert 0 < cover < 1

    def test_total_over_prob(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="A")
        away = TeamSimProfile(team="B")

        result = sim.simulate_game(home, away, n_sims=3000, seed=42)

        total = result.projected_total
        over_prob = result.total_over_prob(total)

        # Over/under the mean should be roughly 50-50
        assert abs(over_prob - 0.5) < 0.10

    def test_first_half_spread(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="A", efg_pct=0.54)
        away = TeamSimProfile(team="B", efg_pct=0.46)

        result = sim.simulate_game(home, away, n_sims=3000, seed=42)

        prob = result.first_half_spread_prob(-2.0)
        assert 0 < prob < 1

    def test_team_total_probs(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="A")
        away = TeamSimProfile(team="B")

        result = sim.simulate_game(home, away, n_sims=3000, seed=42)

        h_avg = np.mean(result.home_scores)
        prob = result.home_team_total_over_prob(h_avg)
        assert abs(prob - 0.5) < 0.10


class TestSimulateSpreadEdge:
    """Test the spread edge convenience method"""

    def test_returns_complete_dict(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="A", efg_pct=0.54)
        away = TeamSimProfile(team="B", efg_pct=0.46)

        edge = sim.simulate_spread_edge(home, away, spread=-4.5, n_sims=2000)

        assert "cover_prob" in edge
        assert "edge" in edge
        assert "kelly_full" in edge
        assert "sim_result" in edge
        assert 0 <= edge["cover_prob"] <= 1


class TestFourFactors:
    """Test Four Factors calculations from box scores"""

    def test_basic_calculation(self):
        ff = calculate_four_factors(
            fgm=25, fga=60, fgm3=8, fga3=20,
            fta=15, ftm=12, oreb=10, to=12,
        )

        assert 0 < ff["efg_pct"] < 1
        assert 0 < ff["to_pct"] < 1
        assert ff["possessions"] > 0

    def test_possession_estimate(self):
        poss = estimate_possessions(fga=60, oreb=10, to=12, fta=15)

        # Should be roughly 60 - 10 + 12 + 0.475 * 15 = 69.125
        assert abs(poss - 69.125) < 0.01


class TestProfileBuilder:
    """Test building sim profiles from stats"""

    def test_build_sim_profile(self):
        ff = {"efg_pct": 0.52, "to_pct": 0.16, "orb_pct": 0.30, "ft_rate": 0.28}
        profile = build_sim_profile("Duke", ff, pace=72.0)

        assert profile.team == "Duke"
        assert profile.efg_pct == 0.52
        assert profile.pace == 72.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
