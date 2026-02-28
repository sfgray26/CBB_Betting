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


class TestLog5Blending:
    """Test that _blend_rate uses Log5 and handles extremes correctly"""

    def test_average_vs_average_returns_d1_avg(self):
        sim = PossessionSimulator()
        # When both rates equal the D1 average, the blended result should ≈ d1_avg
        d1_avg = 0.170
        result = sim._blend_rate(d1_avg, d1_avg, d1_avg)
        assert abs(result - d1_avg) < 0.005

    def test_output_always_in_unit_interval(self):
        sim = PossessionSimulator()
        # Extreme values that would break the old linear formula
        extreme_cases = [
            (0.45, 0.40, 0.170),  # both very high TO rates
            (0.60, 0.58, 0.500),  # elite eFG on both sides
            (0.01, 0.01, 0.170),  # very low rates
            (0.95, 0.95, 0.500),  # near-ceiling rates
        ]
        for off, def_, avg in extreme_cases:
            result = sim._blend_rate(off, def_, avg)
            assert 0.0 < result < 1.0, (
                f"_blend_rate({off}, {def_}, {avg}) = {result} — out of (0,1)"
            )

    def test_better_offense_raises_blended_rate(self):
        sim = PossessionSimulator()
        d1_avg = 0.500
        # Elite offense (0.58 eFG) vs average defense should beat average offense
        elite_off = sim._blend_rate(0.58, d1_avg, d1_avg)
        avg_off = sim._blend_rate(d1_avg, d1_avg, d1_avg)
        assert elite_off > avg_off

    def test_better_defense_lowers_blended_rate(self):
        sim = PossessionSimulator()
        d1_avg = 0.500
        # Average offense vs elite defense (0.44 allowed) should be below average
        elite_def = sim._blend_rate(d1_avg, 0.44, d1_avg)
        avg_def = sim._blend_rate(d1_avg, d1_avg, d1_avg)
        assert elite_def < avg_def

    def test_extreme_inputs_do_not_exceed_old_clamp_bounds(self):
        sim = PossessionSimulator()
        # The safety clamp [0.001, 0.999] should never activate on realistic inputs
        realistic_cases = [
            (0.35, 0.30, 0.170),  # high TO scenario
            (0.55, 0.52, 0.500),  # high eFG scenario
            (0.38, 0.35, 0.280),  # high ORB scenario
        ]
        for off, def_, avg in realistic_cases:
            result = sim._blend_rate(off, def_, avg)
            assert result != 0.001 and result != 0.999, (
                f"Safety clamp activated for ({off}, {def_}, {avg}) — check inputs"
            )

    def test_simulation_scores_remain_realistic_with_log5(self):
        sim = PossessionSimulator()
        home = TeamSimProfile(team="Duke")
        away = TeamSimProfile(team="UNC")

        result = sim.simulate_game(home, away, n_sims=2000, seed=99)

        avg_home = np.mean(result.home_scores)
        avg_away = np.mean(result.away_scores)

        # Scores should still be in the realistic college basketball range
        assert 50 < avg_home < 95, f"Home avg {avg_home:.1f} out of realistic range"
        assert 50 < avg_away < 95, f"Away avg {avg_away:.1f} out of realistic range"

    def test_no_hard_clamp_on_valid_inputs(self):
        """Log5 should not hard-clamp to 0.001/0.999 — it stays in (0,1) naturally."""
        sim = PossessionSimulator()
        # Moderate inputs that are NOT at the boundaries
        result = sim._blend_rate(0.30, 0.25, 0.170)
        # Should be somewhere in (0.1, 0.5) range, NOT at clamp boundaries
        assert 0.001 < result < 0.999
        assert result != 0.001 and result != 0.999

    def test_log5_normalization_against_league_average(self):
        """When off=avg and def=avg, result should equal the league average."""
        sim = PossessionSimulator()
        for avg in [0.170, 0.280, 0.500, 0.300]:
            result = sim._blend_rate(avg, avg, avg)
            assert abs(result - avg) < 0.001, (
                f"Log5({avg}, {avg}, {avg}) = {result}, expected {avg}"
            )


class TestMeanCentering:
    """Tests for SimulationResult.center_on_margin() — the SOS contamination fix."""

    def _make_result(self, n: int = 3000, seed: int = 42) -> SimulationResult:
        sim = PossessionSimulator()
        home = TeamSimProfile(team="AverageHome", efg_pct=0.505, to_pct=0.175, pace=68.0)
        away = TeamSimProfile(team="AverageAway", efg_pct=0.505, to_pct=0.175, pace=68.0)
        return sim.simulate_game(home, away, n_sims=n, seed=seed)

    def test_identity_property(self):
        """Centering to the raw Markov mean should barely change cover_prob."""
        result = self._make_result()
        raw_mean = result.projected_margin
        centered = result.center_on_margin(raw_mean)
        assert abs(centered.projected_margin - raw_mean) < 1e-6

    def test_centering_shifts_mean_to_target(self):
        """After centering, projected_margin should equal the target."""
        result = self._make_result()
        target = 15.0
        centered = result.center_on_margin(target)
        assert abs(centered.projected_margin - target) < 1e-6

    def test_centering_corrects_blowout_cover_prob(self):
        """
        Raw Markov sees two average teams → near-50% cover.
        Centering to +41 (blowout AdjEM margin) at spread=-13.5 → high cover prob.
        """
        sim = PossessionSimulator()
        home = TeamSimProfile(team="Power5", efg_pct=0.540, to_pct=0.165, pace=72.0)
        away = TeamSimProfile(team="MidMajor", efg_pct=0.505, to_pct=0.180, pace=70.0)
        result = sim.simulate_game(home, away, n_sims=5000, seed=42)

        # Before centering: competitive-looking stats → cover prob near 50% at -13.5
        raw_probs = result.spread_cover_probs(-13.5)
        raw_cover = raw_probs["win"]
        assert raw_cover < 0.50, f"Expected raw cover < 0.50, got {raw_cover:.3f}"

        # After centering to blowout margin: cover prob should be high
        centered = result.center_on_margin(41.0)
        centered_probs = centered.spread_cover_probs(-13.5)
        assert centered_probs["win"] > 0.85, (
            f"Expected centered cover > 0.85, got {centered_probs['win']:.3f}"
        )

    def test_push_prob_drops_to_zero_after_float_shift(self):
        """Non-integer centering target destroys integer alignment → push_prob == 0."""
        result = self._make_result(seed=7)
        centered = result.center_on_margin(5.3)  # non-integer shift
        probs = centered.spread_cover_probs(-5.0)  # integer spread
        assert probs["push"] == 0.0, f"Expected push=0.0, got {probs['push']}"

    def test_win_plus_loss_sums_to_one_after_centering(self):
        result = self._make_result()
        centered = result.center_on_margin(8.0)
        probs = centered.spread_cover_probs(-7.5)
        total = probs["win"] + probs["loss"] + probs["push"]
        assert abs(total - 1.0) < 1e-9, f"Probs sum to {total}, expected 1.0"

    def test_scores_become_float64_after_centering(self):
        result = self._make_result()
        centered = result.center_on_margin(10.0)
        assert centered.home_scores.dtype == np.float64

    def test_away_scores_numerically_unchanged(self):
        result = self._make_result()
        centered = result.center_on_margin(10.0)
        np.testing.assert_array_equal(
            centered.away_scores,
            result.away_scores.astype(float),
        )

    def test_n_sims_and_shape_preserved(self):
        n = 2000
        result = self._make_result(n=n)
        centered = result.center_on_margin(-3.0)
        assert centered.n_sims == n
        assert len(centered.home_scores) == n
        assert len(centered.away_scores) == n


class TestSimulateSpreadEdgeWithCentering:
    """Tests for simulate_spread_edge(projected_margin=...) end-to-end."""

    def _sim(self) -> PossessionSimulator:
        return PossessionSimulator()

    def test_none_projected_margin_backward_compatible(self):
        """projected_margin=None must not crash and must return a valid result."""
        sim = self._sim()
        home = TeamSimProfile(team="Home", efg_pct=0.510, to_pct=0.172, pace=70.0)
        away = TeamSimProfile(team="Away", efg_pct=0.500, to_pct=0.180, pace=68.0)

        r = sim.simulate_spread_edge(home, away, spread=-3.5, n_sims=2000,
                                     projected_margin=None)
        assert 0.0 <= r["cover_prob"] <= 1.0
        assert abs(r["cover_prob"] + r["loss_prob"] + r["push_prob"] - 1.0) < 1e-6

    def test_blowout_cover_prob_corrected(self):
        """AdjEM blowout scenario: cover prob corrected from <0.50 → >0.85."""
        sim = self._sim()
        home = TeamSimProfile(team="Power5", efg_pct=0.540, to_pct=0.165, pace=72.0)
        away = TeamSimProfile(team="MidMajor", efg_pct=0.505, to_pct=0.180, pace=70.0)

        raw = sim.simulate_spread_edge(home, away, spread=-13.5, n_sims=5000)
        assert raw["cover_prob"] < 0.50, (
            f"Expected raw cover < 0.50 before centering, got {raw['cover_prob']:.3f}"
        )

        centered = sim.simulate_spread_edge(
            home, away, spread=-13.5, n_sims=5000,
            projected_margin=41.0,
        )
        assert centered["cover_prob"] > 0.85, (
            f"Expected centered cover > 0.85, got {centered['cover_prob']:.3f}"
        )

    def test_close_game_preserves_uncertainty(self):
        """Close game: centering near the spread should keep cover_prob near 50%."""
        sim = self._sim()
        home = TeamSimProfile(team="Home", efg_pct=0.510, to_pct=0.172, pace=70.0)
        away = TeamSimProfile(team="Away", efg_pct=0.505, to_pct=0.178, pace=69.0)

        result = sim.simulate_spread_edge(
            home, away, spread=-3.5, n_sims=4000,
            projected_margin=3.0,  # close game, home slightly favoured
        )
        assert 0.35 < result["cover_prob"] < 0.65, (
            f"Expected cover_prob near 0.50 for close game, got {result['cover_prob']:.3f}"
        )

    def test_push_prob_zero_with_float_centering(self):
        """Non-integer projected_margin shift → no integer alignment → push_prob == 0."""
        sim = self._sim()
        home = TeamSimProfile(team="Home", efg_pct=0.510, to_pct=0.172, pace=70.0)
        away = TeamSimProfile(team="Away", efg_pct=0.505, to_pct=0.178, pace=69.0)

        result = sim.simulate_spread_edge(
            home, away, spread=-5.0, n_sims=2000,
            projected_margin=5.3,  # non-integer target
        )
        assert result["push_prob"] == 0.0, (
            f"Expected push_prob=0.0 after float centering, got {result['push_prob']}"
        )

    def test_logger_emits_shift_magnitude(self, caplog):
        """DEBUG log must mention 'mean-centering' with both team names."""
        import logging
        sim = self._sim()
        home = TeamSimProfile(team="Alpha", efg_pct=0.510, to_pct=0.172, pace=70.0)
        away = TeamSimProfile(team="Beta", efg_pct=0.505, to_pct=0.178, pace=69.0)

        with caplog.at_level(logging.DEBUG, logger="backend.services.possession_sim"):
            sim.simulate_spread_edge(
                home, away, spread=-3.5, n_sims=500,
                projected_margin=12.0,
            )

        combined = " ".join(caplog.messages)
        assert "mean-centering" in combined.lower() or "Markov mean-centering" in combined, (
            f"Expected 'mean-centering' in debug log; got: {caplog.messages}"
        )
        assert "Alpha" in combined or "Beta" in combined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
