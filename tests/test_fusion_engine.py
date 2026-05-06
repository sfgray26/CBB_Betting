"""
Test suite for Bayesian Projection Fusion Engine.

Tests the Marcel update formula and four-state fusion logic for batters and pitchers.
"""

import pytest
from backend.fantasy_baseball.fusion_engine import (
    StabilizationPoints,
    PopulationPrior,
    marcel_update,
    fuse_batter_projection,
    fuse_pitcher_projection,
    FusionResult,
    to_season_counts,
    PitcherCountingStatFormulas,
)


class TestStabilizationPoints:
    """Test stabilization point constants."""

    def test_batter_stabilization_points_exist(self):
        """Verify all batter stabilization points are defined."""
        points = StabilizationPoints()

        # PA thresholds for batters
        assert points.BATTER_K_PERCENT == 60
        assert points.BATTER_BB_PERCENT == 120
        assert points.BATTER_HR_RATE == 170
        assert points.BATTER_ISO == 160
        assert points.BATTER_OBP == 460
        assert points.BATTER_SLG == 320
        assert points.BATTER_AVG == 910
        assert points.BATTER_BARREL_PCT == 50
        assert points.BATTER_XWOBA == 150

    def test_pitcher_stabilization_points_exist(self):
        """Verify all pitcher stabilization points are defined."""
        points = StabilizationPoints()

        # IP thresholds for pitchers
        assert points.PITCHER_K_PERCENT == 70
        assert points.PITCHER_BB_PERCENT == 170
        assert points.PITCHER_ERA == 300
        assert points.PITCHER_WHIP == 300
        assert points.PITCHER_FIP == 170
        assert points.PITCHER_XERA == 150


class TestPopulationPrior:
    """Test population prior constants."""

    def test_batter_priors_exist(self):
        """Verify batter population priors are defined."""
        prior = PopulationPrior()

        assert prior.BATTER_AVG == 0.250
        assert prior.BATTER_OBP == 0.320
        assert prior.BATTER_SLG == 0.410
        assert prior.BATTER_OPS == 0.730
        assert prior.BATTER_HR_PER_PA == 0.035
        assert prior.BATTER_SB_PER_PA == 0.010
        assert prior.BATTER_K_PERCENT == 0.225
        assert prior.BATTER_BB_PERCENT == 0.080

    def test_pitcher_priors_exist(self):
        """Verify pitcher population priors are defined."""
        prior = PopulationPrior()

        assert prior.PITCHER_ERA == 4.50
        assert prior.PITCHER_WHIP == 1.35
        assert prior.PITCHER_K_PER_NINE == 8.5
        assert prior.PITCHER_BB_PER_NINE == 3.0


class TestMarcelUpdate:
    """Test the Marcel update formula for Bayesian shrinkage."""

    def test_midpoint_property(self):
        """At sample_size = stabilization, posterior should be exactly midpoint."""
        prior = 0.250
        observed = 0.300
        stabil = 100

        result = marcel_update(prior, observed, stabil, stabil)
        expected = (prior + observed) / 2

        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_prior_dominance_small_sample(self):
        """Small sample (10% of stabil) should skew >80% toward prior."""
        prior = 0.250
        observed = 0.350
        stabil = 100
        sample = 10  # 10% of stabil

        result = marcel_update(prior, observed, sample, stabil)

        # Weight toward prior = stabil / (stabil + sample) = 100/110 ≈ 0.909
        prior_weight = stabil / (stabil + sample)
        assert prior_weight > 0.80, "Prior should dominate with small sample"

        # Result should be much closer to prior than observed
        assert abs(result - prior) < abs(result - observed)

    def test_observed_dominance_large_sample(self):
        """Large sample (10x stabil) should skew >80% toward observed."""
        prior = 0.250
        observed = 0.350
        stabil = 100
        sample = 1000  # 10x stabil

        result = marcel_update(prior, observed, sample, stabil)

        # Weight toward observed = sample / (stabil + sample) = 1000/1100 ≈ 0.909
        observed_weight = sample / (stabil + sample)
        assert observed_weight > 0.80, "Observed should dominate with large sample"

        # Result should be much closer to observed than prior
        assert abs(result - observed) < abs(result - prior)

    def test_min_sample_enforcement(self):
        """min_sample should prevent division by zero and limit shrinkage."""
        prior = 0.250
        observed = 0.300
        stabil = 100

        # Zero sample should use min_sample
        result = marcel_update(prior, observed, 0, stabil, min_sample=1)

        # With min_sample=1, effective sample is 1
        expected = ((stabil * prior) + (1 * observed)) / (stabil + 1)
        assert abs(result - expected) < 1e-10

    def test_exact_formula_verification(self):
        """Verify exact formula: ((stabil * prior) + (sample * observed)) / (stabil + sample)."""
        prior = 0.275
        observed = 0.295
        sample = 150
        stabil = 320

        result = marcel_update(prior, observed, sample, stabil)
        expected = ((stabil * prior) + (sample * observed)) / (stabil + sample)

        assert abs(result - expected) < 1e-10

    def test_boundary_values(self):
        """Test with edge case values."""
        # All zeros
        assert marcel_update(0, 0, 0, 100, min_sample=1) == 0

        # Very large prior
        result = marcel_update(1.0, 0.0, 100, 100)
        assert result == 0.5

    def test_negative_values_handled(self):
        """Formula should work with any numeric values."""
        # ERA can be treated as negative delta
        result = marcel_update(-1.0, 1.0, 50, 100)
        # stabil*prior + sample*observed = 100*(-1) + 50*1 = -100 + 50 = -50
        # divided by 150 = -1/3
        assert abs(result - (-1/3)) < 1e-10


class TestBatterFusionFourState:
    """Test four-state logic for batter projections."""

    def test_state_1_steamer_statcast_full_fusion(self):
        """Steamer + Statcast: Full Marcel update per component."""
        steamer = {
            'avg': 0.280,
            'obp': 0.350,
            'slg': 0.480,
            'k_percent': 0.220,
            'bb_percent': 0.090,
        }
        statcast = {
            'avg': 0.270,
            'obp': 0.340,
            'slg': 0.460,
            'k_percent': 0.240,
            'bb_percent': 0.080,
            'pa': 500,
            'xwoba': 0.340,
            'woba': 0.330,
        }

        result = fuse_batter_projection(steamer, statcast, sample_size=500)

        assert result.source == 'fusion'
        assert result.components_fused > 0
        # Result should be between steamer and statcast
        assert 0.270 <= result.proj['avg'] <= 0.280 or 0.270 <= result.proj['avg'] <= 0.280

    def test_state_2_steamer_only_unchanged(self):
        """Steamer only: Return Steamer unchanged."""
        steamer = {
            'avg': 0.280,
            'obp': 0.350,
            'slg': 0.480,
            'hr_per_pa': 0.040,
            'sb_per_pa': 0.015,
        }

        result = fuse_batter_projection(steamer, None, sample_size=500)

        assert result.source == 'steamer'
        assert result.components_fused == 0
        assert result.proj['avg'] == 0.280
        assert result.proj['obp'] == 0.350

    def test_state_3_statcast_only_double_shrinkage(self):
        """Statcast only: Fuse with POPULATION_PRIOR using double shrinkage."""
        statcast = {
            'avg': 0.280,
            'obp': 0.360,
            'slg': 0.500,
            'k_percent': 0.200,
            'bb_percent': 0.100,
            'pa': 300,
        }

        result = fuse_batter_projection(None, statcast, sample_size=300)

        assert result.source == 'statcast_shrunk'
        assert result.components_fused > 0
        # Should be closer to prior than pure statcast for low-sample metrics
        prior_avg = 0.250
        assert result.proj['avg'] != statcast['avg']  # Should be shrunk

    def test_state_4_neither_population_prior(self):
        """Neither source: Return pure POPULATION_PRIOR."""
        result = fuse_batter_projection(None, None, sample_size=0)

        assert result.source == 'population_prior'
        assert result.components_fused == 0
        assert result.proj['avg'] == 0.250
        assert result.proj['obp'] == 0.320
        assert result.proj['slg'] == 0.410

    def test_xwoba_override_triggers(self):
        """xwOBA override triggers when |xwOBA - wOBA| > 0.030."""
        statcast = {
            'avg': 0.270,
            'obp': 0.340,
            'slg': 0.460,
            'k_percent': 0.220,
            'bb_percent': 0.080,
            'pa': 400,
            'xwoba': 0.370,  # 0.040 above wOBA
            'woba': 0.330,
        }
        steamer = {
            'avg': 0.280,
            'obp': 0.350,
            'slg': 0.480,
        }

        result = fuse_batter_projection(steamer, statcast, sample_size=400)

        assert result.xwoba_override_detected is True

    def test_xwoba_override_no_trigger(self):
        """xwOBA override does NOT trigger when delta <= 0.030."""
        statcast = {
            'avg': 0.270,
            'obp': 0.340,
            'slg': 0.460,
            'k_percent': 0.220,
            'bb_percent': 0.080,
            'pa': 400,
            'xwoba': 0.350,  # Only 0.020 above wOBA
            'woba': 0.330,
        }
        steamer = {
            'avg': 0.280,
            'obp': 0.350,
            'slg': 0.480,
        }

        result = fuse_batter_projection(steamer, statcast, sample_size=400)

        assert result.xwoba_override_detected is False

    def test_xwoba_likelihood_applied_when_divergence_detected(self):
        """xwoba_likelihood_applied=True and woba_blend in proj when |xwOBA-wOBA|>0.030 and PA>=50"""
        steamer = {'avg': 0.270, 'obp': 0.340, 'slg': 0.450, 'k_percent': 0.22, 'bb_percent': 0.08, 'hr_per_pa': 0.035, 'sb_per_pa': 0.01}
        statcast = {'avg': 0.300, 'obp': 0.370, 'slg': 0.500, 'k_percent': 0.18, 'bb_percent': 0.10, 'xwoba': 0.390, 'woba': 0.350}
        result = fuse_batter_projection(steamer, statcast, sample_size=150)
        assert result.xwoba_likelihood_applied is True
        assert 'woba_blend' in result.proj

    def test_xwoba_likelihood_not_applied_small_sample(self):
        """xwoba_likelihood_applied=False when PA < 50, even with divergence"""
        steamer = {'avg': 0.270, 'obp': 0.340, 'slg': 0.450, 'k_percent': 0.22, 'bb_percent': 0.08, 'hr_per_pa': 0.035, 'sb_per_pa': 0.01}
        statcast = {'avg': 0.300, 'obp': 0.370, 'slg': 0.500, 'k_percent': 0.18, 'bb_percent': 0.10, 'xwoba': 0.390, 'woba': 0.350}
        result = fuse_batter_projection(steamer, statcast, sample_size=30)
        assert result.xwoba_likelihood_applied is False
        assert 'woba_blend' not in result.proj


class TestPitcherFusionFourState:
    """Test four-state logic for pitcher projections."""

    def test_state_1_steamer_statcast_full_fusion(self):
        """Steamer + Statcast: Full Marcel update per component."""
        steamer = {
            'era': 3.50,
            'whip': 1.20,
            'k_percent': 0.250,
            'bb_percent': 0.070,
        }
        statcast = {
            'era': 3.80,
            'whip': 1.25,
            'k_percent': 0.240,
            'bb_percent': 0.075,
            'ip': 150,
            'xera': 3.60,
            'era': 3.80,
        }

        result = fuse_pitcher_projection(steamer, statcast, sample_size=150)

        assert result.source == 'fusion'
        assert result.components_fused > 0

    def test_state_2_steamer_only_unchanged(self):
        """Steamer only: Return Steamer unchanged."""
        steamer = {
            'era': 3.50,
            'whip': 1.20,
            'k_per_nine': 9.0,
            'bb_per_nine': 2.5,
        }

        result = fuse_pitcher_projection(steamer, None, sample_size=100)

        assert result.source == 'steamer'
        assert result.components_fused == 0
        assert result.proj['era'] == 3.50

    def test_state_3_statcast_only_double_shrinkage(self):
        """Statcast only: Fuse with POPULATION_PRIOR using double shrinkage."""
        statcast = {
            'era': 3.20,
            'whip': 1.15,
            'k_percent': 0.280,
            'bb_percent': 0.060,
            'ip': 80,
        }

        result = fuse_pitcher_projection(None, statcast, sample_size=80)

        assert result.source == 'statcast_shrunk'
        assert result.components_fused > 0

    def test_state_4_neither_population_prior(self):
        """Neither source: Return pure POPULATION_PRIOR."""
        result = fuse_pitcher_projection(None, None, sample_size=0)

        assert result.source == 'population_prior'
        assert result.components_fused == 0
        assert result.proj['era'] == 4.50
        assert result.proj['whip'] == 1.35

    def test_xera_override_triggers(self):
        """xERA override triggers when |xERA - ERA| > 0.50."""
        statcast = {
            'era': 4.50,
            'xera': 3.80,  # 0.70 difference
            'whip': 1.30,
            'k_percent': 0.220,
            'bb_percent': 0.080,
            'ip': 100,
        }
        steamer = {
            'era': 3.80,
            'whip': 1.25,
        }

        result = fuse_pitcher_projection(steamer, statcast, sample_size=100)

        assert result.xwoba_override_detected is True

    def test_xera_override_no_trigger(self):
        """xERA override does NOT trigger when delta <= 0.50."""
        statcast = {
            'era': 4.00,
            'xera': 3.70,  # Only 0.30 difference
            'whip': 1.30,
            'k_percent': 0.220,
            'bb_percent': 0.080,
            'ip': 100,
        }
        steamer = {
            'era': 3.80,
            'whip': 1.25,
        }

        result = fuse_pitcher_projection(steamer, statcast, sample_size=100)

        assert result.xwoba_override_detected is False

    def test_xera_likelihood_applied_when_divergence_detected(self):
        """xera_likelihood_applied=True and proj['era'] adjusted when |xERA-ERA|>0.50 and IP>=20"""
        steamer = {'era': 4.50, 'whip': 1.35, 'k_percent': 0.22, 'bb_percent': 0.07, 'k_per_nine': 8.5, 'bb_per_nine': 3.0}
        statcast = {'era': 5.20, 'whip': 1.50, 'k_percent': 0.20, 'bb_percent': 0.08, 'xera': 3.80}
        result = fuse_pitcher_projection(steamer, statcast, sample_size=60)
        assert result.xera_likelihood_applied is True
        assert result.proj['era'] < fuse_pitcher_projection(steamer, {**statcast, 'xera': statcast['era']}, sample_size=60).proj['era'] + 0.5

    def test_xera_likelihood_not_applied_small_sample(self):
        steamer = {'era': 4.50, 'whip': 1.35, 'k_percent': 0.22, 'bb_percent': 0.07, 'k_per_nine': 8.5, 'bb_per_nine': 3.0}
        statcast = {'era': 5.20, 'whip': 1.50, 'k_percent': 0.20, 'bb_percent': 0.08, 'xera': 3.80}
        result = fuse_pitcher_projection(steamer, statcast, sample_size=10)
        assert result.xera_likelihood_applied is False


class TestFusionResult:
    """Test FusionResult dataclass."""

    def test_fusion_result_structure(self):
        """Verify FusionResult has required fields."""
        result = FusionResult(
            proj={'avg': 0.280},
            source='fusion',
            components_fused=5,
            xwoba_override_detected=False
        )

        assert result.proj == {'avg': 0.280}
        assert result.source == 'fusion'
        assert result.components_fused == 5
        assert result.xwoba_override_detected is False
        assert result.xwoba_likelihood_applied is False
        assert result.xera_likelihood_applied is False


class TestMathematicalProperties:
    """Additional mathematical verification tests."""

    def test_marcel_update_is_bounded(self):
        """Marcel update should always return value between prior and observed."""
        prior = 0.250
        observed = 0.350

        for sample in [1, 10, 50, 100, 500]:
            result = marcel_update(prior, observed, sample, 100)
            assert prior <= result <= observed or observed <= result <= prior

    def test_convergence_to_prior(self):
        """As sample -> 0, result -> prior."""
        prior = 0.250
        observed = 0.350

        result = marcel_update(prior, observed, 0, 100, min_sample=1)
        # Should be very close to prior
        assert abs(result - prior) < 0.01

    def test_convergence_to_observed(self):
        """As sample -> infinity, result -> observed."""
        prior = 0.250
        observed = 0.350

        result = marcel_update(prior, observed, 10000, 100)
        # Should be very close to observed
        assert abs(result - observed) < 0.01

    def test_symmetry(self):
        """Marcel update should be symmetric if we swap prior/observed and adjust weights."""
        prior1, observed1 = 0.250, 0.350
        prior2, observed2 = 0.350, 0.250

        result1 = marcel_update(prior1, observed1, 100, 100)
        result2 = marcel_update(prior2, observed2, 100, 100)

        # Results should be symmetric around midpoint
        midpoint = (0.250 + 0.350) / 2
        assert abs(result1 - midpoint) == abs(result2 - midpoint)


class TestToSeasonCounts:
    """Tests for to_season_counts() hybrid counting-stat translation."""

    def _batter_result(self, hr_per_pa=0.05, sb_per_pa=0.01):
        """Helper: FusionResult with batter rate stats."""
        return FusionResult(
            proj={"hr_per_pa": hr_per_pa, "sb_per_pa": sb_per_pa,
                  "avg": 0.280, "obp": 0.350, "slg": 0.480},
            source="fusion",
            components_fused=3,
            xwoba_override_detected=False,
        )

    def _pitcher_result(self, era=3.80, k_per_nine=9.5):
        """Helper: FusionResult with pitcher rate stats."""
        return FusionResult(
            proj={"era": era, "k_per_nine": k_per_nine,
                  "whip": 1.20, "k_percent": 0.28},
            source="fusion",
            components_fused=2,
            xwoba_override_detected=False,
        )

    def test_proj_hr_derived_from_rate(self):
        result = self._batter_result(hr_per_pa=0.05)
        counts = to_season_counts(result, projected_pa=600, projected_ip=0, board_proj={"r": 100, "rbi": 95})
        assert counts["proj_hr"] == 30  # 0.05 * 600

    def test_proj_sb_derived_from_rate(self):
        result = self._batter_result(sb_per_pa=0.02)
        counts = to_season_counts(result, projected_pa=500, projected_ip=0, board_proj={})
        assert counts["proj_sb"] == 10  # 0.02 * 500

    def test_proj_r_passthrough(self):
        result = self._batter_result()
        counts = to_season_counts(result, projected_pa=550, projected_ip=0, board_proj={"r": 85, "rbi": 72})
        assert counts["proj_r"] == 85

    def test_proj_rbi_passthrough(self):
        result = self._batter_result()
        counts = to_season_counts(result, projected_pa=550, projected_ip=0, board_proj={"r": 85, "rbi": 72})
        assert counts["proj_rbi"] == 72

    def test_proj_sv_passthrough(self):
        result = self._pitcher_result()
        counts = to_season_counts(result, projected_pa=0, projected_ip=70, board_proj={"sv": 35})
        assert counts["proj_sv"] == 35

    def test_proj_k_derived_from_k9(self):
        result = self._pitcher_result(k_per_nine=9.0)
        counts = to_season_counts(result, projected_pa=0, projected_ip=180, board_proj={})
        # 9.0 * 180 / 9 = 180
        assert counts["proj_k"] == 180

    def test_proj_w_uses_formula(self):
        result = self._pitcher_result(era=4.50)
        counts = to_season_counts(result, projected_pa=0, projected_ip=180, board_proj={})
        # At league-average ERA (4.50), wins ~ 50% of decisions
        # decisions = round(180 / 8.5) = 21; wins = round(21 * 0.50) = 11 (approx)
        assert 8 <= counts["proj_w"] <= 14

    def test_missing_board_keys_default_zero(self):
        result = self._batter_result()
        counts = to_season_counts(result, projected_pa=600, projected_ip=0, board_proj={})
        assert counts["proj_r"] == 0
        assert counts["proj_rbi"] == 0
        assert counts["proj_sv"] == 0

    def test_projected_hr_never_negative(self):
        result = self._batter_result(hr_per_pa=-0.01)  # bad data
        counts = to_season_counts(result, projected_pa=600, projected_ip=0, board_proj={})
        assert counts["proj_hr"] >= 0

    def test_none_rate_treated_as_zero(self):
        result = FusionResult(
            proj={"hr_per_pa": None, "sb_per_pa": None, "k_per_nine": None, "era": None},
            source="population_prior",
            components_fused=0,
            xwoba_override_detected=False,
        )
        counts = to_season_counts(result, projected_pa=500, projected_ip=100, board_proj={})
        assert counts["proj_hr"] == 0
        assert counts["proj_sb"] == 0
        assert counts["proj_k"] == 0
        assert counts["proj_w"] >= 0  # formula uses league-average ERA fallback


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
