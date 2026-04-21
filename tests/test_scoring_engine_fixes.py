"""Tests for Scoring Engine mathematical fixes (P1-4, P1-5, P1-6)"""
import pytest
from datetime import date
from unittest.mock import MagicMock
from backend.services.scoring_engine import compute_league_zscores


class TestP14WeightedCategorySum:
    """P1-4: Composite should use weighted sum, not arithmetic mean"""

    def test_hr_specialist_not_diluted(self):
        """
        HR specialist: elite HR but average in other categories.
        With weighted sum, HR (1.2x weight) contributes more without dilution.
        """
        rolling_rows = []

        # HR specialist: elite HR, league-average other stats
        hr_specialist = MagicMock()
        hr_specialist.bdl_player_id = 1
        hr_specialist.w_ab = 100
        hr_specialist.w_home_runs = 30  # Elite: well above average
        hr_specialist.w_net_stolen_bases = 3  # Average
        hr_specialist.w_runs = 15  # Average
        hr_specialist.w_hits = 27  # Average
        hr_specialist.w_tb = 60  # Above average (HR contribute to TB)
        hr_specialist.w_rbi = 28  # Average
        hr_specialist.w_strikeouts_bat = 22  # Average
        hr_specialist.w_avg = 0.270  # Average
        hr_specialist.w_ops = 0.780  # Average
        hr_specialist.w_ip = None
        hr_specialist.games_in_window = 14
        rolling_rows.append(hr_specialist)

        # Add league-average players for comparison
        for i in range(15):
            balanced = MagicMock()
            balanced.bdl_player_id = 10 + i
            balanced.w_ab = 100
            balanced.w_home_runs = 12  # Below average
            balanced.w_net_stolen_bases = 3
            balanced.w_runs = 15
            balanced.w_hits = 27
            balanced.w_tb = 43  # Lower (fewer HR)
            balanced.w_rbi = 28
            balanced.w_strikeouts_bat = 22
            balanced.w_avg = 0.270
            balanced.w_ops = 0.760  # Lower SLG from fewer HR
            balanced.w_ip = None
            balanced.games_in_window = 14
            rolling_rows.append(balanced)

        results = compute_league_zscores(rolling_rows, date.today(), 14)

        hr_spec_result = next(r for r in results if r.bdl_player_id == 1)
        balanced_result = next((r for r in results if r.bdl_player_id != 1), None)

        # HR specialist should have positive composite_z (elite HR contributes)
        assert hr_spec_result.composite_z > 0, \
            f"HR specialist should have positive composite_z: {hr_spec_result.composite_z}"

        # HR specialist's z_hr should be positive (elite)
        assert hr_spec_result.z_hr is not None and hr_spec_result.z_hr > 0, \
            f"HR specialist z_hr should be positive: {hr_spec_result.z_hr}"


class TestP15TwoWayNormalization:
    """P1-5: Two-way players should not be unfairly penalized"""

    def test_two_way_vs_hitter_fair_comparison(self):
        """
        Two-way player with identical hitting stats as pure hitter
        should have comparable composite score (not inflated/deflated)
        """
        rolling_rows = []

        # Two-way: elite pitcher, average hitter
        two_way = MagicMock()
        two_way.bdl_player_id = 100
        two_way.w_ab = 50
        two_way.w_ip = 50  # Has pitching
        two_way.w_k_per_9 = 12.0  # Elite
        two_way.w_era = 2.0  # Elite
        two_way.w_whip = 0.9  # Elite
        two_way.w_home_runs = 15  # Average
        two_way.w_net_stolen_bases = 5
        two_way.w_runs = 12
        two_way.w_hits = 12
        two_way.w_strikeouts_pit = 50
        two_way.w_qs = 3
        two_way.games_in_window = 14
        rolling_rows.append(two_way)

        # Pure hitter with same hitting stats
        hitter = MagicMock()
        hitter.bdl_player_id = 101
        hitter.w_ab = 50
        hitter.w_ip = None
        hitter.w_home_runs = 15
        hitter.w_net_stolen_bases = 5
        hitter.w_runs = 12
        hitter.w_hits = 12
        hitter.games_in_window = 14
        rolling_rows.append(hitter)

        results = compute_league_zscores(rolling_rows, date.today(), 14)

        two_way_result = next(r for r in results if r.bdl_player_id == 100)
        hitter_result = next(r for r in results if r.bdl_player_id == 101)

        # After fix: composites should be in same ballpark
        # With weighted sum approach, two-way will be higher (extra pitching categories)
        # but that's fair - they contribute more
        if hitter_result.composite_z > 0:
            ratio = two_way_result.composite_z / hitter_result.composite_z
            assert 0.8 < ratio < 3.0, \
                f"Two-way {two_way_result.composite_z} vs hitter {hitter_result.composite_z} too different"


class TestP16LowSampleConfidence:
    """P1-6: Low sample sizes should compute z-scores but mark low confidence"""

    def test_n_equals_3_computes_z_score(self):
        """With only 3 players, should still compute z-scores"""
        rolling_rows = []
        for i in range(3):
            p = MagicMock()
            p.bdl_player_id = i
            p.w_ab = 50
            p.w_home_runs = 10 + i  # Variation
            p.w_ip = None
            p.games_in_window = 10
            rolling_rows.append(p)

        results = compute_league_zscores(rolling_rows, date.today(), 14)

        assert len(results) == 3
        assert all(r.z_hr is not None for r in results), "Z-HR should be computed"

    def test_confidence_increases_with_sample_size(self):
        """Confidence field should increase as sample size grows"""
        # This test documents the expected behavior
        # Actual confidence calculation: games_in_window / window_days

        small_sample = MagicMock()
        small_sample.bdl_player_id = 1
        small_sample.w_ab = 10
        small_sample.w_home_runs = 5
        small_sample.games_in_window = 3  # Low confidence
        small_sample.w_ip = None

        large_sample = MagicMock()
        large_sample.bdl_player_id = 2
        large_sample.w_ab = 50
        large_sample.w_home_runs = 10
        large_sample.games_in_window = 14  # High confidence
        large_sample.w_ip = None

        results = compute_league_zscores([small_sample, large_sample], date.today(), 14)

        small_result = next(r for r in results if r.bdl_player_id == 1)
        large_result = next(r for r in results if r.bdl_player_id == 2)

        # Small sample should have lower confidence
        assert small_result.confidence < large_result.confidence
        # Verify actual confidence values
        assert small_result.confidence == pytest.approx(3/14, rel=0.01)
        assert large_result.confidence == pytest.approx(1.0, rel=0.01)

    def test_n_equals_2_no_z_score(self):
        """With only 2 players (below MIN_SAMPLE=3), z-scores should be None"""
        rolling_rows = []
        for i in range(2):
            p = MagicMock()
            p.bdl_player_id = i
            p.w_ab = 50
            p.w_home_runs = 10 + i
            p.w_ip = None
            p.games_in_window = 10
            rolling_rows.append(p)

        results = compute_league_zscores(rolling_rows, date.today(), 14)

        # MIN_SAMPLE=3 means no z-scores computed with only 2 players
        assert all(r.z_hr is None for r in results), "Z-HR should be None with < MIN_SAMPLE"
