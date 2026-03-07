"""Tests for A-26 Seed-Spread Kelly Scalars."""

import pytest
from unittest.mock import patch

from backend.betting_model import CBBEdgeModel


class TestSeedSpreadScalars:
    """Test seed-spread Kelly scalar calculation."""

    def setup_method(self):
        """Initialize model for each test."""
        self.model = CBBEdgeModel()

    # ------------------------------------------------------------------
    # Rule 1: #5 seed favored by 6+ points -> 0.75x
    # ------------------------------------------------------------------

    def test_seed5_favored_6_plus_home(self):
        """Home team is #5 seed favored by 6.5 -> 0.75x"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=-6.5
        )
        assert result == 0.75

    def test_seed5_favored_6_plus_away(self):
        """Away team is #5 seed favored by 7 -> 0.75x"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=12, away_seed=5, spread=7.0
        )
        assert result == 0.75

    def test_seed5_favored_5_points_no_scalar(self):
        """#5 seed favored by only 5 -> no scalar (1.0x)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=-5.0
        )
        assert result == 1.0

    # ------------------------------------------------------------------
    # Rule 2: #2 seed favored by 17+ points -> 0.75x
    # ------------------------------------------------------------------

    def test_seed2_favored_17_plus_home(self):
        """Home team is #2 seed favored by 17.5 -> 0.75x"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=2, away_seed=15, spread=-17.5
        )
        assert result == 0.75

    def test_seed2_favored_17_plus_away(self):
        """Away team is #2 seed favored by 18 -> 0.75x"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=15, away_seed=2, spread=18.0
        )
        assert result == 0.75

    def test_seed2_favored_16_points_no_scalar(self):
        """#2 seed favored by only 16 -> no scalar (1.0x)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=2, away_seed=15, spread=-16.0
        )
        assert result == 1.0

    # ------------------------------------------------------------------
    # Rule 3: #8 seed favored by <=3 points -> 0.80x
    # ------------------------------------------------------------------

    def test_seed8_favored_3_points_home(self):
        """Home team is #8 seed favored by 3 -> 0.80x"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=8, away_seed=9, spread=-3.0
        )
        assert result == 0.80

    def test_seed8_favored_2_points_away(self):
        """Away team is #8 seed favored by 2.5 -> 0.80x"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=9, away_seed=8, spread=2.5
        )
        assert result == 0.80

    def test_seed8_favored_4_points_no_scalar(self):
        """#8 seed favored by 4 -> no scalar (1.0x)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=8, away_seed=9, spread=-4.0
        )
        assert result == 1.0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_no_seed_data(self):
        """Missing seed data -> 1.0x (no scalar)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=None, away_seed=5, spread=-6.5
        )
        assert result == 1.0

    def test_no_spread_data(self):
        """Missing spread -> 1.0x (no scalar)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=None
        )
        assert result == 1.0

    def test_pick_em_no_favorite(self):
        """Pick'em spread (0) -> 1.0x (no scalar)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=0.0
        )
        assert result == 1.0

    def test_underdog_seed_no_scalar(self):
        """Seed conditions only apply to FAVORITE"""
        # #5 seed as underdog (getting points) -> no scalar
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=4, spread=6.5  # Home is underdog
        )
        assert result == 1.0

    # ------------------------------------------------------------------
    # Env var overrides
    # ------------------------------------------------------------------

    @patch.dict('os.environ', {'SEED5_FAV6PLUS_SCALAR': '0.60'})
    def test_env_override_seed5(self):
        """SEED5_FAV6PLUS_SCALAR env var overrides default"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=-6.5
        )
        assert result == 0.60

    @patch.dict('os.environ', {'SEED2_FAV17PLUS_SCALAR': '0.50'})
    def test_env_override_seed2(self):
        """SEED2_FAV17PLUS_SCALAR env var overrides default"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=2, away_seed=15, spread=-17.5
        )
        assert result == 0.50

    @patch.dict('os.environ', {'SEED8_FAV3MINUS_SCALAR': '0.70'})
    def test_env_override_seed8(self):
        """SEED8_FAV3MINUS_SCALAR env var overrides default"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=8, away_seed=9, spread=-3.0
        )
        assert result == 0.70
