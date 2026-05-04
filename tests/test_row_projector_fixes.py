"""Tests for ROW Projector mathematical fixes (P0-1, P0-2)"""
import pytest
from datetime import date
from backend.services.row_projector import compute_row_projection, _days_into_season


class TestP01DynamicSeasonDays:
    """P0-1: Season days should be computed dynamically, not hardcoded to 100"""

    def test_days_into_season_opening_day(self):
        """Opening day should return 1"""
        assert _days_into_season(date(2026, 3, 27)) == 1

    def test_days_into_season_day_20(self):
        """Day 20 of season should return 20"""
        assert _days_into_season(date(2026, 4, 15)) == 20

    def test_days_into_season_before_opening_day(self):
        """Before opening day should return 1 (not negative)"""
        assert _days_into_season(date(2026, 3, 1)) == 1

    def test_day_20_projection_no_bias(self):
        """
        Day 20: 60 K over 20 IP should project correctly.
        Buggy version would divide by 100, causing 5x undervaluation.
        """
        rolling_stats = {
            "player_123": {
                "w_strikeouts_pit": 42.0,  # 3.0 K/day over 14 days
                "w_ip": 14.0,              # 1.0 IP/day over 14 days
            }
        }
        season_stats = {
            "player_123": {
                "strikeouts_pit": 60.0,    # 3.0 K/day over 20 days
            }
        }

        result = compute_row_projection(
            rolling_stats_by_player=rolling_stats,
            season_stats_by_player=season_stats,
            games_remaining={"player_123": 1},
            as_of_date=date(2026, 4, 15),
        )

        # Season rate = 60 / 20 = 3.0 K/day (NOT 60/100 = 0.6)
        # Blended = 0.6 × (42/14) + 0.4 × 3.0 = 1.8 + 1.2 = 3.0
        # K_P projection for 1 game ≈ 3.0

        # Verify we're not using buggy 100-day denominator
        # Buggy would give: 0.6 × 3.0 + 0.4 × 0.6 = 2.04
        assert result.K_P > 2.5, f"K_P={result.K_P} too low (buggy 100-day denom?)"
        assert result.K_P < 3.5, f"K_P={result.K_P} too high"

    def test_mid_season_projection_accuracy(self):
        """Day 80: projections should use 80, not 100"""
        # Day 80 of 2026 season = June 14
        as_of = date(2026, 6, 14)

        rolling_stats = {
            "player_456": {
                "w_strikeouts_pit": 63.0,  # 4.5 K/day
                "w_ip": 21.0,              # 1.5 IP/day
            }
        }
        season_stats = {
            "player_456": {
                "strikeouts_pit": 320.0,   # 4.0 K/day over 80 days
            }
        }

        result = compute_row_projection(
            rolling_stats_by_player=rolling_stats,
            season_stats_by_player=season_stats,
            games_remaining={"player_456": 1},
            as_of_date=as_of,
        )

        # Season rate = 320 / 80 = 4.0 (NOT 320/100 = 3.2)
        # Blended = 0.6 × 4.5 + 0.4 × 4.0 = 2.7 + 1.6 = 4.3
        assert result.K_P > 4.0, f"K_P={result.K_P} suggests 100-day denom"


class TestP02OPSFormula:
    """P0-2: OPS should include HBP and SF in OBP calculation"""

    def test_ops_includes_hbp_sf(self):
        """
        Verify HBP and SF are included in OBP calculation.

        Player profile: 350 AB season, 100 H, 35 BB
        Expected HBP: 350 × 0.0067 ≈ 2.345
        Expected SF:  350 × 0.0083 ≈ 2.905

        OBP (no HBP/SF) = (100+35) / 350 = 0.386
        OBP (with HBP/SF) = (100+35+2.345) / (350+35+2.345+2.905) = 137.345 / 390.25 = 0.352
        """
        rolling_stats = {
            "player_789": {
                "w_ab": 350.0,      # 25 AB/day over 14 days
                "w_hits": 100.0,    # .286 average
                "w_tb": 160.0,      # .457 SLG
                "w_walks": 35.0,    # 2.5 BB/day
            }
        }

        result = compute_row_projection(
            rolling_stats_by_player=rolling_stats,
            games_remaining={"player_789": 1},
        )

        # Calculate expected values with HBP/SF
        sum_h = 100.0
        sum_bb = 35.0
        sum_ab = 350.0
        sum_hbp = sum_ab * 0.0067
        sum_sf = sum_ab * 0.0083

        expected_obp = (sum_h + sum_bb + sum_hbp) / (sum_ab + sum_bb + sum_hbp + sum_sf)
        expected_slg = 160.0 / sum_ab
        expected_ops = expected_obp + expected_slg

        # Verify OPS is calculated with HBP/SF
        assert result.OPS > 0, f"OPS should be positive: {result.OPS}"

        # Allow small tolerance for floating point
        assert abs(result.OPS - expected_ops) < 0.01, \
            f"OPS {result.OPS} != expected {expected_ops} (HBP/SF missing?)"

    def test_hbp_sf_constants_sane(self):
        """HBP and SF constants should be within MLB range"""
        from backend.services.row_projector import _MLB_HBP_PER_AB, _MLB_SF_PER_AB

        # MLB ranges: 4-6 HBP per 600 AB, 4-6 SF per 600 AB
        assert 0.005 < _MLB_HBP_PER_AB < 0.010, \
            f"HBP rate {_MLB_HBP_PER_AB} outside MLB range"
        assert 0.005 < _MLB_SF_PER_AB < 0.010, \
            f"SF rate {_MLB_SF_PER_AB} outside MLB range"

    def test_ops_calculation_components(self):
        """Verify OPS = OBP + SLG with proper HBP/SF handling"""
        rolling_stats = {
            "player_ops": {
                "w_ab": 400.0,
                "w_hits": 120.0,    # .300 average
                "w_tb": 200.0,      # .500 SLG
                "w_walks": 40.0,    # .100 BB rate
            }
        }

        result = compute_row_projection(
            rolling_stats_by_player=rolling_stats,
            games_remaining={"player_ops": 1},
        )

        # SLG should be TB / AB = 200 / 400 = 0.500
        expected_slg = 200.0 / 400.0

        # OBP with HBP/SF imputation
        sum_hbp = 400.0 * 0.0067
        sum_sf = 400.0 * 0.0083
        expected_obp = (120 + 40 + sum_hbp) / (400 + 40 + sum_hbp + sum_sf)

        expected_ops = expected_obp + expected_slg

        assert abs(result.OPS - expected_ops) < 0.01, \
            f"OPS calculation incorrect: got {result.OPS}, expected {expected_ops}"
