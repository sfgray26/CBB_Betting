"""
Tests for backend/core/kelly.py

Run with: pytest tests/test_kelly.py -v
"""

import pytest
from backend.core.kelly import (
    kelly_fraction,
    kelly_fraction_with_push,
    portfolio_kelly_divisor,
    kelly_to_units,
    units_to_dollars,
    MAX_KELLY_FRACTION,
    MIN_KELLY_FRACTION,
)


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------


class TestKellyFraction:
    def test_standard_half_kelly_on_minus110(self):
        # win_prob=0.55, decimal_odds=1.909 (≈ -110)
        # full_kelly = (0.55*0.909 - 0.45)/0.909 ≈ 0.0550; half = ≈ 0.0275
        result = kelly_fraction(0.55, 1.909)
        assert result == pytest.approx(0.0275, abs=0.001)

    def test_positive_edge_underdog(self):
        # win_prob=0.60, decimal_odds=2.10 → healthy positive result
        result = kelly_fraction(0.60, 2.10)
        assert result > 0.0

    def test_negative_ev_returns_zero(self):
        # win_prob=0.45 at -110 → negative edge, should return 0
        assert kelly_fraction(0.45, 1.909) == 0.0

    def test_zero_edge_returns_zero(self):
        # win_prob=0.5238 is the breakeven for -110 (≈ 1/1.909)
        breakeven = 1.0 / 1.909
        assert kelly_fraction(breakeven, 1.909) == pytest.approx(0.0, abs=1e-4)

    def test_fractional_divisor_scales_linearly(self):
        # Quarter-Kelly should be half of half-Kelly
        half = kelly_fraction(0.55, 1.909, fractional_divisor=2.0)
        quarter = kelly_fraction(0.55, 1.909, fractional_divisor=4.0)
        assert quarter == pytest.approx(half / 2.0, rel=1e-6)

    def test_max_fraction_cap(self):
        # Even very high edge bets should be capped at MAX_KELLY_FRACTION
        result = kelly_fraction(0.99, 2.0, max_fraction=MAX_KELLY_FRACTION)
        assert result <= MAX_KELLY_FRACTION

    def test_custom_max_fraction_respected(self):
        result = kelly_fraction(0.70, 2.0, max_fraction=0.05)
        assert result <= 0.05

    def test_even_money_positive_edge(self):
        # win_prob=0.55 at even money (+100, decimal=2.0)
        result = kelly_fraction(0.55, 2.0)
        assert result > 0.0

    # --- Input validation ---

    def test_win_prob_zero_raises(self):
        with pytest.raises(ValueError, match="win_prob"):
            kelly_fraction(0.0, 1.909)

    def test_win_prob_one_raises(self):
        with pytest.raises(ValueError, match="win_prob"):
            kelly_fraction(1.0, 1.909)

    def test_win_prob_negative_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(-0.1, 1.909)

    def test_win_prob_above_one_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(1.1, 1.909)

    def test_decimal_odds_below_one_raises(self):
        with pytest.raises(ValueError, match="decimal_odds"):
            kelly_fraction(0.55, 0.9)

    def test_decimal_odds_exactly_one_raises(self):
        # Odds of 1.0 means zero profit per unit → ZeroDivisionError in the formula
        with pytest.raises((ZeroDivisionError, ValueError)):
            kelly_fraction(0.55, 1.0)


# ---------------------------------------------------------------------------
# kelly_fraction_with_push
# ---------------------------------------------------------------------------


class TestKellyFractionWithPush:
    def test_push_increases_fraction_vs_standard(self):
        # Push absorbs mass from the loss side — fraction should be >= standard Kelly
        no_push = kelly_fraction(0.54, 1.909)
        with_push = kelly_fraction_with_push(0.54, 0.08, 1.909)
        assert with_push >= no_push

    def test_zero_push_prob_matches_standard_kelly(self):
        std = kelly_fraction(0.55, 1.909)
        push_zero = kelly_fraction_with_push(0.55, 0.0, 1.909)
        assert push_zero == pytest.approx(std, rel=1e-6)

    def test_negative_ev_with_push_returns_zero(self):
        assert kelly_fraction_with_push(0.40, 0.05, 1.909) == 0.0

    def test_max_fraction_cap_applies(self):
        result = kelly_fraction_with_push(0.90, 0.05, 2.0)
        assert result <= MAX_KELLY_FRACTION

    def test_typical_integer_spread_scenario(self):
        # -7 spread, 54% cover, 8% push, -110 juice → should be positive fraction
        result = kelly_fraction_with_push(0.54, 0.08, 1.909)
        assert result > 0.0

    # --- Validation ---

    def test_win_prob_zero_raises(self):
        with pytest.raises(ValueError, match="win_prob"):
            kelly_fraction_with_push(0.0, 0.05, 1.909)

    def test_push_prob_negative_raises(self):
        with pytest.raises(ValueError, match="push_prob"):
            kelly_fraction_with_push(0.55, -0.01, 1.909)

    def test_push_prob_one_raises(self):
        with pytest.raises(ValueError, match="push_prob"):
            kelly_fraction_with_push(0.55, 1.0, 1.909)

    def test_win_plus_push_eq_one_raises(self):
        with pytest.raises(ValueError, match="win_prob.*push_prob|push_prob.*win_prob"):
            kelly_fraction_with_push(0.70, 0.30, 1.909)

    def test_win_plus_push_above_one_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction_with_push(0.80, 0.30, 1.909)


# ---------------------------------------------------------------------------
# portfolio_kelly_divisor
# ---------------------------------------------------------------------------


class TestPortfolioKellyDivisor:
    def test_no_exposure_returns_base_divisor(self):
        assert portfolio_kelly_divisor(0.0) == pytest.approx(2.0)

    def test_half_exposure_returns_midpoint(self):
        # e=0.075 with target=0.15 → divisor = 2*(1+0.5) = 3.0
        assert portfolio_kelly_divisor(0.075) == pytest.approx(3.0)

    def test_full_exposure_returns_max_divisor(self):
        # e=0.15 → divisor = 2*(1+1.0) = 4.0
        assert portfolio_kelly_divisor(0.15) == pytest.approx(4.0)

    def test_over_exposure_capped_at_max(self):
        # e=0.30 would give 6.0 without cap; should be capped at 4.0
        assert portfolio_kelly_divisor(0.30) == pytest.approx(4.0)

    def test_large_exposure_capped(self):
        assert portfolio_kelly_divisor(1.0) == pytest.approx(4.0)

    def test_custom_base_and_max_divisor(self):
        result = portfolio_kelly_divisor(0.0, base_divisor=3.0, max_divisor=6.0)
        assert result == pytest.approx(3.0)

    def test_divisor_monotonically_increases_with_exposure(self):
        exposures = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.20]
        divisors = [portfolio_kelly_divisor(e) for e in exposures]
        for i in range(len(divisors) - 1):
            assert divisors[i] <= divisors[i + 1]

    # --- Validation ---

    def test_negative_exposure_raises(self):
        with pytest.raises(ValueError, match="concurrent_exposure"):
            portfolio_kelly_divisor(-0.01)

    def test_zero_target_raises(self):
        with pytest.raises(ValueError, match="target_exposure"):
            portfolio_kelly_divisor(0.05, target_exposure=0.0)

    def test_negative_target_raises(self):
        with pytest.raises(ValueError):
            portfolio_kelly_divisor(0.05, target_exposure=-0.1)


# ---------------------------------------------------------------------------
# Utility conversions
# ---------------------------------------------------------------------------


class TestKellyToUnits:
    def test_fraction_to_units(self):
        assert kelly_to_units(0.025) == pytest.approx(2.5)

    def test_zero_fraction(self):
        assert kelly_to_units(0.0) == pytest.approx(0.0)

    def test_max_fraction_to_units(self):
        assert kelly_to_units(MAX_KELLY_FRACTION) == pytest.approx(20.0)

    def test_small_fraction(self):
        assert kelly_to_units(MIN_KELLY_FRACTION) == pytest.approx(0.01)


class TestUnitsToDollars:
    def test_basic_conversion(self):
        assert units_to_dollars(2.5, 1000.0) == pytest.approx(25.0)

    def test_zero_units(self):
        assert units_to_dollars(0.0, 1000.0) == pytest.approx(0.0)

    def test_scaling_with_bankroll(self):
        assert units_to_dollars(1.0, 5000.0) == pytest.approx(50.0)

    def test_half_unit(self):
        assert units_to_dollars(0.5, 5000.0) == pytest.approx(25.0)
