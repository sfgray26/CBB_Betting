"""
Tests for backend/core/odds_math.py

Run with: pytest tests/test_odds_math.py -v
"""

import math
import pytest
from backend.core.odds_math import (
    american_to_decimal,
    implied_prob,
    decimal_to_american,
    remove_vig_shin,
    dynamic_sd,
    FALLBACK_SD,
    DEFAULT_SD_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# american_to_decimal
# ---------------------------------------------------------------------------


class TestAmericanToDecimal:
    def test_even_money_positive(self):
        assert american_to_decimal(100) == pytest.approx(2.0)

    def test_even_money_negative(self):
        assert american_to_decimal(-100) == pytest.approx(2.0)

    def test_standard_juice_minus110(self):
        # 100/110 + 1 = 1.9091
        assert american_to_decimal(-110) == pytest.approx(1.9091, abs=0.0001)

    def test_plus_150_underdog(self):
        # 150/100 + 1 = 2.5
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_minus_200_favorite(self):
        # 100/200 + 1 = 1.5
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_plus_300_long_underdog(self):
        # 300/100 + 1 = 4.0
        assert american_to_decimal(300) == pytest.approx(4.0)

    def test_result_always_above_one(self):
        for odds in [-110, -200, -500, 100, 150, 300]:
            assert american_to_decimal(odds) > 1.0

    def test_float_input_accepted(self):
        # Some API responses may return floats
        assert american_to_decimal(-110.0) == pytest.approx(1.9091, abs=0.0001)

    def test_below_magnitude_raises(self):
        with pytest.raises(ValueError, match="magnitude"):
            american_to_decimal(50)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            american_to_decimal(0)


# ---------------------------------------------------------------------------
# implied_prob
# ---------------------------------------------------------------------------


class TestImpliedProb:
    def test_minus110_implied_prob(self):
        # 110/210 ≈ 0.5238
        assert implied_prob(-110) == pytest.approx(0.5238, abs=0.0001)

    def test_plus150_implied_prob(self):
        assert implied_prob(150) == pytest.approx(0.4, abs=0.0001)

    def test_even_money_is_fifty_percent(self):
        assert implied_prob(100) == pytest.approx(0.5)
        assert implied_prob(-100) == pytest.approx(0.5)

    def test_two_sided_market_overround(self):
        # Standard -110 / -110 market: both sides sum to > 1 (vig)
        total = implied_prob(-110) + implied_prob(-110)
        assert total > 1.0

    def test_implied_prob_between_zero_and_one(self):
        for odds in [-500, -200, -110, 100, 150, 300]:
            p = implied_prob(odds)
            assert 0.0 < p < 1.0


# ---------------------------------------------------------------------------
# decimal_to_american
# ---------------------------------------------------------------------------


class TestDecimalToAmerican:
    def test_even_money(self):
        assert decimal_to_american(2.0) == 100

    def test_minus110_round_trip(self):
        dec = american_to_decimal(-110)
        result = decimal_to_american(dec)
        assert result == pytest.approx(-110, abs=1)

    def test_plus150_round_trip(self):
        dec = american_to_decimal(150)
        assert decimal_to_american(dec) == 150

    def test_minus200_round_trip(self):
        dec = american_to_decimal(-200)
        assert decimal_to_american(dec) == pytest.approx(-200, abs=1)

    def test_favourite_returns_negative(self):
        assert decimal_to_american(1.5) < 0

    def test_underdog_returns_positive(self):
        assert decimal_to_american(2.5) > 0

    def test_below_one_raises(self):
        with pytest.raises(ValueError, match="Decimal odds"):
            decimal_to_american(0.9)


# ---------------------------------------------------------------------------
# remove_vig_shin
# ---------------------------------------------------------------------------


class TestRemoveVigShin:
    def test_output_sums_to_one(self):
        p_a, p_b = remove_vig_shin(-110, -110)
        assert p_a + p_b == pytest.approx(1.0, abs=1e-9)

    def test_symmetric_market_equal_probs(self):
        # Both sides -110 → true probs should be equal (0.5 each)
        p_a, p_b = remove_vig_shin(-110, -110)
        assert p_a == pytest.approx(0.5, abs=0.001)
        assert p_b == pytest.approx(0.5, abs=0.001)

    def test_favourite_has_higher_true_prob(self):
        # -200 favourite vs +170 underdog
        p_fav, p_dog = remove_vig_shin(-200, 170)
        assert p_fav > p_dog

    def test_shin_assigns_less_than_raw_to_favourite(self):
        # After vig removal the true favourite probability must be below the raw
        # implied probability (which overstates the favourite due to vig).
        raw_fav = implied_prob(-200)
        shin_fav, _ = remove_vig_shin(-200, 170)
        assert shin_fav < raw_fav

    def test_true_probs_strictly_between_zero_and_one(self):
        for odds_a, odds_b in [(-110, -110), (-300, 250), (-150, 130)]:
            p_a, p_b = remove_vig_shin(odds_a, odds_b)
            assert 0.0 < p_a < 1.0
            assert 0.0 < p_b < 1.0

    def test_extreme_odds_no_exception(self):
        # Very lopsided market — should not raise
        p_a, p_b = remove_vig_shin(-900, 600)
        assert p_a + p_b == pytest.approx(1.0, abs=1e-6)
        assert p_a > p_b  # heavy favourite

    def test_invalid_odds_raises(self):
        with pytest.raises(ValueError):
            remove_vig_shin(50, -110)

    def test_both_invalid_raises(self):
        with pytest.raises(ValueError):
            remove_vig_shin(50, 50)

    def test_degenerate_overround_falls_back_gracefully(self):
        # When overround < 1.001 (e.g., mispriced +100/+100 market) it should
        # return proportional normalisation rather than raising.
        p_a, p_b = remove_vig_shin(100, 100)
        assert p_a + p_b == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# dynamic_sd
# ---------------------------------------------------------------------------


class TestDynamicSd:
    def test_average_d1_game(self):
        # 173.5 total → sqrt(173.5) * 0.85 ≈ 11.19
        result = dynamic_sd(173.5)
        assert result == pytest.approx(math.sqrt(173.5) * DEFAULT_SD_MULTIPLIER, rel=1e-6)

    def test_slow_pace_game(self):
        result = dynamic_sd(142.0)
        assert result == pytest.approx(math.sqrt(142.0) * 0.85, rel=1e-6)

    def test_fast_pace_game(self):
        result = dynamic_sd(185.0)
        assert result == pytest.approx(math.sqrt(185.0) * 0.85, rel=1e-6)

    def test_custom_multiplier(self):
        result = dynamic_sd(160.0, multiplier=0.90)
        assert result == pytest.approx(math.sqrt(160.0) * 0.90, rel=1e-6)

    def test_zero_total_returns_fallback(self):
        assert dynamic_sd(0.0) == pytest.approx(FALLBACK_SD)

    def test_negative_total_returns_fallback(self):
        assert dynamic_sd(-10.0) == pytest.approx(FALLBACK_SD)

    def test_sd_increases_with_total(self):
        # Higher totals → more variance
        assert dynamic_sd(150.0) < dynamic_sd(170.0) < dynamic_sd(190.0)

    def test_result_positive(self):
        for total in [130.0, 150.0, 165.0, 180.0]:
            assert dynamic_sd(total) > 0.0
