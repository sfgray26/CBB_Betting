"""
Unified Need-Score Service Tests

Tests for the unified need_score service that ensures:
1. Base need_score is identical across all endpoints
2. Statcast boost is calculated separately and transparently
3. Adjusted need_score = base + boost
4. Missing stats return graceful fallback (0.0 or base value, not NaN)

Created: Loop Iteration 11 (2026-06-25)
"""

import pytest
from backend.services.need_score import (
    NeedScoreCalculator,
    NeedScoreResult,
    calculate_need_score,
    get_need_score_calculator,
)


def test_need_score_calculator_singleton():
    """Test that get_need_score_calculator returns singleton instance."""
    calc1 = get_need_score_calculator()
    calc2 = get_need_score_calculator()
    assert calc1 is calc2
    assert isinstance(calc1, NeedScoreCalculator)


def test_base_need_score_with_category_deficits():
    """Test base need_score calculation with category deficits."""
    calculator = NeedScoreCalculator()

    # Player with strong HR/RBI skills
    player_cat_scores = {"hr": 1.2, "rbi": 0.8, "avg": 0.5, "runs": 0.3}
    player_z_score = 1.0

    # Team is weak in HR and RBI
    from backend.schemas import CategoryDeficitOut
    category_deficits = [
        CategoryDeficitOut(category="HR", my_total=45.0, opponent_total=55.0, deficit=10.0, winning=False),
        CategoryDeficitOut(category="RBI", my_total=120.0, opponent_total=128.0, deficit=8.0, winning=False),
    ]

    base_score = calculator.calculate_base_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=player_z_score,
        category_deficits=category_deficits,
        n_cats=5,
    )

    # Base score should be higher than z_score due to category alignment
    assert base_score >= 0.0
    assert isinstance(base_score, float)


def test_statcast_boost_buy_low():
    """Test Statcast boost for BUY_LOW signal."""
    calculator = NeedScoreCalculator()

    boost = calculator.calculate_statcast_boost(["BUY_LOW"])
    assert boost > 0.0  # BUY_LOW should add positive boost


def test_statcast_boost_breakout():
    """Test Statcast boost for BREAKOUT signal."""
    calculator = NeedScoreCalculator()

    boost = calculator.calculate_statcast_boost(["BREAKOUT"])
    assert boost > 0.0  # BREAKOUT should add positive boost


def test_statcast_boost_sell_high():
    """Test Statcast boost for SELL_HIGH signal."""
    calculator = NeedScoreCalculator()

    boost = calculator.calculate_statcast_boost(["SELL_HIGH"])
    assert boost < 0.0  # SELL_HIGH should reduce score


def test_statcast_boost_multiple_signals():
    """Test Statcast boost with multiple signals."""
    calculator = NeedScoreCalculator()

    boost = calculator.calculate_statcast_boost(["BUY_LOW", "BREAKOUT"])
    assert boost > 0.0  # Multiple positive signals should compound


def test_statcast_boost_empty_signals():
    """Test Statcast boost with no signals returns 0.0."""
    calculator = NeedScoreCalculator()

    boost = calculator.calculate_statcast_boost([])
    assert boost == 0.0


def test_adjusted_need_score_equals_base_plus_boost():
    """Test that adjusted_need_score = base_need_score + statcast_boost."""
    calculator = NeedScoreCalculator()

    base = 9.38
    boost = 0.4
    adjusted = calculator.calculate_adjusted_need_score(base, boost)

    # Use approx for floating point comparison
    assert adjusted == pytest.approx(base + boost)
    assert adjusted == pytest.approx(9.78)


def test_calculate_all_returns_complete_result():
    """Test that calculate_all returns NeedScoreResult with all components."""
    calculator = NeedScoreCalculator()

    player_cat_scores = {"hr": 1.0, "rbi": 0.5}
    player_z_score = 1.0

    result = calculator.calculate_all(
        player_cat_scores=player_cat_scores,
        player_z_score=player_z_score,
        category_deficits=[],
        n_cats=5,
        statcast_signals=["BUY_LOW"],
    )

    assert isinstance(result, NeedScoreResult)
    assert hasattr(result, "base_need_score")
    assert hasattr(result, "statcast_boost")
    assert hasattr(result, "adjusted_need_score")
    assert hasattr(result, "statcast_signals")


def test_adjusted_equals_base_plus_boost_in_result():
    """Test that NeedScoreResult.adjusted_need_score = base + boost."""
    calculator = NeedScoreCalculator()

    result = calculator.calculate_all(
        player_cat_scores={"hr": 1.0},
        player_z_score=1.0,
        category_deficits=[],
        n_cats=5,
        statcast_signals=["BUY_LOW"],
    )

    assert result.adjusted_need_score == result.base_need_score + result.statcast_boost


def test_convenience_function_returns_same_result():
    """Test that convenience function calculate_need_score returns same result as calculator."""
    player_cat_scores = {"hr": 1.0, "rbi": 0.5}
    player_z_score = 1.0
    statcast_signals = ["BREAKOUT"]

    # Using convenience function
    result1 = calculate_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=player_z_score,
        category_deficits=[],
        n_cats=5,
        statcast_signals=statcast_signals,
    )

    # Using calculator directly
    calculator = NeedScoreCalculator()
    result2 = calculator.calculate_all(
        player_cat_scores=player_cat_scores,
        player_z_score=player_z_score,
        category_deficits=[],
        n_cats=5,
        statcast_signals=statcast_signals,
    )

    assert result1.base_need_score == result2.base_need_score
    assert result1.statcast_boost == result2.statcast_boost
    assert result1.adjusted_need_score == result2.adjusted_need_score


def test_empty_category_deficits_fallback_to_z_score():
    """Test that empty category_deficits falls back to z_score."""
    calculator = NeedScoreCalculator()

    player_cat_scores = {"hr": 1.0}
    player_z_score = 1.5

    base_score = calculator.calculate_base_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=player_z_score,
        category_deficits=[],  # Empty deficits
        n_cats=5,
    )

    # Should fall back to z_score
    assert base_score == player_z_score


def test_missing_cat_scores_graceful_fallback():
    """Test that missing cat_scores returns valid float (not NaN)."""
    calculator = NeedScoreCalculator()

    base_score = calculator.calculate_base_need_score(
        player_cat_scores={},  # Empty cat_scores
        player_z_score=1.0,
        category_deficits=[],
        n_cats=5,
    )

    # Should return valid float, not NaN
    assert isinstance(base_score, float)
    assert not (base_score != base_score)  # NaN check: NaN != NaN is True


def test_statcast_boost_high_injury_risk():
    """Test Statcast boost for HIGH_INJURY_RISK signal."""
    calculator = NeedScoreCalculator()

    boost = calculator.calculate_statcast_boost(["HIGH_INJURY_RISK"])
    assert boost < 0.0  # HIGH_INJURY_RISK should reduce score


def test_call_count_increments():
    """Test that calculator increments call_count."""
    calculator = NeedScoreCalculator()

    initial_count = calculator._call_count
    calculator.calculate_all(
        player_cat_scores={"hr": 1.0},
        player_z_score=1.0,
        category_deficits=[],
        n_cats=5,
    )

    assert calculator._call_count == initial_count + 1


def test_base_need_score_identical_for_same_inputs():
    """Test that base need_score is identical for same inputs (idempotency)."""
    calculator = NeedScoreCalculator()

    player_cat_scores = {"hr": 1.0, "rbi": 0.5}
    player_z_score = 1.0
    category_deficits = []

    result1 = calculator.calculate_base_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=player_z_score,
        category_deficits=category_deficits,
        n_cats=5,
    )

    result2 = calculator.calculate_base_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=player_z_score,
        category_deficits=category_deficits,
        n_cats=5,
    )

    assert result1 == result2
