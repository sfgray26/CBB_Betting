"""
Integration tests for Phase 2: Rolling Stats → ROW Projection → Category Math.

These tests verify the full end-to-end pipeline:
1. Rolling stats feed into ROW projection
2. ROW projections feed into category math
3. All 18 scoring categories flow through correctly
"""

import pytest

from backend.services.row_projector import ROWProjectionResult, compute_row_projection
from backend.services.category_math import (
    CategoryMathResult,
    compute_all_category_math,
    compute_category_math,
)
from backend.contracts import CategoryStats, CategoryMathSummary
from backend.stat_contract import SCORING_CATEGORY_CODES


# ===========================================================================
# Test: ROW projection feeds category math correctly
# ===========================================================================


def test_row_to_category_math_counting_stats():
    """ROW projection for counting stats flows into category math."""
    # Setup: One hitter with 14 runs over 14 days, 1 game remaining
    rolling = {"player_123": {"w_runs": 14.0}}  # 1 run/day
    games = {"player_123": 1}

    # Compute ROW projection
    row_result = compute_row_projection(rolling, games_remaining=games)

    # Projected runs = 0.6 (blended rate × 1 game)
    assert row_result.R == pytest.approx(0.6, rel=1e-6)

    # Feed into category math
    my_finals = {"R": row_result.R}
    opp_finals = {"R": 0.0}

    math_result = compute_category_math(
        canonical_code="R",
        my_final=my_finals["R"],
        opp_final=opp_finals["R"],
    )

    # I'm projected to win the runs category
    assert math_result.is_winning is True
    assert math_result.margin > 0
    # delta_to_flip can be positive even when winning (safety margin)
    assert math_result.delta_to_flip >= 0


def test_row_to_category_math_ratio_stats():
    """ROW projection for ratio stats (AVG, ERA) flows into category math."""
    # Setup: One pitcher with ERA data
    rolling = {"player_789": {"w_earned_runs": 14.0, "w_ip": 7.0}}  # 7 IP, 14 ER
    games = {"player_789": 1}

    # Compute ROW projection
    row_result = compute_row_projection(rolling, games_remaining=games)

    # ERA = 27 * ER_daily / IP_outs_daily = 27 * 0.6 / 0.9 = 18.0
    assert row_result.ERA == pytest.approx(18.0, rel=0.01)

    # Feed into category math with numerators/denominators for ratio stat
    my_finals = {"ERA": row_result.ERA}
    opp_finals = {"ERA": 4.50}  # Opponent has better ERA

    math_result = compute_category_math(
        canonical_code="ERA",
        my_final=my_finals["ERA"],
        opp_final=opp_finals["ERA"],
        my_numerator=0.6,  # ER_daily (from row projection)
        my_denominator=0.9,  # IP_outs_daily (from row projection)
    )

    # I'm losing the ERA category (lower is better, my ERA is higher)
    assert math_result.is_winning is False
    assert math_result.margin < 0


# ===========================================================================
# Test: Full pipeline with all 18 categories
# ===========================================================================


def test_full_pipeline_all_18_categories():
    """Complete integration test: rolling stats → ROW → category math for all 18 categories."""
    # Setup: Mixed roster with hitters and pitchers
    rolling = {
        # Hitter
        "hitter_123": {
            "w_runs": 14.0,  # 1 run/day
            "w_hits": 28.0,  # 2 hits/day
            "w_home_runs": 7.0,  # 0.5 HR/day
            "w_rbi": 21.0,  # 1.5 RBI/day
            "w_strikeouts_bat": 14.0,  # 1 K/day
            "w_tb": 42.0,  # 3 TB/day
            "w_net_stolen_bases": 3.5,  # 0.25 NSB/day
            "w_ab": 100.0,  # ~7.14 AB/day
            "w_walks": 14.0,  # 1 BB/day
        },
        # Pitcher
        "pitcher_789": {
            "w_strikeouts_pit": 42.0,  # 3 K/day
            "w_qs": 1.0,  # 1 QS over 14 days
            "w_earned_runs": 14.0,  # 1 ER/day
            "w_ip": 7.0,  # 0.5 IP/day
            "w_hits_allowed": 21.0,  # 1.5 H/day
            "w_walks_allowed": 7.0,  # 0.5 BB/day
        },
    }
    games = {"hitter_123": 2, "pitcher_789": 1}

    # Compute ROW projections
    row_result = compute_row_projection(rolling, games_remaining=games)

    # Verify all 18 categories have projections (greenfield = 0.0)
    row_dict = row_result.to_dict()
    assert set(row_dict.keys()) == SCORING_CATEGORY_CODES

    # Greenfield categories are zero
    assert row_dict["W"] == 0.0
    assert row_dict["L"] == 0.0
    assert row_dict["HR_P"] == 0.0
    assert row_dict["NSV"] == 0.0

    # Hitter projections exist
    assert row_dict["R"] > 0
    assert row_dict["H"] > 0
    assert row_dict["HR_B"] > 0

    # Pitcher projections exist
    assert row_dict["K_P"] > 0
    assert row_dict["QS"] > 0
    assert row_dict["ERA"] > 0

    # Feed into category math with opponent projections
    opp_finals = {code: 0.0 for code in SCORING_CATEGORY_CODES}

    # For ratio stats, need numerators/denominators
    my_numerators = {
        "AVG": row_dict["H"] * 2,  # Rough approximation
        "OPS": row_dict["H"] * 2 + row_dict["TB"],  # Rough
        "ERA": row_dict["ERA"] * row_dict["W"],  # ER_total
        "WHIP": row_dict["ERA"] * row_dict["W"],  # Rough
        "K_9": row_dict["K_P"],
    }
    my_denominators = {
        "AVG": row_dict["H"] * 8,  # AB_total
        "OPS": row_dict["H"] * 8,  # AB_total
        "ERA": 1.0,  # IP_total
        "WHIP": 1.0,  # IP_total
        "K_9": 1.0,  # IP_total
    }

    # Compute category math for all categories
    math_results = compute_all_category_math(
        my_finals=row_dict,
        opp_finals=opp_finals,
        my_numerators=my_numerators,
        my_denominators=my_denominators,
    )

    # Verify all 18 categories have results
    assert set(math_results.keys()) == SCORING_CATEGORY_CODES

    # Verify result structure
    for code, result in math_results.items():
        assert isinstance(result, CategoryMathResult)
        assert result.canonical_code == code
        assert isinstance(result.margin, float)
        assert isinstance(result.delta_to_flip, (int, float))
        assert isinstance(result.is_winning, bool)


# ===========================================================================
# Test: Contract integration
# ===========================================================================


def test_category_stats_contract_validates_row_projection():
    """CategoryStats contract validates ROW projection output."""
    row_dict = ROWProjectionResult(
        R=10.0, H=25.0, HR_B=5.0, RBI=20.0, K_B=30.0, TB=40.0, NSB=3.0,
        AVG=0.280, OPS=0.850,
        W=3.0, L=2.0, HR_P=8.0, K_P=60.0, QS=4.0, NSV=-1.0,
        ERA=3.50, WHIP=1.20, K_9=10.5,
    ).to_dict()

    # Should create valid CategoryStats
    stats = CategoryStats(values=row_dict)
    assert len(stats.values) == 18
    assert stats.values["R"] == 10.0


def test_category_stats_contract_rejects_missing_categories():
    """CategoryStats contract rejects partial ROW projection (missing categories)."""
    partial_dict = {"R": 10.0, "H": 25.0}  # Only 2 of 18

    with pytest.raises(ValueError, match="Missing scoring categories"):
        CategoryStats(values=partial_dict)


def test_category_math_summary_contract():
    """CategoryMathSummary contract wraps category math results."""
    # Note: CategoryMathResult from category_math module is a dataclass
    # The contract uses the Pydantic model, so we convert
    from backend.contracts import CategoryMathResult as ContractCategoryMathResult

    contract_results = {
        "R": ContractCategoryMathResult(canonical_code="R", margin=5.0, delta_to_flip=0.0, is_winning=True),
        "ERA": ContractCategoryMathResult(canonical_code="ERA", margin=-0.50, delta_to_flip=2.0, is_winning=False),
    }

    summary = CategoryMathSummary(
        results=contract_results,
        categories_won=1,
        categories_lost=1,
        categories_tied=0,
    )

    assert len(summary.results) == 2
    assert summary.categories_won == 1
    assert summary.categories_lost == 1


# ===========================================================================
# Test: Lower-is-better categories
# ===========================================================================


def test_lower_is_better_margin_calculation():
    """Margin calculation respects lower-is-better categories."""
    # ERA: lower is better
    # My ERA = 3.50, Opp ERA = 4.00 → I'm winning
    result = compute_category_math(
        canonical_code="ERA",
        my_final=3.50,
        opp_final=4.00,
        my_numerator=3.5,  # ER_total (roughly)
        my_denominator=9.0,  # IP_outs (3 IP)
    )

    assert result.is_winning is True
    assert result.margin > 0  # Positive margin means I'm winning


def test_lower_is_better_swap():
    """Swap values shows reverse outcome for lower-is-better."""
    # ERA: My ERA = 4.50, Opp ERA = 3.50 → I'm losing
    result = compute_category_math(
        canonical_code="ERA",
        my_final=4.50,
        opp_final=3.50,
        my_numerator=4.5,
        my_denominator=9.0,
    )

    assert result.is_winning is False
    assert result.margin < 0


# ===========================================================================
# Test: Tied categories
# ===========================================================================


def test_tied_category_returns_zero_margin():
    """Exact tie returns margin = 0 and delta_to_flip = 1."""
    result = compute_category_math(
        canonical_code="R",
        my_final=10.0,
        opp_final=10.0,
    )

    assert result.margin == 0.0
    assert result.delta_to_flip == 1.0  # Need 1 more to win
    assert result.is_winning is False  # margin is not > 0
