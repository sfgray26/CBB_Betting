"""
Tests for L1 pure functions: Phase 3 Workstream D.

Tests for compute_ratio_risk() and compute_category_count_delta().
"""

import pytest

from backend.services.category_math import (
    compute_ratio_risk,
    compute_category_count_delta,
    CategoryMathResult,
    compute_category_math,
)
from backend.stat_contract import SCORING_CATEGORY_CODES


class TestRatioRisk:
    """Tests for compute_ratio_risk()."""

    def test_era_safe_large_cushion(self):
        """ERA SAFE — large cushion (my ERA 3.00, opp ERA 4.50, 20 IP remaining)."""
        # Current: 30 IP, 10 ER → ERA = 3.00
        # Opponent: 4.50 ERA
        # With 20 IP remaining, have huge cushion
        result = compute_ratio_risk(
            my_ip=30.0,
            my_er=10.0,
            my_hits_allowed=30.0,
            my_bb_allowed=10.0,
            opp_era=4.50,
            opp_whip=1.40,
            remaining_ip=20.0,
        )

        assert result["era_risk"] == "SAFE"
        assert result["era_cushion_er"] > 5.0  # Can allow 5+ ER

    def test_era_at_risk_small_cushion(self):
        """ERA AT_RISK — small cushion (my ERA 3.80, opp ERA 4.00, 10 IP remaining)."""
        # Current: 20 IP, 8.44 ER → ERA ≈ 3.80
        # Opponent: 4.00 ERA
        result = compute_ratio_risk(
            my_ip=20.0,
            my_er=8.44,
            my_hits_allowed=20.0,
            my_bb_allowed=8.0,
            opp_era=4.00,
            opp_whip=1.35,
            remaining_ip=10.0,
        )

        assert result["era_risk"] == "AT_RISK"
        assert result["era_cushion_er"] > 0
        assert result["era_cushion_er"] < 5.0

    def test_era_critical_already_losing(self):
        """ERA CRITICAL — already losing with no realistic path to win."""
        # My ERA is terrible, and even with perfect remaining IP, can't catch opp
        result = compute_ratio_risk(
            my_ip=40.0,
            my_er=25.0,  # ERA = 5.63
            my_hits_allowed=50.0,
            my_bb_allowed=15.0,
            opp_era=3.20,
            opp_whip=1.15,
            remaining_ip=5.0,  # Not enough IP to overcome deficit
        )

        assert result["era_risk"] == "CRITICAL"
        assert result["era_cushion_er"] < 0

    def test_whip_safe_large_cushion(self):
        """WHIP SAFE — large cushion."""
        result = compute_ratio_risk(
            my_ip=30.0,
            my_er=10.0,
            my_hits_allowed=25.0,  # WHIP = (25+10)/30 = 1.17
            my_bb_allowed=10.0,
            opp_era=4.50,
            opp_whip=1.40,
            remaining_ip=20.0,
        )

        assert result["whip_risk"] == "SAFE"
        assert result["whip_cushion_baserunners"] > 10.0

    def test_whip_critical_already_losing(self):
        """WHIP CRITICAL — already losing."""
        result = compute_ratio_risk(
            my_ip=20.0,
            my_er=10.0,
            my_hits_allowed=35.0,  # WHIP = (35+15)/20 = 2.50
            my_bb_allowed=15.0,
            opp_era=4.00,
            opp_whip=1.20,
            remaining_ip=10.0,
        )

        assert result["whip_risk"] == "CRITICAL"
        assert result["whip_cushion_baserunners"] < 0

    def test_zero_ip_safe(self):
        """Zero IP → SAFE (no data yet)."""
        result = compute_ratio_risk(
            my_ip=0.0,
            my_er=0.0,
            my_hits_allowed=0.0,
            my_bb_allowed=0.0,
            opp_era=4.00,
            opp_whip=1.30,
            remaining_ip=10.0,
        )

        assert result["era_risk"] == "SAFE"
        assert result["whip_risk"] == "SAFE"
        assert result["era_cushion_er"] == 99.0
        assert result["whip_cushion_baserunners"] == 99.0


class TestCategoryCountDelta:
    """Tests for compute_category_count_delta()."""

    def test_winning_projection(self):
        """compute_category_count_delta — 10+ winning → projected "WIN"."""
        # Create results where we win 10 categories
        results = {}
        for i, code in enumerate(SCORING_CATEGORY_CODES):
            if i < 10:
                # Winning categories
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=5.0,
                    delta_to_flip=10.0,
                    is_winning=True,
                )
            else:
                # Losing categories
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=-5.0,
                    delta_to_flip=10.0,
                    is_winning=False,
                )

        delta = compute_category_count_delta(results)

        assert delta["winning"] == 10
        assert delta["losing"] == 8
        assert delta["tied"] == 0
        assert delta["projected_result"] == "WIN"

    def test_losing_projection(self):
        """compute_category_count_delta — 10+ losing → projected "LOSS"."""
        results = {}
        for i, code in enumerate(SCORING_CATEGORY_CODES):
            if i < 10:
                # Losing categories
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=-5.0,
                    delta_to_flip=10.0,
                    is_winning=False,
                )
            else:
                # Winning categories
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=5.0,
                    delta_to_flip=10.0,
                    is_winning=True,
                )

        delta = compute_category_count_delta(results)

        assert delta["winning"] == 8
        assert delta["losing"] == 10
        assert delta["projected_result"] == "LOSS"

    def test_toss_up_projection(self):
        """compute_category_count_delta — 9-9 split → "TOSS_UP"."""
        results = {}
        for i, code in enumerate(SCORING_CATEGORY_CODES):
            if i < 9:
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=5.0,
                    delta_to_flip=10.0,
                    is_winning=True,
                )
            else:
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=-5.0,
                    delta_to_flip=10.0,
                    is_winning=False,
                )

        delta = compute_category_count_delta(results)

        assert delta["winning"] == 9
        assert delta["losing"] == 9
        assert delta["projected_result"] == "TOSS_UP"

    def test_swing_count_detects_close_margins(self):
        """compute_category_count_delta — swing count detects close margins.

        Swing categories are those that could flip with small changes:
        - Counting stats: losing categories with |margin| < 2.0
        - Ratio stats: categories with |margin| < 0.1 (winning or losing)
        """
        results = {}
        for i, code in enumerate(SCORING_CATEGORY_CODES):
            if i < 8:
                # 8 comfortable wins (not swing)
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=10.0,
                    delta_to_flip=20.0,
                    is_winning=True,
                )
            elif i < 14:
                # 6 close losses (swing - these are losing by < 2)
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=-1.0,  # Close loss
                    delta_to_flip=2.0,
                    is_winning=False,
                )
            elif i < 16:
                # 2 close ratio wins (swing - ERA, WHIP with < 0.1 margin)
                results[code] = CategoryMathResult(
                    canonical_code="ERA" if i == 14 else "WHIP",
                    margin=0.08,
                    delta_to_flip=0.15,
                    is_winning=True,
                )
            else:
                # 2 solid losses (not swing - losing by > 2)
                results[code] = CategoryMathResult(
                    canonical_code=code,
                    margin=-5.0,
                    delta_to_flip=10.0,
                    is_winning=False,
                )

        delta = compute_category_count_delta(results)

        # 6 close counting losses + 2 close ratio wins = 8 swing categories
        assert delta["swing"] == 8
        assert delta["winning"] == 10

    def test_ratio_swing_threshold(self):
        """Ratio stats use smaller threshold for swing detection."""
        results = {
            "ERA": CategoryMathResult(
                canonical_code="ERA",
                margin=0.05,  # Small positive margin -> swing
                delta_to_flip=0.10,
                is_winning=True,
            ),
            "WHIP": CategoryMathResult(
                canonical_code="WHIP",
                margin=0.08,  # Small positive margin -> swing
                delta_to_flip=0.15,
                is_winning=True,
            ),
            "R": CategoryMathResult(
                canonical_code="R",
                margin=-3.0,  # Solid loss, not close -> not swing
                delta_to_flip=5.0,
                is_winning=False,
            ),
        }

        delta = compute_category_count_delta(results)

        # ERA and WHIP should be swing (margin < 0.1), R should not
        assert delta["swing"] == 2
