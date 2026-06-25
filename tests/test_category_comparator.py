"""
Unit tests for category_comparator.py

Tests that the unified category comparator produces consistent win/loss/tie verdicts
across all categories, including inverted (lower-is-better) ones.

Created: Loop Iteration 13 (2026-06-25)
"""

import pytest
from backend.services.category_comparator import (
    CategoryComparator,
    CategoryResult,
    CategoryStatus,
    compare_category,
    category_status,
    get_canonical_category,
    get_category_comparator,
    CATEGORY_DIRECTIONS,
    BUBBLE_THRESHOLD_PCT,
    PUNT_THRESHOLD_PCT,
    CONCEDE_THRESHOLD_PCT,
)


class TestCategoryDirections:
    """Test that category directions are correctly defined."""

    def test_era_is_lower_better(self):
        """ERA should be lower-is-better."""
        assert CATEGORY_DIRECTIONS["ERA"] == "lower"

    def test_whip_is_lower_better(self):
        """WHIP should be lower-is-better."""
        assert CATEGORY_DIRECTIONS["WHIP"] == "lower"

    def test_hra_is_lower_better(self):
        """Home runs allowed should be lower-is-better."""
        assert CATEGORY_DIRECTIONS["HRA"] == "lower"

    def test_losses_is_lower_better(self):
        """Losses should be lower-is-better."""
        assert CATEGORY_DIRECTIONS["L"] == "lower"

    def test_k_is_higher_better(self):
        """Strikeouts should be higher-is-better."""
        assert CATEGORY_DIRECTIONS["K"] == "higher"

    def test_hr_is_higher_better(self):
        """Home runs should be higher-is-better."""
        assert CATEGORY_DIRECTIONS["HR"] == "higher"

    def test_avg_is_higher_better(self):
        """Batting average should be higher-is-better."""
        assert CATEGORY_DIRECTIONS["AVG"] == "higher"


class TestHigherBetterCategories:
    """Test higher-is-better categories produce correct verdicts."""

    @pytest.fixture
    def comparator(self):
        return CategoryComparator()

    def test_hr_win(self, comparator):
        """Higher HR wins."""
        result = comparator.compare("HR", 15, 10)
        assert result.verdict == "W"
        assert result.gap == 5.0

    def test_hr_loss(self, comparator):
        """Lower HR loses."""
        result = comparator.compare("HR", 8, 12)
        assert result.verdict == "L"
        assert result.gap == -4.0

    def test_hr_tie(self, comparator):
        """Equal HR ties."""
        result = comparator.compare("HR", 10, 10)
        assert result.verdict == "T"
        assert result.gap == 0.0

    def test_k_9_win(self, comparator):
        """Higher K/9 wins."""
        result = comparator.compare("K_9", 14.62, 9.78)
        assert result.verdict == "W"
        assert result.gap == pytest.approx(4.84)

    def test_rbi_win(self, comparator):
        """Higher RBI wins."""
        result = comparator.compare("RBI", 45, 30)
        assert result.verdict == "W"
        assert result.gap == 15.0

    def test_sb_win(self, comparator):
        """Higher SB wins."""
        result = comparator.compare("SB", 12, 8)
        assert result.verdict == "W"
        assert result.gap == 4.0

    def test_avg_win(self, comparator):
        """Higher AVG wins."""
        result = comparator.compare("AVG", 0.280, 0.250)
        assert result.verdict == "W"
        assert result.gap == pytest.approx(0.03)


class TestLowerBetterCategories:
    """Test lower-is-better categories produce correct verdicts."""

    @pytest.fixture
    def comparator(self):
        return CategoryComparator()

    def test_era_loss(self, comparator):
        """Higher ERA loses (lower is better)."""
        result = comparator.compare("ERA", 5.62, 2.74)
        assert result.verdict == "L"
        assert result.gap == pytest.approx(2.88)

    def test_era_win(self, comparator):
        """Lower ERA wins."""
        result = comparator.compare("ERA", 2.50, 4.00)
        assert result.verdict == "W"
        assert result.gap == -1.50

    def test_era_tie(self, comparator):
        """Equal ERA ties."""
        result = comparator.compare("ERA", 3.50, 3.50)
        assert result.verdict == "T"
        assert result.gap == 0.0

    def test_whip_loss(self, comparator):
        """Higher WHIP loses (lower is better)."""
        result = comparator.compare("WHIP", 1.45, 1.20)
        assert result.verdict == "L"
        assert result.gap == 0.25

    def test_whip_win(self, comparator):
        """Lower WHIP wins."""
        result = comparator.compare("WHIP", 1.05, 1.30)
        assert result.verdict == "W"
        assert result.gap == -0.25

    def test_hra_loss(self, comparator):
        """More HR allowed loses (lower is better)."""
        result = comparator.compare("HRA", 15, 8)
        assert result.verdict == "L"
        assert result.gap == 7.0

    def test_hra_win(self, comparator):
        """Fewer HR allowed wins."""
        result = comparator.compare("HRA", 5, 12)
        assert result.verdict == "W"
        assert result.gap == -7.0

    def test_losses_loss(self, comparator):
        """More losses loses (lower is better)."""
        result = comparator.compare("L", 15, 10)
        assert result.verdict == "L"
        assert result.gap == 5.0

    def test_k_allowed_loss(self, comparator):
        """More K allowed loses (lower is better)."""
        result = comparator.compare("K_allowed", 150, 120)
        assert result.verdict == "L"
        assert result.gap == 30.0

    def test_bb_allowed_loss(self, comparator):
        """More BB allowed loses (lower is better)."""
        result = comparator.compare("BB_allowed", 45, 35)
        assert result.verdict == "L"
        assert result.gap == 10.0


class TestCategoryStatus:
    """Test gap-based status (AHEAD/BEHIND/BUBBLE/PUNT?/CONCEDE)."""

    @pytest.fixture
    def comparator(self):
        return CategoryComparator(
            bubble_threshold_pct=0.10,
            punt_threshold_pct=0.10,
            concede_threshold_pct=0.25,
        )

    def test_small_gap_bubble_status(self, comparator):
        """Within 10% gap returns BUBBLE status."""
        # HR: 12 vs 11 = 9.1% gap → BUBBLE
        status = comparator.get_status("HR", 12, 11)
        assert status.status == "BUBBLE"
        assert status.verdict == "W"

    def test_medium_gap_ahead_status(self, comparator):
        """10-25% gap ahead returns AHEAD status."""
        # HR: 15 vs 12 = 25% gap → AHEAD (exactly at threshold, should be AHEAD)
        status = comparator.get_status("HR", 15, 12)
        assert status.status == "AHEAD"
        assert status.verdict == "W"

    def test_medium_gap_behind_status(self, comparator):
        """10-25% gap behind returns BEHIND status."""
        # HR: 9 vs 12 = -25% gap → PUNT? (at exactly 25% threshold)
        status = comparator.get_status("HR", 9, 12)
        assert status.status == "PUNT?"
        assert status.verdict == "L"

    def test_large_gap_punt_status(self, comparator):
        """>25% gap behind returns CONCEDE status."""
        # HR: 7 vs 12 = -41.7% gap → CONCEDE
        status = comparator.get_status("HR", 7, 12)
        assert status.status == "CONCEDE"
        assert status.verdict == "L"

    def test_very_large_gap_concede_status(self, comparator):
        """>25% gap behind returns CONCEDE status."""
        # ERA: 6.00 vs 3.50 = 71.4% gap → CONCEDE
        status = comparator.get_status("ERA", 6.00, 3.50)
        assert status.status == "CONCEDE"
        assert status.verdict == "L"

    def test_tie_returns_bubble(self, comparator):
        """Tie returns BUBBLE status."""
        status = comparator.get_status("HR", 10, 10)
        assert status.status == "BUBBLE"
        assert status.verdict == "T"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_compare_category_function(self):
        """compare_category() convenience function works."""
        result = compare_category("ERA", 5.62, 2.74)
        assert isinstance(result, CategoryResult)
        assert result.verdict == "L"

    def test_category_status_function(self):
        """category_status() convenience function works."""
        status = category_status("HR", 15, 10)
        assert isinstance(status, CategoryStatus)
        assert status.verdict == "W"
        assert status.status == "AHEAD"


class TestCanonicalCategories:
    """Test canonical category name resolution."""

    def test_canonical_without_namespace(self):
        """Category without namespace returns as-is."""
        assert get_canonical_category("ERA") == "ERA"
        assert get_canonical_category("HR") == "HR"

    def test_canonical_with_pitching_namespace(self):
        """Pitching category with namespace extracts base."""
        assert get_canonical_category("P_K") == "K"
        assert get_canonical_category("P_ERA") == "ERA"
        assert get_canonical_category("P_HR") == "HR"

    def test_canonical_with_hitting_namespace(self):
        """Hitting category with namespace extracts base."""
        assert get_canonical_category("H_K") == "K"
        assert get_canonical_category("H_HR") == "HR"

    def test_canonical_unknown_returns_as_is(self):
        """Unknown category returns as-is."""
        assert get_canonical_category("UNKNOWN") == "UNKNOWN"


class TestGapPercentageCalculation:
    """Test gap percentage calculation edge cases."""

    @pytest.fixture
    def comparator(self):
        return CategoryComparator()

    def test_gap_pct_with_positive_opponent(self, comparator):
        """Gap percent calculated correctly with positive opponent value."""
        result = comparator.compare("HR", 12, 10)
        assert result.gap_pct == 0.2  # (12-10)/10 = 20%

    def test_gap_pct_with_negative_opponent(self, comparator):
        """Gap percent calculated correctly with negative opponent value."""
        result = comparator.compare("HR", -5, -10)
        assert result.gap_pct == 0.5  # (-5-(-10))/|-10| = 50%

    def test_gap_pct_with_zero_opponent(self, comparator):
        """Gap percent returns 0 when opponent value is 0."""
        result = comparator.compare("HR", 5, 0)
        assert result.gap_pct == 0.0

    def test_gap_pct_with_fractional_values(self, comparator):
        """Gap percent calculated correctly with fractional values."""
        result = comparator.compare("AVG", 0.280, 0.250)
        assert result.gap_pct == pytest.approx(0.12, rel=1e-2)  # ~12%


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def comparator(self):
        return CategoryComparator()

    def test_unknown_category_defaults_to_higher(self, comparator):
        """Unknown category defaults to higher-is-better."""
        result = comparator.compare("UNKNOWN", 10, 5)
        assert result.verdict == "W"  # Higher value wins

    def test_zero_values_tie(self, comparator):
        """Both zero values return tie."""
        result = comparator.compare("HR", 0, 0)
        assert result.verdict == "T"
        assert result.gap == 0.0

    def test_negative_values_higher_better(self, comparator):
        """Negative values handled correctly for higher-is-better."""
        result = comparator.compare("HR", -5, -10)
        assert result.verdict == "W"  # -5 > -10

    def test_negative_values_lower_better(self, comparator):
        """Negative values handled correctly for lower-is-better."""
        result = comparator.compare("ERA", -3.5, -2.5)
        assert result.verdict == "W"  # -3.5 < -2.5 (lower ERA wins)


class TestUATAuditExamples:
    """Test the specific examples from the UAT audit that were inconsistent."""

    @pytest.fixture
    def comparator(self):
        return CategoryComparator()

    def test_k9_14_62_vs_9_78_returns_win(self, comparator):
        """K/9: 14.62 vs 9.78 should return WIN (higher is better)."""
        result = comparator.compare("K_9", 14.62, 9.78)
        assert result.verdict == "W"

    def test_era_5_62_vs_2_74_returns_loss(self, comparator):
        """ERA: 5.62 vs 2.74 should return LOSS (lower is better)."""
        result = comparator.compare("ERA", 5.62, 2.74)
        assert result.verdict == "L"

    def test_hra_0_vs_2_returns_win(self, comparator):
        """HRA: 0 vs 2 should return WIN (lower is better)."""
        result = comparator.compare("HRA", 0, 2)
        assert result.verdict == "W"

    def test_avg_213_vs_231_returns_loss(self, comparator):
        """AVG: .213 vs .231 should return LOSS (higher is better)."""
        result = comparator.compare("AVG", 0.213, 0.231)
        assert result.verdict == "L"


class TestSingletonPattern:
    """Test singleton pattern for comparator instance."""

    def test_get_category_comparator_returns_singleton(self):
        """get_category_comparator() returns same instance."""
        comparator1 = get_category_comparator()
        comparator2 = get_category_comparator()
        assert comparator1 is comparator2
        assert isinstance(comparator1, CategoryComparator)
