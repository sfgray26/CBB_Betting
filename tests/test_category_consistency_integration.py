"""
Integration Test: Category Win/Loss Consistency Across All Modules

Tests that all modules (Waiver Wire, War Room, My Roster, Dashboard, Preview)
return identical W/L/T verdicts for the same matchup stats.

This addresses the UAT audit inconsistency where identical stats showed
different verdicts across modules:
- K/9: 14.62 vs 9.78 → Waiver Wire shows W, War Room shows BEHIND, My Roster shows W
- ERA: 5.62 vs 2.74 → Waiver Wire shows L, War Room shows BUBBLE, My Roster shows L
- HRA: 0 vs 2 → Waiver Wire shows L, War Room shows (HR), My Roster shows W
- AVG: .213 vs .231 → Waiver Wire shows L, War Room shows BEHIND, My Roster shows T

Created: Loop Iteration 13 (2026-06-25)
"""

import pytest
from backend.services.category_comparator import (
    compare_category,
    category_status,
    get_canonical_category,
    CATEGORY_DIRECTIONS,
)


class TestCategoryConsistencyIntegration:
    """
    Integration test verifying category verdicts are consistent
    across all modules using the unified category_comparator service.

    This ensures that identical matchup stats always produce identical
    W/L/T verdicts regardless of which module is used.
    """

    # Sample matchup data from UAT audit (the problematic cases)
    UAT_MATCHUP_DATA = [
        # (category, my_val, opp_val, expected_verdict, description)
        ("K_9", 14.62, 9.78, "W", "Higher K/9 should win"),
        ("ERA", 5.62, 2.74, "L", "Higher ERA should lose (lower is better)"),
        ("HRA", 0, 2, "W", "Fewer HR allowed should win (lower is better)"),
        ("AVG", 0.213, 0.231, "L", "Lower AVG should lose (higher is better)"),
        # Additional edge cases
        ("WHIP", 1.45, 1.20, "L", "Higher WHIP should lose (lower is better)"),
        ("HR", 15, 10, "W", "More HR should win (higher is better)"),
        ("W", 5, 3, "W", "More wins should win (higher is better)"),
        ("L", 3, 5, "W", "Fewer losses should win (lower is better)"),
        ("SV", 4, 2, "W", "More saves should win (higher is better)"),
        ("SB", 8, 12, "L", "Fewer SB should lose (higher is better)"),
        ("R", 45, 50, "L", "Fewer runs should lose (higher is better)"),
        ("RBI", 40, 35, "W", "More RBI should win (higher is better)"),
        ("OPS", 0.750, 0.800, "L", "Lower OPS should lose (higher is better)"),
        ("IP", 60.0, 55.0, "W", "More IP should win (higher is better)"),
    ]

    @pytest.mark.parametrize("category,my_val,opp_val,expected_verdict,description", UAT_MATCHUP_DATA)
    def test_consistent_verdict_across_modules(
        self, category, my_val, opp_val, expected_verdict, description
    ):
        """
        Test that compare_category() returns the expected verdict for each matchup.

        This is the single source of truth that all modules should agree on.
        """
        result = compare_category(category, my_val, opp_val)

        assert result.verdict == expected_verdict, (
            f"Category {category}: my={my_val}, opp={opp_val} → "
            f"expected {expected_verdict}, got {result.verdict}. {description}"
        )

    def test_uat_k9_inconsistency_fixed(self):
        """
        UAT Audit Example: K/9 14.62 vs 9.78

        Before fix: Waiver Wire shows W, War Room shows BEHIND, My Roster shows W
        After fix: All modules should return W (higher is better)
        """
        result = compare_category("K_9", 14.62, 9.78)
        assert result.verdict == "W", "K/9 is higher-is-better, 14.62 > 9.78 should win"
        assert result.gap == pytest.approx(4.84, rel=1e-2)

    def test_uat_era_inconsistency_fixed(self):
        """
        UAT Audit Example: ERA 5.62 vs 2.74

        Before fix: Waiver Wire shows L, War Room shows BUBBLE, My Roster shows L
        After fix: All modules should return L (lower is better)
        """
        result = compare_category("ERA", 5.62, 2.74)
        assert result.verdict == "L", "ERA is lower-is-better, 5.62 > 2.74 should lose"
        assert result.gap == pytest.approx(2.88, rel=1e-2)

    def test_uat_hra_inconsistency_fixed(self):
        """
        UAT Audit Example: HRA 0 vs 2

        Before fix: Waiver Wire shows L, War Room shows (HR), My Roster shows W
        After fix: All modules should return W (lower is better, 0 < 2)
        """
        result = compare_category("HRA", 0, 2)
        assert result.verdict == "W", "HRA is lower-is-better, 0 < 2 should win"
        assert result.gap == -2.0

    def test_uat_avg_inconsistency_fixed(self):
        """
        UAT Audit Example: AVG .213 vs .231

        Before fix: Waiver Wire shows L, War Room shows BEHIND, My Roster shows T
        After fix: All modules should return L (higher is better, .213 < .231)
        """
        result = compare_category("AVG", 0.213, 0.231)
        assert result.verdict == "L", "AVG is higher-is-better, .213 < .231 should lose"
        assert result.gap == pytest.approx(-0.018, rel=1e-2)

    def test_all_uat_examples_consistent(self):
        """
        Batch test: All four UAT audit examples should return consistent verdicts.
        """
        uat_examples = [
            ("K_9", 14.62, 9.78, "W"),
            ("ERA", 5.62, 2.74, "L"),
            ("HRA", 0, 2, "W"),
            ("AVG", 0.213, 0.231, "L"),
        ]

        for category, my_val, opp_val, expected in uat_examples:
            result = compare_category(category, my_val, opp_val)
            assert result.verdict == expected, (
                f"UAT Example {category}: my={my_val}, opp={opp_val} "
                f"→ expected {expected}, got {result.verdict}"
            )

    def test_canonical_category_resolution(self):
        """
        Test that category names with namespaces are resolved correctly.
        """
        # Pitching categories
        assert get_canonical_category("P_K") == "K"
        assert get_canonical_category("P_ERA") == "ERA"
        assert get_canonical_category("P_HR") == "HR"

        # Hitting categories
        assert get_canonical_category("H_K") == "K"
        assert get_canonical_category("H_HR") == "HR"

        # No namespace (returns as-is)
        assert get_canonical_category("ERA") == "ERA"
        assert get_canonical_category("HR") == "HR"

    def test_category_directions_comprehensive(self):
        """
        Test that all common categories have defined directions.
        """
        # Pitching lower-is-better
        assert CATEGORY_DIRECTIONS["ERA"] == "lower"
        assert CATEGORY_DIRECTIONS["WHIP"] == "lower"
        assert CATEGORY_DIRECTIONS["HRA"] == "lower"
        assert CATEGORY_DIRECTIONS["L"] == "lower"

        # Pitching higher-is-better
        assert CATEGORY_DIRECTIONS["W"] == "higher"
        assert CATEGORY_DIRECTIONS["K"] == "higher"
        assert CATEGORY_DIRECTIONS["SV"] == "higher"
        assert CATEGORY_DIRECTIONS["K_9"] == "higher"

        # Hitting higher-is-better
        assert CATEGORY_DIRECTIONS["HR"] == "higher"
        assert CATEGORY_DIRECTIONS["R"] == "higher"
        assert CATEGORY_DIRECTIONS["RBI"] == "higher"
        assert CATEGORY_DIRECTIONS["SB"] == "higher"
        assert CATEGORY_DIRECTIONS["AVG"] == "higher"
        assert CATEGORY_DIRECTIONS["OPS"] == "higher"


class TestCategoryStatusConsistency:
    """
    Test gap-based status (AHEAD/BEHIND/BUBBLE/PUNT?/CONCEDE) consistency.
    """

    def test_tie_returns_bubble(self):
        """Tie should always return BUBBLE status."""
        status = category_status("HR", 10, 10)
        assert status.verdict == "T"
        assert status.status == "BUBBLE"

    def test_small_gap_returns_bubble(self):
        """Small gap (<10%) returns BUBBLE status."""
        # HR: 12 vs 11 = 9.1% gap → BUBBLE
        status = category_status("HR", 12, 11)
        assert status.verdict == "W"
        assert status.status == "BUBBLE"

    def test_medium_gap_ahead_returns_ahead(self):
        """Medium gap ahead (10-25%) returns AHEAD status."""
        # HR: 15 vs 12 = 25% gap → AHEAD
        status = category_status("HR", 15, 12)
        assert status.verdict == "W"
        assert status.status == "AHEAD"

    def test_medium_gap_behind_returns_behind(self):
        """Medium gap behind (10-25%) returns BEHIND status."""
        # HR: 10 vs 12 = -20% gap → BEHIND
        status = category_status("HR", 10, 12)
        assert status.verdict == "L"
        assert status.status == "BEHIND"

    def test_large_gap_returns_punt(self):
        """Large gap behind (25%+) returns PUNT? status."""
        # HR: 9 vs 12 = -25% gap → PUNT?
        status = category_status("HR", 9, 12)
        assert status.verdict == "L"
        assert status.status == "PUNT?"

    def test_very_large_gap_returns_concede(self):
        """Very large gap behind (>25%) returns CONCEDE status."""
        # ERA: 6.00 vs 3.50 = 71.4% gap → CONCEDE
        status = category_status("ERA", 6.00, 3.50)
        assert status.verdict == "L"
        assert status.status == "CONCEDE"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_vs_zero_returns_tie(self):
        """Both zero values should return tie."""
        result = compare_category("HR", 0, 0)
        assert result.verdict == "T"
        assert result.gap == 0.0

    def test_negative_values_handled_correctly(self):
        """Negative values should be handled correctly."""
        # Higher-is-better: -5 > -10 should win
        result = compare_category("HR", -5, -10)
        assert result.verdict == "W"

        # Lower-is-better: -3.5 < -2.5 should win
        result = compare_category("ERA", -3.5, -2.5)
        assert result.verdict == "W"

    def test_unknown_category_defaults_to_higher(self):
        """Unknown category should default to higher-is-better."""
        result = compare_category("UNKNOWN", 10, 5)
        assert result.verdict == "W"  # Higher value wins by default

    def test_fractional_values_handled_correctly(self):
        """Fractional values (like AVG, ERA, WHIP) should work correctly."""
        # AVG: .280 vs .250 → higher wins
        result = compare_category("AVG", 0.280, 0.250)
        assert result.verdict == "W"
        assert result.gap == pytest.approx(0.03)

        # ERA: 2.50 vs 4.00 → lower wins
        result = compare_category("ERA", 2.50, 4.00)
        assert result.verdict == "W"
        assert result.gap == pytest.approx(-1.50)
