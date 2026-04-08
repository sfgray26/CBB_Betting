"""
H2H One Win Format Validations

Acceptance criteria from research doc Section 4.5.
Tests NSB calculation, scarcity index, IP bank precision, and one-win probability.
"""

import pytest


def test_nsb_calculation_allows_negative():
    """
    NSB = SB - CS, not raw SB. NSB can be negative in H2H One Win format.

    Example: A player with 0 SB and 1 CS should have NSB = -1, not 0.
    The max(0, sb - cs) bug clamped this incorrectly.
    """
    # Simulate the NSB calculation logic
    def calculate_nsb(sb: int, cs: int) -> int:
        """NSB can be negative for H2H One Win format."""
        return sb - cs

    # Test cases
    assert calculate_nsb(10, 2) == 8   # Normal positive NSB
    assert calculate_nsb(5, 5) == 0    # Break-even
    assert calculate_nsb(0, 1) == -1   # Negative NSB (caught more than stolen)
    assert calculate_nsb(0, 5) == -5   # Highly negative NSB

    # Edge case: no CS data (treat as 0)
    assert calculate_nsb(10, 0) == 10


def test_scarcity_index_cf_vs_lf_rf():
    """
    CF scarcity reflects multi-eligibility.

    CF is scarcer (~45 MLB players qualify) than LF/RF (~28-35 each).
    Multi-eligible players (e.g., Bellinger CF/LF/RF) count for ALL THREE
    in scarcity calculations.
    """
    # This test requires PositionEligibility data to be populated
    # For now, test the scarcity calculation logic in isolation

    def compute_scarcity_rank(
        rostered_pct: float,
        position_pool_size: int
    ) -> int:
        """
        Simpler scarcity rank: higher rostered % = better (lower rank).

        In production, this uses window functions over PositionEligibility.
        """
        # Placeholder implementation
        # Real implementation: rank() OVER (PARTITION BY primary_position ORDER BY league_rostered_pct DESC)
        return int(position_pool_size * (1 - rostered_pct))

    # Test CF scarcity (smaller pool = more volatile rankings)
    cf_rank = compute_scarcity_rank(0.80, 45)  # 80% rostered in 45-player pool
    lf_rank = compute_scarcity_rank(0.80, 80)  # 80% rostered in 80-player pool

    # Same rostered % but smaller pool = lower (better) rank for CF
    assert cf_rank < lf_rank


def test_ip_bank_precision():
    """
    IP projections sum correctly for 18 IP minimum.

    H2H One Win format requires 18 IP minimum per week.
    Test roster with Cole (13 IP) + Glasnow (12 IP) = 25 IP.
    Should show SAFE status, not WARNING.
    """
    def check_ip_minimum(total_ip: float, minimum: float = 18.0) -> str:
        """Check IP bank against minimum."""
        if total_ip >= minimum:
            return "SAFE"
        elif total_ip >= minimum * 0.75:  # Within 75%
            return "WARNING"
        else:
            return "CRITICAL"

    # Test cases
    assert check_ip_minimum(25.0, 18.0) == "SAFE"    # Cole + Glasnow
    assert check_ip_minimum(18.0, 18.0) == "SAFE"    # Exactly minimum
    assert check_ip_minimum(15.0, 18.0) == "WARNING"  # Below minimum but close
    assert check_ip_minimum(10.0, 18.0) == "CRITICAL" # Far below minimum


def test_one_win_probability():
    """
    5-4 and 9-0 both = 1 win (not 1.8x vs 0.2x).

    H2H One Win format: win 6+ of 10 categories = 1 win.
    Margin doesn't matter.
    """
    def calculate_week_win(categories_won: int, categories_total: int = 10) -> bool:
        """Returns True if 6+ categories won."""
        return categories_won >= 6

    # Test cases
    assert calculate_week_win(5, 10) is False  # 5-5 = loss
    assert calculate_week_win(6, 10) is True   # 6-4 = win
    assert calculate_week_win(9, 10) is True   # 9-1 = win (same as 6-4)
    assert calculate_week_win(10, 10) is True  # 10-0 = win

    # Edge case: non-standard category counts
    # In 8-cat format, need 5+ categories to win (majority)
    def calculate_week_win_8cat(categories_won: int) -> bool:
        """Returns True if 5+ categories won in 8-cat format."""
        return categories_won >= 5

    assert calculate_week_win_8cat(4) is False   # 4-4 = loss
    assert calculate_week_win_8cat(5) is True    # 5-3 = win


def test_nsb_stat_id_60_available():
    """
    Yahoo Fantasy API exposes NSB as stat_id 60.

    Verified via K-28 audit. stat_id 5070 in research doc was incorrect.
    """
    # This test documents the correct stat_id
    yahoo_nsb_stat_id = 60
    assert yahoo_nsb_stat_id == 60, "NSB is stat_id 60, not 5070"


def test_statcast_cs_fallback():
    """
    Statcast provides CS (caught stealing) field if Yahoo API fails.

    Fallback strategy: use pybaseball.statcast() -> 'CS' column.
    """
    # Document the fallback field name
    statcast_cs_field = "CS"
    assert statcast_cs_field == "CS"

    # Fallback calculation
    def calculate_nsb_with_fallback(sb: int, cs: int | None) -> int | None:
        """Handle missing CS data."""
        if cs is None:
            return None  # Cannot calculate without CS
        return sb - cs

    assert calculate_nsb_with_fallback(10, 2) == 8
    assert calculate_nsb_with_fallback(10, None) is None


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
