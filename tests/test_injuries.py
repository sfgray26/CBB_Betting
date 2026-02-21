"""
Tests for injury impact estimation.
Run with: pytest tests/test_injuries.py -v
"""

import pytest
from backend.services.injuries import estimate_impact, TIER_IMPACT


class TestInjuryImpact:
    """Test usage-weighted injury impact with bounded multiplier"""

    def test_no_usage_rate_returns_base_tier(self):
        for tier, base in TIER_IMPACT.items():
            assert estimate_impact(tier) == base

    def test_average_starter_usage_is_neutral_multiplier(self):
        # D1 average starter usage = 22%; multiplier should be ~1.0
        impact = estimate_impact("star", usage_rate=22.0)
        base = TIER_IMPACT["star"]
        assert abs(impact - base * 1.0) < 0.05

    def test_high_usage_increases_impact(self):
        # 35% usage → multiplier = 35/22 = 1.59
        impact = estimate_impact("star", usage_rate=35.0)
        base = TIER_IMPACT["star"]
        expected = base * (35.0 / 22.0)
        assert abs(impact - expected) < 0.05

    def test_high_usage_capped_at_1_8x(self):
        # 44%+ usage → multiplier capped at 1.8
        impact_44 = estimate_impact("star", usage_rate=44.0)
        impact_60 = estimate_impact("star", usage_rate=60.0)
        base = TIER_IMPACT["star"]
        cap = base * 1.8
        assert abs(impact_44 - cap) < 0.05
        assert abs(impact_60 - cap) < 0.05

    def test_low_usage_floored_at_0_5x(self):
        # 10% usage → multiplier = max(0.5, 10/22) = max(0.5, 0.45) = 0.5
        impact = estimate_impact("star", usage_rate=10.0)
        base = TIER_IMPACT["star"]
        assert abs(impact - base * 0.5) < 0.05

    def test_zero_usage_rate_returns_base(self):
        # usage_rate=0 is treated as "not provided" by the guard `> 0`
        impact = estimate_impact("starter", usage_rate=0.0)
        assert impact == TIER_IMPACT["starter"]

    def test_all_tiers_scale_consistently(self):
        usage = 33.0  # 1.5× baseline; clamped below 1.8
        for tier, base in TIER_IMPACT.items():
            impact = estimate_impact(tier, usage_rate=usage)
            expected_multiplier = min(1.8, max(0.5, usage / 22.0))
            assert abs(impact - base * expected_multiplier) < 0.01

    def test_unknown_tier_uses_default(self):
        # Unknown tier → default base 0.2
        impact = estimate_impact("unknown_tier")
        assert impact == 0.2

    def test_impact_never_negative(self):
        for tier in TIER_IMPACT:
            for usage in [5.0, 10.0, 22.0, 35.0, 50.0]:
                assert estimate_impact(tier, usage_rate=usage) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
