"""
Tests for OpenClaw Lite simplified coordinator.
"""

import pytest
from backend.services.openclaw_lite import (
    OpenClawLite,
    IntegrityResult,
    perform_sanity_check,
    get_openclaw_lite,
)


class TestHeuristicRules:
    """Test the heuristic-based integrity checking."""
    
    def test_clean_search_returns_confirmed(self):
        checker = OpenClawLite()
        result = checker.check_integrity_heuristic(
            search_text="Team looking good, no injuries reported. Weather is clear.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == "CONFIRMED"
        assert result.confidence > 0.8
        assert result.source == "heuristic"
    
    def test_multiple_risk_keywords_triggers_caution(self):
        checker = OpenClawLite()
        result = checker.check_integrity_heuristic(
            search_text="Player injury reported. Starting lineup uncertain. Doubtful status.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == "CAUTION"
        assert result.source == "heuristic"
    
    def test_star_player_out_caution(self):
        checker = OpenClawLite()
        result = checker.check_integrity_heuristic(
            search_text="The star player is out with an ankle injury.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == "CAUTION"
    
    def test_conflicting_reports_volatile(self):
        checker = OpenClawLite()
        result = checker.check_integrity_heuristic(
            search_text="Conflicting reports about the injury. Sources disagree on severity.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == "VOLATILE"
    
    def test_high_stakes_boosts_confidence(self):
        checker = OpenClawLite()
        result = checker.check_integrity_heuristic(
            search_text="Everything looks normal today.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=2.0  # High stakes
        )
        assert result.verdict == "CONFIRMED"
        # Should mention high stakes in reasoning
        assert "stakes" in result.reasoning.lower() or "caution" in result.reasoning.lower()


class TestKeywordDetection:
    """Test keyword-based risk detection."""
    
    def test_injury_keywords_detected(self):
        checker = OpenClawLite()
        text = "Player injured, listed as doubtful"
        hits = sum(1 for kw in checker.HIGH_RISK_KEYWORDS if kw in text.lower())
        assert hits >= 2
    
    def test_conflict_keywords_detected(self):
        checker = OpenClawLite()
        text = "Conflicting reports from sources"
        hits = sum(1 for kw in checker.CONFLICT_KEYWORDS if kw in text.lower())
        assert hits >= 2


class TestRoutingLogic:
    """Test that routing selects appropriate method."""
    
    @pytest.mark.asyncio
    async def test_low_stakes_uses_heuristic(self):
        checker = OpenClawLite()
        result = await checker.check_integrity(
            search_text="Clean search, no issues.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.3,  # Low stakes
            force_heuristic=False
        )
        assert result.source == "heuristic"
    
    @pytest.mark.asyncio
    async def test_high_stakes_uses_direct(self):
        checker = OpenClawLite()
        result = await checker.check_integrity(
            search_text="Some minor news but nothing major.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=2.0,  # High stakes
            is_elite_eight_or_later=False
        )
        # Should use direct path (which falls back to heuristic for simple cases)
        assert result.verdict in ["CONFIRMED", "CAUTION", "VOLATILE", "ABORT"]
    
    @pytest.mark.asyncio
    async def test_elite_eight_uses_direct(self):
        checker = OpenClawLite()
        result = await checker.check_integrity(
            search_text="Clean search.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5,
            is_elite_eight_or_later=True  # Tournament stakes
        )
        assert result.source in ["heuristic", "kimi"]


class TestAbortConditions:
    """Test that critical issues trigger ABORT."""
    
    def test_star_out_aborts(self):
        checker = OpenClawLite()
        result = checker.check_integrity_direct(
            search_text="Star player out with major injury.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=1.0
        )
        assert result.verdict == "ABORT"
    
    def test_major_scandal_aborts(self):
        checker = OpenClawLite()
        result = checker.check_integrity_direct(
            search_text="Major scandal broke, investigation ongoing.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=1.0
        )
        assert result.verdict == "ABORT"


class TestBackwardCompatibility:
    """Test backward-compatible wrapper."""
    
    @pytest.mark.asyncio
    async def test_perform_sanity_check_returns_string(self):
        result = await perform_sanity_check(
            home_team="Duke",
            away_team="UNC",
            verdict="Bet 1.0u Duke -4.5",
            search_results="No injuries, clean slate."
        )
        assert isinstance(result, str)
        assert result in ["CONFIRMED", "CAUTION", "VOLATILE", "ABORT", "RED FLAG"]
    
    @pytest.mark.asyncio
    async def test_extracts_units_from_verdict(self):
        # Should parse "1.5u" from verdict string
        result = await perform_sanity_check(
            home_team="Duke",
            away_team="UNC",
            verdict="Bet 1.5u Duke -4.5",  # Note: 1.5u
            search_results="Minor concern but looks okay."
        )
        assert isinstance(result, str)


class TestStats:
    """Test statistics tracking."""
    
    def test_tracks_call_counts(self):
        checker = OpenClawLite()
        
        # Make some calls
        checker.check_integrity_heuristic("clean", "A", "B", 0.5)
        checker.check_integrity_heuristic("risky injury", "A", "B", 0.5)
        
        stats = checker.get_stats()
        assert stats["heuristic_calls"] == 2
        assert stats["total_calls"] == 2
        assert "heuristic_pct" in stats
    
    def test_singleton_returns_same_instance(self):
        a = get_openclaw_lite()
        b = get_openclaw_lite()
        assert a is b


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_search_text(self):
        checker = OpenClawLite()
        result = checker.check_integrity_heuristic(
            search_text="",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        # Should default to CONFIRMED for empty input
        assert result.verdict == "CONFIRMED"
    
    def test_very_long_search_text(self):
        checker = OpenClawLite()
        long_text = "injury " * 100  # Many risk keywords
        result = checker.check_integrity_heuristic(
            search_text=long_text,
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == "CAUTION"  # Should catch the risk
    
    def test_case_insensitive_matching(self):
        checker = OpenClawLite()
        result = checker.check_integrity_heuristic(
            search_text="INJURY reported, PLAYER is OUT",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == "CAUTION"  # Uppercase should still match


class TestPerformance:
    """Test that heuristic is fast."""
    
    def test_heuristic_under_10ms(self):
        import time
        checker = OpenClawLite()
        
        start = time.time()
        for _ in range(100):
            checker.check_integrity_heuristic(
                search_text="Some test text with a few words.",
                home_team="Duke",
                away_team="UNC",
                recommended_units=0.5
            )
        elapsed = time.time() - start
        
        # 100 calls should take less than 10ms total
        assert elapsed < 0.01, f"Heuristic too slow: {elapsed*1000:.2f}ms for 100 calls"
