"""
Tests for OpenClaw Lite v3.0 — Simplified Integrity Coordination Service
"""

import pytest
import asyncio
from backend.services.openclaw_lite import (
    OpenClawLite,
    IntegrityResult,
    IntegrityVerdict,
    perform_sanity_check,
    async_perform_sanity_check,
    get_openclaw_lite,
    get_escalation_queue,
    HighStakesEscalationQueue,
    TelemetrySnapshot,
)


class TestHeuristicRules:
    """Test the heuristic-based integrity checking."""
    
    def test_clean_search_returns_confirmed(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="Team looking good, no injuries reported. Weather is clear.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == IntegrityVerdict.CONFIRMED.value
        assert result.confidence > 0.8
        assert result.source == "heuristic"
    
    def test_multiple_risk_keywords_triggers_caution(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="Player injury reported. Starting lineup uncertain. Doubtful status.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == IntegrityVerdict.CAUTION.value
        assert result.source == "heuristic"
    
    def test_star_player_out_caution(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="The star player is out with an ankle injury.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        # Should be CAUTION (not ABORT) because it's not "star player out" exactly
        assert result.verdict == IntegrityVerdict.CAUTION.value
    
    def test_conflicting_reports_volatile(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="Conflicting reports about the injury. Sources disagree on severity.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == IntegrityVerdict.VOLATILE.value
    
    def test_high_stakes_clean_returns_confirmed(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="Everything looks normal today.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=2.0  # High stakes
        )
        # Clean search with high stakes should still be CONFIRMED
        assert result.verdict == IntegrityVerdict.CONFIRMED.value


class TestAbortConditions:
    """Test that critical issues trigger ABORT."""
    
    def test_critical_star_player_out_aborts(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="Star player out for the season.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=1.0
        )
        assert result.verdict == IntegrityVerdict.ABORT.value
    
    def test_season_ending_injury_aborts(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="Season ending injury confirmed, player done for the year.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=1.0
        )
        assert result.verdict == IntegrityVerdict.ABORT.value


class TestKeywordDetection:
    """Test keyword-based risk detection."""
    
    def test_injury_keywords_detected(self):
        checker = OpenClawLite()
        text = "Player injured, listed as doubtful"
        hits = sum(1 for kw in checker.HIGH_RISK_KEYWORDS if kw in text.lower())
        assert hits >= 2
    
    def test_conflict_keywords_detected(self):
        checker = OpenClawLite()
        text = "Conflicting reports. Unclear situation with speculation."
        hits = sum(1 for kw in checker.CONFLICT_KEYWORDS if kw in text.lower())
        assert hits >= 2


class TestAsyncFunctionality:
    """Test async operations."""
    
    @pytest.mark.asyncio
    async def test_async_check_returns_result(self):
        checker = OpenClawLite(enable_telemetry=False)
        result = await checker.check_integrity(
            search_text="Clean search, no issues.",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.3,
            game_key="test_1"
        )
        assert isinstance(result, IntegrityResult)
        assert result.verdict == IntegrityVerdict.CONFIRMED.value
    
    @pytest.mark.asyncio
    async def test_concurrent_checks(self):
        checker = OpenClawLite(enable_telemetry=True)
        
        # Run 10 checks concurrently
        tasks = [
            checker.check_integrity(
                search_text="Test",
                home_team=f"Home{i}",
                away_team=f"Away{i}",
                recommended_units=0.5,
                game_key=f"test_{i}"
            )
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert all(isinstance(r, IntegrityResult) for r in results)
        
        # Check telemetry
        telemetry = checker.get_telemetry()
        assert telemetry["total_checks"] == 10


class TestBackwardCompatibility:
    """Test backward-compatible wrapper."""
    
    def test_perform_sanity_check_returns_string(self):
        result = perform_sanity_check(
            home_team="Duke",
            away_team="UNC",
            verdict="Bet 1.0u Duke -4.5",
            search_results="No injuries, clean slate."
        )
        assert isinstance(result, str)
        assert result in [v.value for v in IntegrityVerdict]
    
    def test_perform_sanity_check_extracts_units(self):
        # Should parse "1.5u" from verdict string
        result = perform_sanity_check(
            home_team="Duke",
            away_team="UNC",
            verdict="Bet 1.5u Duke -4.5",  # Note: 1.5u
            search_results="Minor concern but looks okay."
        )
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_async_perform_sanity_check(self):
        result = await async_perform_sanity_check(
            home_team="Duke",
            away_team="UNC",
            verdict="Bet 1.0u Duke -4.5",
            search_results="No issues."
        )
        assert isinstance(result, str)
        assert result in [v.value for v in IntegrityVerdict]


class TestTelemetry:
    """Test telemetry and metrics tracking."""
    
    def test_telemetry_tracks_checks(self):
        checker = OpenClawLite(enable_telemetry=True)
        
        # Make some calls
        checker._check_integrity_heuristic_sync("clean", "A", "B", 0.5)
        checker._check_integrity_heuristic_sync("injury out", "A", "B", 0.5)
        
        # Note: telemetry is only recorded via check_integrity (async)
        # Direct sync calls don't record telemetry
        telemetry = checker.get_telemetry()
        assert telemetry["total_checks"] == 0  # No async calls made
    
    @pytest.mark.asyncio
    async def test_telemetry_records_via_async(self):
        checker = OpenClawLite(enable_telemetry=True)
        
        await checker.check_integrity("Everything looks good.", "A", "B", 0.5, "test1")
        await checker.check_integrity("Player is out with injury and is doubtful.", "A", "B", 0.5, "test2")
        
        telemetry = checker.get_telemetry()
        assert telemetry["total_checks"] == 2
        # Check verdict distribution - one confirmed, one caution
        verdict_dist = telemetry.get("verdict_distribution", {})
        assert verdict_dist.get("confirmed") == 1
        assert verdict_dist.get("caution") == 1
    
    def test_telemetry_disabled(self):
        checker = OpenClawLite(enable_telemetry=False)
        assert checker.get_telemetry() is None


class TestEscalationQueue:
    """Test high-stakes escalation queue."""
    
    def test_enqueue_creates_file(self, tmp_path):
        queue = HighStakesEscalationQueue(queue_dir=str(tmp_path / "escalations"))
        
        queue_id = queue.enqueue(
            game_key="Duke@UNC",
            home_team="UNC",
            away_team="Duke",
            recommended_units=2.0,
            integrity_verdict="CAUTION",
            reason="High stakes test"
        )
        
        assert queue_id is not None
        
        # Check file was created
        queue_file = queue.queue_dir / f"{queue_id}.json"
        assert queue_file.exists()
    
    def test_get_pending_returns_list(self, tmp_path):
        queue = HighStakesEscalationQueue(queue_dir=str(tmp_path / "escalations"))
        
        queue.enqueue(
            game_key="Duke@UNC",
            home_team="UNC",
            away_team="Duke",
            recommended_units=2.0,
            integrity_verdict="CAUTION",
            reason="Test"
        )
        
        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0]["game_key"] == "Duke@UNC"
    
    def test_resolve_marks_completed(self, tmp_path):
        queue = HighStakesEscalationQueue(queue_dir=str(tmp_path / "escalations"))
        
        queue_id = queue.enqueue(
            game_key="Duke@UNC",
            home_team="UNC",
            away_team="Duke",
            recommended_units=2.0,
            integrity_verdict="CAUTION",
            reason="Test"
        )
        
        success = queue.resolve(queue_id, "APPROVED", "test_user")
        assert success is True


class TestHighStakesEscalation:
    """Test automatic high-stakes escalation."""
    
    @pytest.mark.asyncio
    async def test_high_units_escalates(self, tmp_path):
        # Use temp directory for escalation queue
        import backend.services.openclaw_lite as ocl
        original_queue_dir = ocl.HighStakesEscalationQueue.__init__
        
        checker = OpenClawLite(enable_telemetry=False)
        # Override queue dir to temp path
        checker.escalation_queue = HighStakesEscalationQueue(queue_dir=str(tmp_path / "escalations"))
        
        result = await checker.check_integrity(
            search_text="Everything looks good.",
            home_team="UNC",
            away_team="Duke",
            recommended_units=2.0,  # High stakes
            game_key="Duke@UNC"
        )
        
        # Check escalation file was created
        pending = checker.escalation_queue.get_pending()
        assert len(pending) == 1
        assert pending[0]["recommended_units"] == 2.0
    
    @pytest.mark.xfail(
        reason="escalate_if_needed paused 2026-04-21 — always returns None until OpenClaw re-enabled",
        strict=True,
    )
    def test_escalate_if_needed_helper(self, tmp_path):
        import backend.services.openclaw_lite as ocl

        # Override the default queue location
        queue = HighStakesEscalationQueue(queue_dir=str(tmp_path / "escalations"))

        # Low stakes - should not escalate
        queue_id = ocl.escalate_if_needed(
            game_key="A@B",
            home_team="B",
            away_team="A",
            recommended_units=0.5,
            integrity_verdict="CONFIRMED",
            is_neutral=False
        )
        assert queue_id is None

        # High stakes - should escalate
        queue_id = ocl.escalate_if_needed(
            game_key="A@B",
            home_team="B",
            away_team="A",
            recommended_units=2.0,
            integrity_verdict="CONFIRMED",
            is_neutral=False
        )
        assert queue_id is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_search_text(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        # Should default to CONFIRMED for empty input
        assert result.verdict == IntegrityVerdict.CONFIRMED.value
    
    def test_very_long_search_text(self):
        checker = OpenClawLite()
        long_text = "Player injury reported. Doubtful status. Starting lineup uncertain. " * 50
        result = checker._check_integrity_heuristic_sync(
            search_text=long_text,
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == IntegrityVerdict.CAUTION.value
    
    def test_case_insensitive_matching(self):
        checker = OpenClawLite()
        result = checker._check_integrity_heuristic_sync(
            search_text="INJURY reported, PLAYER is OUT",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5
        )
        assert result.verdict == IntegrityVerdict.CAUTION.value
    
    @pytest.mark.asyncio
    async def test_error_handling_returns_caution(self):
        checker = OpenClawLite()
        # Pass None as search_text to trigger error
        result = await checker.check_integrity(
            search_text=None,  # type: ignore
            home_team="A",
            away_team="B",
            recommended_units=0.5,
            game_key="test"
        )
        # Should return CAUTION on error, not crash
        assert result.verdict == IntegrityVerdict.CAUTION.value


class TestPerformance:
    """Test that heuristic is fast."""
    
    def test_heuristic_under_10ms(self):
        import time
        checker = OpenClawLite()
        
        start = time.time()
        for _ in range(100):
            checker._check_integrity_heuristic_sync(
                search_text="Some test text with a few words.",
                home_team="Duke",
                away_team="UNC",
                recommended_units=0.5
            )
        elapsed = time.time() - start
        
        # 100 calls should take less than 10ms total
        assert elapsed < 0.01, f"Heuristic too slow: {elapsed*1000:.2f}ms for 100 calls"


class TestSingleton:
    """Test singleton pattern."""
    
    def test_singleton_returns_same_instance(self):
        a = get_openclaw_lite()
        b = get_openclaw_lite()
        assert a is b
    
    def test_singleton_preserves_telemetry_setting(self):
        # Reset singleton
        import backend.services.openclaw_lite as ocl
        ocl._lite_instance = None
        
        # Create with telemetry enabled
        a = get_openclaw_lite(enable_telemetry=True)
        assert a.telemetry is not None
        
        # Get again - should return same instance
        b = get_openclaw_lite(enable_telemetry=False)  # Setting ignored for existing
        assert a is b
        assert b.telemetry is not None  # Preserved from first call
