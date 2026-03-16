"""
OpenClaw Lite v3.0 — Simplified Integrity Coordination Service

A production-ready, lightweight integrity checking system that replaces
the complex v2.0 coordinator with a focused, high-performance implementation.

Key Features:
- Fast heuristic-based integrity checks (<1ms)
- Async-native design for concurrent processing
- Built-in telemetry and metrics
- High-stakes escalation queue
- No external LLM dependencies

Migration Notes:
- v2.1: Initial heuristic-based implementation
- v3.0: Added async support, telemetry, escalation queue
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("openclaw_lite")


class IntegrityVerdict(Enum):
    """Standard integrity verdicts."""
    CONFIRMED = "CONFIRMED"
    CAUTION = "CAUTION"
    VOLATILE = "VOLATILE"
    ABORT = "ABORT"
    RED_FLAG = "RED FLAG"


@dataclass
class IntegrityResult:
    """Result of an integrity check."""
    verdict: str  # CONFIRMED, CAUTION, VOLATILE, ABORT, RED FLAG
    confidence: float  # 0.0-1.0
    reasoning: str
    source: str  # "heuristic", "kimi"
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TelemetrySnapshot:
    """Snapshot of OpenClaw performance metrics."""
    total_checks: int = 0
    heuristic_checks: int = 0
    escalated_checks: int = 0
    fallback_checks: int = 0
    
    # Verdict distribution
    confirmed_count: int = 0
    caution_count: int = 0
    volatile_count: int = 0
    abort_count: int = 0
    red_flag_count: int = 0
    
    # Performance
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Error tracking
    errors: int = 0
    last_error: Optional[str] = None
    
    def record_check(self, result: IntegrityResult):
        """Record a check result."""
        self.total_checks += 1
        
        if result.source == "heuristic":
            self.heuristic_checks += 1
        elif result.source == "kimi":
            self.escalated_checks += 1
        
        # Count verdicts
        verdict = result.verdict.upper()
        if "CONFIRMED" in verdict:
            self.confirmed_count += 1
        elif "CAUTION" in verdict:
            self.caution_count += 1
        elif "VOLATILE" in verdict:
            self.volatile_count += 1
        elif "ABORT" in verdict:
            self.abort_count += 1
        elif "RED FLAG" in verdict:
            self.red_flag_count += 1
        
        # Track latency
        self.total_latency_ms += result.latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_checks
        self.max_latency_ms = max(self.max_latency_ms, result.latency_ms)
    
    def record_error(self, error: str):
        """Record an error."""
        self.errors += 1
        self.last_error = error
    
    def to_dict(self) -> Dict:
        return {
            "total_checks": self.total_checks,
            "heuristic_checks": self.heuristic_checks,
            "escalated_checks": self.escalated_checks,
            "fallback_checks": self.fallback_checks,
            "verdict_distribution": {
                "confirmed": self.confirmed_count,
                "caution": self.caution_count,
                "volatile": self.volatile_count,
                "abort": self.abort_count,
                "red_flag": self.red_flag_count,
            },
            "performance": {
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "max_latency_ms": round(self.max_latency_ms, 2),
            },
            "errors": self.errors,
            "last_error": self.last_error,
        }


class HighStakesEscalationQueue:
    """
    File-based queue for high-stakes games requiring manual review.
    
    Games are flagged when:
    - recommended_units >= 1.5
    - Tournament Elite Eight or later
    - VOLATILE verdict received
    """
    
    def __init__(self, queue_dir: str = ".openclaw/escalation_queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
    
    def enqueue(self, game_key: str, home_team: str, away_team: str, 
                recommended_units: float, integrity_verdict: Optional[str],
                reason: str) -> str:
        """
        Add a game to the escalation queue.
        
        Returns:
            queue_id: Unique identifier for this escalation
        """
        queue_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{game_key.replace('@', '_')}"
        
        entry = {
            "queue_id": queue_id,
            "timestamp": datetime.utcnow().isoformat(),
            "game_key": game_key,
            "home_team": home_team,
            "away_team": away_team,
            "recommended_units": recommended_units,
            "integrity_verdict": integrity_verdict,
            "escalation_reason": reason,
            "status": "pending_review",
        }
        
        queue_file = self.queue_dir / f"{queue_id}.json"
        with open(queue_file, 'w') as f:
            json.dump(entry, f, indent=2)
        
        logger.warning(
            "HIGH-STAKES ESCALATION: %s @ %s (%.2fu) - %s",
            away_team, home_team, recommended_units, reason
        )
        
        return queue_id
    
    def get_pending(self) -> List[Dict]:
        """Get all pending escalations."""
        pending = []
        for f in self.queue_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    entry = json.load(fp)
                    if entry.get("status") == "pending_review":
                        pending.append(entry)
            except Exception as e:
                logger.warning(f"Failed to read escalation file {f}: {e}")
        
        return sorted(pending, key=lambda x: x.get("timestamp", ""))
    
    def resolve(self, queue_id: str, resolution: str, reviewer: str) -> bool:
        """Mark an escalation as resolved."""
        queue_file = self.queue_dir / f"{queue_id}.json"
        if not queue_file.exists():
            return False
        
        try:
            with open(queue_file) as f:
                entry = json.load(f)
            
            entry["status"] = "resolved"
            entry["resolution"] = resolution
            entry["reviewer"] = reviewer
            entry["resolved_at"] = datetime.utcnow().isoformat()
            
            with open(queue_file, 'w') as f:
                json.dump(entry, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to resolve escalation {queue_id}: {e}")
            return False


class OpenClawLite:
    """
    Lightweight, high-performance integrity checker.
    
    Uses tuned heuristics for fast decisions with optional escalation
    for high-stakes scenarios.
    """
    
    # Risk keyword taxonomies.
    # WORD_BOUNDARY_KEYWORDS: single-word terms susceptible to substring false
    # positives (e.g. "out" in "outstanding", "timeout", "shout").  These are
    # matched with \b word-boundary regex instead of a plain substring check.
    WORD_BOUNDARY_KEYWORDS = {"out", "miss", "star", "rest"}

    HIGH_RISK_KEYWORDS = [
        "injury", "injured", "out", "doubtful", "questionable",
        "suspension", "suspended", "arrest", "investigation",
        "lineup", "starting", "star", "key player", "miss",
        "absence", "absent", "won't play", "will not play",
        "ruled out", "done for season", "medical", "trainer",
        "ankle", "knee", "concussion", "protocol"
    ]
    
    MODERATE_RISK_KEYWORDS = [
        "questionable", "game time decision", "gtp", "uncertain",
        "unclear", "monitor", "limited", "restriction", "rest",
        "fatigue", "tired", "heavy minutes", "back-to-back",
        "travel", "delay", "late arrival", "weather", "missed"
    ]
    
    CONFLICT_KEYWORDS = [
        "conflicting", "disagree", "contradictory", "unclear",
        "rumor", "speculation", "reportedly", "sources say",
        "confusion", "uncertain", "unknown", "mystery"
    ]
    
    CRITICAL_ABORT_KEYWORDS = [
        "star player out", "key player out", "major injury",
        "season ending", "surgery", "career ending"
    ]
    
    ABORT_KEYWORDS = [
        "scandal", "investigation", "arrested", "arrest", 
        "suspended indefinitely", "banned", "cheating"
    ]
    
    # High-stakes thresholds
    HIGH_STAKES_UNITS = 1.5
    ELITE_EIGHT_ROUND = 4
    
    def __init__(self, enable_telemetry: bool = True):
        self.telemetry = TelemetrySnapshot() if enable_telemetry else None
        self.escalation_queue = HighStakesEscalationQueue()
        self._semaphore = asyncio.Semaphore(8)  # Max concurrent checks
    
    def _kw_match(self, kw: str, text_lower: str) -> bool:
        """
        Return True when keyword ``kw`` matches in ``text_lower``.

        Single-word terms listed in WORD_BOUNDARY_KEYWORDS are matched with
        ``\\b`` word boundaries so that short words like "out" do not fire on
        "outstanding", "timeout", or "shout".  All other keywords use a plain
        substring check (which is fine for multi-word phrases or longer unique
        terms like "concussion", "injury", etc.).
        """
        if kw in self.WORD_BOUNDARY_KEYWORDS:
            return bool(re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        return kw in text_lower

    def _check_integrity_heuristic_sync(
        self,
        search_text: str,
        home_team: str,
        away_team: str,
        recommended_units: float = 0.0,
        is_elite_eight_or_later: bool = False
    ) -> IntegrityResult:
        """
        Synchronous heuristic check - core decision logic.

        This is the primary path for all integrity checks.
        Returns in <1ms for typical inputs.
        """
        import time
        start = time.time()

        text_lower = search_text.lower()

        # 1. Critical abort conditions (check first for speed)
        for critical in self.CRITICAL_ABORT_KEYWORDS:
            if critical in text_lower:
                return IntegrityResult(
                    verdict=IntegrityVerdict.ABORT.value,
                    confidence=0.9,
                    reasoning=f"Critical signal detected: {critical}",
                    source="heuristic",
                    latency_ms=(time.time() - start) * 1000
                )
        
        # 2. Serious abort conditions
        abort_hits = sum(1 for kw in self.ABORT_KEYWORDS if kw in text_lower)
        if abort_hits >= 1:
            return IntegrityResult(
                verdict=IntegrityVerdict.ABORT.value,
                confidence=0.85,
                reasoning="Serious issue detected",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 3. Keyword counting (uses word-boundary matching for ambiguous short words)
        high_risk_hits = sum(1 for kw in self.HIGH_RISK_KEYWORDS if self._kw_match(kw, text_lower))
        moderate_hits = sum(1 for kw in self.MODERATE_RISK_KEYWORDS if self._kw_match(kw, text_lower))
        conflict_hits = sum(1 for kw in self.CONFLICT_KEYWORDS if self._kw_match(kw, text_lower))
        
        # 4. Late-breaking uncertainty
        late_breaking = any(x in text_lower for x in ["late", "just", "breaking", "developing"])
        uncertainty = any(x in text_lower for x in ["uncertain", "unclear", "unknown", "mystery", "monitor", "status"])
        missed_status = "missed" in text_lower and any(x in text_lower for x in ["practice", "shootaround", "warmup"])
        
        if late_breaking and uncertainty:
            return IntegrityResult(
                verdict=IntegrityVerdict.VOLATILE.value,
                confidence=0.70,
                reasoning="Late-breaking uncertainty",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        if missed_status and uncertainty:
            return IntegrityResult(
                verdict=IntegrityVerdict.VOLATILE.value,
                confidence=0.75,
                reasoning="Key player missed activity with uncertain status",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 5. Conflicting information
        if conflict_hits >= 2 or "conflicting" in text_lower:
            return IntegrityResult(
                verdict=IntegrityVerdict.VOLATILE.value,
                confidence=0.70,
                reasoning="Conflicting or uncertain information detected",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 6. Multiple high-risk signals
        if high_risk_hits >= 3:
            return IntegrityResult(
                verdict=IntegrityVerdict.CAUTION.value,
                confidence=0.75,
                reasoning=f"Multiple risk indicators ({high_risk_hits} high-risk signals)",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 7. Uncertain status
        if uncertainty and ("questionable" in text_lower or "doubtful" in text_lower):
            return IntegrityResult(
                verdict=IntegrityVerdict.VOLATILE.value,
                confidence=0.65,
                reasoning="Uncertain player status",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 8. Star player issues
        if ("star" in text_lower or "key" in text_lower) and \
           ("out" in text_lower or "doubtful" in text_lower or "questionable" in text_lower):
            return IntegrityResult(
                verdict=IntegrityVerdict.CAUTION.value,
                confidence=0.80,
                reasoning="Key player status concerns",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 9. Suspension
        if "suspension" in text_lower or "suspended" in text_lower:
            return IntegrityResult(
                verdict=IntegrityVerdict.CAUTION.value,
                confidence=0.75,
                reasoning="Suspension affecting availability",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 10. Moderate risk accumulation
        if high_risk_hits >= 2 or moderate_hits >= 3:
            return IntegrityResult(
                verdict=IntegrityVerdict.CAUTION.value,
                confidence=0.70,
                reasoning=f"Elevated risk profile ({high_risk_hits} risk, {moderate_hits} moderate signals)",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 11. Fatigue/schedule concerns
        if any(x in text_lower for x in ["back-to-back", "fatigue", "tired", "heavy minutes", "delay", "travel"]):
            return IntegrityResult(
                verdict=IntegrityVerdict.CAUTION.value,
                confidence=0.65,
                reasoning="Schedule or fatigue concerns",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 12. Single moderate risk
        if moderate_hits >= 1 and high_risk_hits >= 1:
            return IntegrityResult(
                verdict=IntegrityVerdict.CAUTION.value,
                confidence=0.60,
                reasoning="Risk factors present",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 13. High stakes with any concern
        if recommended_units >= self.HIGH_STAKES_UNITS and (high_risk_hits >= 1 or moderate_hits >= 1):
            return IntegrityResult(
                verdict=IntegrityVerdict.CAUTION.value,
                confidence=0.75,
                reasoning=f"High-stakes bet with risk signals",
                source="heuristic",
                latency_ms=(time.time() - start) * 1000
            )
        
        # 14. Clean
        return IntegrityResult(
            verdict=IntegrityVerdict.CONFIRMED.value,
            confidence=0.90,
            reasoning="No concerning signals in search results",
            source="heuristic",
            latency_ms=(time.time() - start) * 1000
        )
    
    async def check_integrity(
        self,
        search_text: str,
        home_team: str,
        away_team: str,
        recommended_units: float = 0.0,
        is_elite_eight_or_later: bool = False,
        game_key: Optional[str] = None
    ) -> IntegrityResult:
        """
        Async integrity check with concurrency control and telemetry.
        
        This is the primary entry point for production use.
        """
        async with self._semaphore:
            try:
                result = self._check_integrity_heuristic_sync(
                    search_text, home_team, away_team,
                    recommended_units, is_elite_eight_or_later
                )
                
                # Check for high-stakes escalation
                needs_escalation = (
                    recommended_units >= self.HIGH_STAKES_UNITS or
                    is_elite_eight_or_later or
                    "VOLATILE" in result.verdict
                )
                
                if needs_escalation and game_key:
                    reason = []
                    if recommended_units >= self.HIGH_STAKES_UNITS:
                        reason.append(f"high stakes ({recommended_units}u)")
                    if is_elite_eight_or_later:
                        reason.append("tournament game")
                    if "VOLATILE" in result.verdict:
                        reason.append("volatile verdict")
                    
                    self.escalation_queue.enqueue(
                        game_key=game_key,
                        home_team=home_team,
                        away_team=away_team,
                        recommended_units=recommended_units,
                        integrity_verdict=result.verdict,
                        reason="; ".join(reason)
                    )
                
                # Record telemetry
                if self.telemetry:
                    self.telemetry.record_check(result)
                
                return result
                
            except Exception as e:
                logger.error(f"Integrity check failed for {game_key}: {e}")
                if self.telemetry:
                    self.telemetry.record_error(str(e))
                
                return IntegrityResult(
                    verdict=IntegrityVerdict.CAUTION.value,
                    confidence=0.5,
                    reasoning=f"Check failed: {str(e)}",
                    source="heuristic",
                    latency_ms=0.0
                )
    
    def get_telemetry(self) -> Optional[Dict]:
        """Get current telemetry snapshot."""
        return self.telemetry.to_dict() if self.telemetry else None
    
    def reset_telemetry(self):
        """Reset telemetry counters."""
        if self.telemetry:
            self.telemetry = TelemetrySnapshot()


# Global instance
_lite_instance: Optional[OpenClawLite] = None


def get_openclaw_lite(enable_telemetry: bool = True) -> OpenClawLite:
    """Get or create the singleton OpenClaw Lite instance."""
    global _lite_instance
    if _lite_instance is None:
        _lite_instance = OpenClawLite(enable_telemetry=enable_telemetry)
    return _lite_instance


# ============================================================================
# Backward-Compatible API
# ============================================================================

def perform_sanity_check(
    home_team: str,
    away_team: str,
    verdict: str,
    search_results: str,
    is_elite_eight_or_later: bool = False,
    game_key: Optional[str] = None
) -> str:
    """
    Backward-compatible synchronous wrapper for integrity checks.
    
    Used by:
    - backend/services/scout.py
    - backend/services/analysis.py
    
    Args:
        home_team: Home team name
        away_team: Away team name
        verdict: Model verdict string (e.g., "Bet 1.0u Duke -4")
        search_results: Search results text to analyze
        is_elite_eight_or_later: Whether this is a tournament game
        game_key: Unique game identifier for escalation
    
    Returns:
        Verdict string: "CONFIRMED", "CAUTION", "VOLATILE", "ABORT", or "RED FLAG"
    """
    # Parse recommended units from verdict
    units_match = re.search(r'(\d+\.?\d*)u', verdict)
    recommended_units = float(units_match.group(1)) if units_match else 0.0
    
    # Run check
    checker = get_openclaw_lite()
    result = checker._check_integrity_heuristic_sync(
        search_text=search_results,
        home_team=home_team,
        away_team=away_team,
        recommended_units=recommended_units,
        is_elite_eight_or_later=is_elite_eight_or_later
    )
    
    # Handle high-stakes escalation (fire and forget)
    needs_escalation = (
        recommended_units >= OpenClawLite.HIGH_STAKES_UNITS or
        is_elite_eight_or_later or
        "VOLATILE" in result.verdict
    )
    
    if needs_escalation:
        reason = []
        if recommended_units >= OpenClawLite.HIGH_STAKES_UNITS:
            reason.append(f"high stakes ({recommended_units}u)")
        if is_elite_eight_or_later:
            reason.append("tournament game")
        if "VOLATILE" in result.verdict:
            reason.append("volatile verdict")
        
        game_key = game_key or f"{away_team}@{home_team}"
        checker.escalation_queue.enqueue(
            game_key=game_key,
            home_team=home_team,
            away_team=away_team,
            recommended_units=recommended_units,
            integrity_verdict=result.verdict,
            reason="; ".join(reason)
        )
    
    return result.verdict


async def async_perform_sanity_check(
    home_team: str,
    away_team: str,
    verdict: str,
    search_results: str,
    is_elite_eight_or_later: bool = False,
    game_key: Optional[str] = None
) -> str:
    """
    Async version of perform_sanity_check for concurrent processing.
    
    Use this in async contexts (like analysis.py integrity sweep).
    """
    units_match = re.search(r'(\d+\.?\d*)u', verdict)
    recommended_units = float(units_match.group(1)) if units_match else 0.0
    
    checker = get_openclaw_lite()
    result = await checker.check_integrity(
        search_text=search_results,
        home_team=home_team,
        away_team=away_team,
        recommended_units=recommended_units,
        is_elite_eight_or_later=is_elite_eight_or_later,
        game_key=game_key or f"{away_team}@{home_team}"
    )
    
    return result.verdict


# ============================================================================
# High-Stakes Escalation Helper
# ============================================================================

def get_escalation_queue() -> HighStakesEscalationQueue:
    """Get the escalation queue instance."""
    return HighStakesEscalationQueue()


def escalate_if_needed(
    game_key: str,
    home_team: str,
    away_team: str,
    recommended_units: float,
    integrity_verdict: Optional[str],
    is_neutral: bool = False
) -> Optional[str]:
    """
    Manually trigger escalation for a game.
    
    Returns queue_id if escalated, None otherwise.
    """
    needs_escalation = (
        recommended_units >= OpenClawLite.HIGH_STAKES_UNITS or
        "VOLATILE" in (integrity_verdict or "")
    )
    
    if not needs_escalation:
        return None
    
    reason = []
    if recommended_units >= OpenClawLite.HIGH_STAKES_UNITS:
        reason.append(f"high stakes ({recommended_units}u)")
    if "VOLATILE" in (integrity_verdict or ""):
        reason.append("volatile verdict")
    
    queue = get_escalation_queue()
    return queue.enqueue(
        game_key=game_key,
        home_team=home_team,
        away_team=away_team,
        recommended_units=recommended_units,
        integrity_verdict=integrity_verdict,
        reason="; ".join(reason)
    )
