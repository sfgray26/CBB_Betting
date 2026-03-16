"""
OpenClaw Morning Brief Generator

Generates a clean, intelligence-rich morning brief for Discord #openclaw-briefs channel.
Designed to be concise, actionable, and not noisy.

Author: Kimi CLI / Claude Code
Document: OPCL-001 Phase 1
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from backend.services.discord_notifier import send_to_channel, _COLOR_GREEN, _COLOR_YELLOW, _COLOR_RED, _COLOR_BLUE

logger = logging.getLogger("openclaw_briefs")


@dataclass
class SlateSummary:
    """Summary of today's betting slate."""
    total_games: int = 0
    bet_tier: int = 0
    consider_tier: int = 0
    pass_tier: int = 0
    high_stakes_count: int = 0  # >= 1.5u


@dataclass
class IntegrityStatus:
    """Integrity check summary."""
    all_confirmed: bool = True
    caution_count: int = 0
    volatile_count: int = 0
    abort_count: int = 0
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


@dataclass
class SharpMoneyAlert:
    """Sharp money movement alert."""
    game: str = ""
    line_move: str = ""  # e.g., "-3.5 → -4.5"
    pattern: str = ""  # "steam", "opener_gap", "rlm"
    confidence: float = 0.0
    recommendation: str = ""


@dataclass
class TournamentStatus:
    """Tournament tracking info."""
    days_to_first_four: Optional[int] = None
    days_to_round_of_64: Optional[int] = None
    current_round: Optional[str] = None
    cinderella_teams: List[str] = None
    
    def __post_init__(self):
        if self.cinderella_teams is None:
            self.cinderella_teams = []


class MorningBriefGenerator:
    """
    Generates the OpenClaw morning intelligence brief.
    
    Designed to be:
    - Clean: Well-formatted, easy to scan
    - Concise: Only valuable information
    - Actionable: Clear recommendations
    - Not noisy: One message per day, only when relevant
    """
    
    # Tournament dates (2026)
    FIRST_FOUR_DATE = datetime(2026, 3, 18)
    ROUND_OF_64_DATE = datetime(2026, 3, 20)
    
    def __init__(self):
        self.today = datetime.now(timezone.utc)
        self.date_str = self.today.strftime("%B %d, %Y")
        self.weekday = self.today.strftime("%A")
    
    def generate_brief(self, 
                       slate: Optional[SlateSummary] = None,
                       integrity: Optional[IntegrityStatus] = None,
                       sharp_alerts: Optional[List[SharpMoneyAlert]] = None,
                       tournament: Optional[TournamentStatus] = None,
                       escalation_count: int = 0) -> Dict[str, Any]:
        """
        Generate the morning brief embed.
        
        Args:
            slate: Today's slate summary
            integrity: Integrity check status
            sharp_alerts: Sharp money alerts from overnight
            tournament: Tournament tracking info
            escalation_count: Number of pending escalations
        
        Returns:
            Discord embed dict
        """
        # Determine overall color based on status
        color = self._determine_color(slate, integrity, escalation_count)
        
        # Build sections
        sections = []
        
        # 1. Header with slate summary
        header = self._build_header(slate)
        sections.append(header)
        
        # 2. Integrity status (only if not all confirmed)
        if integrity and not integrity.all_confirmed:
            integrity_section = self._build_integrity_section(integrity)
            sections.append(integrity_section)
        
        # 3. Sharp money alerts (only if present)
        if sharp_alerts:
            sharp_section = self._build_sharp_section(sharp_alerts)
            sections.append(sharp_section)
        
        # 4. Tournament countdown/status
        tournament_section = self._build_tournament_section(tournament)
        if tournament_section:
            sections.append(tournament_section)
        
        # 5. Escalation queue status (only if pending)
        if escalation_count > 0:
            escalation_section = f"🚨 **{escalation_count} high-stakes game(s)** pending manual review"
            sections.append(escalation_section)
        
        # Combine into description
        description = "\n\n".join(sections)
        
        # Build embed
        embed = {
            "title": f"🌅 OpenClaw Morning Brief — {self.weekday}, {self.date_str}",
            "description": description,
            "color": color,
            "footer": {
                "text": "OpenClaw Intelligence v3.0 | OPCL-001"
            },
            "timestamp": self.today.isoformat(),
        }
        
        return embed
    
    def _determine_color(self, slate: Optional[SlateSummary], 
                         integrity: Optional[IntegrityStatus],
                         escalation_count: int) -> int:
        """Determine embed color based on overall status."""
        # Red if abort or high escalations
        if (integrity and integrity.abort_count > 0) or escalation_count >= 3:
            return _COLOR_RED
        
        # Yellow if volatile or some escalations
        if (integrity and integrity.volatile_count > 0) or escalation_count > 0:
            return _COLOR_YELLOW
        
        # Green if bets available
        if slate and slate.bet_tier > 0:
            return _COLOR_GREEN
        
        # Default blue
        return _COLOR_BLUE
    
    def _build_header(self, slate: Optional[SlateSummary]) -> str:
        """Build the header section with slate summary."""
        if not slate:
            return "📊 **Today's Slate**: Data pending analysis"
        
        lines = [f"📊 **Today's Slate**: {slate.total_games} games"]
        
        if slate.bet_tier > 0:
            lines.append(f"• **{slate.bet_tier} BET** recommendation(s)")
        if slate.consider_tier > 0:
            lines.append(f"• **{slate.consider_tier} CONSIDER** — monitor for line moves")
        if slate.pass_tier > 0:
            lines.append(f"• **{slate.pass_tier} PASS** — no edge found")
        if slate.high_stakes_count > 0:
            lines.append(f"• **{slate.high_stakes_count} high-stakes** (≥1.5u)")
        
        return "\n".join(lines)
    
    def _build_integrity_section(self, integrity: IntegrityStatus) -> str:
        """Build integrity status section."""
        parts = ["🔍 **Integrity Status**:"]
        
        if integrity.abort_count > 0:
            parts.append(f"⚠️ **{integrity.abort_count} ABORT** — Do not bet")
        if integrity.volatile_count > 0:
            parts.append(f"🟡 **{integrity.volatile_count} VOLATILE** — High uncertainty")
        if integrity.caution_count > 0:
            parts.append(f"⚡ **{integrity.caution_count} CAUTION** — Monitor closely")
        
        if integrity.issues:
            parts.append(f"Issues: {', '.join(integrity.issues[:3])}")  # Max 3 issues
        
        return "\n".join(parts)
    
    def _build_sharp_section(self, alerts: List[SharpMoneyAlert]) -> str:
        """Build sharp money alerts section."""
        parts = ["⚡ **Sharp Money Overnight**:"]
        
        for alert in alerts[:3]:  # Max 3 alerts
            emoji = "🔥" if alert.confidence >= 0.8 else "📈"
            parts.append(
                f"{emoji} **{alert.game}**: {alert.line_move} "
                f"({alert.pattern})"
            )
            if alert.recommendation:
                parts.append(f"   → {alert.recommendation}")
        
        return "\n".join(parts)
    
    def _build_tournament_section(self, tournament: Optional[TournamentStatus]) -> Optional[str]:
        """Build tournament status section."""
        if tournament and tournament.current_round:
            # Active tournament
            cinderella_str = ""
            if tournament.cinderella_teams:
                cinderella_str = f" | Cinderella watch: {', '.join(tournament.cinderella_teams[:2])}"
            
            return f"🏀 **Tournament**: {tournament.current_round}{cinderella_str}"
        
        # Countdown mode
        days_to_first = self._days_to(self.FIRST_FOUR_DATE)
        days_to_64 = self._days_to(self.ROUND_OF_64_DATE)
        
        if days_to_first is not None and days_to_first > 0:
            if days_to_first == 1:
                return "🏀 **Tournament**: First Four starts TOMORROW!"
            return f"🏀 **Tournament**: {days_to_first} days to First Four"
        
        if days_to_64 is not None and days_to_64 > 0:
            if days_to_64 == 1:
                return "🏀 **Tournament**: Round of 64 starts TOMORROW!"
            return f"🏀 **Tournament**: {days_to_64} days to Round of 64"
        
        return None
    
    def _days_to(self, target_date: datetime) -> Optional[int]:
        """Calculate days until target date."""
        today = self.today.date()
        target = target_date.date()
        delta = (target - today).days
        return delta if delta >= 0 else None


class BriefDataCollector:
    """
    Collects data for the morning brief from various sources.
    
    This is the data layer - it queries the database, OpenClaw,
    and other services to gather the information needed for the brief.
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger("openclaw_briefs.collector")
    
    async def collect_slate_summary(self) -> SlateSummary:
        """
        Collect today's slate summary from predictions.
        
        Returns:
            SlateSummary with today's game counts
        """
        try:
            from backend.models import Prediction, SessionLocal
            
            if not self.db:
                self.db = SessionLocal()
            
            today = datetime.now(timezone.utc).date()
            
            # Query today's predictions
            predictions = self.db.query(Prediction).filter(
                Prediction.game_date == today
            ).all()
            
            slate = SlateSummary(total_games=len(predictions))
            
            for pred in predictions:
                verdict = pred.verdict or ""
                if "BET" in verdict.upper():
                    slate.bet_tier += 1
                    # Check if high stakes
                    units = pred.recommended_units or 0
                    if units >= 1.5:
                        slate.high_stakes_count += 1
                elif "CONSIDER" in verdict.upper():
                    slate.consider_tier += 1
                else:
                    slate.pass_tier += 1
            
            return slate
            
        except Exception as e:
            self.logger.error(f"Failed to collect slate summary: {e}")
            return SlateSummary()
    
    async def collect_integrity_status(self) -> IntegrityStatus:
        """
        Collect integrity check status from OpenClaw.
        
        Returns:
            IntegrityStatus with summary of checks
        """
        try:
            # This would integrate with openclaw_lite telemetry
            # For now, return clean status
            # TODO: Wire to actual OpenClaw telemetry
            return IntegrityStatus(all_confirmed=True)
            
        except Exception as e:
            self.logger.error(f"Failed to collect integrity status: {e}")
            return IntegrityStatus(all_confirmed=True)
    
    async def collect_sharp_alerts(self) -> List[SharpMoneyAlert]:
        """
        Collect sharp money alerts from overnight.
        
        Returns:
            List of SharpMoneyAlert objects
        """
        try:
            # This would query sharp_money service
            # For now, return empty
            # TODO: Wire to sharp_money detector
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to collect sharp alerts: {e}")
            return []
    
    async def collect_escalation_count(self) -> int:
        """
        Get count of pending escalations.
        
        Returns:
            Number of pending high-stakes escalations
        """
        try:
            from backend.services.openclaw_lite import HighStakesEscalationQueue
            
            queue = HighStakesEscalationQueue()
            pending = queue.get_pending()
            return len(pending)
            
        except Exception as e:
            self.logger.error(f"Failed to collect escalation count: {e}")
            return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_and_send_morning_brief(db_session=None) -> bool:
    """
    Generate and send the morning brief to Discord.
    
    This is the main entry point for the daily brief.
    
    Args:
        db_session: Optional database session
    
    Returns:
        True if sent successfully
    """
    logger.info("Generating morning brief...")
    
    try:
        # Collect data
        collector = BriefDataCollector(db_session)
        
        slate = await collector.collect_slate_summary()
        integrity = await collector.collect_integrity_status()
        sharp_alerts = await collector.collect_sharp_alerts()
        escalation_count = await collector.collect_escalation_count()
        
        # Generate brief
        generator = MorningBriefGenerator()
        embed = generator.generate_brief(
            slate=slate,
            integrity=integrity,
            sharp_alerts=sharp_alerts,
            escalation_count=escalation_count
        )
        
        # Send to Discord
        success = send_to_channel("openclaw-briefs", embed=embed)
        
        if success:
            logger.info("Morning brief sent successfully")
        else:
            logger.warning("Failed to send morning brief to Discord")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to generate morning brief: {e}")
        return False


def generate_brief_sync(db_session=None) -> bool:
    """
    Synchronous wrapper for generate_and_send_morning_brief.
    
    Use this for cron jobs or non-async contexts.
    """
    import asyncio
    return asyncio.run(generate_and_send_morning_brief(db_session))


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow running directly for testing
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test mode: generate sample brief
    if "--test" in sys.argv:
        print("Generating test brief...")
        
        generator = MorningBriefGenerator()
        
        # Create sample data
        slate = SlateSummary(
            total_games=8,
            bet_tier=2,
            consider_tier=1,
            pass_tier=5,
            high_stakes_count=1
        )
        
        integrity = IntegrityStatus(
            all_confirmed=False,
            caution_count=1,
            issues=["Gonzaga: Key player questionable"]
        )
        
        sharp_alerts = [
            SharpMoneyAlert(
                game="Duke @ UNC",
                line_move="-2.5 → -3.5",
                pattern="steam",
                confidence=0.85,
                recommendation="Wait for line to stabilize"
            )
        ]
        
        embed = generator.generate_brief(
            slate=slate,
            integrity=integrity,
            sharp_alerts=sharp_alerts,
            escalation_count=0
        )
        
        print("Embed generated:")
        import json
        print(json.dumps(embed, indent=2))
        
        # Send if requested
        if "--send" in sys.argv:
            print("\nSending to Discord...")
            success = send_to_channel("openclaw-briefs", embed=embed)
            print(f"Send result: {'SUCCESS' if success else 'FAILED'}")
    
    else:
        # Normal mode: generate and send real brief
        print("Generating and sending morning brief...")
        success = generate_brief_sync()
        sys.exit(0 if success else 1)
