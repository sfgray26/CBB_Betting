"""
OpenClaw Morning Brief Generator — IMPROVED VERSION

Actually works with real data sources.
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from backend.services.discord_notifier import send_to_channel, _COLOR_GREEN, _COLOR_YELLOW, _COLOR_RED, _COLOR_BLUE
from backend.models import Prediction, Game, BetLog, SessionLocal

logger = logging.getLogger("openclaw_briefs")


@dataclass
class SlateSummary:
    """Summary of today's betting slate."""
    total_games: int = 0
    bet_tier: int = 0
    consider_tier: int = 0
    pass_tier: int = 0
    high_stakes_count: int = 0
    avg_edge: float = 0.0
    total_units: float = 0.0


class ImprovedMorningBriefGenerator:
    """
    Generates an actually useful morning brief with real data.
    """
    
    def __init__(self):
        self.today = datetime.now(timezone.utc)
        self.date_str = self.today.strftime("%B %d, %Y")
        self.weekday = self.today.strftime("%A")
    
    def collect_real_data(self) -> Dict[str, Any]:
        """Collect actual data from the database."""
        try:
            db = SessionLocal()
            today = self.today.date()
            
            # Get today's predictions
            predictions = db.query(Prediction).filter(
                Prediction.prediction_date == today
            ).all()
            
            slate = SlateSummary(total_games=len(predictions))
            edges = []
            
            for pred in predictions:
                verdict = pred.verdict or ""
                units = pred.recommended_units or 0
                edge = pred.edge_conservative or 0
                
                if "BET" in verdict.upper():
                    slate.bet_tier += 1
                    slate.total_units += units
                    edges.append(edge)
                    if units >= 1.0:
                        slate.high_stakes_count += 1
                elif "CONSIDER" in verdict.upper():
                    slate.consider_tier += 1
                else:
                    slate.pass_tier += 1
            
            if edges:
                slate.avg_edge = sum(edges) / len(edges)
            
            # Get yesterday's results if available
            yesterday = today - timedelta(days=1)
            yesterday_bets = db.query(BetLog).join(Game).filter(
                Game.game_date >= yesterday,
                Game.game_date < today,
                BetLog.outcome.isnot(None)
            ).all()
            
            yday_results = {
                'total': len(yesterday_bets),
                'wins': sum(1 for b in yesterday_bets if b.outcome == 1),
                'losses': sum(1 for b in yesterday_bets if b.outcome == 0),
                'profit': sum(b.profit_loss_units or 0 for b in yesterday_bets)
            }
            
            db.close()
            
            return {
                'slate': slate,
                'yesterday': yday_results,
                'tournament_days': self._days_to_tournament()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect real data: {e}")
            return {
                'slate': SlateSummary(),
                'yesterday': {'total': 0, 'wins': 0, 'losses': 0, 'profit': 0},
                'tournament_days': None
            }
    
    def _days_to_tournament(self) -> Optional[int]:
        """Calculate days to March Madness."""
        first_four = datetime(2026, 3, 18).date()
        today = self.today.date()
        days = (first_four - today).days
        return days if days >= 0 else None
    
    def generate_brief(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the morning brief embed."""
        slate = data.get('slate', SlateSummary())
        yday = data.get('yesterday', {})
        tourney_days = data.get('tournament_days')
        
        # Determine color
        if slate.bet_tier > 0:
            color = _COLOR_GREEN
        elif slate.consider_tier > 0:
            color = _COLOR_YELLOW
        else:
            color = _COLOR_BLUE
        
        # Build description
        sections = []
        
        # 1. Today's slate
        if slate.bet_tier > 0:
            sections.append(
                f"🎯 **{slate.bet_tier} BET{'S' if slate.bet_tier > 1 else ''}** today "
                f"({slate.avg_edge:.1%} avg edge, {slate.total_units:.2f}u total)"
            )
            if slate.high_stakes_count > 0:
                sections.append(f"🔥 **{slate.high_stakes_count} high-stakes** (≥1.0u)")
        elif slate.consider_tier > 0:
            sections.append(f"🟡 **{slate.consider_tier} CONSIDER** — watch for line moves")
        else:
            sections.append("⚪ No bets today — model finding no edge")
        
        # 2. Yesterday's results
        if yday.get('total', 0) > 0:
            profit = yday.get('profit', 0)
            profit_emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
            sections.append(
                f"\n📊 **Yesterday**: {yday.get('wins', 0)}-{yday.get('losses', 0)} "
                f"| {profit_emoji} {profit:+.2f}u"
            )
        
        # 3. Tournament countdown
        if tourney_days is not None:
            if tourney_days == 0:
                sections.append("\n🏀 **TOURNAMENT STARTS TODAY!**")
            elif tourney_days == 1:
                sections.append("\n🏀 **First Four TOMORROW!**")
            elif tourney_days > 0:
                sections.append(f"\n🏀 **{tourney_days} days** to March Madness")
        
        description = "\n".join(sections)
        
        # Add action items
        if slate.bet_tier > 0:
            description += "\n\n➡️ Check #cbb-bets for full picks"
        elif slate.consider_tier > 0:
            description += "\n\n➡️ Monitor lines for potential moves toward model"
        
        return {
            "title": f"🌅 Morning Brief — {self.weekday}, {self.date_str}",
            "description": description,
            "color": color,
            "footer": {"text": "CBB Edge v9"},
            "timestamp": self.today.isoformat(),
        }


def generate_and_send_morning_brief_improved() -> bool:
    """
    Generate and send the improved morning brief.

    PAUSED (2026-04-21): OpenClaw morning briefs are disabled to reduce
    Discord noise while the baseball module is being implemented.
    """
    logger.info("Morning brief generation skipped — OpenClaw paused")
    return False


# Canonical alias — callers should use this name
generate_and_send_morning_brief = generate_and_send_morning_brief_improved


if __name__ == "__main__":
    # Test mode
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if "--test" in sys.argv:
        print("Testing improved morning brief...")
        generator = ImprovedMorningBriefGenerator()
        data = generator.collect_real_data()
        embed = generator.generate_brief(data)
        import json
        print(json.dumps(embed, indent=2))
    else:
        success = generate_and_send_morning_brief_improved()
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
