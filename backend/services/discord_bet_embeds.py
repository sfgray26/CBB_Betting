"""
Improved Discord Bet Notifications — Clear, Actionable, Complete

Key improvements:
1. ALL bet details in ONE message (no more clicking to see picks)
2. Clear ACTION section with exact pick
3. Summary shows ALL bets with key details
4. No more useless notification IDs
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Color codes
_COLOR_GREEN = 0x2ECC71
_COLOR_YELLOW = 0xF1C40F
_COLOR_RED = 0xE74C3C
_COLOR_BLUE = 0x3498DB
_COLOR_GOLD = 0xFFD700


def create_bet_summary_embed(bets: List[Dict], summary: Dict) -> Optional[Dict]:
    """
    Create a SUPER CLEAR bet summary that shows ALL picks immediately.
    """
    n_bets = len(bets)
    if n_bets == 0:
        return None
    
    today = datetime.now(timezone.utc).strftime("%b %d, %Y")
    
    # Build the bet list - show ALL key info in description
    bet_lines = []
    total_units = 0
    
    for i, bet in enumerate(bets, 1):
        home = bet.get('home_team', 'Home')
        away = bet.get('away_team', 'Away')
        bet_side = bet.get('bet_side', 'home')
        spread = bet.get('spread', 0)
        edge = bet.get('edge_conservative', 0) or 0
        units = bet.get('recommended_units', 0) or 0
        
        team_to_bet = home if bet_side == 'home' else away
        spread_val = spread if bet_side == 'home' else -spread
        spread_str = f"{spread_val:+.1f}"
        
        total_units += units
        
        # Format: BET #1: Duke -4.5 (3.2% edge, 1.25u)
        bet_lines.append(
            f"**BET #{i}: {team_to_bet} {spread_str}**\n"
            f"└ Edge: {edge:.1%} | Stake: {units:.2f}u"
        )
    
    # Simple, clear description
    description = "\n\n".join(bet_lines)
    description += f"\n\n💰 **Total: {total_units:.2f} units**"
    
    # Color: Gold for high stakes, Green for standard
    max_units = max(b.get('recommended_units', 0) or 0 for b in bets)
    color = _COLOR_GOLD if max_units >= 1.0 else _COLOR_GREEN
    
    return {
        "title": f"🎯 {n_bets} BET{'S' if n_bets > 1 else ''} FOR {today.upper()}",
        "description": description,
        "color": color,
        "footer": {"text": "CBB Edge v9"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_detailed_bet_embed(bet: Dict, bet_number: int = 1) -> Dict:
    """
    Create a detailed embed for a single bet with ALL information.
    
    This shows the full analysis for users who want details.
    """
    home = bet.get('home_team', 'Home')
    away = bet.get('away_team', 'Away')
    bet_side = bet.get('bet_side', 'home')
    spread = bet.get('spread', 0)
    edge = bet.get('edge_conservative', 0) or 0
    units = bet.get('recommended_units', 0) or 0
    margin = bet.get('projected_margin', 0) or 0
    snr = bet.get('snr')
    verdict = bet.get('verdict', '')
    
    team_to_bet = home if bet_side == 'home' else away
    opponent = away if bet_side == 'home' else home  # noqa: F841
    spread_val = spread if bet_side == 'home' else -spread
    spread_str = f"{spread_val:+.1f}"
    
    # Get notes if available
    notes = bet.get('matchup_notes', [])
    note_text = notes[0] if notes else "Model edge detected"
    
    # Format edge with visual indicator
    if edge >= 0.05:
        edge_emoji = "🔥"
    elif edge >= 0.035:
        edge_emoji = "✅"
    else:
        edge_emoji = "📊"
    
    return {
        "title": f"{bet_number}. BET: {team_to_bet} {spread_str}",
        "description": f"**{away} @ {home}**\n{edge_emoji} {note_text}",
        "color": _COLOR_GREEN,
        "fields": [
            {
                "name": "🎯 ACTION",
                "value": f"**Bet {team_to_bet} {spread_str}**\nEdge: {edge:.1%} | Stake: {units:.2f}u",
                "inline": False
            },
            {
                "name": "📊 Model Projection",
                "value": f"Margin: {margin:+.1f} pts\nSNR: {snr:.0%}" if snr else f"Margin: {margin:+.1f} pts",
                "inline": True
            },
            {
                "name": "💰 Sizing",
                "value": f"{units:.2f} units\nKelly: {bet.get('kelly_fractional', 0):.1%}",
                "inline": True
            },
            {
                "name": "🔍 Verdict",
                "value": verdict,
                "inline": True
            },
        ],
        "footer": {"text": "CBB Edge v9"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_bet_now_alert_embed(game_data: Dict, line_movement: Dict) -> Dict:
    """
    URGENT alert when a line move creates a new BET opportunity.
    
    This is for real-time line monitoring.
    """
    home = game_data.get('home_team', 'Home')
    away = game_data.get('away_team', 'Away')
    old_spread = line_movement.get('old_spread', 0)
    new_spread = line_movement.get('new_spread', 0)
    delta = new_spread - old_spread
    edge = line_movement.get('new_edge', 0)
    
    # Determine which side to bet based on line move
    if delta > 0:  # Line moved toward home
        bet_team = home
        bet_spread = new_spread
        move_desc = f"Home line improved {delta:+.1f} pts"
    else:  # Line moved toward away
        bet_team = away
        bet_spread = -new_spread
        move_desc = f"Away line improved {abs(delta):+.1f} pts"
    
    return {
        "title": f"🚨 BET NOW: {away} @ {home}",
        "description": f"Line movement created a new betting opportunity!\n\n**{move_desc}**",
        "color": _COLOR_GOLD,
        "fields": [
            {
                "name": "⏰ ACTION REQUIRED",
                "value": f"**Bet {bet_team} {bet_spread:+.1f}**",
                "inline": False
            },
            {
                "name": "📈 Line Movement",
                "value": f"{old_spread:+.1f} → {new_spread:+.1f} ({delta:+.1f} pts)",
                "inline": True
            },
            {
                "name": "🎯 Model Edge",
                "value": f"{edge:.1%}",
                "inline": True
            },
            {
                "name": "💡 Why Bet Now?",
                "value": "Line moved toward our side — better price than model entry point.",
                "inline": False
            },
        ],
        "footer": {"text": "⚡ Real-Time Alert — Act quickly!"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_daily_results_embed(results: List[Dict], date_str: str = None) -> Dict:
    """
    End-of-day summary showing results of all bets.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%b %d, %Y")
    
    total_bets = len(results)
    if total_bets == 0:
        return {
            "title": f"📊 Daily Results — {date_str}",
            "description": "No bets placed today.",
            "color": _COLOR_BLUE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    wins = sum(1 for r in results if r.get('outcome') == 1)
    losses = sum(1 for r in results if r.get('outcome') == 0)
    pushes = sum(1 for r in results if r.get('outcome') == -1)
    
    total_profit = sum(r.get('profit_loss_units', 0) or 0 for r in results)
    
    if total_profit > 0:
        color = _COLOR_GREEN
        result_emoji = "🟢"
    elif total_profit < 0:
        color = _COLOR_RED
        result_emoji = "🔴"
    else:
        color = _COLOR_BLUE
        result_emoji = "⚪"
    
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    description = f"**{total_bets} bets | {wins}-{losses}-{pushes} | {win_rate:.0%} WR**\n\n"
    
    # List each result
    for r in results:
        team = r.get('team', 'Unknown')
        outcome = r.get('outcome', 0)
        profit = r.get('profit_loss_units', 0) or 0
        
        if outcome == 1:
            emoji = "✅"
            outcome_str = "WIN"
        elif outcome == 0:
            emoji = "❌"
            outcome_str = "LOSS"
        else:
            emoji = "🔄"
            outcome_str = "PUSH"
        
        description += f"{emoji} {team}: {outcome_str} ({profit:+.2f}u)\n"
    
    description += f"\n**{result_emoji} Total P&L: {total_profit:+.2f}u**"
    
    return {
        "title": f"📊 Daily Results — {date_str}",
        "description": description,
        "color": color,
        "footer": {"text": "CBB Edge Results"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Export for use in discord_notifier.py
__all__ = [
    'create_bet_summary_embed',
    'create_detailed_bet_embed',
    'create_bet_now_alert_embed',
    'create_daily_results_embed',
]
