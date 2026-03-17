"""
Simplified Discord Notification System — Clean, Useful, Actionable

Consolidates down to 3 essential channels:
1. cbb-bets — Daily picks, morning briefs, live bet alerts
2. cbb-results — End-of-day results, P&L tracking, weekly summaries  
3. cbb-alerts — System issues, line movements, urgent notifications

All fantasy/openclaw channels removed (user said fantasy baseball is paused).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_DISCORD_API_BASE = "https://discord.com/api/v10"

# Simplified color scheme
_COLOR_GREEN = 0x2ECC71   # wins, bets
_COLOR_RED = 0xE74C3C     # losses, alerts
_COLOR_BLUE = 0x3498DB    # info, briefs
_COLOR_GOLD = 0xFFD700    # high stakes

# Essential channels only — with backward compatibility
CHANNEL_BETS = os.getenv("DISCORD_CHANNEL_CBB_BETS") or os.getenv("DISCORD_CHANNEL_CBB_BRIEF")
CHANNEL_RESULTS = os.getenv("DISCORD_CHANNEL_CBB_RESULTS") or os.getenv("DISCORD_CHANNEL_CBB_BRIEF") or os.getenv("DISCORD_CHANNEL_ID")
CHANNEL_ALERTS = os.getenv("DISCORD_CHANNEL_CBB_ALERTS") or os.getenv("DISCORD_CHANNEL_CBB_BRIEF")
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")


def _bot_token() -> Optional[str]:
    return BOT_TOKEN


def _channel_id(channel_type: str) -> Optional[str]:
    """Get channel ID for notification type."""
    mapping = {
        "bets": CHANNEL_BETS,
        "morning": CHANNEL_BETS,  # Morning brief goes to bets channel
        "results": CHANNEL_RESULTS,
        "alert": CHANNEL_ALERTS,
        "urgent": CHANNEL_ALERTS,
    }
    return mapping.get(channel_type)


def _post(channel_id: str, content: Optional[str] = None, embeds: Optional[List[Dict]] = None) -> bool:
    """Post a message to Discord."""
    if not _bot_token() or not channel_id:
        return False
    
    url = f"{_DISCORD_API_BASE}/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {_bot_token()}",
        "Content-Type": "application/json",
    }
    
    payload = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Discord post failed: {e}")
        return False


# =============================================================================
# MORNING BRIEF — What to bet today
# =============================================================================

def send_morning_brief(
    bets: List[Dict],
    slate_summary: Dict,
    chaos_level: Optional[float] = None
) -> bool:
    """
    Send morning briefing with today's bets.
    
    Args:
        bets: List of bet recommendations
        slate_summary: Summary of today's slate
        chaos_level: Tournament bracket chaos level (if applicable)
    """
    channel = _channel_id("morning")
    if not channel:
        logger.warning("Morning brief: No Discord channel configured")
        return False
    
    today = datetime.now(timezone.utc).strftime("%A, %B %d")
    
    # Build bet list
    bet_lines = []
    total_units = 0
    
    for i, bet in enumerate(bets[:10], 1):  # Max 10 bets
        team = bet.get("team", "Unknown")
        spread = bet.get("spread", 0)
        edge = bet.get("edge", 0) * 100
        units = bet.get("units", 0)
        total_units += units
        
        emoji = "🔥" if edge >= 5 else "✅" if edge >= 3.5 else "📊"
        bet_lines.append(f"{emoji} **{team} {spread:+.1f}** — {edge:.1f}% edge, {units:.2f}u")
    
    if not bet_lines:
        bet_lines.append("📭 No bets today — all games pass the model")
    
    # Add tournament bracket info if during tournament
    tournament_info = ""
    if chaos_level is not None:
        mode = "CHALK" if chaos_level < 0.3 else "BALANCED" if chaos_level < 0.7 else "CHAOS"
        tournament_info = f"\n🏀 **Tournament Mode:** {mode} (chaos={chaos_level:.1f})"
    
    embed = {
        "title": f"🌅 MORNING BRIEF — {today}",
        "description": (
            f"**{slate_summary.get('n_games', 0)} games** on the slate\n"
            f"**{len(bets)} bets** recommended ({total_units:.2f}u total)"
            f"{tournament_info}\n\n"
            + "\n".join(bet_lines[:5])  # Show first 5
        ),
        "color": _COLOR_BLUE,
        "footer": {
            "text": f"V9.1 Model | CLV: {slate_summary.get('avg_clv', 0):.1f}%"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Add link to dashboard
    dashboard_url = os.getenv("DASHBOARD_URL", "")
    if dashboard_url and len(bets) > 5:
        embed["description"] += f"\n\n🔗 [View all {len(bets)} bets]({dashboard_url})"
    
    return _post(channel, embeds=[embed])


# =============================================================================
# LIVE BET ALERT — When line moves create opportunity
# =============================================================================

def send_live_bet_alert(
    game: str,
    line_movement: Dict,
    model_edge: float
) -> bool:
    """
    Send urgent alert when a line movement creates a betting opportunity.
    """
    channel = _channel_id("bets")
    if not channel:
        return False
    
    team = line_movement.get("team", "Unknown")
    old_line = line_movement.get("old_line", 0)
    new_line = line_movement.get("new_line", 0)
    
    embed = {
        "title": f"⚡ LIVE ALERT — {game}",
        "description": (
            f"**Line moved toward our side!**\n\n"
            f"📊 **{team}**: {old_line:+.1f} → {new_line:+.1f}\n"
            f"🎯 Model edge: {model_edge*100:.1f}%"
        ),
        "color": _COLOR_GOLD,
        "footer": {"text": "⏰ Act quickly — lines move fast"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return _post(channel, embeds=[embed])


# =============================================================================
# END OF DAY RESULTS — How did we do?
# =============================================================================

def send_eod_results(
    results: List[Dict],
    daily_pl: float,
    season_pl: Optional[float] = None
) -> bool:
    """
    Send end-of-day results summary.
    
    Args:
        results: List of bet outcomes
        daily_pl: Daily P&L in units
        season_pl: Season-to-date P&L (optional)
    """
    channel = _channel_id("results")
    if not channel:
        logger.warning("EOD results: No Discord channel configured")
        return False
    
    today = datetime.now(timezone.utc).strftime("%A, %B %d")
    
    wins = sum(1 for r in results if r.get("outcome") == "win")
    losses = sum(1 for r in results if r.get("outcome") == "loss")
    pushes = sum(1 for r in results if r.get("outcome") == "push")
    
    # Result summary line
    if daily_pl > 0:
        result_emoji = "🟢"
        color = _COLOR_GREEN
    elif daily_pl < 0:
        result_emoji = "🔴"
        color = _COLOR_RED
    else:
        result_emoji = "⚪"
        color = _COLOR_BLUE
    
    # Individual results (max 10)
    result_lines = []
    for r in results[:10]:
        team = r.get("team", "Unknown")
        outcome = r.get("outcome", "")
        pl = r.get("pl", 0)
        
        if outcome == "win":
            emoji = "✅"
        elif outcome == "loss":
            emoji = "❌"
        else:
            emoji = "🔄"
        
        result_lines.append(f"{emoji} {team}: {pl:+.2f}u")
    
    description = (
        f"**{wins}-{losses}-{pushes}** | **{result_emoji} {daily_pl:+.2f}u**\n\n"
        + "\n".join(result_lines)
    )
    
    if season_pl is not None:
        season_emoji = "🟢" if season_pl > 0 else "🔴" if season_pl < 0 else "⚪"
        description += f"\n\n📊 **Season P&L:** {season_emoji} {season_pl:+.2f}u"
    
    embed = {
        "title": f"📊 END OF DAY — {today}",
        "description": description,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return _post(channel, embeds=[embed])


# =============================================================================
# TOURNAMENT UPDATE — Bracket progress, upsets, etc.
# =============================================================================

def send_tournament_update(
    day: str,
    upsets: List[Dict],
    bracket_status: str,
    cinderella_teams: Optional[List[str]] = None
) -> bool:
    """
    Send tournament-specific update (upsets, bracket busters, etc.)
    """
    channel = _channel_id("bets")
    if not channel:
        return False
    
    lines = [f"🏀 **TOURNAMENT UPDATE — {day}**\n"]
    
    if upsets:
        lines.append(f"⚡ **{len(upsets)} Upsets Today:**")
        for upset in upsets[:5]:
            winner = upset.get("winner", "Unknown")
            loser = upset.get("loser", "Unknown")
            w_seed = upset.get("winner_seed", "?")
            l_seed = upset.get("loser_seed", "?")
            lines.append(f"  #{w_seed} {winner} beats #{l_seed} {loser}")
        lines.append("")
    
    lines.append(f"**Bracket Status:** {bracket_status}")
    
    if cinderella_teams:
        lines.append(f"\n🧚 **Cinderella Watch:** {', '.join(cinderella_teams)}")
    
    embed = {
        "title": "🏀 MARCH MADNESS UPDATE",
        "description": "\n".join(lines),
        "color": _COLOR_BLUE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return _post(channel, embeds=[embed])


# =============================================================================
# SYSTEM ALERT — Something is wrong
# =============================================================================

def send_system_alert(
    alert_type: str,
    message: str,
    severity: str = "warning"
) -> bool:
    """
    Send system alert (data issues, errors, etc.)
    """
    channel = _channel_id("alert")
    if not channel:
        return False
    
    colors = {
        "info": _COLOR_BLUE,
        "warning": _COLOR_GOLD,
        "error": _COLOR_RED,
    }
    
    emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🚨"}.get(severity, "⚠️")
    
    embed = {
        "title": f"{emoji} SYSTEM ALERT — {alert_type.upper()}",
        "description": message,
        "color": colors.get(severity, _COLOR_GOLD),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return _post(channel, embeds=[embed])


# =============================================================================
# WEEKLY SUMMARY — Sunday recap
# =============================================================================

def send_weekly_summary(
    week_pl: float,
    week_record: str,
    best_bet: Optional[Dict] = None,
    worst_bet: Optional[Dict] = None,
    trends: Optional[List[str]] = None
) -> bool:
    """
    Send weekly summary on Sunday.
    """
    channel = _channel_id("results")
    if not channel:
        return False
    
    lines = [
        f"📊 **WEEKLY RECAP**",
        f"",
        f"**Record:** {week_record}",
        f"**P&L:** {week_pl:+.2f}u",
        f"",
    ]
    
    if best_bet:
        lines.append(f"🌟 **Best Bet:** {best_bet.get('team')} ({best_bet.get('pl', 0):+.2f}u)")
    
    if worst_bet:
        lines.append(f"💩 **Worst Bet:** {worst_bet.get('team')} ({worst_bet.get('pl', 0):+.2f}u)")
    
    if trends:
        lines.append(f"\n📈 **Trends:**")
        for trend in trends[:3]:
            lines.append(f"  • {trend}")
    
    color = _COLOR_GREEN if week_pl > 0 else _COLOR_RED if week_pl < 0 else _COLOR_BLUE
    
    embed = {
        "title": "📊 WEEKLY SUMMARY",
        "description": "\n".join(lines),
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return _post(channel, embeds=[embed])


# =============================================================================
# TEST FUNCTION
# =============================================================================

def send_test_message() -> bool:
    """Send test message to verify Discord is working."""
    if not _bot_token():
        logger.error("Discord test: No bot token configured")
        return False
    
    # Try to send to each configured channel
    results = []
    
    for name, channel_id in [
        ("Bets", CHANNEL_BETS),
        ("Results", CHANNEL_RESULTS),
        ("Alerts", CHANNEL_ALERTS),
    ]:
        if channel_id:
            success = _post(
                channel_id,
                embeds=[{
                    "title": f"🧪 TEST — {name} Channel",
                    "description": "Discord notification system is working!",
                    "color": _COLOR_GREEN,
                }]
            )
            results.append(f"{name}: {'✅' if success else '❌'}")
    
    logger.info(f"Discord test results: {', '.join(results)}")
    return all("✅" in r for r in results)
