"""
Discord notification service for CBB Edge + Fantasy Baseball.

Sends notifications to multiple Discord channels based on message type.

Required env vars:
  DISCORD_BOT_TOKEN — Discord bot token (bot must be a member of the server)

Optional env vars (channel IDs):
  DISCORD_CHANNEL_CBB_BETS
  DISCORD_CHANNEL_CBB_BRIEF
  DISCORD_CHANNEL_CBB_ALERTS
  DISCORD_CHANNEL_CBB_TOURNAMENT
  DISCORD_CHANNEL_FANTASY_LINEUPS
  DISCORD_CHANNEL_FANTASY_WAIVERS
  DISCORD_CHANNEL_FANTASY_NEWS
  DISCORD_CHANNEL_FANTASY_DRAFT
  DISCORD_CHANNEL_OPENCLAW_BRIEFS
  DISCORD_CHANNEL_OPENCLAW_ESCALATIONS
  DISCORD_CHANNEL_OPENCLAW_HEALTH
  DISCORD_CHANNEL_SYSTEM_ERRORS
  DISCORD_CHANNEL_SYSTEM_LOGS
  DISCORD_CHANNEL_DATA_ALERTS
  DISCORD_CHANNEL_GENERAL
  DISCORD_CHANNEL_ADMIN_COMMANDS

Legacy support:
  DISCORD_CHANNEL_ID — Fallback for backward compatibility

If DISCORD_BOT_TOKEN is not set all functions silently no-op.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from backend.services.scout import (
    generate_scouting_report,
    generate_morning_briefing_narrative,
    generate_health_narrative,
)

logger = logging.getLogger(__name__)

_DISCORD_API_BASE = "https://discord.com/api/v10"

# Embed colours (decimal integers, not hex strings)
_COLOR_GREEN = 0x2ECC71   # bets found
_COLOR_YELLOW = 0xF1C40F  # considers only
_COLOR_GREY = 0x95A5A6    # all pass
_COLOR_RED = 0xE74C3C     # errors / alerts
_COLOR_BLUE = 0x3498DB    # info / briefs
_COLOR_ORANGE = 0xE67E22  # warnings

# Channel configuration mapping
CHANNEL_MAP = {
    # 🏀 CBB EDGE
    "cbb-bets": "DISCORD_CHANNEL_CBB_BETS",
    "cbb-morning-brief": "DISCORD_CHANNEL_CBB_BRIEF",
    "cbb-alerts": "DISCORD_CHANNEL_CBB_ALERTS",
    "cbb-tournament": "DISCORD_CHANNEL_CBB_TOURNAMENT",
    
    # ⚾ FANTASY BASEBALL
    "fantasy-lineups": "DISCORD_CHANNEL_FANTASY_LINEUPS",
    "fantasy-waivers": "DISCORD_CHANNEL_FANTASY_WAIVERS",
    "fantasy-news": "DISCORD_CHANNEL_FANTASY_NEWS",
    "fantasy-draft": "DISCORD_CHANNEL_FANTASY_DRAFT",
    
    # 🎯 OPENCLAW INTEL
    "openclaw-briefs": "DISCORD_CHANNEL_OPENCLAW_BRIEFS",
    "openclaw-escalations": "DISCORD_CHANNEL_OPENCLAW_ESCALATIONS",
    "openclaw-health": "DISCORD_CHANNEL_OPENCLAW_HEALTH",
    
    # ⚙️ SYSTEM OPS
    "system-errors": "DISCORD_CHANNEL_SYSTEM_ERRORS",
    "system-logs": "DISCORD_CHANNEL_SYSTEM_LOGS",
    "data-alerts": "DISCORD_CHANNEL_DATA_ALERTS",
    
    # 💬 GENERAL
    "general": "DISCORD_CHANNEL_GENERAL",
    "admin-commands": "DISCORD_CHANNEL_ADMIN_COMMANDS",
}

# Legacy fallback channel (original bets channel)
_LEGACY_CHANNEL_ID = "1477436117426110615"


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def _bot_token() -> Optional[str]:
    return os.getenv("DISCORD_BOT_TOKEN")


def _get_channel_id(channel_name: str) -> Optional[str]:
    """Get channel ID for a named channel from environment variables."""
    env_var = CHANNEL_MAP.get(channel_name)
    if env_var:
        channel_id = os.getenv(env_var)
        if channel_id:
            return channel_id
    
    # Fallback to legacy channel ID for backward compatibility
    if channel_name in ("cbb-bets", "general"):
        legacy = os.getenv("DISCORD_CHANNEL_ID", _LEGACY_CHANNEL_ID)
        if legacy:
            return legacy
    
    return None


def _post_to_channel(channel_id: str, payload: dict, mention_admin: bool = False) -> bool:
    """POST a message payload to a specific channel. Returns True on success."""
    token = _bot_token()
    if not token:
        logger.debug("DISCORD_BOT_TOKEN not set — skipping Discord notification")
        return False
    
    if not channel_id:
        logger.debug("No channel ID configured — skipping notification")
        return False

    url = f"{_DISCORD_API_BASE}/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    
    # Add admin mention if requested
    if mention_admin:
        content = payload.get("content", "")
        payload["content"] = "@admin " + content
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code not in (200, 201):
            logger.warning(
                "Discord API returned %d: %s", resp.status_code, resp.text[:300]
            )
            return False
        return True
    except requests.RequestException as exc:
        logger.warning("Discord POST failed: %s", exc)
        return False


def send_to_channel(
    channel_name: str, 
    message: str = None, 
    embed: dict = None, 
    embeds: list = None,
    mention_admin: bool = False
) -> bool:
    """
    Send a message to a named channel.
    
    Args:
        channel_name: One of the keys in CHANNEL_MAP (e.g., "cbb-bets", "fantasy-lineups")
        message: Plain text content
        embed: Single embed dict
        embeds: List of embed dicts
        mention_admin: Whether to mention @admin
    
    Returns:
        True if sent successfully, False otherwise
    """
    channel_id = _get_channel_id(channel_name)
    if not channel_id:
        logger.warning(f"Channel '{channel_name}' not configured")
        return False
    
    payload = {}
    if message:
        payload["content"] = message
    if embed:
        payload["embeds"] = [embed]
    if embeds:
        payload["embeds"] = embeds
    
    if not payload:
        logger.warning("No content to send")
        return False
    
    return _post_to_channel(channel_id, payload, mention_admin)


def route_notification(
    message_type: str,
    content: str = None,
    embed: dict = None,
    embeds: list = None,
    severity: str = "normal"
) -> bool:
    """
    Route a notification to the appropriate channel based on message type.
    
    Args:
        message_type: Type of message (determines channel routing)
        content: Plain text message
        embed: Single embed dict
        embeds: List of embed dicts
        severity: "normal", "warning", "critical" — affects @admin mention
    
    Returns:
        True if sent successfully
    """
    # Routing map: message_type -> channel_name
    routing = {
        # CBB Betting
        "bet_recommendation": "cbb-bets",
        "morning_brief": "cbb-morning-brief",
        "line_movement": "cbb-alerts",
        "sharp_signal": "cbb-alerts",
        "tournament_update": "cbb-tournament",
        
        # Fantasy Baseball
        "lineup_recommendation": "fantasy-lineups",
        "waiver_suggestion": "fantasy-waivers",
        "injury_alert": "fantasy-news",
        "draft_pick": "fantasy-draft",
        
        # OpenClaw
        "research_brief": "openclaw-briefs",
        "high_stakes_escalation": "openclaw-escalations",
        "system_health": "openclaw-health",
        
        # System
        "critical_error": "system-errors",
        "routine_log": "system-logs",
        "data_degradation": "data-alerts",
        
        # Fallback
        "general": "general",
    }
    
    channel_name = routing.get(message_type, "general")
    mention_admin = severity in ("warning", "critical")
    
    return send_to_channel(
        channel_name=channel_name,
        message=content,
        embed=embed,
        embeds=embeds,
        mention_admin=mention_admin
    )


# ---------------------------------------------------------------------------
# Legacy Support (Backward Compatible)
# ---------------------------------------------------------------------------


def _legacy_channel_id() -> str:
    """Get the legacy default channel ID for backward compatibility."""
    return os.getenv("DISCORD_CHANNEL_ID", _LEGACY_CHANNEL_ID)


def _post(payload: dict) -> bool:
    """Legacy POST function — sends to original bets channel."""
    channel_id = _legacy_channel_id()
    return _post_to_channel(channel_id, payload)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _pick_str(bet: Dict) -> str:
    """Build human-readable pick, e.g. 'Duke -4.5' or 'UNC (away)'."""
    home = bet.get("home_team", "Home")
    away = bet.get("away_team", "Away")
    spread = bet.get("spread")
    side = bet.get("bet_side", "home")
    team = away if side == "away" else home

    if spread is None:
        return team

    val = (-spread) if side == "away" else spread
    sign = "+" if val > 0 else ""
    return f"{team} {sign}{val:.1f}"


def _tier(verdict: str) -> str:
    """Extract '[T3]' → 'T3', or '—' if not present."""
    m = re.search(r'\[T(\d+)\]', verdict or "")
    return f"T{m.group(1)}" if m else "—"


def _odds_str(bet_odds) -> str:
    if bet_odds is None:
        return "—"
    return f"+{bet_odds:.0f}" if bet_odds >= 0 else f"{bet_odds:.0f}"


def _bet_embed(bet: Dict) -> Dict:
    """Build a clear, actionable bet embed for Discord."""
    pick = _pick_str(bet)
    edge = bet.get("edge_conservative", 0.0) or 0.0
    units = bet.get("recommended_units", 0.0) or 0.0
    margin = bet.get("projected_margin", 0.0) or 0.0
    kelly = bet.get("kelly_fractional", 0.0) or 0.0
    verdict = bet.get("verdict", "")
    snr = bet.get("snr")
    
    # Extract team info for clarity
    home_team = bet.get("home_team", "Home")
    away_team = bet.get("away_team", "Away")
    bet_side = bet.get("bet_side", "home")
    spread = bet.get("spread")
    
    # Determine which team we're betting on
    team_to_bet = home_team if bet_side == "home" else away_team
    opponent = away_team if bet_side == "home" else home_team
    home_away = "HOME" if bet_side == "home" else "AWAY"
    
    # Format the spread for display from bettor's perspective
    if spread is not None:
        if bet_side == "home":
            spread_val = spread
        else:
            spread_val = -spread
        spread_str = f"{spread_val:+.1f}"
    else:
        spread_str = "ML"

    # Generate LLM-based scouting insight
    insight = generate_scouting_report(
        home_team=home_team,
        away_team=away_team,
        matchup_notes=bet.get("matchup_notes", []),
        verdict=verdict,
        edge=edge
    )

    # Use the V9 integrity_verdict computed during analysis Pass 2.
    integrity = bet.get("integrity_verdict") or "Not run"

    snr_str = f"{snr:.0%}" if snr is not None else "N/A"

    # Build clear action line
    action_line = f"**Take {team_to_bet} {spread_str}**"

    return {
        "title": f"BET: {team_to_bet}",
        "description": f"{action_line}\n{away_team} @ {home_team}",
        "color": _COLOR_GREEN,
        "fields": [
            {"name": "Betting",      "value": f"{team_to_bet} {spread_str}", "inline": True},
            {"name": "Side",         "value": f"{home_away} ({bet_side.upper()})", "inline": True},
            {"name": "Opponent",     "value": opponent,                       "inline": True},
            {"name": "Edge",         "value": f"{edge:.1%}",                  "inline": True},
            {"name": "Stake",        "value": f"{units:.2f}u",                "inline": True},
            {"name": "Odds",         "value": _odds_str(bet.get("bet_odds")), "inline": True},
            {"name": "Proj. Margin", "value": f"{margin:+.1f} pts",           "inline": True},
            {"name": "Kelly",        "value": f"{kelly:.1%}",                 "inline": True},
            {"name": "Tier",         "value": _tier(verdict),                 "inline": True},
            {"name": "Source SNR",   "value": snr_str,                        "inline": True},
            {"name": "Model Insight", "value": insight,                       "inline": False},
            {"name": "V9 Integrity", "value": integrity,                      "inline": True},
        ],
    }


# ---------------------------------------------------------------------------
# Public API (Legacy Functions — Updated for Multi-Channel)
# ---------------------------------------------------------------------------


def send_todays_bets(
    bet_details: Optional[List[Dict]],
    summary: Dict,
) -> None:
    """
    Send today's betting slate to Discord #cbb-bets channel.
    """
    if not _bot_token():
        return

    n_bets = summary.get("bets_recommended", 0)
    n_considered = summary.get("games_considered", 0)
    n_analyzed = summary.get("games_analyzed", 0)
    n_pass = max(0, n_analyzed - n_bets - n_considered)
    duration = summary.get("duration_seconds", 0)
    today = datetime.now(timezone.utc).strftime("%b %d, %Y")

    if n_bets > 0:
        color = _COLOR_GREEN
        status_line = f"**{n_bets} BET{'s' if n_bets > 1 else ''}** found on today's slate!"
    elif n_considered > 0:
        color = _COLOR_YELLOW
        status_line = (
            f"No BETs today. "
            f"{n_considered} CONSIDER game{'s' if n_considered > 1 else ''} — "
            "watch for line movement toward the model's side."
        )
    else:
        color = _COLOR_GREY
        status_line = "PASS on all games today. No edges found."

    summary_embed = {
        "title": f"CBB Edge — {today}",
        "description": status_line,
        "color": color,
        "fields": [
            {"name": "Games Analyzed", "value": str(n_analyzed),   "inline": True},
            {"name": "BET",            "value": str(n_bets),        "inline": True},
            {"name": "CONSIDER",       "value": str(n_considered),  "inline": True},
            {"name": "PASS",           "value": str(n_pass),        "inline": True},
            {"name": "Run Time",       "value": f"{duration:.0f}s", "inline": True},
        ],
        "footer": {"text": "CBB Edge Analyzer v9"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Send summary to cbb-bets channel
    success = send_to_channel("cbb-bets", embed=summary_embed)
    if not success:
        return

    if not bet_details:
        return

    # Sort highest-edge first
    ordered = sorted(bet_details, key=lambda b: b.get("edge_conservative") or 0.0, reverse=True)

    # Send each bet as an individual message
    for bet in ordered:
        try:
            logger.info("Sending Discord embed for %s @ %s", bet.get("away_team"), bet.get("home_team"))
            send_to_channel("cbb-bets", embed=_bet_embed(bet))
        except Exception as e:
            logger.error("Failed to send bet embed for %s: %s", bet.get("home_team"), e)


def send_morning_brief(summary_embed: dict) -> bool:
    """
    Send morning briefing to #cbb-morning-brief channel.
    
    Args:
        summary_embed: Pre-built embed dict with slate summary
    
    Returns:
        True if sent successfully
    """
    return send_to_channel("cbb-morning-brief", embed=summary_embed)


def send_health_briefing(summary: Dict) -> None:
    """
    Send a system health report to Discord #openclaw-health channel.
    """
    if not _bot_token():
        return

    perf = summary.get("performance", {})
    port = summary.get("portfolio", {})
    sys_status = summary.get("system", {})
    
    status_str = (
        f"Performance: {perf.get('status')} (MAE: {perf.get('mean_mae', 'N/A')}), "
        f"Portfolio: {port.get('status')} (Drawdown: {port.get('current_drawdown_pct', 0.0):.1%}), "
        f"System: {sys_status.get('status')} (Tests: {sys_status.get('passed', False)})"
    )
    
    narrative = generate_health_narrative(summary)

    color_map = {"GREEN": 0x2ECC71, "YELLOW": 0xF1C40F, "RED": 0xE74C3C}
    overall_status = "GREEN"
    if "RED" in [perf.get("status"), port.get("status"), sys_status.get("status")]:
        overall_status = "RED"
    elif "YELLOW" in [perf.get("status"), port.get("status"), sys_status.get("status")]:
        overall_status = "YELLOW"

    embed = {
        "title": "🛰️ Performance Sentinel — Health Report",
        "description": f"*{narrative}*",
        "color": color_map.get(overall_status, 0x95A5A6),
        "fields": [
            {
                "name": "Model Accuracy (30d)", 
                "value": f"MAE: `{perf.get('mean_mae', 'N/A')}`\nStatus: {perf.get('status')}", 
                "inline": True
            },
            {
                "name": "Portfolio Health", 
                "value": f"Drawdown: `{port.get('current_drawdown_pct', 0.0):.1%}`\nStatus: {port.get('status')}", 
                "inline": True
            },
            {
                "name": "System Integrity", 
                "value": f"Pytest: `{'PASSED' if sys_status.get('passed') else 'FAILED'}`\nStatus: {sys_status.get('status')}", 
                "inline": True
            }
        ],
        "footer": {"text": "CBB Edge Sentinel Unit"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Send to openclaw-health channel
    send_to_channel("openclaw-health", embed=embed)


def send_verdict_flip_alert(movement) -> None:
    """
    Send a real-time alert when a line move flips a PASS to a BET.
    Goes to #cbb-alerts channel.
    """
    if not _bot_token() or not movement.fresh_analysis:
        return

    analysis = movement.fresh_analysis
    edge = analysis.edge_conservative or 0.0
    units = analysis.recommended_units or 0.0

    calcs = (analysis.full_analysis or {}).get("calculations", {})
    bet_side = calcs.get("bet_side", "home")
    pick_team = movement.home_team if bet_side == "home" else movement.away_team
    if movement.new_value is not None:
        side_spread = movement.new_value if bet_side == "home" else -movement.new_value
        sign = "+" if side_spread > 0 else ""
        pick_str = f"{pick_team} {sign}{side_spread:.1f}"
    else:
        pick_str = pick_team

    old_str = f"{movement.old_value:+.1f}" if movement.old_value is not None else "N/A"
    new_str = f"{movement.new_value:+.1f}" if movement.new_value is not None else "N/A"

    insight = generate_scouting_report(
        home_team=movement.home_team,
        away_team=movement.away_team,
        matchup_notes=analysis.notes or [],
        verdict=analysis.verdict,
        edge=edge,
    )

    embed = {
        "title": f"LINE FLIP: {movement.away_team} @ {movement.home_team}",
        "description": "New **BET** opportunity detected via Real-Time Pulse.",
        "color": _COLOR_GREEN,
        "fields": [
            {"name": "New Pick",      "value": f"**{pick_str}**",             "inline": True},
            {"name": "Edge",          "value": f"{edge:.1%}",                  "inline": True},
            {"name": "Stake",         "value": f"{units:.2f}u",                "inline": True},
            {"name": "Movement",      "value": f"{old_str} -> **{new_str}**",  "inline": True},
            {"name": "Model Insight", "value": f"*{insight}*",                 "inline": False},
        ],
        "footer": {"text": "Level 5 Real-Time Pulse"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Send to cbb-alerts channel
    send_to_channel("cbb-alerts", embed=embed)


def send_source_health_alert(
    n_active: int,
    kenpom_up: bool,
    barttorvik_up: bool,
    evanmiya_status: str,
    kenpom_teams: int = 0,
    barttorvik_teams: int = 0,
    evanmiya_teams: int = 0,
) -> None:
    """
    Fire a Discord alert when fewer than 2 of 3 rating sources are active.
    Goes to #data-alerts channel.
    """
    if not _bot_token():
        return

    em_icon = "✅" if evanmiya_status == "UP" else ("⚠️" if evanmiya_status == "DROPPED" else "❌")
    status_line = (
        f"KenPom: {'✅' if kenpom_up else '❌'} ({kenpom_teams} teams) | "
        f"BartTorvik: {'✅' if barttorvik_up else '❌'} ({barttorvik_teams} teams) | "
        f"EvanMiya: {em_icon} ({evanmiya_teams} teams)"
    )

    severity = "CRITICAL" if n_active < 2 else "WARNING"
    impact = (
        "Model on **1 source only** — margin shrinkage active, wider CI, fewer BET verdicts. "
        "**Do not trust outputs until fixed.**"
        if n_active < 2
        else "Model on **2 of 3 sources** — slightly degraded accuracy."
    )

    embed = {
        "title": f"⚠️ RATINGS SOURCE {severity}: {n_active}/3 Active",
        "description": impact,
        "color": _COLOR_RED,
        "fields": [
            {"name": "Source Status", "value": status_line, "inline": False},
            {
                "name": "Action Required",
                "value": (
                    "Check Railway logs. "
                    "BartTorvik → `barttorvik.com/2026_super_standings.json` · "
                    "EvanMiya → Cloudflare block, Playwright fallback active."
                ),
                "inline": False,
            },
        ],
        "footer": {"text": "CBB Edge — Source Health Monitor"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Send to data-alerts channel
    send_to_channel("data-alerts", embed=embed)


# ---------------------------------------------------------------------------
# New Channel-Specific Functions
# ---------------------------------------------------------------------------


def send_high_stakes_escalation(
    game_key: str,
    home_team: str,
    away_team: str,
    recommended_units: float,
    integrity_verdict: str,
    reason: str,
    queue_id: str
) -> bool:
    """
    Send high-stakes escalation alert to #openclaw-escalations channel.
    
    Returns:
        True if sent successfully
    """
    embed = {
        "title": "🚨 HIGH-STAKES ESCALATION",
        "description": f"Game: {away_team} @ {home_team}",
        "color": _COLOR_RED,
        "fields": [
            {"name": "Recommended Size", "value": f"{recommended_units:.2f} units", "inline": True},
            {"name": "Integrity Verdict", "value": integrity_verdict, "inline": True},
            {"name": "Escalation Reason", "value": reason, "inline": False},
            {"name": "Queue ID", "value": f"`{queue_id}`", "inline": False},
            {"name": "Action Required", "value": "Manual review before tipoff", "inline": False},
        ],
        "footer": {"text": "OpenClaw Escalation Queue"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return send_to_channel("openclaw-escalations", embed=embed, mention_admin=True)


def send_fantasy_lineup(lineup_data: dict) -> bool:
    """
    Send daily fantasy lineup to #fantasy-lineups channel.
    
    Args:
        lineup_data: Dict with hitters, pitchers, projections
    """
    hitters = lineup_data.get("hitters", [])
    pitchers = lineup_data.get("pitchers", [])
    total_proj = lineup_data.get("total_projected", 0)
    
    hitter_text = "\n".join([
        f"{h['position']}: {h['name']} ({h['team']}) — {h['projection']:.1f}"
        for h in hitters[:10]  # Top 10 hitters
    ])
    
    pitcher_text = "\n".join([
        f"{p['position']}: {p['name']} ({p['team']}) — {p['projection']:.1f}"
        for p in pitchers[:4]  # Top 4 pitchers
    ])
    
    embed = {
        "title": f"⚾ Today's Optimal Lineup — {datetime.now(timezone.utc).strftime('%B %d')}",
        "description": f"Total Projected: **{total_proj:.1f} points**",
        "color": _COLOR_BLUE,
        "fields": [
            {"name": "Hitters", "value": f"```{hitter_text}```", "inline": False},
            {"name": "Pitchers", "value": f"```{pitcher_text}```", "inline": False},
        ],
        "footer": {"text": "Fantasy Baseball Optimizer"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return send_to_channel("fantasy-lineups", embed=embed)


def send_system_error(error_message: str, details: str = None) -> bool:
    """
    Send critical system error to #system-errors channel with @admin mention.
    """
    embed = {
        "title": "❌ SYSTEM ERROR",
        "description": error_message,
        "color": _COLOR_RED,
        "fields": [
            {"name": "Timestamp", "value": datetime.now(timezone.utc).isoformat(), "inline": True},
        ],
        "footer": {"text": "CBB Edge System Monitor"},
    }
    
    if details:
        embed["fields"].append({"name": "Details", "value": f"```{details[:1000]}```", "inline": False})
    
    return send_to_channel("system-errors", embed=embed, mention_admin=True)


def send_routine_log(message: str) -> bool:
    """
    Send routine operation log to #system-logs channel.
    """
    return send_to_channel("system-logs", message=message)
