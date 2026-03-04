"""
Discord notification service for CBB Edge.

Sends daily bet recommendations to a Discord channel using the Discord Bot API.

Required env var:
  DISCORD_BOT_TOKEN   — Discord bot token (bot must be a member of the server)

Optional env var:
  DISCORD_CHANNEL_ID  — Override the default channel ID
                        (default: 1477436117426110615)

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

_DEFAULT_CHANNEL_ID = "1477436117426110615"
_DISCORD_API_BASE = "https://discord.com/api/v10"

# Embed colours (decimal integers, not hex strings)
_COLOR_GREEN = 0x2ECC71   # bets found
_COLOR_YELLOW = 0xF1C40F  # considers only
_COLOR_GREY = 0x95A5A6    # all pass


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _bot_token() -> Optional[str]:
    return os.getenv("DISCORD_BOT_TOKEN")


def _channel_id() -> str:
    return os.getenv("DISCORD_CHANNEL_ID", _DEFAULT_CHANNEL_ID)


def _post(payload: dict) -> bool:
    """POST a message payload to the configured channel. Returns True on success."""
    token = _bot_token()
    if not token:
        logger.debug("DISCORD_BOT_TOKEN not set — skipping Discord notification")
        return False

    url = f"{_DISCORD_API_BASE}/channels/{_channel_id()}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
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
    pick = _pick_str(bet)
    edge = bet.get("edge_conservative", 0.0) or 0.0
    units = bet.get("recommended_units", 0.0) or 0.0
    margin = bet.get("projected_margin", 0.0) or 0.0
    kelly = bet.get("kelly_fractional", 0.0) or 0.0
    verdict = bet.get("verdict", "")
    snr = bet.get("snr")

    # Generate LLM-based scouting insight
    insight = generate_scouting_report(
        home_team=bet.get("home_team", "Home"),
        away_team=bet.get("away_team", "Away"),
        matchup_notes=bet.get("matchup_notes", []),
        verdict=verdict,
        edge=edge
    )

    # Use the V9 integrity_verdict computed during analysis Pass 2.
    # This is the exact verdict that was used for Kelly sizing — no second
    # DDGS search needed, and Discord stays consistent with the model.
    integrity = bet.get("integrity_verdict") or "Not run"

    snr_str = f"{snr:.0%}" if snr is not None else "N/A"

    return {
        "title": f"PICK: {pick}",
        "description": f"{bet.get('away_team', 'Away')} @ {bet.get('home_team', 'Home')}",
        "color": _COLOR_GREEN,
        "fields": [
            {"name": "Edge",         "value": f"{edge:.1%}",         "inline": True},
            {"name": "Stake",        "value": f"{units:.2f}u",       "inline": True},
            {"name": "Odds",         "value": _odds_str(bet.get("bet_odds")), "inline": True},
            {"name": "Proj. Margin", "value": f"{margin:+.1f} pts",  "inline": True},
            {"name": "Kelly",        "value": f"{kelly:.1%}",        "inline": True},
            {"name": "Tier",         "value": _tier(verdict),        "inline": True},
            {"name": "Source SNR",   "value": snr_str,               "inline": True},
            {"name": "Model Insight", "value": f"*{insight}*",       "inline": False},
            {"name": "V9 Integrity Verdict", "value": f"**{integrity}**", "inline": False},
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_todays_bets(
    bet_details: Optional[List[Dict]],
    summary: Dict,
) -> None:
    """
    Send today's betting slate to Discord.

    Args:
        bet_details: List of bet dicts from the analysis summary. Each dict
                     contains home_team, away_team, spread, bet_side,
                     edge_conservative, recommended_units, bet_odds,
                     kelly_fractional, projected_margin, verdict.
                     May be None or [] when no bets were found.
        summary:     The _summary() dict from run_nightly_analysis().
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

    success = _post({"embeds": [summary_embed]})
    if not success:
        return  # If summary failed, don't try individual bets

    if not bet_details:
        return

    # Sort highest-edge first
    ordered = sorted(bet_details, key=lambda b: b.get("edge_conservative") or 0.0, reverse=True)

    # Send each bet as an individual message to ensure delivery and avoid char limits
    for bet in ordered:
        try:
            logger.info("Sending Discord embed for %s @ %s", bet.get("away_team"), bet.get("home_team"))
            _post({"embeds": [_bet_embed(bet)]})
        except Exception as e:
            logger.error("Failed to send bet embed for %s: %s", bet.get("home_team"), e)


def send_health_briefing(summary: Dict) -> None:
    """
    Send a system health report to Discord.
    """
    if not _bot_token():
        return

    # Use LLM to write a professional narrative
    perf = summary.get("performance", {})
    port = summary.get("portfolio", {})
    sys_status = summary.get("system", {})
    
    status_str = (
        f"Performance: {perf.get('status')} (MAE: {perf.get('mean_mae', 'N/A')}), "
        f"Portfolio: {port.get('status')} (Drawdown: {port.get('current_drawdown_pct', 0.0):.1%}), "
        f"System: {sys_status.get('status')} (Tests: {sys_status.get('passed', False)})"
    )
    
    # Use dedicated health narrative logic
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

    _post({"embeds": [embed]})

