"""
Tiered escalation coordinator — O-9

Detects high-stakes BET-tier games that warrant a Kimi CLI second opinion.
Currently logging-only: flags triggers in structured logs so the operator
can manually escalate. Kimi API routing will be wired post-March 18.

Escalation triggers (any one sufficient):
  - recommended_units >= ESCALATION_UNITS_THRESHOLD (default 1.5)
  - is_neutral=True (tournament/neutral-site proxy for round >= 4)
  - integrity_verdict contains "VOLATILE"
"""

import logging
from datetime import datetime

from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)

_UNITS_THRESHOLD_DEFAULT = "1.5"


def escalate_if_needed(
    game_key: str,
    home_team: str,
    away_team: str,
    recommended_units: float,
    integrity_verdict: str | None,
    is_neutral: bool = False,
) -> bool:
    """
    Check if a BET-tier game meets escalation criteria and log if so.

    Args:
        game_key: Unique game identifier (e.g. "Duke_NC State_2026-03-18").
        home_team: Home team name.
        away_team: Away team name.
        recommended_units: Model-recommended bet size in percentage-point units.
        integrity_verdict: OpenClaw verdict string or None.
        is_neutral: True when game is at a neutral site (tournament proxy).

    Returns:
        True if any escalation trigger fired, False otherwise.
    """
    triggers = []

    units_threshold = get_float_env("ESCALATION_UNITS_THRESHOLD", _UNITS_THRESHOLD_DEFAULT)
    if recommended_units >= units_threshold:
        triggers.append(f"units={recommended_units:.2f} >= {units_threshold:.2f}")

    if is_neutral:
        triggers.append("neutral_site=True (tournament game)")

    if integrity_verdict and "VOLATILE" in str(integrity_verdict).upper():
        triggers.append(f"integrity_verdict={integrity_verdict}")

    if triggers:
        logger.warning(
            "ESCALATION_FLAGGED [%s @ %s] key=%s — triggers: %s "
            "— Kimi second opinion recommended (manual until API wired)",
            away_team,
            home_team,
            game_key,
            " | ".join(triggers),
        )
        return True

    return False


def send_line_movement_alert(
    game_key: str,
    away_team: str,
    home_team: str,
    old_spread: float,
    new_spread: float,
    delta: float,
    new_edge: float,
    abandoned: bool = False,
    game_time: str = None,
    min_bet_edge: float = 0.025,
) -> bool:
    """
    Send a Discord alert for significant line movement.
    
    NOISE REDUCTION: Only sends alerts for:
    1. BET NOW - Line moved favorably toward us AND edge is still strong
    2. ABANDON - Edge collapsed (for those already in the bet)
    
    Does NOT send: Hold, edge thinning, edge still playable, etc.
    
    Args:
        game_key: Game identifier.
        away_team: Away team name.
        home_team: Home team name.
        old_spread: Original spread from bet log.
        new_spread: Current consensus spread.
        delta: Movement in points.
        new_edge: Fresh model edge at the new line.
        abandoned: True if edge dropped below MIN_BET_EDGE.
        game_time: Optional game start time string.
        min_bet_edge: Minimum edge threshold (default 2.5%).
        
    Returns:
        True if sent successfully.
    """
    from backend.services.discord_notifier import _post, _COLOR_GREEN, _COLOR_RED
    
    # Only send for TWO scenarios:
    # 1. BET NOW - Line moved in our favor (delta >= 1.5) AND edge is strong
    # 2. ABANDON - Edge collapsed completely
    
    # Format spread with + for positive values
    def fmt_spread(val: float) -> str:
        return f"{val:+.1f}" if val != 0 else "0.0"

    if delta >= 1.5 and new_edge >= min_bet_edge:
        # LINE MOVED IN OUR FAVOR - BET NOW SIGNAL
        embed = {
            "title": f"🎯 BET NOW: {away_team} @ {home_team}",
            "description": f"Line moved toward us — better price available!\n🕐 **Tip-off:** {game_time or 'TBD'}",
            "color": _COLOR_GREEN,
            "fields": [
                {"name": "📈 Line Movement", "value": f"{fmt_spread(old_spread)} → {fmt_spread(new_spread)} ({delta:+.1f} pts toward us)", "inline": False},
                {"name": "🎯 Model Edge", "value": f"{new_edge:.2%}", "inline": True},
                {"name": "✅ Action", "value": "**BET NOW** — Better price than our entry\nIf you haven't bet: ADD NOW\nIf you already bet: You got worse line but hold", "inline": False},
            ],
            "footer": {"text": f"Line Monitor • {datetime.utcnow().strftime('%H:%M UTC')}"},
            "timestamp": datetime.utcnow().isoformat()
        }
        return _post({"embeds": [embed]})
        
    elif abandoned:
        # EDGE COLLAPSED - ABANDON (for those already in)
        embed = {
            "title": f"🚨 ABANDON: {away_team} @ {home_team}",
            "description": f"Edge collapsed below threshold.\n🕐 **Tip-off:** {game_time or 'TBD'}",
            "color": _COLOR_RED,
            "fields": [
                {"name": "📉 Edge Collapsed", "value": f"{new_edge:.1%} (below {min_bet_edge:.1%} minimum)", "inline": False},
                {"name": "❌ Action", "value": "If you already bet: Consider hedging or live exit\nIf you haven't bet: DO NOT ADD", "inline": False},
            ],
            "footer": {"text": f"Line Monitor • {datetime.utcnow().strftime('%H:%M UTC')}"},
            "timestamp": datetime.utcnow().isoformat()
        }
        return _post({"embeds": [embed]})
    
    # All other scenarios (line moved against us, edge thinning, etc.) - NO ALERT
    return False
