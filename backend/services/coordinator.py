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
    from backend.services.discord_notifier import _post, _COLOR_YELLOW, _COLOR_GREY, _COLOR_RED
    
    # Determine action and styling based on situation
    if abandoned:
        status = "🚨 **ABANDON** — Edge Collapsed"
        color = _COLOR_RED
        action = "❌ DO NOT ADD"
        recommendation = f"Edge fell to {new_edge:.1%} (below {min_bet_edge:.1%} threshold). If you already bet, monitor for live exit. Do not add new exposure."
    elif delta >= 1.5:  # Line moved in our favor (better price)
        status = "✅ **LINE_MOVED_FAVORABLE** — Better Price"
        color = _COLOR_YELLOW
        action = "🎯 BET NOW (if no position)"
        recommendation = f"Line moved {delta:+.1f}pts toward us — you can get a better price than our entry. Edge: {new_edge:.1%}. If you haven't bet yet, ADD NOW. If you already bet, you got worse line but hold."
    elif delta <= -1.5:  # Line moved against us (worse price)
        if new_edge >= 0.04:  # Still strong edge
            status = "⚠️ **LINE_MOVED_AGAINST** — Still Playable"
            color = _COLOR_YELLOW
            action = "📊 HOLD EXISTING ONLY"
            recommendation = f"Line moved {delta:+.1f}pts against us, but edge remains {new_edge:.1%}. If you already bet: HOLD. If you haven't bet: AVOID — better prices are gone."
        elif new_edge >= min_bet_edge:  # Edge thinning but playable
            status = "⚠️ **LINE_MOVED_AGAINST** — Edge Thinning"
            color = _COLOR_GREY
            action = "🛑 NO NEW ADDS"
            recommendation = f"Line moved {delta:+.1f}pts against us. Edge down to {new_edge:.1%}. If you already bet: HOLD. If you haven't bet: PASS — margin too thin."
        else:
            status = "🚨 **LINE_MOVED_AGAINST** — Edge Gone"
            color = _COLOR_RED
            action = "❌ ABANDON"
            recommendation = f"Line moved {delta:+.1f}pts against us and edge collapsed to {new_edge:.1%}. Do not add. If you already bet, consider hedging."
    else:
        return False  # Shouldn't happen given threshold check
    
    # Format spread with + for positive values
    def fmt_spread(val: float) -> str:
        return f"{val:+.1f}" if val != 0 else "0.0"

    # Build time info
    time_field = ""
    if game_time:
        time_field = f"\n🕐 **Tip-off:** {game_time}"

    embed = {
        "title": f"Line Monitor: {away_team} @ {home_team}",
        "description": f"{status}{time_field}",
        "color": color,
        "fields": [
            {"name": "📊 Line Movement", "value": f"{fmt_spread(old_spread)} → {fmt_spread(new_spread)} ({delta:+.1f} pts)", "inline": False},
            {"name": "🎯 New Model Edge", "value": f"{new_edge:.2%}", "inline": True},
            {"name": "✅ Action", "value": action, "inline": False},
            {"name": "📝 Recommendation", "value": recommendation, "inline": False},
        ],
        "footer": {"text": f"Game: {game_key} • Monitor time: {datetime.utcnow().strftime('%H:%M UTC')}"},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return _post({"embeds": [embed]})
