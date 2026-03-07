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
