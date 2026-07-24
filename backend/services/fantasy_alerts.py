"""Fantasy Baseball operational alert hooks."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from backend.models import DBAlert, SessionLocal

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_YAHOO_AUTH_ALERT_TYPE = "YAHOO_AUTH_OUTAGE"


def report_yahoo_auth_outage(
    *,
    failure_count: int,
    threshold: int,
    circuit_open_until: Optional[datetime],
    status_code: int = 403,
) -> bool:
    """
    Persist and dispatch a critical alert when Yahoo auth is circuit-open.

    The alert intentionally excludes access/refresh token values. Discord routing
    uses existing DISCORD_* environment variables and no-ops when not configured.
    """
    now = datetime.now(_ET)
    message = (
        "Yahoo Fantasy auth outage: repeated auth failures opened the Yahoo "
        f"auth circuit ({failure_count}/{threshold}, last HTTP {status_code})."
    )
    recommendation = (
        "Verify Yahoo developer app Fantasy Sports authorization, re-run OAuth "
        "if needed, then update Railway Yahoo token variables before redeploy."
    )

    persisted = _persist_yahoo_auth_alert(
        message=message,
        recommendation=recommendation,
        failure_count=failure_count,
        threshold=threshold,
        now=now,
    )
    sent = _send_yahoo_auth_discord_alert(
        message=message,
        recommendation=recommendation,
        circuit_open_until=circuit_open_until,
        now=now,
    )
    return persisted or sent


def _persist_yahoo_auth_alert(
    *,
    message: str,
    recommendation: str,
    failure_count: int,
    threshold: int,
    now: datetime,
) -> bool:
    db: Session = SessionLocal()
    try:
        cutoff = now - timedelta(hours=1)
        existing = (
            db.query(DBAlert)
            .filter(
                DBAlert.alert_type == _YAHOO_AUTH_ALERT_TYPE,
                DBAlert.created_at >= cutoff,
                DBAlert.acknowledged == False,
            )
            .first()
        )
        if existing:
            existing.message = message
            existing.current_value = float(failure_count)
        else:
            db.add(
                DBAlert(
                    alert_type=_YAHOO_AUTH_ALERT_TYPE,
                    severity="CRITICAL",
                    message=f"{message} {recommendation}",
                    threshold=float(threshold),
                    current_value=float(failure_count),
                )
            )
        db.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("Failed to persist Yahoo auth outage alert: %s", exc)
        return False
    finally:
        db.close()


def _send_yahoo_auth_discord_alert(
    *,
    message: str,
    recommendation: str,
    circuit_open_until: Optional[datetime],
    now: datetime,
) -> bool:
    try:
        from backend.services.discord_notifier import send_to_channel
    except Exception as exc:  # noqa: BLE001
        logger.debug("Discord notifier unavailable for Yahoo auth alert: %s", exc)
        return False

    until_text = "unknown"
    if circuit_open_until is not None:
        until_text = circuit_open_until.astimezone(timezone.utc).isoformat()

    embed = {
        "title": "Yahoo Fantasy Auth Outage",
        "description": message,
        "color": 0xE74C3C,
        "fields": [
            {"name": "Impact", "value": "Fantasy Yahoo reads/writes are disabled while the circuit is open.", "inline": False},
            {"name": "Circuit open until", "value": until_text, "inline": True},
            {"name": "Action required", "value": recommendation, "inline": False},
        ],
        "footer": {"text": "CBB Edge Fantasy"},
        "timestamp": now.astimezone(timezone.utc).isoformat(),
    }
    return send_to_channel("data-alerts", embed=embed, mention_admin=True)
