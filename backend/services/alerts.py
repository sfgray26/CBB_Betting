"""
Alert checking and dispatch.

Public API:
  check_performance_alerts(db)  → List[Alert]
  persist_alerts(db, alerts)    → None  (deduplicates within 24 h)
  send_alert(alert, channels)   → None  (email / SMS — skips if not configured)
  run_alert_check()             → List[Alert]  (entry point for scheduler)
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.models import BetLog, DBAlert, SessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    alert_type: str
    severity: str             # INFO | WARNING | CRITICAL
    message: str
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Alert logic
# ---------------------------------------------------------------------------

def check_performance_alerts(db: Session) -> List[Alert]:
    """
    Evaluate all alert conditions against settled bets.

    Conditions checked:
      1. CRITICAL  — mean CLV (last 50) < -0.5%
      2. WARNING   — mean CLV (last 50) < 0%
      3. WARNING   — drawdown > 15%
      4. INFO      — drawdown > 8%
      5. INFO      — win rate (last 100) < 48%
      6. WARNING   — ROI (last 50) < -5%
      7. WARNING   — 10+ consecutive losses
      8. INFO      — 7-9 consecutive losses
    """
    alerts: List[Alert] = []

    bets = (
        db.query(BetLog)
        .filter(BetLog.outcome.isnot(None), BetLog.outcome != -1)
        .order_by(BetLog.timestamp.asc())
        .all()
    )

    if not bets:
        return alerts

    total = len(bets)
    pl_list = [b.profit_loss_dollars or 0.0 for b in bets]

    # 1-2. CLV alerts — last 50 bets with CLV data
    last_50_clv = [b.clv_prob for b in bets[-50:] if b.clv_prob is not None]
    if len(last_50_clv) >= 10:
        mean_50_clv = sum(last_50_clv) / len(last_50_clv)
        if mean_50_clv < -0.005:
            alerts.append(Alert(
                alert_type="CLV_NEGATIVE",
                severity="CRITICAL",
                message=(
                    f"Mean CLV over last {len(last_50_clv)} bets is "
                    f"{mean_50_clv:.2%} — stop betting immediately"
                ),
                threshold=-0.005,
                current_value=round(mean_50_clv, 5),
                recommendation=(
                    "Stop betting. The model is systematically losing to the market. "
                    "Review line-shopping, model weights, and recent data quality."
                ),
            ))
        elif mean_50_clv < 0.0:
            alerts.append(Alert(
                alert_type="CLV_DECLINING",
                severity="WARNING",
                message=(
                    f"Mean CLV over last {len(last_50_clv)} bets dropped to {mean_50_clv:.2%}"
                ),
                threshold=0.0,
                current_value=round(mean_50_clv, 5),
                recommendation=(
                    "Consider halving bet sizes until CLV recovers. "
                    "Review recent games where you lost closing value."
                ),
            ))

    # 3-4. Drawdown
    running = peak = 0.0
    for pl in pl_list:
        running += pl
        if running > peak:
            peak = running

    current_dd = (peak - running) / peak if peak > 0 else 0.0

    if current_dd > 0.15:
        alerts.append(Alert(
            alert_type="DRAWDOWN_HIGH",
            severity="WARNING",
            message=f"Drawdown reached {current_dd:.1%} of bankroll peak",
            threshold=0.15,
            current_value=round(current_dd, 4),
            recommendation=(
                "Reduce bet sizes by 50% until drawdown falls below 10%. "
                "Do not chase losses."
            ),
        ))
    elif current_dd > 0.08:
        alerts.append(Alert(
            alert_type="DRAWDOWN_ELEVATED",
            severity="INFO",
            message=f"Drawdown at {current_dd:.1%} — within normal range",
            threshold=0.08,
            current_value=round(current_dd, 4),
            recommendation="Monitor. No action required unless it continues to grow.",
        ))

    # 5. Win rate (last 100)
    last_100 = bets[-100:]
    if len(last_100) >= 30:
        wr = sum(1 for b in last_100 if b.outcome == 1) / len(last_100)
        if wr < 0.48:
            alerts.append(Alert(
                alert_type="WIN_RATE_LOW",
                severity="INFO",
                message=(
                    f"Win rate over last {len(last_100)} bets: {wr:.1%} "
                    f"(break-even ~52.4% at -110)"
                ),
                threshold=0.48,
                current_value=round(wr, 4),
                recommendation=(
                    "Win rate is a noisy metric. Focus on CLV as the primary edge indicator. "
                    "Expected variance at this sample size is ±5-8%."
                ),
            ))

    # 6. ROI (last 50)
    last_50 = bets[-50:]
    if len(last_50) >= 20:
        l50_pl = sum(b.profit_loss_dollars or 0.0 for b in last_50)
        l50_risked = sum(b.bet_size_dollars or 0.0 for b in last_50)
        if l50_risked > 0:
            roi_50 = l50_pl / l50_risked
            if roi_50 < -0.05:
                alerts.append(Alert(
                    alert_type="ROI_NEGATIVE",
                    severity="WARNING",
                    message=f"ROI over last {len(last_50)} bets: {roi_50:.1%}",
                    threshold=-0.05,
                    current_value=round(roi_50, 4),
                    recommendation=(
                        "Edge may be deteriorating. Cross-check CLV and model inputs. "
                        "Consider a betting pause until you identify the cause."
                    ),
                ))

    # 7-8. Consecutive losses
    streak = 0
    for b in reversed(bets):
        if b.outcome == 0:
            streak += 1
        else:
            break

    if streak >= 10:
        alerts.append(Alert(
            alert_type="LOSING_STREAK",
            severity="WARNING",
            message=f"{streak} consecutive losses",
            threshold=10,
            current_value=float(streak),
            recommendation=(
                "Check for systematic model error or market regime change. "
                "Review whether your edges are still valid against current lines."
            ),
        ))
    elif streak >= 7:
        alerts.append(Alert(
            alert_type="LOSING_STREAK",
            severity="INFO",
            message=f"{streak} consecutive losses (within normal variance at this sample size)",
            threshold=7,
            current_value=float(streak),
            recommendation="No action needed. Variance is expected. Monitor CLV.",
        ))

    return alerts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_alerts(db: Session, alerts: List[Alert]) -> None:
    """
    Upsert alerts into the database, deduplicating by type within 24 hours.
    Updates the message and current_value if the alert already exists.
    """
    cutoff = datetime.utcnow() - timedelta(hours=24)

    for alert in alerts:
        existing = (
            db.query(DBAlert)
            .filter(
                DBAlert.alert_type == alert.alert_type,
                DBAlert.created_at >= cutoff,
                DBAlert.acknowledged == False,
            )
            .first()
        )

        if existing:
            existing.message = alert.message
            existing.current_value = alert.current_value
        else:
            db.add(DBAlert(
                alert_type=alert.alert_type,
                severity=alert.severity,
                message=alert.message,
                threshold=alert.threshold,
                current_value=alert.current_value,
            ))

    db.commit()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def send_alert(alert: Alert, channels: Optional[List[str]] = None) -> None:
    """
    Dispatch an alert over configured channels.
    Silently skips any channel whose credentials are missing.
    """
    if channels is None:
        channels = ["email"]

    for channel in channels:
        try:
            if channel == "email":
                _send_email(alert)
            elif channel == "sms":
                _send_sms(alert)
        except Exception as exc:
            logger.error("Alert dispatch failed (%s): %s", channel, exc)


def _send_email(alert: Alert) -> None:
    sg_key = os.getenv("SENDGRID_API_KEY")
    to_email = os.getenv("ALERT_EMAIL")
    if not sg_key or not to_email:
        logger.debug("Email alerts not configured — skipping")
        return

    import sendgrid
    from sendgrid.helpers.mail import Mail

    body = (
        f"Alert Type:     {alert.alert_type}\n"
        f"Severity:       {alert.severity}\n\n"
        f"{alert.message}\n\n"
        f"Recommendation: {alert.recommendation}\n\n"
        f"Current Value:  {alert.current_value}\n"
        f"Threshold:      {alert.threshold}\n"
        f"Generated at:   {datetime.utcnow().isoformat()}"
    )

    msg = Mail(
        from_email="alerts@cbb-edge.com",
        to_emails=to_email,
        subject=f"[CBB Edge] {alert.severity}: {alert.alert_type.replace('_', ' ')}",
        plain_text_content=body,
    )
    sendgrid.SendGridAPIClient(api_key=sg_key).send(msg)
    logger.info("Alert email sent: %s", alert.alert_type)


def _send_sms(alert: Alert) -> None:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    tok = os.getenv("TWILIO_AUTH_TOKEN")
    frm = os.getenv("TWILIO_FROM_NUMBER")
    to = os.getenv("TWILIO_TO_NUMBER")
    if not all([sid, tok, frm, to]):
        logger.debug("SMS alerts not configured — skipping")
        return

    from twilio.rest import Client

    body = f"CBB Edge {alert.severity}: {alert.message} — {alert.recommendation}"
    Client(sid, tok).messages.create(body=body[:160], from_=frm, to=to)
    logger.info("Alert SMS sent: %s", alert.alert_type)


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------

def run_alert_check() -> List[Alert]:
    """
    Full alert pipeline: check → persist → send critical ones.
    Called by the daily snapshot job.
    """
    db = SessionLocal()
    try:
        alerts = check_performance_alerts(db)
        if alerts:
            persist_alerts(db, alerts)
            for a in alerts:
                if a.severity == "CRITICAL":
                    send_alert(a, channels=["email", "sms"])
            logger.info(
                "Alert check complete: %d alerts (%d critical)",
                len(alerts),
                sum(1 for a in alerts if a.severity == "CRITICAL"),
            )
        return alerts
    finally:
        db.close()
