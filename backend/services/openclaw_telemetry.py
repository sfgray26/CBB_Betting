"""
OpenClaw Telemetry Dashboard — Quiet System Health Monitoring

Designed to be non-noisy:
- Only alerts when there are actual issues
- Daily summary mode (one clean status message per day)
- Anomaly detection (don't report "all good" constantly)

Author: Kimi CLI / Claude Code
Document: OPCL-001 Phase 1
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from backend.services.discord_notifier import send_to_channel, _COLOR_GREEN, _COLOR_YELLOW, _COLOR_RED

logger = logging.getLogger("openclaw_telemetry")


# State file for tracking last status (to avoid duplicate alerts)
STATE_FILE = Path(".openclaw/telemetry_state.json")


@dataclass
class SystemMetrics:
    """Core system health metrics."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Data sources
    kenpom_status: str = "unknown"  # "ok", "degraded", "down"
    barttorvik_status: str = "unknown"
    odds_monitor_status: str = "unknown"
    active_data_sources: int = 0
    
    # Model performance
    predictions_today: int = 0
    integrity_checks_24h: int = 0
    avg_integrity_latency_ms: float = 0.0
    
    # Sharp money
    sharp_signals_24h: int = 0
    high_confidence_signals: int = 0
    
    # Escalations
    pending_escalations: int = 0
    resolved_today: int = 0
    
    # System health
    db_response_ms: Optional[float] = None
    discord_connected: bool = False
    last_odds_poll_minutes: Optional[int] = None


@dataclass
class Anomaly:
    """Detected system anomaly."""
    severity: str  # "warning", "critical"
    component: str
    message: str
    metric_value: Optional[Any] = None
    threshold: Optional[Any] = None


class TelemetryCollector:
    """
    Collects system telemetry from various sources.
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger("openclaw_telemetry.collector")
    
    async def collect(self) -> SystemMetrics:
        """Collect all system metrics."""
        metrics = SystemMetrics()
        
        try:
            # Data sources
            await self._check_data_sources(metrics)
            
            # Predictions and integrity
            await self._check_predictions(metrics)
            
            # Sharp money
            await self._check_sharp_money(metrics)
            
            # Escalations
            await self._check_escalations(metrics)
            
            # System health
            await self._check_system_health(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting telemetry: {e}")
        
        return metrics
    
    async def _check_data_sources(self, metrics: SystemMetrics):
        """Check data source health."""
        try:
            # Check ratings status endpoint or internal state
            # For now, set to ok if we can query the DB
            from backend.models import SessionLocal
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
            
            metrics.active_data_sources = 2  # KenPom + BartTorvik
            metrics.kenpom_status = "ok"
            metrics.barttorvik_status = "ok"
            
        except Exception as e:
            self.logger.warning(f"Data source check failed: {e}")
            metrics.active_data_sources = 0
    
    async def _check_predictions(self, metrics: SystemMetrics):
        """Check prediction stats."""
        try:
            from backend.models import Prediction, SessionLocal
            from datetime import date
            
            db = SessionLocal()
            
            # Today's predictions
            today = date.today()
            metrics.predictions_today = db.query(Prediction).filter(
                Prediction.game_date == today
            ).count()
            
            # Integrity checks (from telemetry if available)
            # This would integrate with openclaw_lite telemetry
            metrics.integrity_checks_24h = 0  # TODO: Wire to actual telemetry
            
            db.close()
            
        except Exception as e:
            self.logger.warning(f"Prediction check failed: {e}")
    
    async def _check_sharp_money(self, metrics: SystemMetrics):
        """Check sharp money signals."""
        try:
            # This would query the sharp money detector
            # For now, leave at 0
            metrics.sharp_signals_24h = 0  # TODO: Wire to sharp_money service
            metrics.high_confidence_signals = 0
            
        except Exception as e:
            self.logger.warning(f"Sharp money check failed: {e}")
    
    async def _check_escalations(self, metrics: SystemMetrics):
        """Check escalation queue."""
        try:
            from backend.services.openclaw_lite import HighStakesEscalationQueue
            
            queue = HighStakesEscalationQueue()
            pending = queue.get_pending()
            metrics.pending_escalations = len(pending)
            
            # Count resolved today (from file timestamps)
            # This is approximate
            resolved_today = 0
            for f in queue.queue_dir.glob("*.json"):
                try:
                    with open(f) as fp:
                        entry = json.load(fp)
                        if entry.get("status") == "resolved":
                            resolved_at = entry.get("resolved_at", "")
                            if resolved_at.startswith(datetime.now(timezone.utc).strftime("%Y-%m-%d")):
                                resolved_today += 1
                except:
                    pass
            
            metrics.resolved_today = resolved_today
            
        except Exception as e:
            self.logger.warning(f"Escalation check failed: {e}")
    
    async def _check_system_health(self, metrics: SystemMetrics):
        """Check core system health."""
        try:
            # DB response time
            import time
            from backend.models import SessionLocal
            
            start = time.time()
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
            metrics.db_response_ms = (time.time() - start) * 1000
            
            # Discord connectivity (will be tested when we try to send)
            metrics.discord_connected = True  # Assumed, actual check on send
            
        except Exception as e:
            self.logger.warning(f"System health check failed: {e}")
            metrics.db_response_ms = None


class AnomalyDetector:
    """
    Detects anomalies in system metrics.
    
    Only reports actual issues, not "everything is fine" noise.
    """
    
    # Thresholds for anomaly detection
    THRESHOLDS = {
        "data_sources_min": 2,  # Should have at least 2 sources
        "db_response_max_ms": 1000,  # DB should respond in <1s
        "pending_escalations_max": 5,  # Alert if >5 pending
        "odds_poll_stale_minutes": 30,  # Alert if odds >30 min old
    }
    
    def detect(self, metrics: SystemMetrics) -> List[Anomaly]:
        """
        Detect anomalies in metrics.
        
        Returns:
            List of detected anomalies (empty if all good)
        """
        anomalies = []
        
        # Check data sources
        if metrics.active_data_sources < self.THRESHOLDS["data_sources_min"]:
            anomalies.append(Anomaly(
                severity="critical",
                component="data_sources",
                message=f"Only {metrics.active_data_sources} data sources active (need {self.THRESHOLDS['data_sources_min']})",
                metric_value=metrics.active_data_sources,
                threshold=self.THRESHOLDS["data_sources_min"]
            ))
        
        # Check DB response
        if metrics.db_response_ms and metrics.db_response_ms > self.THRESHOLDS["db_response_max_ms"]:
            anomalies.append(Anomaly(
                severity="warning",
                component="database",
                message=f"Database slow: {metrics.db_response_ms:.0f}ms response time",
                metric_value=metrics.db_response_ms,
                threshold=self.THRESHOLDS["db_response_max_ms"]
            ))
        
        # Check escalations
        if metrics.pending_escalations > self.THRESHOLDS["pending_escalations_max"]:
            anomalies.append(Anomaly(
                severity="warning",
                component="escalations",
                message=f"{metrics.pending_escalations} escalations pending review",
                metric_value=metrics.pending_escalations,
                threshold=self.THRESHOLDS["pending_escalations_max"]
            ))
        
        # Check odds freshness
        if metrics.last_odds_poll_minutes and metrics.last_odds_poll_minutes > self.THRESHOLDS["odds_poll_stale_minutes"]:
            anomalies.append(Anomaly(
                severity="warning",
                component="odds_monitor",
                message=f"Odds data stale: {metrics.last_odds_poll_minutes} min since last poll",
                metric_value=metrics.last_odds_poll_minutes,
                threshold=self.THRESHOLDS["odds_poll_stale_minutes"]
            ))
        
        return anomalies


class TelemetryDashboard:
    """
    Quiet telemetry dashboard for Discord.
    
    Philosophy: Only speak when something needs attention.
    """
    
    def __init__(self):
        self.collector = TelemetryCollector()
        self.detector = AnomalyDetector()
        self.logger = logging.getLogger("openclaw_telemetry.dashboard")
    
    async def check_and_alert(self, force_summary: bool = False) -> bool:
        """
        Check system health and alert if needed.
        
        Args:
            force_summary: If True, send daily summary even if no anomalies
        
        Returns:
            True if message was sent
        """
        # Collect metrics
        metrics = await self.collector.collect()
        
        # Detect anomalies
        anomalies = self.detector.detect(metrics)
        
        # Decide whether to send
        should_send = False
        alert_only = False
        
        if anomalies:
            # Always send if there are anomalies
            should_send = True
            alert_only = True
            self.logger.info(f"Detected {len(anomalies)} anomalies, sending alert")
        
        elif force_summary:
            # Send daily summary
            should_send = True
            alert_only = False
            self.logger.info("Sending daily summary")
        
        elif self._should_send_daily_summary():
            # Time for daily summary
            should_send = True
            alert_only = False
            self.logger.info("Daily summary time, sending status")
        
        if not should_send:
            self.logger.debug("No anomalies and not summary time — skipping")
            return False
        
        # Generate and send embed
        embed = self._generate_embed(metrics, anomalies, alert_only)
        success = send_to_channel("openclaw-health", embed=embed)
        
        if success:
            self._update_state()
        
        return success
    
    def _should_send_daily_summary(self) -> bool:
        """Check if we should send the daily summary (once per day at 12 PM)."""
        try:
            if not STATE_FILE.exists():
                return True
            
            with open(STATE_FILE) as f:
                state = json.load(f)
            
            last_sent = state.get("last_summary_sent", "")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            # Only send once per day
            return last_sent != today
            
        except Exception as e:
            self.logger.warning(f"Error checking state: {e}")
            return True
    
    def _update_state(self):
        """Update state file after sending."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            state = {}
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    state = json.load(f)
            
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            state["last_summary_sent"] = today
            state["last_check"] = datetime.now(timezone.utc).isoformat()
            
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Error updating state: {e}")
    
    def _generate_embed(self, metrics: SystemMetrics, anomalies: List[Anomaly], 
                        alert_only: bool) -> Dict[str, Any]:
        """Generate the telemetry embed."""
        
        if anomalies and alert_only:
            # Alert mode: highlight issues
            return self._generate_alert_embed(metrics, anomalies)
        else:
            # Summary mode: clean daily status
            return self._generate_summary_embed(metrics)
    
    def _generate_alert_embed(self, metrics: SystemMetrics, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Generate alert embed for anomalies."""
        # Determine color based on worst severity
        has_critical = any(a.severity == "critical" for a in anomalies)
        color = _COLOR_RED if has_critical else _COLOR_YELLOW
        
        # Build description
        lines = [f"⚠️ **{len(anomalies)} issue(s) detected**:\n"]
        
        for anomaly in anomalies:
            emoji = "🔴" if anomaly.severity == "critical" else "🟡"
            lines.append(f"{emoji} **{anomaly.component}**: {anomaly.message}")
        
        # Add quick metrics
        lines.append("\n📊 Quick status:")
        lines.append(f"• Data sources: {metrics.active_data_sources}/2")
        lines.append(f"• Predictions today: {metrics.predictions_today}")
        lines.append(f"• Pending escalations: {metrics.pending_escalations}")
        
        description = "\n".join(lines)
        
        return {
            "title": "🚨 OpenClaw System Alert",
            "description": description,
            "color": color,
            "footer": {
                "text": "OpenClaw Telemetry | Anomaly detected"
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _generate_summary_embed(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Generate clean daily summary embed."""
        
        # Determine overall status
        status_emoji = "✅" if metrics.active_data_sources >= 2 else "⚠️"
        
        # Build fields
        fields = [
            {
                "name": "Data Sources",
                "value": f"{metrics.active_data_sources}/2 active\nKP: {metrics.kenpom_status}\nBT: {metrics.barttorvik_status}",
                "inline": True
            },
            {
                "name": "Today's Activity",
                "value": f"Predictions: {metrics.predictions_today}\nIntegrity: {metrics.integrity_checks_24h} checks\nSharp: {metrics.sharp_signals_24h} signals",
                "inline": True
            },
            {
                "name": "Escalations",
                "value": f"Pending: {metrics.pending_escalations}\nResolved today: {metrics.resolved_today}",
                "inline": True
            }
        ]
        
        # Add DB response if available
        if metrics.db_response_ms:
            db_status = "✅" if metrics.db_response_ms < 500 else "⚠️"
            fields.append({
                "name": "Database",
                "value": f"{db_status} {metrics.db_response_ms:.0f}ms",
                "inline": True
            })
        
        return {
            "title": f"{status_emoji} OpenClaw System Status",
            "description": "Daily health summary — all systems operational" if metrics.active_data_sources >= 2 else "⚠️ Degraded performance detected",
            "color": _COLOR_GREEN if metrics.active_data_sources >= 2 else _COLOR_YELLOW,
            "fields": fields,
            "footer": {
                "text": "OpenClaw Telemetry | Daily Summary"
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def check_system_health(force_summary: bool = False) -> bool:
    """
    Check system health and send alert if needed.

    PAUSED (2026-04-21): OpenClaw telemetry and Discord health alerts are
    disabled to reduce noise while the baseball module is being implemented.

    Args:
        force_summary: Ignored while paused.

    Returns:
        False — OpenClaw is paused.
    """
    logger.info("OpenClaw telemetry check skipped — paused until baseball module is complete")
    return False


def check_system_health_sync(force_summary: bool = False) -> bool:
    """Synchronous wrapper for check_system_health."""
    logger.info("OpenClaw telemetry check skipped — paused until baseball module is complete")
    return False


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="OpenClaw Telemetry")
    parser.add_argument("--force-summary", action="store_true", 
                        help="Force daily summary even if no issues")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: generate sample embed")
    
    args = parser.parse_args()
    
    if args.test:
        print("Generating test telemetry...")
        
        # Create sample metrics
        metrics = SystemMetrics(
            active_data_sources=2,
            kenpom_status="ok",
            barttorvik_status="ok",
            predictions_today=8,
            integrity_checks_24h=12,
            sharp_signals_24h=3,
            pending_escalations=1,
            resolved_today=2,
            db_response_ms=45.5
        )
        
        dashboard = TelemetryDashboard()
        
        # Test summary mode
        print("\n=== Summary Mode ===")
        embed = dashboard._generate_summary_embed(metrics)
        print(json.dumps(embed, indent=2))
        
        # Test alert mode
        print("\n=== Alert Mode ===")
        anomalies = [
            Anomaly("warning", "database", "DB slow: 1200ms response", 1200, 1000),
            Anomaly("critical", "data_sources", "Only 1 source active", 1, 2)
        ]
        embed = dashboard._generate_alert_embed(metrics, anomalies)
        print(json.dumps(embed, indent=2))
        
        print("\nUse --force-summary to send to Discord")
    
    else:
        print("Checking system health...")
        success = check_system_health_sync(force_summary=args.force_summary)
        print(f"Result: {'SENT' if success else 'SKIPPED'}")
        sys.exit(0 if success else 1)
