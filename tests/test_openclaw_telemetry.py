"""
Tests for OpenClaw Telemetry Dashboard.

Run with: python -m pytest tests/test_openclaw_telemetry.py -v
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, mock_open, AsyncMock

from backend.services.openclaw_telemetry import (
    SystemMetrics,
    Anomaly,
    AnomalyDetector,
    TelemetryCollector,
    TelemetryDashboard,
    check_system_health,
)


class TestSystemMetrics:
    """Test SystemMetrics dataclass."""
    
    def test_default_metrics(self):
        """Test default metric values."""
        metrics = SystemMetrics()
        
        assert metrics.active_data_sources == 0
        assert metrics.predictions_today == 0
        assert metrics.pending_escalations == 0
        assert metrics.timestamp is not None


class TestAnomalyDetector:
    """Test the AnomalyDetector class."""
    
    def test_no_anomalies_when_healthy(self):
        """Test that healthy metrics produce no anomalies."""
        detector = AnomalyDetector()
        
        metrics = SystemMetrics(
            active_data_sources=2,
            db_response_ms=100,
            pending_escalations=0
        )
        
        anomalies = detector.detect(metrics)
        
        assert len(anomalies) == 0
    
    def test_detect_low_data_sources(self):
        """Test detection of insufficient data sources."""
        detector = AnomalyDetector()
        
        metrics = SystemMetrics(
            active_data_sources=1,  # Below threshold of 2
            db_response_ms=100
        )
        
        anomalies = detector.detect(metrics)
        
        assert len(anomalies) == 1
        assert anomalies[0].component == "data_sources"
        assert anomalies[0].severity == "critical"
    
    def test_detect_slow_db(self):
        """Test detection of slow database response."""
        detector = AnomalyDetector()
        
        metrics = SystemMetrics(
            active_data_sources=2,
            db_response_ms=1500  # Above threshold of 1000
        )
        
        anomalies = detector.detect(metrics)
        
        assert len(anomalies) == 1
        assert anomalies[0].component == "database"
        assert anomalies[0].severity == "warning"
    
    def test_detect_many_pending_escalations(self):
        """Test detection of too many pending escalations."""
        detector = AnomalyDetector()
        
        metrics = SystemMetrics(
            active_data_sources=2,
            pending_escalations=10  # Above threshold of 5
        )
        
        anomalies = detector.detect(metrics)
        
        assert len(anomalies) == 1
        assert anomalies[0].component == "escalations"
    
    def test_detect_multiple_anomalies(self):
        """Test detection of multiple simultaneous issues."""
        detector = AnomalyDetector()
        
        metrics = SystemMetrics(
            active_data_sources=0,  # Critical
            db_response_ms=2000,     # Warning
            pending_escalations=10   # Warning
        )
        
        anomalies = detector.detect(metrics)
        
        assert len(anomalies) == 3
        severities = [a.severity for a in anomalies]
        assert "critical" in severities
        assert "warning" in severities


class TestTelemetryDashboard:
    """Test the TelemetryDashboard class."""
    
    @pytest.mark.asyncio
    @patch("backend.services.openclaw_telemetry.send_to_channel")
    async def test_alert_sent_on_anomalies(self, mock_send):
        """Test that alert is sent when anomalies detected."""
        mock_send.return_value = True
        
        dashboard = TelemetryDashboard()
        
        # Mock collector to return metrics with anomalies (use AsyncMock for async function)
        dashboard.collector.collect = AsyncMock(return_value=SystemMetrics(
            active_data_sources=0,  # Will trigger anomaly
        ))
        
        sent = await dashboard.check_and_alert()
        
        assert sent is True
        mock_send.assert_called_once()
        
        # Verify alert format
        args, kwargs = mock_send.call_args
        assert args[0] == "openclaw-health"  # First positional arg
        embed = kwargs.get("embed", {})
        assert "Alert" in embed.get("title", "") or "issue" in embed.get("description", "").lower()
    
    @pytest.mark.asyncio
    @patch("backend.services.openclaw_telemetry.send_to_channel")
    async def test_no_alert_when_healthy(self, mock_send):
        """Test that no alert sent when system healthy (not summary time)."""
        mock_send.return_value = True
        
        dashboard = TelemetryDashboard()
        
        # Mock collector to return healthy metrics (use AsyncMock)
        dashboard.collector.collect = AsyncMock(return_value=SystemMetrics(
            active_data_sources=2,
            db_response_ms=50
        ))
        
        # Mock should_send_daily_summary to return False
        dashboard._should_send_daily_summary = MagicMock(return_value=False)
        
        sent = await dashboard.check_and_alert()
        
        # Should not send (quiet mode)
        assert sent is False
        mock_send.assert_not_called()
    
    @pytest.mark.asyncio
    @patch("backend.services.openclaw_telemetry.send_to_channel")
    async def test_summary_sent_when_forced(self, mock_send):
        """Test that summary is sent when force_summary=True."""
        mock_send.return_value = True
        
        dashboard = TelemetryDashboard()
        
        dashboard.collector.collect = AsyncMock(return_value=SystemMetrics(
            active_data_sources=2,
            predictions_today=8,
            pending_escalations=1
        ))
        
        sent = await dashboard.check_and_alert(force_summary=True)
        
        assert sent is True
        mock_send.assert_called_once()
        
        # Verify summary format
        args, kwargs = mock_send.call_args
        embed = kwargs.get("embed", {})
        assert "Status" in embed.get("title", "")
    
    def test_generate_alert_embed(self):
        """Test alert embed generation."""
        dashboard = TelemetryDashboard()
        
        metrics = SystemMetrics(active_data_sources=2)
        anomalies = [
            Anomaly("warning", "test", "Test warning", 10, 5)
        ]
        
        embed = dashboard._generate_alert_embed(metrics, anomalies)
        
        assert "Alert" in embed["title"]
        assert embed["color"] > 0
        assert "1 issue" in embed["description"].lower() or "issue(s)" in embed["description"]
    
    def test_generate_summary_embed(self):
        """Test summary embed generation."""
        dashboard = TelemetryDashboard()
        
        metrics = SystemMetrics(
            active_data_sources=2,
            predictions_today=10,
            pending_escalations=2,
            db_response_ms=45.5
        )
        
        embed = dashboard._generate_summary_embed(metrics)
        
        assert "Status" in embed["title"]
        assert embed["color"] > 0
        assert "fields" in embed
        
        # Should have data sources field
        field_names = [f["name"] for f in embed["fields"]]
        assert "Data Sources" in field_names or any("source" in n.lower() for n in field_names)


class TestTelemetryCollector:
    """Test the TelemetryCollector class."""
    
    @pytest.mark.asyncio
    async def test_collect_returns_metrics(self):
        """Test that collect returns SystemMetrics."""
        collector = TelemetryCollector()
        
        metrics = await collector.collect()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp is not None


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    @patch("backend.services.openclaw_telemetry.send_to_channel")
    async def test_check_system_health(self, mock_send):
        """Test end-to-end system health check."""
        mock_send.return_value = True
        
        # Force summary to ensure something happens
        result = await check_system_health(force_summary=True)
        
        # Should complete without error
        assert isinstance(result, bool)


class TestCLIMode:
    """Test CLI functionality."""
    
    def test_module_runs_as_script(self):
        """Test that module can be run directly."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "-m", "backend.services.openclaw_telemetry", "--test"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should generate test output
        assert result.returncode == 0
        assert "test" in result.stdout.lower() or "Summary" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
