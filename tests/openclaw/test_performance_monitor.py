"""
Tests for OpenClaw PerformanceMonitor.

Validates:
- CLV decay detection thresholds (5% elevated, 8% warning, 15% critical)
- Win rate tracking vs expected
- Discord embed generation
- Guardian freeze compliance
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backend.services.openclaw.performance_monitor import (
    PerformanceMonitor,
    DecayReport,
    WinRateReport,
    DecaySeverity
)
from backend.services.openclaw.database import OpenClawDB


class TestDecaySeverityThresholds:
    """Validate decay severity thresholds per spec."""
    
    def test_normal_threshold(self):
        """< 5% decay = NORMAL"""
        assert DecaySeverity.NORMAL == PerformanceMonitor._get_severity(0.04)
        assert DecaySeverity.NORMAL == PerformanceMonitor._get_severity(0.00)
        
    def test_elevated_threshold(self):
        """5-8% decay = ELEVATED"""
        assert DecaySeverity.ELEVATED == PerformanceMonitor._get_severity(0.05)
        assert DecaySeverity.ELEVATED == PerformanceMonitor._get_severity(0.07)
        
    def test_warning_threshold(self):
        """8-15% decay = WARNING"""
        assert DecaySeverity.WARNING == PerformanceMonitor._get_severity(0.08)
        assert DecaySeverity.WARNING == PerformanceMonitor._get_severity(0.14)
        
    def test_critical_threshold(self):
        """> 15% decay = CRITICAL"""
        assert DecaySeverity.CRITICAL == PerformanceMonitor._get_severity(0.15)
        assert DecaySeverity.CRITICAL == PerformanceMonitor._get_severity(0.25)


class TestDecayReport:
    """Test DecayReport data structure and formatting."""
    
    def test_decay_report_creation(self):
        """DecayReport can be created with all fields."""
        report = DecayReport(
            sport='cbb',
            current_decay_pct=12.5,
            severity=DecaySeverity.WARNING,
            window_hours=48,
            sample_size=45,
            trend='degrading',
            confidence=0.85
        )
        
        assert report.sport == 'cbb'
        assert report.current_decay_pct == 12.5
        assert report.severity == DecaySeverity.WARNING
        
    def test_discord_embed_generation(self):
        """Discord embed contains all required fields."""
        report = DecayReport(
            sport='cbb',
            current_decay_pct=15.2,
            severity=DecaySeverity.CRITICAL,
            window_hours=48,
            sample_size=50,
            trend='degrading',
            confidence=0.90
        )
        
        embed = report.to_discord_embed()
        
        assert 'Performance Monitor' in embed['title']
        assert 'CRITICAL' in embed['fields'][1]['value']
        assert '15.2%' in embed['fields'][0]['value']
        assert embed['color'] == 0xFF0000  # Red for critical


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock(spec=OpenClawDB)
        db.is_guardian_active.return_value = True
        return db
    
    @pytest.fixture
    def monitor(self, mock_db):
        """Create monitor with mock DB."""
        return PerformanceMonitor(sport='cbb', db=mock_db)
    
    def test_check_clv_decay_insufficient_samples(self, monitor, mock_db):
        """Returns insufficient_data when sample size too small."""
        mock_db.get_clv_window.return_value = [
            {'edge_percent': 0.05, 'closing_edge': 0.03}
        ] * 10  # Only 10 samples
        
        report = monitor.check_clv_decay(window_hours=48, min_samples=20)
        
        assert report.trend == 'insufficient_data'
        assert report.sample_size == 10
        
    def test_check_clv_decay_normal(self, monitor, mock_db):
        """Detects normal decay levels."""
        mock_db.get_clv_window.return_value = [
            {'edge_percent': 0.05, 'closing_edge': 0.048}  # 4% decay
        ] * 30
        
        report = monitor.check_clv_decay()
        
        assert report.severity == DecaySeverity.NORMAL
        assert abs(report.current_decay_pct - 4.0) < 0.5
        
    def test_check_clv_decay_critical(self, monitor, mock_db):
        """Detects critical decay levels."""
        mock_db.get_clv_window.return_value = [
            {'edge_percent': 0.05, 'closing_edge': 0.02}  # 60% decay
        ] * 50
        
        report = monitor.check_clv_decay()
        
        assert report.severity == DecaySeverity.CRITICAL
        
    def test_check_clv_decay_trend_improving(self, monitor, mock_db):
        """Detects improving trend in second half."""
        # First half: high decay, Second half: low decay
        data = [{'edge_percent': 0.05, 'closing_edge': 0.02}] * 25  # 60% decay
        data += [{'edge_percent': 0.05, 'closing_edge': 0.048}] * 25  # 4% decay
        mock_db.get_clv_window.return_value = data
        
        report = monitor.check_clv_decay()
        
        assert report.trend == 'improving'
        
    def test_win_rate_underperformance(self, monitor, mock_db):
        """Detects significant underperformance."""
        games = []
        for i in range(40):
            game = {
                'edge_at_bet': 0.03,
                'predicted_spread': -3.0,
                'spread_result': -2.0 if i < 15 else 5.0  # First 15 win, rest lose
            }
            games.append(game)
        
        mock_db.get_game_outcomes.return_value = games
        
        report = monitor.check_win_rate()
        
        assert report.actual_win_rate < report.expected_win_rate
        assert report.sample_size > 0


class TestGuardianCompliance:
    """Verify read-only operation during Guardian freeze."""
    
    def test_guardian_date_calculation(self):
        """Guardian freeze date is Apr 7, 2026."""
        db = OpenClawDB()
        assert db.GUARDIAN_LIFT_DATE.year == 2026
        assert db.GUARDIAN_LIFT_DATE.month == 4
        assert db.GUARDIAN_LIFT_DATE.day == 7
        
    def test_save_blocked_during_freeze(self):
        """Write operations blocked during Guardian freeze."""
        db = OpenClawDB()
        
        # Mock as if we're before Apr 7
        db._read_only = True
        
        from backend.services.openclaw.database import PerformanceMetric
        metric = PerformanceMetric(
            sport='cbb',
            metric_type='accuracy',
            value=0.55,
            sample_size=100,
            window_days=7,
            calculated_at=datetime.now()
        )
        
        result = db.save_metric(metric)
        assert result == False  # Blocked
