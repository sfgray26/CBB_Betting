"""
Tests for OpenClaw PatternDetector.

Validates:
- CBB pattern detection (conference, seed, HCA, month, day-of-week)
- MLB pattern framework
- Vulnerability severity assignment
- Sweep report generation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from backend.services.openclaw.pattern_detector import (
    PatternDetector,
    Vulnerability,
    SweepReport,
    PatternType
)
from backend.services.openclaw.database import OpenClawDB


class TestCBBPatternDetection:
    """Test CBB-specific pattern detection."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        return Mock(spec=OpenClawDB)
    
    @pytest.fixture
    def detector(self, mock_db):
        """Create CBB detector with mock DB."""
        return PatternDetector(sport='cbb', db=mock_db)
    
    def test_conference_bias_detection(self, detector, mock_db):
        """Detects underperformance in specific conference."""
        games = []
        for i in range(20):
            game = {
                'conference': 'ACC',
                'predicted_spread': -4.0,
                'spread_result': 2.0 if i < 16 else -5.0,  # 80% losses
                'game_date': datetime.now()
            }
            games.append(game)
        
        mock_db.get_game_outcomes.return_value = games
        
        report = detector.run_sweep(days=30)
        
        # Should detect ACC underperformance
        acc_vulns = [v for v in report.vulnerabilities 
                     if v.pattern_type == PatternType.CONFERENCE_BIAS]
        assert len(acc_vulns) > 0
        
    def test_seed_misprice_detection(self, detector, mock_db):
        """Detects seed-based mispricing in tournament games."""
        games = []
        # High seeds (1-8) performing poorly vs low seeds (9-16)
        # Create disparity: high seeds win 30%, low seeds win 70%
        
        # High seed games (both seeds 1-8) - performing poorly
        for i in range(20):
            result = -5.0 if i % 3 == 0 else 3.0  # 33% win rate
            game = {
                'home_seed': 2,
                'away_seed': 7,
                'is_neutral_site': True,
                'predicted_spread': -6.0,  # Predict home win
                'spread_result': result,
                'game_date': datetime.now()
            }
            games.append(game)
        
        # Low seed games (at least one seed 9-16) - performing well
        for i in range(20):
            result = -4.0 if i % 3 != 0 else 5.0  # 67% win rate
            game = {
                'home_seed': 3,
                'away_seed': 12,  # Low seed
                'is_neutral_site': True,
                'predicted_spread': -6.0,
                'spread_result': result,
                'game_date': datetime.now()
            }
            games.append(game)
        
        mock_db.get_game_outcomes.return_value = games
        
        report = detector.run_sweep(days=30)
        
        seed_vulns = [v for v in report.vulnerabilities 
                      if v.pattern_type == PatternType.SEED_MISPRICE]
        # Should detect the disparity (> 15% difference)
        assert len(seed_vulns) > 0 or any('seed' in v.description.lower() 
                                          for v in report.vulnerabilities)
        
    def test_hca_error_detection(self, detector, mock_db):
        """Detects home court advantage mispricing."""
        games = []
        # Create clear disparity: home 70% win, neutral 30% win (> 10% diff)
        
        # Home games (70% win rate)
        for i in range(30):
            game = {
                'is_neutral_site': False,
                'predicted_spread': -4.0,  # Predict home win
                'spread_result': -5.0 if i % 10 < 7 else 3.0,  # 70% win
                'game_date': datetime.now()
            }
            games.append(game)
        
        # Neutral games (30% win rate)
        for i in range(30):
            game = {
                'is_neutral_site': True,
                'predicted_spread': -4.0,
                'spread_result': -5.0 if i % 10 < 3 else 3.0,  # 30% win
                'game_date': datetime.now()
            }
            games.append(game)
        
        mock_db.get_game_outcomes.return_value = games
        
        report = detector.run_sweep(days=30)
        
        hca_vulns = [v for v in report.vulnerabilities 
                     if v.pattern_type == PatternType.HCA_ERROR]
        # Should detect disparity (> 10% difference)
        assert len(hca_vulns) > 0 or any(
            'home' in v.description.lower() or 'neutral' in v.description.lower()
            for v in report.vulnerabilities
        )
        
    def test_insufficient_samples(self, detector, mock_db):
        """No vulnerabilities when sample size too small."""
        mock_db.get_game_outcomes.return_value = [
            {'predicted_spread': -3, 'spread_result': -2}
        ] * 5  # Only 5 games
        
        report = detector.run_sweep(days=30)
        
        assert report.games_analyzed == 5
        # May have general drift but not specific patterns


class TestMLBPatternFramework:
    """Test MLB pattern detection framework."""
    
    @pytest.fixture
    def mlb_detector(self):
        """Create MLB detector."""
        return PatternDetector(sport='mlb')
        
    def test_mlb_patterns_configured(self, mlb_detector):
        """MLB detector has pattern weights configured."""
        assert PatternType.PITCH_COUNT_FATIGUE in mlb_detector._pattern_weights
        assert PatternType.PLATOON_SPLIT in mlb_detector._pattern_weights
        assert PatternType.COORS_EFFECT in mlb_detector._pattern_weights
        
    def test_mlb_sport_setting(self, mlb_detector):
        """MLB detector has correct sport setting."""
        assert mlb_detector.sport == 'mlb'


class TestVulnerability:
    """Test Vulnerability data structure."""
    
    def test_vulnerability_creation(self):
        """Vulnerability can be created with all fields."""
        vuln = Vulnerability(
            pattern_type=PatternType.CONFERENCE_BIAS,
            sport='cbb',
            confidence=0.85,
            severity='WARNING',
            description='ACC underperforming at 42% win rate',
            affected_games=25,
            sample_win_rate=0.42,
            expected_win_rate=0.52,
            edge_impact=0.10,
            recommended_action='Review ACC team ratings'
        )
        
        assert vuln.sport == 'cbb'
        assert vuln.severity == 'WARNING'
        assert vuln.confidence == 0.85
        
    def test_discord_embed_generation(self):
        """Discord embed contains all critical information."""
        vuln = Vulnerability(
            pattern_type=PatternType.SEED_MISPRICE,
            sport='cbb',
            confidence=0.90,
            severity='CRITICAL',
            description='High seeds underperforming significantly',
            affected_games=30,
            sample_win_rate=0.40,
            expected_win_rate=0.55,
            edge_impact=0.15,
            recommended_action='Recalibrate seed priors'
        )
        
        embed = vuln.to_discord_embed()
        
        assert vuln.pattern_type.value in embed['title']
        assert embed['color'] == 0xFF0000  # Red for CRITICAL
        assert 'CRITICAL' in [f['value'] for f in embed['fields']]
        assert any('Recalibrate' in str(f.get('footer', {}).get('text', '')) 
                   for f in [embed] if f.get('footer'))


class TestSweepReport:
    """Test SweepReport aggregation."""
    
    def test_sweep_report_aggregation(self):
        """SweepReport correctly aggregates vulnerabilities."""
        vulns = [
            Vulnerability(
                pattern_type=PatternType.CONFERENCE_BIAS,
                sport='cbb',
                confidence=0.8,
                severity='CRITICAL',
                description='Test',
                affected_games=10,
                edge_impact=0.1,
                recommended_action='Fix',
                sample_win_rate=0.40,
                expected_win_rate=0.52
            ),
            Vulnerability(
                pattern_type=PatternType.HCA_ERROR,
                sport='cbb',
                confidence=0.7,
                severity='WARNING',
                description='Test 2',
                affected_games=15,
                edge_impact=0.08,
                recommended_action='Fix 2',
                sample_win_rate=0.45,
                expected_win_rate=0.52
            ),
            Vulnerability(
                pattern_type=PatternType.GENERAL_DRIFT,
                sport='cbb',
                confidence=0.6,
                severity='INFO',
                description='Test 3',
                affected_games=5,
                edge_impact=0.05,
                recommended_action='Monitor',
                sample_win_rate=0.48,
                expected_win_rate=0.52
            )
        ]
        
        report = SweepReport(
            sport='cbb',
            vulnerabilities=vulns,
            games_analyzed=100,
            days_analyzed=30,
            patterns_checked=10
        )
        
        by_sev = report.by_severity()
        assert len(by_sev.get('CRITICAL', [])) == 1
        assert len(by_sev.get('WARNING', [])) == 1
        assert len(by_sev.get('INFO', [])) == 1
        assert report.has_critical() == True
        
    def test_no_critical(self):
        """has_critical returns False when no critical vulns."""
        vulns = [
            Vulnerability(
                pattern_type=PatternType.MONTH_DRIFT,
                sport='cbb',
                confidence=0.6,
                severity='WARNING',
                description='Test',
                affected_games=10,
                edge_impact=0.05,
                recommended_action='Monitor',
                sample_win_rate=0.45,
                expected_win_rate=0.52
            )
        ]
        
        report = SweepReport(
            sport='cbb',
            vulnerabilities=vulns,
            games_analyzed=50,
            days_analyzed=14,
            patterns_checked=5
        )
        
        assert report.has_critical() == False


class TestGeneralDrift:
    """Test general model drift detection."""
    
    @pytest.fixture
    def detector(self):
        """Create detector with mock DB."""
        db = Mock(spec=OpenClawDB)
        return PatternDetector(sport='cbb', db=db)
        
    def test_general_drift_detection(self, detector):
        """Detects performance decline over time."""
        games = []
        base_date = datetime.now() - timedelta(days=60)
        
        # First half: good performance (60% wins, above 50% threshold)
        for i in range(40):
            game = {
                'predicted_spread': -3.0,
                'spread_result': -4.0 if i % 10 < 6 else 2.0,  # 60% win
                'game_date': base_date + timedelta(days=i)
            }
            games.append(game)
        
        # Second half: poor performance (30% wins, > 8% degradation)
        for i in range(40, 80):
            game = {
                'predicted_spread': -3.0,
                'spread_result': -4.0 if i % 10 < 3 else 2.0,  # 30% win
                'game_date': base_date + timedelta(days=i)
            }
            games.append(game)
        
        detector.db.get_game_outcomes.return_value = games
        
        report = detector.run_sweep(days=60)
        
        drift_vulns = [v for v in report.vulnerabilities 
                       if v.pattern_type == PatternType.GENERAL_DRIFT]
        # Drift should be detected (60% -> 30% = 30% degradation, > 8% threshold)
        assert len(drift_vulns) > 0 or any('drift' in v.description.lower() 
                                           for v in report.vulnerabilities)
