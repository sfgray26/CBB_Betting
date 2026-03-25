"""
Performance Monitor Agent

Monitors model performance metrics and detects decay patterns:
- CLV decay detection (15% = CRITICAL, 8% = WARNING)
- Win rate tracking vs expected
- Prediction accuracy by sport/market
- Monte Carlo simulation quality

Runs every 2 hours during active season.
Reports via Discord without blocking main analysis loop.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import logging
import statistics

from .database import OpenClawDB

logger = logging.getLogger(__name__)


class DecaySeverity(Enum):
    """CLV decay severity levels."""
    NORMAL = "normal"      # < 5% decay
    ELEVATED = "elevated"  # 5-8% decay
    WARNING = "warning"    # 8-15% decay
    CRITICAL = "critical"  # > 15% decay


@dataclass
class DecayReport:
    """CLV decay analysis report."""
    sport: str
    current_decay_pct: float
    severity: DecaySeverity
    window_hours: int
    sample_size: int
    trend: str  # 'improving', 'stable', 'degrading'
    confidence: float  # 0.0-1.0
    details: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_discord_embed(self) -> Dict[str, Any]:
        """Convert to Discord embed format."""
        color_map = {
            DecaySeverity.NORMAL: 0x00FF00,      # Green
            DecaySeverity.ELEVATED: 0xFFFF00,    # Yellow
            DecaySeverity.WARNING: 0xFFA500,     # Orange
            DecaySeverity.CRITICAL: 0xFF0000,    # Red
        }
        
        return {
            'title': f'🔍 Performance Monitor: {self.sport.upper()}',
            'description': f'CLV Decay Analysis ({self.window_hours}h window)',
            'color': color_map.get(self.severity, 0x808080),
            'fields': [
                {
                    'name': 'Decay Rate',
                    'value': f'{self.current_decay_pct:.1f}%',
                    'inline': True
                },
                {
                    'name': 'Severity',
                    'value': self.severity.value.upper(),
                    'inline': True
                },
                {
                    'name': 'Sample Size',
                    'value': str(self.sample_size),
                    'inline': True
                },
                {
                    'name': 'Trend',
                    'value': self.trend.title(),
                    'inline': True
                },
                {
                    'name': 'Confidence',
                    'value': f'{self.confidence*100:.0f}%',
                    'inline': True
                }
            ],
            'timestamp': self.generated_at.isoformat()
        }


@dataclass
class WinRateReport:
    """Win rate vs expected analysis."""
    sport: str
    actual_win_rate: float
    expected_win_rate: float
    sample_size: int
    variance_from_expected: float
    is_significant: bool  # p < 0.05
    market_breakdown: Dict[str, Tuple[float, int]]  # market -> (win_rate, n)
    period_days: int = 14


class PerformanceMonitor:
    """
    Monitors model performance and detects decay patterns.
    
    This agent runs independently and reports findings without blocking
the main betting analysis loop.
    
    Usage:
        monitor = PerformanceMonitor(sport='cbb')
        
        # Check CLV decay
        decay = monitor.check_clv_decay(window_hours=48)
        if decay.severity in (DecaySeverity.WARNING, DecaySeverity.CRITICAL):
            send_alert(decay.to_discord_embed())
        
        # Check win rates
        win_rate = monitor.check_win_rate(days=14)
        if win_rate.is_significant and win_rate.variance_from_expected < -0.05:
            logger.warning(f"Underperforming by {abs(win_rate.variance_from_expected)*100:.1f}%")
    """
    
    # Decay thresholds per spec
    DECAY_ELEVATED = 0.05   # 5%
    DECAY_WARNING = 0.08    # 8%
    DECAY_CRITICAL = 0.15   # 15%
    
    @classmethod
    def _get_severity(cls, decay_pct: float) -> DecaySeverity:
        """Classify decay percentage into severity level."""
        if decay_pct >= cls.DECAY_CRITICAL:
            return DecaySeverity.CRITICAL
        elif decay_pct >= cls.DECAY_WARNING:
            return DecaySeverity.WARNING
        elif decay_pct >= cls.DECAY_ELEVATED:
            return DecaySeverity.ELEVATED
        else:
            return DecaySeverity.NORMAL
    
    # Minimum sample sizes for statistical validity
    MIN_CLV_SAMPLES = 20
    MIN_WIN_RATE_SAMPLES = 30
    
    def __init__(
        self,
        sport: str = 'cbb',
        db: Optional[OpenClawDB] = None
    ):
        self.sport = sport
        self.db = db or OpenClawDB()
        self._last_check: Optional[datetime] = None
        
    def check_clv_decay(
        self,
        window_hours: int = 48,
        min_samples: Optional[int] = None
    ) -> DecayReport:
        """
        Analyze CLV decay over specified window.
        
        CLV (Closing Line Value) decay measures how much edge is lost
        from prediction time to game start. High decay suggests:
        - Market efficiency improving
        - Model predictions leaking
        - Sharp money correcting lines
        
        Args:
            window_hours: Analysis window (default 48h for tournament pace)
            min_samples: Minimum games for valid analysis
            
        Returns:
            DecayReport with severity assessment and trend
        """
        min_samples = min_samples or self.MIN_CLV_SAMPLES
        
        # Fetch CLV data from database
        clv_data = self.db.get_clv_window(self.sport, window_hours)
        
        if len(clv_data) < min_samples:
            logger.warning(
                f"Insufficient CLV samples: {len(clv_data)} < {min_samples}"
            )
            return DecayReport(
                sport=self.sport,
                current_decay_pct=0.0,
                severity=DecaySeverity.NORMAL,
                window_hours=window_hours,
                sample_size=len(clv_data),
                trend='insufficient_data',
                confidence=0.0,
                details={'error': f'Only {len(clv_data)} samples available'}
            )
        
        # Calculate decay metrics
        decays = []
        for row in clv_data:
            predicted = row.get('edge_percent', 0)
            closing = row.get('closing_edge', 0)
            if predicted != 0:
                decay = (predicted - closing) / abs(predicted)
                decays.append(decay)
        
        if not decays:
            return DecayReport(
                sport=self.sport,
                current_decay_pct=0.0,
                severity=DecaySeverity.NORMAL,
                window_hours=window_hours,
                sample_size=0,
                trend='no_data',
                confidence=0.0
            )
        
        # Calculate statistics
        current_decay = statistics.mean(decays)
        decay_std = statistics.stdev(decays) if len(decays) > 1 else 0
        
        # Determine severity
        abs_decay = abs(current_decay)
        if abs_decay >= self.DECAY_CRITICAL:
            severity = DecaySeverity.CRITICAL
        elif abs_decay >= self.DECAY_WARNING:
            severity = DecaySeverity.WARNING
        elif abs_decay >= self.DECAY_ELEVATED:
            severity = DecaySeverity.ELEVATED
        else:
            severity = DecaySeverity.NORMAL
        
        # Calculate trend (compare first half to second half)
        mid = len(decays) // 2
        if mid > 0:
            first_half = statistics.mean(decays[:mid])
            second_half = statistics.mean(decays[mid:])
            
            if second_half > first_half * 1.1:
                trend = 'degrading'
            elif second_half < first_half * 0.9:
                trend = 'improving'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        # Confidence based on sample size
        confidence = min(1.0, len(decays) / (min_samples * 2))
        
        report = DecayReport(
            sport=self.sport,
            current_decay_pct=current_decay * 100,
            severity=severity,
            window_hours=window_hours,
            sample_size=len(decays),
            trend=trend,
            confidence=confidence,
            details={
                'decay_std': decay_std,
                'decay_median': statistics.median(decays),
                'min_decay': min(decays),
                'max_decay': max(decays)
            }
        )
        
        self._last_check = datetime.now()
        
        # Log findings
        if severity in (DecaySeverity.WARNING, DecaySeverity.CRITICAL):
            logger.warning(
                f"CLV {severity.value.upper()} detected: "
                f"{current_decay*100:.1f}% decay over {window_hours}h"
            )
        else:
            logger.info(
                f"CLV check: {current_decay*100:.1f}% decay ({severity.value})"
            )
        
        return report
    
    def check_win_rate(
        self,
        days: int = 14,
        min_edge: float = 0.02
    ) -> WinRateReport:
        """
        Compare actual win rate to model expectation.
        
        Identifies if model is over/under-performing its own predictions,
        which could indicate:
        - Calibration drift
        - Edge erosion
        - Variance (normal)
        
        Args:
            days: Analysis period
            min_edge: Minimum edge threshold for inclusion
            
        Returns:
            WinRateReport with variance analysis
        """
        # Fetch game outcomes with predictions
        games = self.db.get_game_outcomes(
            self.sport,
            days=days
        )
        
        if len(games) < self.MIN_WIN_RATE_SAMPLES:
            return WinRateReport(
                sport=self.sport,
                actual_win_rate=0.0,
                expected_win_rate=0.0,
                sample_size=len(games),
                variance_from_expected=0.0,
                is_significant=False,
                market_breakdown={},
                period_days=days
            )
        
        # Analyze by market
        wins = 0
        total_ev = 0.0
        market_results: Dict[str, List[Tuple[bool, float]]] = {
            'spread': [],
            'total': [],
            'moneyline': []
        }
        
        for game in games:
            # Skip if edge below threshold
            edge = game.get('edge_at_bet', 0)
            if abs(edge) < min_edge:
                continue
            
            # Determine if this was a win
            predicted_spread = game.get('predicted_spread')
            actual_spread = game.get('spread_result')
            
            if predicted_spread is not None and actual_spread is not None:
                win = (predicted_spread > 0 and actual_spread > 0) or \
                      (predicted_spread < 0 and actual_spread < 0)
                
                # Model's estimated win probability from edge
                # Rough approximation: edge * 25 + 50 (for small edges)
                est_win_prob = min(0.95, max(0.05, 0.5 + edge * 25))
                
                market_results['spread'].append((win, est_win_prob))
                if win:
                    wins += 1
                total_ev += est_win_prob
        
        total_bets = sum(len(v) for v in market_results.values())
        
        if total_bets == 0:
            return WinRateReport(
                sport=self.sport,
                actual_win_rate=0.0,
                expected_win_rate=0.0,
                sample_size=0,
                variance_from_expected=0.0,
                is_significant=False,
                market_breakdown={},
                period_days=days
            )
        
        actual_wr = wins / total_bets
        expected_wr = total_ev / total_bets
        variance = actual_wr - expected_wr
        
        # Statistical significance (binomial test approximation)
        if total_bets > 0:
            se = (expected_wr * (1 - expected_wr) / total_bets) ** 0.5
            z_score = abs(variance) / se if se > 0 else 0
            is_significant = z_score > 1.96  # p < 0.05
        else:
            is_significant = False
        
        # Market breakdown
        breakdown = {}
        for market, results in market_results.items():
            if results:
                m_wins = sum(1 for w, _ in results if w)
                breakdown[market] = (m_wins / len(results), len(results))
        
        report = WinRateReport(
            sport=self.sport,
            actual_win_rate=actual_wr,
            expected_win_rate=expected_wr,
            sample_size=total_bets,
            variance_from_expected=variance,
            is_significant=is_significant,
            market_breakdown=breakdown,
            period_days=days
        )
        
        if is_significant:
            direction = "overperforming" if variance > 0 else "underperforming"
            logger.warning(
                f"Win rate {direction} by {abs(variance)*100:.1f}% "
                f"({actual_wr*100:.1f}% actual vs {expected_wr*100:.1f}% expected)"
            )
        
        return report
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive health summary for Discord status.
        
        Returns:
            Dict with status, alerts, and recommendations
        """
        summary = {
            'sport': self.sport,
            'checked_at': datetime.now().isoformat(),
            'status': 'healthy',
            'alerts': [],
            'metrics': {}
        }
        
        # CLV decay check
        clv = self.check_clv_decay()
        summary['metrics']['clv_decay'] = {
            'value': f"{clv.current_decay_pct:.1f}%",
            'severity': clv.severity.value
        }
        
        if clv.severity == DecaySeverity.CRITICAL:
            summary['status'] = 'critical'
            summary['alerts'].append({
                'level': 'CRITICAL',
                'message': f"CLV decay at {clv.current_decay_pct:.1f}% — market may be adapting"
            })
        elif clv.severity == DecaySeverity.WARNING:
            summary['status'] = 'warning'
            summary['alerts'].append({
                'level': 'WARNING',
                'message': f"Elevated CLV decay: {clv.current_decay_pct:.1f}%"
            })
        
        # Win rate check
        wr = self.check_win_rate()
        if wr.is_significant and wr.variance_from_expected < -0.05:
            summary['alerts'].append({
                'level': 'WARNING',
                'message': f"Win rate underperforming by {abs(wr.variance_from_expected)*100:.1f}%"
            })
        
        summary['metrics']['win_rate'] = {
            'actual': f"{wr.actual_win_rate*100:.1f}%",
            'expected': f"{wr.expected_win_rate*100:.1f}%"
        }
        
        return summary
