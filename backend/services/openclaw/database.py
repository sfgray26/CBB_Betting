"""
OpenClaw Database Layer

Time-series storage for:
- model_performance_metrics (accuracy, CLV, prediction quality)
- vulnerability_reports (detected patterns, confidence, recommendations)
- learning_journal (experimental features and results)
- roadmap_state (prioritized improvements, status tracking)

All operations are read-only during Guardian freeze period.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single model performance measurement."""
    sport: str
    metric_type: str  # 'accuracy', 'clv', 'win_rate', 'mae'
    value: float
    sample_size: int
    window_days: int
    calculated_at: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VulnerabilityReport:
    """Detected pattern/vulnerability in model performance."""
    sport: str
    pattern_type: str  # 'clv_decay', 'conference_bias', 'hca', etc.
    confidence: float  # 0.0-1.0
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    description: str
    affected_games: int
    expected_impact: float  # projected edge reduction
    recommended_action: str
    detected_at: datetime
    resolved_at: Optional[datetime] = None


class OpenClawDB:
    """
    Database interface for OpenClaw monitoring data.
    
    During Guardian freeze (until Apr 7, 2026), this operates in READ-ONLY mode.
    After freeze lifts, write operations are enabled for self-improvement.
    """
    
    GUARDIAN_LIFT_DATE = datetime(2026, 4, 7)
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url
        self._read_only = datetime.now() < self.GUARDIAN_LIFT_DATE
        self._cache: Dict[str, Any] = {}
        
    def is_guardian_active(self) -> bool:
        """Check if Guardian freeze is still in effect."""
        return datetime.now() < self.GUARDIAN_LIFT_DATE
    
    # === READ OPERATIONS (Always Allowed) ===
    
    def get_recent_metrics(
        self,
        sport: str,
        metric_type: str,
        days: int = 7
    ) -> List[PerformanceMetric]:
        """
        Fetch performance metrics for analysis.
        
        Args:
            sport: 'cbb' or 'mlb'
            metric_type: 'accuracy', 'clv', 'win_rate', 'mae'
            days: lookback window
            
        Returns:
            List of performance metrics ordered by date
        """
        # Query from model_performance_metrics table
        query = """
            SELECT sport, metric_type, value, sample_size, window_days, 
                   calculated_at, metadata
            FROM model_performance_metrics
            WHERE sport = %s 
              AND metric_type = %s
              AND calculated_at >= NOW() - INTERVAL '%s days'
            ORDER BY calculated_at ASC
        """
        
        logger.debug(f"Fetching {metric_type} for {sport} over {days} days")
        
        # Placeholder - actual DB connection handled by caller
        return []
    
    def get_clv_window(
        self,
        sport: str,
        window_hours: int = 48
    ) -> List[Dict[str, Any]]:
        """
        Get CLV data for decay detection.
        
        Returns list of {timestamp, edge_percent, closing_edge} for trend analysis.
        """
        query = """
            SELECT 
                game_time as timestamp,
                predicted_edge as edge_percent,
                actual_clv as closing_edge
            FROM clv_snapshots
            WHERE sport = %s
              AND game_time >= NOW() - INTERVAL '%s hours'
            ORDER BY game_time ASC
        """
        
        logger.debug(f"Fetching CLV window for {sport}: {window_hours}h")
        return []
    
    def get_game_outcomes(
        self,
        sport: str,
        filters: Optional[Dict[str, Any]] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Fetch game outcomes for pattern analysis.
        
        Args:
            sport: 'cbb' or 'mlb'
            filters: Optional filters (conference, seed_range, day_of_week, etc.)
            days: lookback period
            
        Returns:
            List of game outcome records
        """
        query = """
            SELECT 
                g.game_id,
                g.home_team,
                g.away_team,
                g.home_score,
                g.away_score,
                g.spread_result,
                g.total_result,
                g.game_date,
                g.conference,
                g.home_seed,
                g.away_seed,
                g.is_neutral_site,
                p.predicted_spread,
                p.predicted_total,
                p.edge_at_bet
            FROM games g
            JOIN predictions p ON g.game_id = p.game_id
            WHERE g.sport = %s
              AND g.game_date >= CURRENT_DATE - INTERVAL '%s days'
        """
        
        if filters:
            if 'conference' in filters:
                query += f" AND g.conference = '{filters['conference']}'"
            if 'seed_range' in filters:
                min_s, max_s = filters['seed_range']
                query += f" AND g.home_seed BETWEEN {min_s} AND {max_s}"
        
        query += " ORDER BY g.game_date ASC"
        
        logger.debug(f"Fetching {sport} outcomes: {days} days, filters={filters}")
        return []
    
    def get_active_vulnerabilities(
        self,
        sport: Optional[str] = None,
        min_severity: str = 'WARNING'
    ) -> List[VulnerabilityReport]:
        """
        Get unresolved vulnerability reports.
        
        Args:
            sport: Filter by sport, or None for all
            min_severity: 'INFO', 'WARNING', or 'CRITICAL'
            
        Returns:
            List of active vulnerability reports
        """
        severity_order = {'INFO': 0, 'WARNING': 1, 'CRITICAL': 2}
        min_level = severity_order.get(min_severity, 1)
        
        query = """
            SELECT sport, pattern_type, confidence, severity, description,
                   affected_games, expected_impact, recommended_action,
                   detected_at, resolved_at
            FROM vulnerability_reports
            WHERE resolved_at IS NULL
        """
        
        if sport:
            query += f" AND sport = '{sport}'"
        
        logger.debug(f"Fetching vulnerabilities: sport={sport}, min_severity={min_severity}")
        return []
    
    # === WRITE OPERATIONS (Guardian-gated) ===
    
    def save_metric(self, metric: PerformanceMetric) -> bool:
        """Save performance metric. Blocked during Guardian freeze."""
        if self._read_only:
            logger.warning(f"Guardian active: Cannot save metric (read-only mode)")
            return False
        
        # Insert into model_performance_metrics
        logger.info(f"Saved metric: {metric.sport}/{metric.metric_type} = {metric.value:.3f}")
        return True
    
    def save_vulnerability(self, vuln: VulnerabilityReport) -> bool:
        """Save vulnerability report. Blocked during Guardian freeze."""
        if self._read_only:
            logger.warning(f"Guardian active: Cannot save vulnerability (read-only mode)")
            return False
        
        # Insert into vulnerability_reports
        logger.info(f"Saved vulnerability: {vuln.pattern_type} ({vuln.severity})")
        return True
    
    def resolve_vulnerability(
        self,
        pattern_type: str,
        sport: str
    ) -> bool:
        """Mark vulnerability as resolved. Blocked during Guardian freeze."""
        if self._read_only:
            logger.warning(f"Guardian active: Cannot resolve vulnerability (read-only mode)")
            return False
        
        # Update vulnerability_reports set resolved_at = NOW()
        logger.info(f"Resolved vulnerability: {pattern_type} ({sport})")
        return True
