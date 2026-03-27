"""
Dashboard Service — Phase B Foundation

Aggregates all dashboard data for the fantasy baseball home screen.

Usage:
    from backend.services.dashboard_service import DashboardService
    service = DashboardService()
    dashboard = await service.get_dashboard(user_id, team_key)
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from sqlalchemy.orm import Session

from backend.models import UserPreferences, SessionLocal, PlayerDailyMetric
from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer
from backend.services.waiver_edge_detector import WaiverEdgeDetector
from backend.services.data_reliability_engine import (
    get_reliability_engine,
    DataQualityTier,
    DataSource,
)
from backend.fantasy_baseball.yahoo_client import YahooFantasyClient, YahooAuthError

logger = logging.getLogger(__name__)


@dataclass
class LineupGap:
    """Identifies an unfilled lineup slot."""
    position: str
    severity: str  # "critical", "warning", "info"
    message: str
    suggested_add: Optional[str] = None


@dataclass
class StreakPlayer:
    """Player with hot/cold streak information."""
    player_id: str
    name: str
    team: str
    positions: List[str]
    trend: str  # "hot", "cold", "neutral"
    trend_score: float  # z-score
    last_7_avg: float
    last_14_avg: float
    last_30_avg: float
    reason: str


@dataclass
class WaiverTarget:
    """Prioritized waiver wire target."""
    player_id: str
    name: str
    team: str
    positions: List[str]
    percent_owned: float
    priority_score: float
    tier: str  # "must_add", "strong_add", "streamer"
    reason: str


@dataclass
class InjuryFlag:
    """Injury alert for a rostered player."""
    player_id: str
    name: str
    status: str  # "IL", "IL10", "IL60", "DTD", "OUT"
    injury_note: Optional[str]
    severity: str  # "critical", "warning", "info"
    estimated_return: Optional[str]
    action_needed: str


@dataclass
class MatchupPreview:
    """This week's matchup outlook."""
    week_number: int
    opponent_team_name: str
    opponent_record: str
    my_projected_categories: Dict[str, float]
    opponent_projected_categories: Dict[str, float]
    win_probability: float
    category_advantages: List[str]
    category_disadvantages: List[str]


@dataclass
class ProbablePitcherInfo:
    """Pitcher start information."""
    name: str
    team: str
    opponent: str
    game_date: str
    is_two_start: bool
    matchup_quality: str  # "favorable", "neutral", "unfavorable"
    stream_score: float
    reason: str


@dataclass
class DashboardData:
    """Complete dashboard payload."""
    timestamp: str
    user_id: str
    
    # B1.1: Lineup Gaps
    lineup_gaps: List[LineupGap]
    lineup_filled_count: int
    lineup_total_count: int
    
    # B1.2: Hot/Cold Streaks
    hot_streaks: List[StreakPlayer]
    cold_streaks: List[StreakPlayer]
    
    # B1.3: Waiver Targets
    waiver_targets: List[WaiverTarget]
    
    # B1.4: Injury Flags
    injury_flags: List[InjuryFlag]
    healthy_count: int
    injured_count: int
    
    # B1.5: Matchup Preview
    matchup_preview: Optional[MatchupPreview]
    
    # B1.6: Probable Pitchers
    probable_pitchers: List[ProbablePitcherInfo]
    two_start_pitchers: List[ProbablePitcherInfo]
    
    # Settings
    preferences: Dict[str, Any]


class DashboardService:
    """
    Aggregates dashboard data from multiple sources with reliability validation.
    
    This service coordinates:
    - Yahoo API for roster/matchup data (with fallback handling)
    - Statcast for streak analysis (with freshness validation)
    - WaiverEdgeDetector for FA recommendations
    - DailyLineupOptimizer for lineup gaps
    - DataReliabilityEngine for quality scoring
    """
    
    def __init__(self):
        self.lineup_optimizer = DailyLineupOptimizer()
        self.waiver_detector = WaiverEdgeDetector()
        self.reliability_engine = get_reliability_engine()
        self._yahoo_client: Optional[YahooFantasyClient] = None
    
    def _get_yahoo_client(self) -> Optional[YahooFantasyClient]:
        """Get Yahoo client with lazy initialization and error handling."""
        if self._yahoo_client is None:
            try:
                self._yahoo_client = YahooFantasyClient()
                self.reliability_engine.record_source_success(DataSource.YAHOO_API)
            except YahooAuthError as e:
                logger.warning(f"Yahoo auth not available: {e}")
                self.reliability_engine.record_source_failure(DataSource.YAHOO_API, str(e))
                return None
        return self._yahoo_client
    
    async def get_dashboard(
        self,
        user_id: str,
        team_key: Optional[str] = None,
        db: Optional[Session] = None
    ) -> DashboardData:
        """
        Build complete dashboard data for a user.
        
        Args:
            user_id: Unique user identifier
            team_key: Yahoo team key (optional, will try to detect)
            db: Database session (optional, will create if not provided)
        
        Returns:
            DashboardData with all panels populated
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            # Load user preferences
            prefs = self._get_or_create_preferences(db, user_id)
            
            # Gather all dashboard components in parallel where possible
            lineup_gaps, filled, total = await self._get_lineup_gaps(user_id, team_key)
            hot_streaks, cold_streaks = await self._get_streaks(user_id)
            waiver_targets = await self._get_waiver_targets(user_id, prefs)
            injury_flags, healthy, injured = await self._get_injury_flags(user_id)
            matchup = await self._get_matchup_preview(user_id, team_key)
            pitchers, two_starts = await self._get_probable_pitchers(user_id)
            
            return DashboardData(
                timestamp=datetime.utcnow().isoformat(),
                user_id=user_id,
                lineup_gaps=lineup_gaps,
                lineup_filled_count=filled,
                lineup_total_count=total,
                hot_streaks=hot_streaks[:5],  # Top 5 hot
                cold_streaks=cold_streaks[:5],  # Top 5 cold
                waiver_targets=waiver_targets[:5],  # Top 5 targets
                injury_flags=injury_flags,
                healthy_count=healthy,
                injured_count=injured,
                matchup_preview=matchup,
                probable_pitchers=pitchers,
                two_start_pitchers=two_starts,
                preferences=self._prefs_to_dict(prefs)
            )
        
        finally:
            if close_db:
                db.close()
    
    def _get_or_create_preferences(self, db: Session, user_id: str) -> UserPreferences:
        """Get existing preferences or create defaults."""
        prefs = db.query(UserPreferences).filter_by(user_id=user_id).first()
        if prefs is None:
            prefs = UserPreferences(user_id=user_id)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)
        return prefs
    
    def _prefs_to_dict(self, prefs: UserPreferences) -> Dict[str, Any]:
        """Convert preferences to dictionary."""
        return {
            "notifications": prefs.notifications,
            "dashboard_layout": prefs.dashboard_layout,
            "streak_settings": prefs.streak_settings,
            "waiver_preferences": prefs.waiver_preferences,
        }
    
    async def _get_lineup_gaps(
        self,
        user_id: str,
        team_key: Optional[str]
    ) -> tuple[List[LineupGap], int, int]:
        """
        B1.1: Detect unfilled lineup positions using Yahoo API.
        
        Returns:
            (gaps list, filled count, total slots)
        """
        client = self._get_yahoo_client()
        if not client:
            logger.warning("Yahoo client unavailable - cannot detect lineup gaps")
            return [], 0, 9
        
        try:
            # Get roster from Yahoo
            roster = client.get_roster(team_key) if team_key else []
            
            # Validate roster data
            validation = self.reliability_engine.validate_yahoo_roster(
                roster, timestamp=datetime.utcnow()
            )
            
            if not validation.is_valid:
                logger.warning(f"Roster validation failed: {validation.errors}")
            
            # Define required positions for Yahoo H2H
            required_positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "Util"]
            
            # Get active players (not on IL)
            active_players = [
                p for p in roster 
                if p.get("selected_position") not in ("IL", "IL10", "IL60")
            ]
            
            # Map players to positions they can fill
            gaps = []
            filled_count = 0
            
            for req_pos in required_positions:
                # Find a player eligible for this position
                eligible = [
                    p for p in active_players 
                    if req_pos in p.get("positions", []) or 
                    (req_pos == "Util" and any(pos in ["C", "1B", "2B", "3B", "SS", "OF"] for pos in p.get("positions", [])))
                ]
                
                if eligible:
                    filled_count += 1
                else:
                    # Gap detected
                    severity = "critical" if req_pos in ("C", "SS") else "warning"
                    gaps.append(LineupGap(
                        position=req_pos,
                        severity=severity,
                        message=f"No eligible player for {req_pos} slot",
                        suggested_add=None  # Would need waiver wire analysis
                    ))
            
            return gaps, filled_count, len(required_positions)
            
        except Exception as e:
            logger.error(f"Failed to get lineup gaps: {e}")
            self.reliability_engine.record_source_failure(DataSource.YAHOO_API, str(e))
            return [], 0, 9
    
    async def _get_streaks(
        self, 
        user_id: str, 
        db: Optional[Session] = None
    ) -> tuple[List[StreakPlayer], List[StreakPlayer]]]:
        """
        B1.2: Calculate hot/cold streaks from Statcast data.
        
        Uses 7/14/30 day rolling windows from player_daily_metrics.
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            # Get player's rostered players first
            client = self._get_yahoo_client()
            roster = []
            if client:
                try:
                    roster = client.get_roster()
                except Exception as e:
                    logger.warning(f"Could not fetch roster for streaks: {e}")
            
            if not roster:
                return [], []
            
            # Get player IDs from roster
            roster_names = {p.get("name", "").lower() for p in roster}
            
            # Query recent metrics from database
            recent_date = datetime.utcnow().date() - timedelta(days=1)
            metrics = db.query(PlayerDailyMetric).filter(
                PlayerDailyMetric.metric_date >= recent_date - timedelta(days=30),
                PlayerDailyMetric.sport == "mlb"
            ).all()
            
            hot = []
            cold = []
            
            for player in roster:
                player_name = player.get("name", "").lower()
                
                # Find metrics for this player
                player_metrics = [
                    m for m in metrics 
                    if m.player_name.lower() == player_name
                ]
                
                if not player_metrics:
                    continue
                
                # Get most recent metric
                latest = max(player_metrics, key=lambda m: m.metric_date)
                
                # Validate data quality
                validation = self.reliability_engine.validate_statcast_data(
                    latest.player_id,
                    {
                        "player_id": latest.player_id,
                        "player_name": latest.player_name,
                        "game_date": latest.metric_date.isoformat(),
                        "exit_velocity_avg": latest.bat_speed or 0,
                    },
                    timestamp=datetime.combine(latest.metric_date, datetime.min.time())
                )
                
                # Only use data if quality is acceptable
                if validation.quality_tier in (DataQualityTier.TIER_4_STALE, DataQualityTier.TIER_5_UNAVAILABLE):
                    logger.debug(f"Skipping stale data for {latest.player_name}")
                    continue
                
                # Calculate trend
                z_score = latest.z_score_recent or 0
                
                # Get rolling averages
                rolling = latest.rolling_window or {}
                last_7 = rolling.get("7d", {}).get("avg", 0)
                last_14 = rolling.get("14d", {}).get("avg", 0)
                last_30 = rolling.get("30d", {}).get("avg", 0)
                
                streak_player = StreakPlayer(
                    player_id=latest.player_id,
                    name=latest.player_name,
                    team=latest.player_name,  # Would need to get from roster
                    positions=player.get("positions", []),
                    trend="hot" if z_score > 0.5 else "cold" if z_score < -0.5 else "neutral",
                    trend_score=z_score,
                    last_7_avg=last_7,
                    last_14_avg=last_14,
                    last_30_avg=last_30,
                    reason=f"z-score: {z_score:.2f} (data quality: {validation.quality_tier.value})"
                )
                
                if z_score > 0.5:
                    hot.append(streak_player)
                elif z_score < -0.5:
                    cold.append(streak_player)
            
            # Sort by trend score
            hot.sort(key=lambda x: x.trend_score, reverse=True)
            cold.sort(key=lambda x: x.trend_score)
            
            return hot, cold
            
        finally:
            if close_db:
                db.close()
    
    async def _get_waiver_targets(
        self,
        user_id: str,
        prefs: UserPreferences
    ) -> List[WaiverTarget]:
        """
        B1.3: Get prioritized waiver wire recommendations.
        """
        targets = []
        
        # TODO: Integrate with WaiverEdgeDetector
        # Apply user's waiver_preferences filters
        
        return targets
    
    async def _get_injury_flags(self, user_id: str) -> tuple[List[InjuryFlag], int, int]:
        """
        B1.4: Detect injured players on roster using Yahoo API.
        
        Returns:
            (injury flags, healthy count, injured count)
        """
        client = self._get_yahoo_client()
        if not client:
            logger.warning("Yahoo client unavailable - cannot detect injuries")
            return [], 0, 0
        
        try:
            roster = client.get_roster()
            
            # Validate roster data
            validation = self.reliability_engine.validate_yahoo_roster(roster)
            if not validation.is_valid:
                logger.warning(f"Roster validation failed for injury check: {validation.errors}")
            
            flags = []
            healthy = 0
            injured = 0
            
            # Status mappings
            injury_statuses = {"IL", "IL10", "IL60", "DTD", "OUT", "NA"}
            
            for player in roster:
                status = player.get("status", "")
                selected_pos = player.get("selected_position", "")
                
                # Check if player is injured
                is_injured = status in injury_statuses or selected_pos in ("IL", "IL10", "IL60")
                
                if is_injured:
                    injured += 1
                    
                    # Determine severity
                    if status in ("IL", "IL60") or selected_pos in ("IL", "IL60"):
                        severity = "critical"
                        action = "Move to IL slot immediately"
                    elif status == "IL10" or selected_pos == "IL10":
                        severity = "warning"
                        action = "Consider moving to IL slot"
                    elif status == "DTD":
                        severity = "warning"
                        action = "Check lineup status before lock"
                    else:
                        severity = "info"
                        action = "Monitor status"
                    
                    flags.append(InjuryFlag(
                        player_id=player.get("player_id", ""),
                        name=player.get("name", "Unknown"),
                        status=status or selected_pos or "OUT",
                        injury_note=player.get("injury_note"),
                        severity=severity,
                        estimated_return=None,  # Would need additional data source
                        action_needed=action
                    ))
                else:
                    healthy += 1
            
            return flags, healthy, injured
            
        except Exception as e:
            logger.error(f"Failed to get injury flags: {e}")
            self.reliability_engine.record_source_failure(DataSource.YAHOO_API, str(e))
            return [], 0, 0
    
    async def _get_matchup_preview(
        self,
        user_id: str,
        team_key: Optional[str]
    ) -> Optional[MatchupPreview]:
        """
        B1.5: Get this week's matchup analysis.
        """
        # TODO: Integrate with Yahoo scoreboard API
        # Use MCMC simulator for projections
        
        return None
    
    async def _get_probable_pitchers(
        self,
        user_id: str
    ) -> tuple[List[ProbablePitcherInfo], List[ProbablePitcherInfo]]:
        """
        B1.6: Get probable pitchers and two-start SPs.
        """
        pitchers = []
        two_starts = []
        
        # TODO: Use DailyLineupOptimizer.flag_pitcher_starts()
        # Cross-reference with free agents for streamer suggestions
        
        return pitchers, two_starts
    
    # ---------------------------------------------------------------------
    # User Preferences CRUD
    # ---------------------------------------------------------------------
    
    def get_preferences(self, user_id: str, db: Optional[Session] = None) -> Dict[str, Any]:
        """Get user preferences."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            prefs = self._get_or_create_preferences(db, user_id)
            return self._prefs_to_dict(prefs)
        finally:
            if close_db:
                db.close()
    
    def update_preferences(
        self,
        user_id: str,
        updates: Dict[str, Any],
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            updates: Dict with keys matching preference fields
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            prefs = self._get_or_create_preferences(db, user_id)
            
            # Update allowed fields
            if "notifications" in updates:
                prefs.notifications = {**prefs.notifications, **updates["notifications"]}
            if "dashboard_layout" in updates:
                prefs.dashboard_layout = {**prefs.dashboard_layout, **updates["dashboard_layout"]}
            if "projection_weights" in updates:
                prefs.projection_weights = {**prefs.projection_weights, **updates["projection_weights"]}
            if "streak_settings" in updates:
                prefs.streak_settings = {**prefs.streak_settings, **updates["streak_settings"]}
            if "waiver_preferences" in updates:
                prefs.waiver_preferences = {**prefs.waiver_preferences, **updates["waiver_preferences"]}
            
            db.commit()
            db.refresh(prefs)
            
            return self._prefs_to_dict(prefs)
        finally:
            if close_db:
                db.close()


# Singleton instance
_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service() -> DashboardService:
    """Get singleton dashboard service."""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service
