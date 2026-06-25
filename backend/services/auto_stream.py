"""
Auto-Stream Service — Automated roster actions for streaming recommendations.

Background scheduler runs daily at 6 AM ET to evaluate streaming recommendations
and execute ADD or ADD_DROP actions based on user configuration.

Usage:
    from backend.services.auto_stream import AutoStreamService, get_auto_stream_service
    service = get_auto_stream_service()
    config = await service.get_config(user_id)
    result = await service.execute_scheduled_run(user_id, db)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from backend.models import SessionLocal, UserPreferences
from backend.services.yahoo_actions import YahooActionsService, ActionType
from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client

logger = logging.getLogger(__name__)

# Advisory lock ID for Auto-Stream scheduled job (100_042)
AUTO_STREAM_LOCK_ID = 100_042

# Recommendation tiers (from streaming-recommendations)
RecommendationTier = Literal["EXCELLENT", "GOOD", "AVERAGE", "AVOID"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


class AutoStreamStatus(str, Enum):
    """Status of Auto-Stream feature."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PAUSED = "paused"


@dataclass
class AutoStreamConfig:
    """User configuration for Auto-Stream."""
    enabled: bool = False
    drop_priority: List[str] = field(default_factory=list)
    min_confidence: ConfidenceLevel = "HIGH"
    min_recommendation: RecommendationTier = "EXCELLENT"
    max_adds_per_week: int = 2
    user_id: str = ""

    # Computed fields (not stored)
    executed_this_week: int = 0
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None

    def tier_meets_minimum(self, tier: RecommendationTier) -> bool:
        """Check if a recommendation tier meets the minimum threshold."""
        tier_order = ["EXCELLENT", "GOOD", "AVERAGE", "AVOID"]
        min_idx = tier_order.index(self.min_recommendation)
        tier_idx = tier_order.index(tier)
        return tier_idx <= min_idx

    def confidence_meets_minimum(self, confidence: ConfidenceLevel) -> bool:
        """Check if confidence meets the minimum threshold."""
        conf_order = ["HIGH", "MEDIUM", "LOW"]
        min_idx = conf_order.index(self.min_confidence)
        conf_idx = conf_order.index(confidence)
        return conf_idx <= min_idx


@dataclass
class AutoStreamAction:
    """Record of an Auto-Stream action attempt."""
    timestamp: str
    player_name: str
    player_id: str
    action: ActionType
    success: bool
    transaction_id: Optional[str] = None
    error: Optional[str] = None
    reason_skipped: Optional[str] = None


@dataclass
class AutoStreamResult:
    """Result of an Auto-Stream execution run."""
    executed: List[AutoStreamAction] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    run_timestamp: str = ""
    next_run_at: str = ""


@dataclass
class AutoStreamStatusResponse:
    """Response for GET /api/fantasy/auto-stream/status."""
    enabled: bool
    config: AutoStreamConfig
    last_run_at: Optional[str]
    next_run_at: Optional[str]
    executed_this_week: int
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    recent_log: List[Dict[str, Any]] = field(default_factory=list)


class AutoStreamService:
    """
    Service for automated streaming recommendations execution.

    Features:
    - Configuration storage in UserPreferences
    - Daily 6 AM ET scheduled execution
    - Validation of roster space and drop candidates
    - Integration with YahooActionsService for roster mutations
    - Action logging and status reporting
    """

    def __init__(self) -> None:
        self._run_log: List[AutoStreamAction] = []
        self._weekly_count: int = 0
        self._week_start: datetime = self._get_week_start()

    @staticmethod
    def _get_week_start() -> datetime:
        """Get Monday 00:00 ET of current scoring week."""
        now = datetime.now(ZoneInfo("America/New_York"))
        # Monday is weekday 0
        days_since_monday = now.weekday()
        week_start = (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return week_start

    def _reset_weekly_count_if_needed(self) -> None:
        """Reset weekly counter if we've crossed into a new scoring week."""
        current_week_start = self._get_week_start()
        if current_week_start > self._week_start:
            self._weekly_count = 0
            self._week_start = current_week_start

    def _get_next_run_time(self) -> str:
        """Calculate next scheduled run time (6 AM ET tomorrow)."""
        now = datetime.now(ZoneInfo("America/New_York"))
        tomorrow = (now + timedelta(days=1)).replace(
            hour=6, minute=0, second=0, microsecond=0
        )
        return tomorrow.isoformat()

    async def get_config(self, user_id: str, db: Optional[Session] = None) -> AutoStreamConfig:
        """
        Retrieve Auto-Stream configuration for a user.

        Args:
            user_id: User identifier
            db: Optional database session

        Returns:
            AutoStreamConfig with user's settings or defaults
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            prefs = db.query(UserPreferences).filter_by(user_id=user_id).first()

            if prefs and prefs.auto_stream_config:
                config_dict = prefs.auto_stream_config
                return AutoStreamConfig(
                    enabled=config_dict.get("enabled", False),
                    drop_priority=config_dict.get("drop_priority", []),
                    min_confidence=config_dict.get("min_confidence", "HIGH"),
                    min_recommendation=config_dict.get("min_recommendation", "EXCELLENT"),
                    max_adds_per_week=config_dict.get("max_adds_per_week", 2),
                    user_id=user_id,
                    executed_this_week=self._weekly_count,
                    last_run_at=self._get_last_run_timestamp(),
                    next_run_at=self._get_next_run_time(),
                )
            else:
                # Return default disabled config
                return AutoStreamConfig(
                    enabled=False,
                    drop_priority=[],
                    min_confidence="HIGH",
                    min_recommendation="EXCELLENT",
                    max_adds_per_week=2,
                    user_id=user_id,
                    executed_this_week=self._weekly_count,
                    last_run_at=self._get_last_run_timestamp(),
                    next_run_at=self._get_next_run_time(),
                )
        finally:
            if close_db:
                db.close()

    async def update_config(
        self,
        user_id: str,
        enabled: bool,
        drop_priority: List[str],
        min_confidence: ConfidenceLevel,
        min_recommendation: RecommendationTier,
        max_adds_per_week: int,
        db: Session,
    ) -> AutoStreamConfig:
        """
        Update Auto-Stream configuration for a user.

        Args:
            user_id: User identifier
            enabled: Whether Auto-Stream is enabled
            drop_priority: Ordered list of player IDs to drop first
            min_confidence: Minimum confidence level (HIGH/MEDIUM/LOW)
            min_recommendation: Minimum recommendation tier (EXCELLENT/GOOD/AVERAGE)
            max_adds_per_week: Maximum adds per scoring week
            db: Database session

        Returns:
            Updated AutoStreamConfig
        """
        # Validate max_adds_per_week
        if max_adds_per_week < 1 or max_adds_per_week > 10:
            raise ValueError("max_adds_per_week must be between 1 and 10")

        # Validate confidence level
        if min_confidence not in ("HIGH", "MEDIUM", "LOW"):
            raise ValueError("min_confidence must be HIGH, MEDIUM, or LOW")

        # Validate recommendation tier
        if min_recommendation not in ("EXCELLENT", "GOOD", "AVERAGE", "AVOID"):
            raise ValueError("min_recommendation must be EXCELLENT, GOOD, AVERAGE, or AVOID")

        # Validate drop_priority players are on roster
        if drop_priority:
            client = get_yahoo_client()
            try:
                roster = client.get_roster()
                roster_keys = {p.get("player_key") for p in roster}
                invalid_keys = set(drop_priority) - roster_keys
                if invalid_keys:
                    raise ValueError(f"drop_priority contains players not on roster: {invalid_keys}")
            except Exception as e:
                logger.warning(f"Could not validate drop_priority against roster: {e}")
                # Continue anyway - validation will fail at execution time

        # Get or create UserPreferences
        prefs = db.query(UserPreferences).filter_by(user_id=user_id).first()
        if not prefs:
            prefs = UserPreferences(user_id=user_id)
            db.add(prefs)

        # Update auto_stream_config
        prefs.auto_stream_config = {
            "enabled": enabled,
            "drop_priority": drop_priority,
            "min_confidence": min_confidence,
            "min_recommendation": min_recommendation,
            "max_adds_per_week": max_adds_per_week,
            "updated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        }
        prefs.updated_at = datetime.now(ZoneInfo("America/New_York"))

        db.commit()
        db.refresh(prefs)

        logger.info(f"Auto-Stream config updated for user {user_id}: enabled={enabled}")

        return await self.get_config(user_id, db)

    def _get_last_run_timestamp(self) -> Optional[str]:
        """Get timestamp of most recent action in log."""
        if self._run_log:
            return self._run_log[-1].timestamp
        return None

    async def get_status(self, user_id: str, db: Optional[Session] = None) -> AutoStreamStatusResponse:
        """
        Get current Auto-Stream status for a user.

        Args:
            user_id: User identifier
            db: Optional database session

        Returns:
            AutoStreamStatusResponse with current state
        """
        config = await self.get_config(user_id, db)

        return AutoStreamStatusResponse(
            enabled=config.enabled,
            config=config,
            last_run_at=config.last_run_at,
            next_run_at=config.next_run_at,
            executed_this_week=config.executed_this_week,
            recent_log=[
                {
                    "timestamp": a.timestamp,
                    "player_name": a.player_name,
                    "action": a.action,
                    "success": a.success,
                    "error": a.error,
                    "reason_skipped": a.reason_skipped,
                }
                for a in self._run_log[-10:]
            ],
        )

    async def execute_scheduled_run(
        self,
        user_id: str,
        target_date: str,
        db: Session,
    ) -> AutoStreamResult:
        """
        Execute Auto-Stream scheduled run for a user.

        Args:
            user_id: User identifier
            target_date: Target date for streaming recommendations (YYYY-MM-DD)
            db: Database session

        Returns:
            AutoStreamResult with executed, skipped, and errors lists
        """
        self._reset_weekly_count_if_needed()

        config = await self.get_config(user_id, db)

        if not config.enabled:
            logger.info(f"Auto-Stream disabled for user {user_id}, skipping execution")
            return AutoStreamResult(
                run_timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),
                next_run_at=self._get_next_run_time(),
            )

        result = AutoStreamResult(
            run_timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),
            next_run_at=self._get_next_run_time(),
        )

        # Check weekly limit
        if self._weekly_count >= config.max_adds_per_week:
            result.skipped.append({
                "reason": "weekly_limit_reached",
                "message": f"Already executed {self._weekly_count}/{config.max_adds_per_week} adds this week",
            })
            return result

        logger.info(f"Auto-Stream execution for user {user_id} on {target_date}")

        # Fetch streaming recommendations from ProbablePitcherSnapshot
        from backend.models import ProbablePitcherSnapshot, PlayerIDMapping
        from sqlalchemy import func

        # Parse target date and set 7-day window
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            result.errors.append({
                "reason": "invalid_date_format",
                "message": f"Invalid date format: {target_date}. Use YYYY-MM-DD",
            })
            return result

        end_dt = target_dt + timedelta(days=7)

        # Query probable pitchers for the window
        query = db.query(
            ProbablePitcherSnapshot.bdl_player_id,
            ProbablePitcherSnapshot.pitcher_name,
            ProbablePitcherSnapshot.team,
            ProbablePitcherSnapshot.game_date,
            ProbablePitcherSnapshot.quality_score,
            ProbablePitcherSnapshot.is_confirmed,
        ).filter(
            ProbablePitcherSnapshot.game_date >= target_dt,
            ProbablePitcherSnapshot.game_date <= end_dt,
            ProbablePitcherSnapshot.bdl_player_id.isnot(None),
            ProbablePitcherSnapshot.quality_score.isnot(None),
        ).order_by(ProbablePitcherSnapshot.game_date, ProbablePitcherSnapshot.team)

        rows = query.all()

        # Group by pitcher to find 2-starters
        pitcher_starts: dict[int, list] = {}
        for r in rows:
            pid = r.bdl_player_id
            if pid not in pitcher_starts:
                pitcher_starts[pid] = []
            pitcher_starts[pid].append({
                "pitcher_name": r.pitcher_name,
                "quality_score": float(r.quality_score or 0),
                "is_confirmed": r.is_confirmed,
            })

        # Calculate recommendation tier and confidence for each 2-starter
        recommendations = []
        for pid, starts in pitcher_starts.items():
            if len(starts) >= 2:
                avg_quality = sum(s["quality_score"] for s in starts[:2]) / 2
                confirmed_count = sum(1 for s in starts[:2] if s.get("is_confirmed"))

                # Determine confidence
                if confirmed_count == 2:
                    confidence = "HIGH"
                elif confirmed_count == 1:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"

                # Determine recommendation tier
                if avg_quality >= 1.0 and confidence == "HIGH":
                    recommendation = "EXCELLENT"
                elif avg_quality >= 0.3 and confidence in ("HIGH", "MEDIUM"):
                    recommendation = "GOOD"
                elif confidence == "LOW":
                    recommendation = "AVOID"
                elif avg_quality >= -0.3:
                    recommendation = "AVERAGE"
                else:
                    recommendation = "AVOID"

                recommendations.append({
                    "bdl_player_id": pid,
                    "pitcher_name": starts[0]["pitcher_name"],
                    "avg_quality": avg_quality,
                    "confidence": confidence,
                    "recommendation": recommendation,
                })

        # Sort by quality descending
        recommendations.sort(key=lambda x: x["avg_quality"], reverse=True)

        # Filter by user thresholds
        qualifying = []
        for rec in recommendations:
            if config.tier_meets_minimum(rec["recommendation"]) and config.confidence_meets_minimum(rec["confidence"]):
                qualifying.append(rec)

        if not qualifying:
            result.skipped.append({
                "reason": "no_qualifying_pitchers",
                "message": "No pitchers meet minimum recommendation and confidence thresholds",
            })
            return result

        # Get user's roster to check for space and existing players
        yahoo_client = get_yahoo_client()
        roster = []
        try:
            roster = yahoo_client.get_roster()
        except Exception as e:
            result.errors.append({
                "reason": "roster_fetch_failed",
                "message": f"Failed to fetch roster: {e}",
            })
            return result

        roster_player_keys = {p.get("player_key") for p in roster if p.get("player_key")}
        roster_size = len(roster)

        # Get Yahoo player keys for qualifying pitchers
        yahoo_keys_map = {}
        for rec in qualifying:
            mapping = db.query(PlayerIDMapping.yahoo_key).filter_by(bdl_id=rec["bdl_player_id"]).first()
            if mapping and mapping.yahoo_key:
                yahoo_keys_map[rec["bdl_player_id"]] = mapping.yahoo_key
            else:
                result.skipped.append({
                    "reason": "no_yahoo_mapping",
                    "message": f"No Yahoo mapping found for BDL player {rec['bdl_player_id']} ({rec['pitcher_name']})",
                })

        # Check roster space (27 = full for standard Yahoo leagues)
        MAX_ROSTER_SIZE = 27
        has_roster_space = roster_size < MAX_ROSTER_SIZE

        # Process each qualifying pitcher
        for rec in qualifying:
            bdl_id = rec["bdl_player_id"]
            yahoo_key = yahoo_keys_map.get(bdl_id)

            if not yahoo_key:
                continue

            # Skip if already on roster
            if yahoo_key in roster_player_keys:
                result.skipped.append({
                    "reason": "already_on_roster",
                    "message": f"{rec['pitcher_name']} is already on roster",
                })
                continue

            # Check weekly limit
            if self._weekly_count >= config.max_adds_per_week:
                result.skipped.append({
                    "reason": "weekly_limit_reached",
                    "message": f"Reached weekly limit of {config.max_adds_per_week} adds",
                })
                break

            # Execute action
            try:
                actions_service = YahooActionsService()
                action_type: ActionType = "ADD"

                drop_player_id = None
                position = "P"  # Default pitcher slot

                if not has_roster_space:
                    # Need to drop someone - use drop_priority
                    if config.drop_priority:
                        # Find first drop_priority player on roster
                        for drop_key in config.drop_priority:
                            if drop_key in roster_player_keys:
                                drop_player_id = drop_key
                                action_type = "ADD_DROP"
                                break

                    if not drop_player_id:
                        result.skipped.append({
                            "reason": "no_roster_space",
                            "message": f"Roster full and no drop candidates for {rec['pitcher_name']}",
                        })
                        continue

                # Execute the action
                action_result = await actions_service.execute_action(
                    action=action_type,
                    add_player_id=yahoo_key,
                    drop_player_id=drop_player_id,
                    position=position,
                )

                if action_result.success:
                    self._weekly_count += 1
                    action_record = AutoStreamAction(
                        timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),
                        player_name=rec["pitcher_name"],
                        player_id=yahoo_key,
                        action=action_type,
                        success=True,
                        transaction_id=action_result.transaction_id,
                    )
                    self._run_log.append(action_record)
                    result.executed.append(action_record)
                    logger.info(f"Auto-Stream: {action_type} {rec['pitcher_name']} (txn: {action_result.transaction_id})")
                else:
                    action_record = AutoStreamAction(
                        timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),
                        player_name=rec["pitcher_name"],
                        player_id=yahoo_key,
                        action=action_type,
                        success=False,
                        error=action_result.error or "Unknown error",
                    )
                    result.errors.append({
                        "player_name": rec["pitcher_name"],
                        "action": action_type,
                        "error": action_result.error or "Unknown error",
                    })
                    logger.warning(f"Auto-Stream: Failed {action_type} {rec['pitcher_name']}: {action_result.error}")

            except Exception as e:
                result.errors.append({
                    "player_name": rec["pitcher_name"],
                    "action": action_type,
                    "error": str(e),
                })
                logger.error(f"Auto-Stream: Exception executing action for {rec['pitcher_name']}: {e}")

        return result


# Singleton instance
_auto_stream_service: Optional[AutoStreamService] = None


def get_auto_stream_service() -> AutoStreamService:
    """Get singleton AutoStreamService instance."""
    global _auto_stream_service
    if _auto_stream_service is None:
        _auto_stream_service = AutoStreamService()
    return _auto_stream_service
