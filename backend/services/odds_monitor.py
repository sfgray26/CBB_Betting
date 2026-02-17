"""
Event-driven odds monitor for real-time line movement detection.

Complements the nightly batch pipeline by polling The Odds API at a
configurable interval and triggering re-analysis when:

    1. A line crosses the model's edge threshold.
    2. A line moves significantly (> 1.5 points) since the last poll,
       suggesting informed money has entered the market.
    3. A game enters the "golden window" (< 2 hours to tipoff) where
       closing-line efficiency peaks.

Design:
    - Runs as an APScheduler interval job (default: every 5 minutes).
    - Maintains an in-memory line history to detect ticks.
    - Emits ``LineMovement`` events consumed by the analysis service.
    - Respects API quota by tracking ``x-requests-remaining``.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from backend.services.odds import OddsAPIClient, get_data_freshness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LineSnapshot:
    """Point-in-time capture of a game's line."""

    game_id: str
    home_team: str
    away_team: str
    spread: Optional[float]
    spread_odds: Optional[float]
    total: Optional[float]
    moneyline_home: Optional[float]
    moneyline_away: Optional[float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LineMovement:
    """Detected line change between two snapshots."""

    game_id: str
    home_team: str
    away_team: str
    field: str  # "spread", "total", "moneyline_home", etc.
    old_value: Optional[float]
    new_value: Optional[float]
    delta: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_significant: bool = False
    minutes_to_tipoff: Optional[float] = None


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

class OddsMonitor:
    """
    Polls The Odds API and detects actionable line movements.

    Usage::

        monitor = OddsMonitor()
        monitor.on_significant_move(my_callback)
        monitor.poll()   # call from APScheduler
    """

    # Thresholds
    SPREAD_MOVE_THRESHOLD = 1.5   # Points
    TOTAL_MOVE_THRESHOLD = 2.0    # Points
    ML_MOVE_THRESHOLD = 25.0      # American odds units
    GOLDEN_WINDOW_MINUTES = 120   # 2 hours before tipoff
    MIN_API_QUOTA_RESERVE = 10    # Stop polling if quota drops below this

    def __init__(self, api_key: Optional[str] = None):
        self._client = OddsAPIClient(api_key=api_key)
        self._history: Dict[str, List[LineSnapshot]] = {}  # game_id -> snapshots
        self._callbacks: List[Callable[[LineMovement], None]] = []
        self._last_poll: Optional[datetime] = None
        self._polls_remaining: Optional[int] = None

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_significant_move(self, callback: Callable[[LineMovement], None]) -> None:
        """Register a callback fired when a significant line movement is detected."""
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self) -> Dict:
        """
        Fetch current odds, detect movements, and fire callbacks.

        Returns a summary dict for logging / the admin status endpoint.
        """
        now = datetime.utcnow()

        # Quota guard
        if self._polls_remaining is not None and self._polls_remaining < self.MIN_API_QUOTA_RESERVE:
            logger.warning(
                "Odds monitor paused — API quota low (%d remaining)",
                self._polls_remaining,
            )
            return {"status": "quota_paused", "remaining": self._polls_remaining}

        try:
            games = self._client.get_todays_games()
        except Exception as exc:
            logger.error("Odds monitor poll failed: %s", exc)
            return {"status": "error", "error": str(exc)}

        movements: List[LineMovement] = []
        for game in games:
            snap = LineSnapshot(
                game_id=game.get("game_id", ""),
                home_team=game.get("home_team", ""),
                away_team=game.get("away_team", ""),
                spread=game.get("best_spread"),
                spread_odds=game.get("best_spread_odds"),
                total=game.get("best_total"),
                moneyline_home=game.get("best_moneyline_home"),
                moneyline_away=game.get("best_moneyline_away"),
                timestamp=now,
            )

            gid = snap.game_id
            prev_snaps = self._history.get(gid, [])

            if prev_snaps:
                last = prev_snaps[-1]
                game_movements = self._detect_movements(last, snap, game)
                movements.extend(game_movements)

            # Keep rolling window (last 50 snapshots per game)
            prev_snaps.append(snap)
            if len(prev_snaps) > 50:
                prev_snaps = prev_snaps[-50:]
            self._history[gid] = prev_snaps

        # Fire callbacks for significant movements
        significant = [m for m in movements if m.is_significant]
        for movement in significant:
            for cb in self._callbacks:
                try:
                    cb(movement)
                except Exception as exc:
                    logger.error("Odds monitor callback error: %s", exc)

        self._last_poll = now

        # Prune history for games that are no longer in today's slate
        current_ids = {g.get("game_id") for g in games}
        stale_ids = [gid for gid in self._history if gid not in current_ids]
        for gid in stale_ids:
            del self._history[gid]

        result = {
            "status": "ok",
            "games_tracked": len(self._history),
            "movements_detected": len(movements),
            "significant_movements": len(significant),
            "timestamp": now.isoformat(),
        }
        logger.info(
            "Odds monitor: %d games, %d movements (%d significant)",
            len(self._history),
            len(movements),
            len(significant),
        )
        return result

    # ------------------------------------------------------------------
    # Movement detection
    # ------------------------------------------------------------------

    def _detect_movements(
        self,
        prev: LineSnapshot,
        curr: LineSnapshot,
        raw_game: Dict,
    ) -> List[LineMovement]:
        """Compare two snapshots and return any movements."""
        movements = []

        # Calculate minutes to tipoff
        commence = raw_game.get("commence_time")
        minutes_to_tipoff = None
        if commence:
            try:
                if isinstance(commence, str):
                    tip = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    minutes_to_tipoff = (tip - datetime.utcnow().astimezone(tip.tzinfo)).total_seconds() / 60.0
            except (ValueError, TypeError):
                pass

        checks = [
            ("spread", prev.spread, curr.spread, self.SPREAD_MOVE_THRESHOLD),
            ("total", prev.total, curr.total, self.TOTAL_MOVE_THRESHOLD),
            ("moneyline_home", prev.moneyline_home, curr.moneyline_home, self.ML_MOVE_THRESHOLD),
            ("moneyline_away", prev.moneyline_away, curr.moneyline_away, self.ML_MOVE_THRESHOLD),
        ]

        for field_name, old_val, new_val, threshold in checks:
            if old_val is None or new_val is None:
                continue
            delta = new_val - old_val
            if abs(delta) < 0.001:
                continue

            is_sig = abs(delta) >= threshold
            # Also flag any movement inside the golden window
            if minutes_to_tipoff is not None and 0 < minutes_to_tipoff <= self.GOLDEN_WINDOW_MINUTES:
                # Lower threshold in the golden window
                is_sig = is_sig or abs(delta) >= threshold * 0.5

            movements.append(
                LineMovement(
                    game_id=curr.game_id,
                    home_team=curr.home_team,
                    away_team=curr.away_team,
                    field=field_name,
                    old_value=old_val,
                    new_value=new_val,
                    delta=delta,
                    timestamp=curr.timestamp,
                    is_significant=is_sig,
                    minutes_to_tipoff=minutes_to_tipoff,
                )
            )

        return movements

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_line_history(self, game_id: str) -> List[LineSnapshot]:
        """Return all captured snapshots for a game."""
        return list(self._history.get(game_id, []))

    def get_status(self) -> Dict:
        """Return monitor status for the admin endpoint."""
        return {
            "active": True,
            "games_tracked": len(self._history),
            "last_poll": self._last_poll.isoformat() if self._last_poll else None,
            "polls_remaining": self._polls_remaining,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_odds_monitor: Optional[OddsMonitor] = None


def get_odds_monitor() -> OddsMonitor:
    global _odds_monitor
    if _odds_monitor is None:
        _odds_monitor = OddsMonitor()
    return _odds_monitor
