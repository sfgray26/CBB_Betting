"""
Portfolio-level Kelly sizing and concurrent exposure management.

Prevents the "30 bets on Saturday" problem where independent Kelly
sizing ignores shared bankroll risk.  Implements:

    1. Simultaneous Kelly — scales individual Kelly fractions based on
       total capital deployed across all concurrent bets.
    2. Dynamic bankroll tracking — uses current bankroll (not starting)
       for unit sizing.
    3. Max-drawdown circuit breaker — halts new bets when cumulative
       drawdown exceeds the configured threshold.
    4. Correlation-aware bucketing — groups bets by conference to
       apply a small correlation penalty (conference outcomes are
       weakly correlated through officiating crews, travel, etc.).
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PortfolioPosition:
    """A single pending bet in the portfolio."""

    game_id: int
    pick: str
    kelly_fractional: float  # Raw fractional Kelly before portfolio adjustment
    recommended_units: float
    edge_conservative: float
    conference: Optional[str] = None
    scheduled_time: Optional[datetime] = None


@dataclass
class PortfolioState:
    """Snapshot of the current portfolio exposure."""

    current_bankroll: float
    starting_bankroll: float
    positions: List[PortfolioPosition] = field(default_factory=list)
    drawdown_pct: float = 0.0
    total_exposure_pct: float = 0.0
    is_halted: bool = False
    halt_reason: Optional[str] = None


@dataclass
class AdjustedSizing:
    """Output of portfolio-adjusted sizing for a single bet."""

    raw_kelly: float
    portfolio_kelly: float
    raw_units: float
    adjusted_units: float
    scaling_factor: float
    reason: str


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class PortfolioManager:
    """
    Manages bankroll-aware bet sizing across simultaneous positions.

    The key insight: if you have N simultaneous bets each sized at f%,
    your total exposure is N*f%.  If that exceeds a threshold (e.g. 15%),
    each individual bet must be scaled down proportionally.
    """

    def __init__(
        self,
        starting_bankroll: Optional[float] = None,
        max_total_exposure_pct: float = 15.0,
        max_single_bet_pct: float = 3.0,
        max_drawdown_pct: Optional[float] = None,
        conference_correlation: float = 0.05,
    ):
        self.starting_bankroll = starting_bankroll or float(
            os.getenv("STARTING_BANKROLL", "1000")
        )
        self.current_bankroll = self.starting_bankroll
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_single_bet_pct = max_single_bet_pct
        self.max_drawdown_pct = max_drawdown_pct or float(
            os.getenv("MAX_DRAWDOWN_PCT", "15.0")
        )
        self.conference_correlation = conference_correlation

        self._pending_positions: List[PortfolioPosition] = []

    # ------------------------------------------------------------------
    # Bankroll state
    # ------------------------------------------------------------------

    def update_bankroll(self, current: float) -> None:
        """Update the current bankroll (after wins/losses settle)."""
        self.current_bankroll = current

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as a percentage of starting bankroll."""
        if self.starting_bankroll <= 0:
            return 0.0
        return max(
            0.0,
            (self.starting_bankroll - self.current_bankroll)
            / self.starting_bankroll
            * 100.0,
        )

    @property
    def is_halted(self) -> bool:
        """True if drawdown exceeds the max-drawdown circuit breaker."""
        return self.drawdown_pct >= self.max_drawdown_pct

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add_position(self, position: PortfolioPosition) -> None:
        """Register a pending bet."""
        self._pending_positions.append(position)

    def clear_settled(self, settled_game_ids: List[int]) -> None:
        """Remove positions for games that have been settled."""
        self._pending_positions = [
            p for p in self._pending_positions if p.game_id not in settled_game_ids
        ]

    def clear_all(self) -> None:
        """Reset all pending positions (e.g. end of day)."""
        self._pending_positions.clear()

    @property
    def total_exposure_pct(self) -> float:
        """Sum of all pending position sizes as % of current bankroll."""
        return sum(p.recommended_units for p in self._pending_positions)

    # ------------------------------------------------------------------
    # Portfolio-adjusted sizing
    # ------------------------------------------------------------------

    def adjust_kelly(
        self,
        raw_kelly_frac: float,
        raw_units: float,
        conference: Optional[str] = None,
    ) -> AdjustedSizing:
        """
        Scale a raw Kelly fraction to account for total portfolio exposure.

        The algorithm:
        1. Check circuit breaker (drawdown halt).
        2. Cap single-bet exposure at ``max_single_bet_pct``.
        3. If adding this bet would push total exposure above
           ``max_total_exposure_pct``, scale it down so the cap is
           not breached.
        4. Apply a small correlation penalty if other bets share the
           same conference.
        """
        # 1. Circuit breaker
        if self.is_halted:
            return AdjustedSizing(
                raw_kelly=raw_kelly_frac,
                portfolio_kelly=0.0,
                raw_units=raw_units,
                adjusted_units=0.0,
                scaling_factor=0.0,
                reason=f"HALTED: drawdown {self.drawdown_pct:.1f}% >= {self.max_drawdown_pct}%",
            )

        # 2. Single-bet cap
        capped_units = min(raw_units, self.max_single_bet_pct)
        scaling = capped_units / raw_units if raw_units > 0 else 1.0

        # 3. Total exposure cap
        current_total = self.total_exposure_pct
        headroom = max(0.0, self.max_total_exposure_pct - current_total)

        if capped_units > headroom:
            if headroom <= 0:
                return AdjustedSizing(
                    raw_kelly=raw_kelly_frac,
                    portfolio_kelly=0.0,
                    raw_units=raw_units,
                    adjusted_units=0.0,
                    scaling_factor=0.0,
                    reason=f"No headroom: total exposure {current_total:.1f}% >= cap {self.max_total_exposure_pct}%",
                )
            capped_units = headroom
            scaling = capped_units / raw_units if raw_units > 0 else 0.0

        # 4. Conference correlation penalty
        if conference:
            same_conf_count = sum(
                1
                for p in self._pending_positions
                if p.conference and p.conference.lower() == conference.lower()
            )
            if same_conf_count > 0:
                corr_penalty = 1.0 - self.conference_correlation * same_conf_count
                corr_penalty = max(corr_penalty, 0.5)  # Floor at 50%
                capped_units *= corr_penalty
                scaling *= corr_penalty

        adjusted_kelly = raw_kelly_frac * scaling

        reason_parts = []
        if scaling < 1.0:
            reason_parts.append(f"scaled {scaling:.2f}x")
        if current_total > 0:
            reason_parts.append(f"existing exposure {current_total:.1f}%")
        reason = "; ".join(reason_parts) if reason_parts else "no adjustment needed"

        return AdjustedSizing(
            raw_kelly=raw_kelly_frac,
            portfolio_kelly=adjusted_kelly,
            raw_units=raw_units,
            adjusted_units=round(capped_units, 4),
            scaling_factor=round(scaling, 4),
            reason=reason,
        )

    def get_state(self) -> PortfolioState:
        """Return a snapshot of the current portfolio state."""
        return PortfolioState(
            current_bankroll=self.current_bankroll,
            starting_bankroll=self.starting_bankroll,
            positions=list(self._pending_positions),
            drawdown_pct=self.drawdown_pct,
            total_exposure_pct=self.total_exposure_pct,
            is_halted=self.is_halted,
            halt_reason=(
                f"Drawdown {self.drawdown_pct:.1f}% >= {self.max_drawdown_pct}%"
                if self.is_halted
                else None
            ),
        )

    def load_from_db(self, db: Session) -> None:
        """
        Reconstruct portfolio state from the database.

        Reads pending (unsettled) bets from bet_logs to restore exposure
        tracking after a server restart.
        """
        from backend.models import BetLog

        pending_bets = (
            db.query(BetLog)
            .filter(BetLog.outcome.is_(None))
            .all()
        )

        self._pending_positions.clear()
        for bet in pending_bets:
            self._pending_positions.append(
                PortfolioPosition(
                    game_id=bet.game_id,
                    pick=bet.pick,
                    kelly_fractional=bet.kelly_fractional or 0.0,
                    recommended_units=bet.bet_size_units or 0.0,
                    edge_conservative=bet.conservative_edge or 0.0,
                )
            )

        # Reconstruct current bankroll from last settled bet
        last_settled = (
            db.query(BetLog)
            .filter(BetLog.outcome.isnot(None))
            .order_by(BetLog.timestamp.desc())
            .first()
        )
        if last_settled and last_settled.bankroll_at_bet:
            self.current_bankroll = (
                last_settled.bankroll_at_bet + (last_settled.profit_loss_dollars or 0)
            )

        logger.info(
            "Portfolio loaded: %d pending positions, bankroll $%.2f, drawdown %.1f%%",
            len(self._pending_positions),
            self.current_bankroll,
            self.drawdown_pct,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_portfolio_manager: Optional[PortfolioManager] = None


def get_portfolio_manager() -> PortfolioManager:
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager()
    return _portfolio_manager
