"""
Portfolio-level Kelly sizing and concurrent exposure management.

Prevents the "30 bets on Saturday" problem where independent Kelly
sizing ignores shared bankroll risk.  Implements:

    1. Simultaneous Kelly multiplier — mathematically constrains
       worst-case drawdown when N bets resolve concurrently.
    2. Dynamic bankroll tracking — uses current bankroll (not starting)
       for unit sizing.
    3. Max-drawdown circuit breaker — halts new bets when cumulative
       drawdown exceeds the configured threshold.
    4. Correlation-aware bucketing — groups bets by conference to
       apply a small correlation penalty (conference outcomes are
       weakly correlated through officiating crews, travel, etc.).

The key mathematical fix vs naive fractional Kelly:

    Standard Kelly assumes *sequential* betting — bet, resolve, recalculate.
    When N bets are placed simultaneously, the worst-case loss is the sum
    of all individual bet sizes.  The simultaneous Kelly multiplier
    scales each bet's Kelly fraction so that even under a total-loss
    scenario, the bankroll drawdown stays within survival parameters.

    Multiplier = min(1.0, max_risk / (N * avg_kelly))
"""

import logging
import os
import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
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
        max_total_exposure_pct: float = 20.0,
        max_single_bet_pct: float = 4.0,
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

    # ------------------------------------------------------------------
    # Simultaneous Kelly math
    # ------------------------------------------------------------------

    def simultaneous_kelly_multiplier(self, n_new_bets: int = 1) -> float:
        """
        Compute the simultaneous Kelly multiplier.

        Standard Kelly assumes sequential betting.  When N bets are
        concurrent, the worst-case drawdown is the sum of all bet sizes.

        The multiplier constrains total risk so that a total-loss
        scenario (all N bets lose) keeps drawdown within
        ``max_total_exposure_pct``.

        Formula:
            M = max_risk / (N * avg_kelly)

        Where:
            max_risk = max_total_exposure_pct (as fraction, e.g. 0.15)
            N = total concurrent bets (existing + new)
            avg_kelly = average Kelly fraction across pending bets

        Returns a multiplier in (0, 1].  All individual Kelly fractions
        should be multiplied by this before sizing.
        """
        n_existing = len(self._pending_positions)
        n_total = n_existing + n_new_bets

        if n_total <= 1:
            return 1.0

        # Average Kelly across existing positions
        if n_existing > 0:
            avg_kelly = sum(p.kelly_fractional for p in self._pending_positions) / n_existing
        else:
            avg_kelly = 0.05  # Assume moderate Kelly for new-only batches

        avg_kelly = max(avg_kelly, 0.01)  # Floor to prevent division by zero

        max_risk_frac = self.max_total_exposure_pct / 100.0
        multiplier = max_risk_frac / (n_total * avg_kelly)

        return min(1.0, max(0.05, multiplier))

    def worst_case_drawdown(self) -> float:
        """
        Calculate worst-case drawdown if ALL pending bets lose simultaneously.

        Returns percentage of current bankroll.
        """
        if self.current_bankroll <= 0:
            return 100.0
        total_at_risk = sum(
            p.recommended_units * (self.current_bankroll / 100.0)
            for p in self._pending_positions
        )
        return (total_at_risk / self.current_bankroll) * 100.0

    def adjust_kelly(
        self,
        raw_kelly_frac: float,
        raw_units: float,
        conference: Optional[str] = None,
    ) -> AdjustedSizing:
        """
        Scale a raw Kelly fraction using simultaneous Kelly optimization.

        The algorithm:
        1. Check circuit breaker (drawdown halt).
        2. Apply simultaneous Kelly multiplier based on concurrent bet count.
        3. Cap single-bet exposure at ``max_single_bet_pct``.
        4. If adding this bet would push total exposure above
           ``max_total_exposure_pct``, scale to fit.
        5. Apply conference correlation penalty.
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

        # 2. Simultaneous Kelly multiplier
        sim_mult = self.simultaneous_kelly_multiplier(n_new_bets=1)
        scaled_units = raw_units * sim_mult
        scaling = sim_mult

        # 3. Single-bet cap
        if scaled_units > self.max_single_bet_pct:
            scaling *= self.max_single_bet_pct / scaled_units
            scaled_units = self.max_single_bet_pct

        # 4. Total exposure cap
        current_total = self.total_exposure_pct
        headroom = max(0.0, self.max_total_exposure_pct - current_total)

        if scaled_units > headroom:
            if headroom <= 0:
                return AdjustedSizing(
                    raw_kelly=raw_kelly_frac,
                    portfolio_kelly=0.0,
                    raw_units=raw_units,
                    adjusted_units=0.0,
                    scaling_factor=0.0,
                    reason=f"No headroom: total exposure {current_total:.1f}% >= cap {self.max_total_exposure_pct}%",
                )
            scaling *= headroom / scaled_units
            scaled_units = headroom

        # 5. Conference correlation penalty
        if conference:
            same_conf_count = sum(
                1
                for p in self._pending_positions
                if p.conference and p.conference.lower() == conference.lower()
            )
            if same_conf_count > 0:
                corr_penalty = 1.0 - self.conference_correlation * same_conf_count
                corr_penalty = max(corr_penalty, 0.5)  # Floor at 50%
                scaled_units *= corr_penalty
                scaling *= corr_penalty

        adjusted_kelly = raw_kelly_frac * scaling

        reason_parts = []
        if sim_mult < 1.0:
            reason_parts.append(f"sim_kelly {sim_mult:.2f}x ({len(self._pending_positions)} concurrent)")
        if scaling < sim_mult:
            reason_parts.append(f"capped to {scaled_units:.2f}%")
        if current_total > 0:
            reason_parts.append(f"existing exposure {current_total:.1f}%")
        reason = "; ".join(reason_parts) if reason_parts else "no adjustment needed"

        return AdjustedSizing(
            raw_kelly=raw_kelly_frac,
            portfolio_kelly=adjusted_kelly,
            raw_units=raw_units,
            adjusted_units=round(scaled_units, 4),
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
