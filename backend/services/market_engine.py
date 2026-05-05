"""
PR 4.3/4.4 — Market Signals Engine

Pure computation module for Yahoo ownership-based market intelligence.
No module-level I/O. All DB queries live in daily_ingestion._compute_market_signals().

Signals computed:
  - ownership_velocity   : change in %owned per day over 7d window
  - ownership_delta_7d   : raw %owned change over 7 days
  - ownership_delta_30d  : raw %owned change over 30 days
  - add_drop_ratio       : add_rate_7d / drop_rate_7d (>1 = net buying pressure)
  - market_score         : 0-100 contrarian signal (50 = neutral)
  - market_tag           : BUY_LOW / SELL_HIGH / HOT_PICKUP / SLEEPER / FAIR
  - market_urgency       : ACT_NOW / THIS_WEEK / MONITOR

Algorithm (compute_market_score):
  skill_signal    = 2.0 * (skill_gap_percentile - 0.5)       # [-1, +1]
  market_awareness = min(abs(ownership_velocity) / 5.0, 1.0)  # [0, 1]
  contrarian      = skill_signal * (1.0 - market_awareness)
  if confidence < 0.5:
      contrarian *= 0.5                                        # PR 4.4 gate
  market_score    = clamp(50.0 + contrarian * 50.0, 0, 100)

Interpretation:
  market_score > 65 : undervalued relative to market (buy signal)
  market_score < 35 : overvalued relative to market (sell signal)
  35-65             : fairly priced

Consumed by:
  - daily_ingestion._compute_market_signals() (lock 100_038, 8:30 AM ET)
  - waiver_edge_detector.get_top_moves() (tiebreaker, PR 4.5)

Architecture principle: market_score is a TIEBREAKER, not a primary signal.
Max weight in ranking: 10%. Never overrides skill (composite_z / need_score).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.services.config_service import get_threshold as _get_threshold

# ---------------------------------------------------------------------------
# Config-driven tag thresholds
# ---------------------------------------------------------------------------

_BUY_LOW_SKILL_PCT:  float = _get_threshold("market.buy_low.skill_pct",  default=0.85)
_SELL_HIGH_SKILL_PCT: float = _get_threshold("market.sell_high.skill_pct", default=0.15)
_HOT_VEL_THRESHOLD:  float = _get_threshold("market.hot.velocity",        default=5.0)
_HOT_SKILL_FLOOR:    float = _get_threshold("market.hot.skill_floor",     default=0.60)
_SLEEPER_OWNED_CAP:  float = _get_threshold("market.sleeper.owned_cap",   default=15.0)
_SLEEPER_SKILL_FLOOR: float = _get_threshold("market.sleeper.skill_floor", default=0.70)
_SELL_VEL_THRESHOLD: float = _get_threshold("market.sell.velocity",       default=3.0)
_STABLE_VEL_CAP:     float = _get_threshold("market.stable.velocity_cap", default=2.0)

# velocity = pct_owned_change_per_day; 5.0 = rapid pickup (+5% owned/day)
_VEL_NORMALIZATION:  float = _get_threshold("market.velocity_normalization", default=5.0)

# Confidence gate: below this threshold, contrarian signal is halved
_CONF_GATE: float = _get_threshold("market.confidence_gate", default=0.5)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class MarketResult:
    """
    Market signal output for a single player on a single date.

    Fields:
      market_score    -- 0-100 contrarian signal (50 = fairly priced)
      market_tag      -- BUY_LOW | SELL_HIGH | HOT_PICKUP | SLEEPER | FAIR
      market_urgency  -- ACT_NOW | THIS_WEEK | MONITOR
    """
    market_score:   float
    market_tag:     str
    market_urgency: str


# Tag constants
BUY_LOW    = "BUY_LOW"
SELL_HIGH  = "SELL_HIGH"
HOT_PICKUP = "HOT_PICKUP"
SLEEPER    = "SLEEPER"
FAIR       = "FAIR"

ACT_NOW    = "ACT_NOW"
THIS_WEEK  = "THIS_WEEK"
MONITOR    = "MONITOR"


# ---------------------------------------------------------------------------
# Pure math functions (no I/O)
# ---------------------------------------------------------------------------

def compute_ownership_velocity(
    current_pct: Optional[float],
    pct_7d_ago: Optional[float],
) -> float:
    """
    Ownership velocity: change in %owned per day over the 7-day window.

    Positive = player being added; Negative = player being dropped.
    Returns 0.0 when either input is None (data absent).
    """
    if current_pct is None or pct_7d_ago is None:
        return 0.0
    return (current_pct - pct_7d_ago) / 7.0


def compute_ownership_deltas(
    current_pct: Optional[float],
    pct_7d_ago: Optional[float],
    pct_30d_ago: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """
    Raw ownership deltas: (delta_7d, delta_30d).

    delta_7d  = current - 7d_ago
    delta_30d = current - 30d_ago
    Returns None for a delta when either endpoint is missing.
    """
    delta_7d  = (current_pct - pct_7d_ago)  if (current_pct is not None and pct_7d_ago  is not None) else None
    delta_30d = (current_pct - pct_30d_ago) if (current_pct is not None and pct_30d_ago is not None) else None
    return delta_7d, delta_30d


def compute_add_drop_ratio(
    add_rate_7d: Optional[float],
    drop_rate_7d: Optional[float],
) -> Optional[float]:
    """
    Add/drop ratio: >1.0 = net buying pressure, <1.0 = net selling.

    Returns None when either rate is missing.
    Returns None when drop_rate_7d is 0 (avoid division by zero).
    """
    if add_rate_7d is None or drop_rate_7d is None:
        return None
    if drop_rate_7d <= 0.0:
        return None
    return add_rate_7d / drop_rate_7d


def classify_market_tag(
    skill_gap_percentile: float,
    ownership_velocity: float,
    owned_pct: float,
) -> tuple[str, str]:
    """
    Classify player's market situation into (tag, urgency).

    Tag logic (evaluated in priority order):
      BUY_LOW    : high skill percentile + stable ownership (undervalued gem)
      SELL_HIGH  : low skill percentile + rising ownership (sell into hype)
      HOT_PICKUP : rising fast + decent skill (FAAB target before price rises)
      SLEEPER    : low ownership + good skill (hidden value)
      FAIR       : everything else

    Parameters
    ----------
    skill_gap_percentile : float in [0, 1] — player's composite_z percentile rank
    ownership_velocity   : float — %owned change per day (from compute_ownership_velocity)
    owned_pct            : float — current %owned (0-100)
    """
    if skill_gap_percentile > _BUY_LOW_SKILL_PCT and abs(ownership_velocity) < _STABLE_VEL_CAP:
        return BUY_LOW, ACT_NOW

    if skill_gap_percentile < _SELL_HIGH_SKILL_PCT and ownership_velocity > _SELL_VEL_THRESHOLD:
        return SELL_HIGH, THIS_WEEK

    if ownership_velocity > _HOT_VEL_THRESHOLD and skill_gap_percentile > _HOT_SKILL_FLOOR:
        return HOT_PICKUP, ACT_NOW

    if owned_pct < _SLEEPER_OWNED_CAP and skill_gap_percentile > _SLEEPER_SKILL_FLOOR:
        return SLEEPER, THIS_WEEK

    return FAIR, MONITOR


def compute_market_score(
    skill_gap: float,
    skill_gap_percentile: float,
    ownership_velocity: float,
    owned_pct: float,
    confidence: float,
) -> MarketResult:
    """
    Compute market_score (0-100) and classify tag/urgency.

    Algorithm:
      skill_signal    = 2.0 * (skill_gap_percentile - 0.5)
      market_awareness = min(abs(ownership_velocity) / _VEL_NORMALIZATION, 1.0)
      contrarian      = skill_signal * (1.0 - market_awareness)

    PR 4.4 confidence gate:
      if confidence < _CONF_GATE: contrarian *= 0.5

    market_score = clamp(50.0 + contrarian * 50.0, 0.0, 100.0)

    Interpretation:
      >65  : player undervalued relative to market → buy signal
      <35  : player overvalued relative to market  → sell signal
      35-65: fairly priced

    Parameters
    ----------
    skill_gap            : composite_z - expected_z (informational, not used in formula)
    skill_gap_percentile : float [0, 1] — player's percentile rank in their cohort
    ownership_velocity   : float — %owned change per day
    owned_pct            : float — current %owned (0-100)
    confidence           : float [0, 1] — from player_scores or opportunity engine
    """
    skill_signal     = 2.0 * (skill_gap_percentile - 0.5)
    market_awareness = min(abs(ownership_velocity) / max(_VEL_NORMALIZATION, 1e-9), 1.0)
    contrarian       = skill_signal * (1.0 - market_awareness)

    # PR 4.4: Confidence gate — dampen signal when sample is thin
    if confidence < _CONF_GATE:
        contrarian *= 0.5

    market_score = max(0.0, min(100.0, 50.0 + contrarian * 50.0))
    tag, urgency = classify_market_tag(skill_gap_percentile, ownership_velocity, owned_pct)

    return MarketResult(
        market_score=round(market_score, 2),
        market_tag=tag,
        market_urgency=urgency,
    )
