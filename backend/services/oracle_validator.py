"""
K-15: Oracle Validation System

Flags predictions where our model's projected margin diverges significantly
from the rating-system consensus (KenPom + BartTorvik average).

Divergence is expressed as a z-score relative to a calibrated oracle SD.
The flagging threshold tightens as game time approaches:

  ≥24h before tip  →  z ≥ ORACLE_THRESHOLD_Z_EARLY  (default 2.0)
  4–24h            →  z ≥ ORACLE_THRESHOLD_Z_MID    (default 2.5)
  <4h              →  z ≥ ORACLE_THRESHOLD_Z_LATE   (default 3.0)

Flagged predictions are surfaced at GET /admin/oracle/flagged.

Usage:
    from backend.services.oracle_validator import calculate_oracle_divergence
    result = calculate_oracle_divergence(
        model_spread=analysis.projected_margin,
        kenpom_home=ratings["kenpom"]["home"],
        kenpom_away=ratings["kenpom"]["away"],
        barttorvik_home=ratings["barttorvik"]["home"],
        barttorvik_away=ratings["barttorvik"]["away"],
        hours_to_tipoff=hours_to_tipoff,
    )
    if result and result.flagged:
        ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from backend.utils.env_utils import get_float_env

# ---------------------------------------------------------------------------
# Calibrated SD for rating-system disagreement.
# In CBB, the raw AdjEM margin from two independent systems typically agrees
# within ±3-5 points per team.  A 4-point SD covers ~68% of normal spreads.
# ---------------------------------------------------------------------------
ORACLE_SD: float = get_float_env("ORACLE_SD", "4.0")

# Time-weighted thresholds — tighten as game approaches.
# Higher z at game time means we only flag truly irreconcilable divergences,
# while accepting more uncertainty early in the day.
ORACLE_THRESHOLD_Z_EARLY: float = get_float_env("ORACLE_THRESHOLD_Z_EARLY", "2.0")
ORACLE_THRESHOLD_Z_MID: float = get_float_env("ORACLE_THRESHOLD_Z_MID", "2.5")
ORACLE_THRESHOLD_Z_LATE: float = get_float_env("ORACLE_THRESHOLD_Z_LATE", "3.0")


@dataclass
class OracleResult:
    """
    Result of comparing our model's spread to the rating-system consensus.

    Fields
    ------
    oracle_spread       Consensus spread (avg of available rating differentials).
                        Positive = home favoured, same sign convention as
                        projected_margin.
    model_spread        Our model's projected_margin at analysis time.
    divergence_points   |model_spread - oracle_spread| in raw points.
    divergence_z        divergence_points / ORACLE_SD — normalised signal.
    threshold_z         The z threshold in effect at this hours_to_tipoff.
    flagged             True when divergence_z >= threshold_z.
    sources             Rating systems that contributed to the consensus.
    """

    oracle_spread: float
    model_spread: float
    divergence_points: float
    divergence_z: float
    threshold_z: float
    flagged: bool
    sources: list[str]

    def to_dict(self) -> dict:
        return {
            "oracle_spread": round(self.oracle_spread, 3),
            "model_spread": round(self.model_spread, 3),
            "divergence_points": round(self.divergence_points, 3),
            "divergence_z": round(self.divergence_z, 3),
            "threshold_z": self.threshold_z,
            "flagged": self.flagged,
            "sources": self.sources,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_threshold(hours_to_tipoff: Optional[float]) -> float:
    """Return the z threshold for the given hours-to-tipoff window."""
    if hours_to_tipoff is None or hours_to_tipoff >= 24:
        return ORACLE_THRESHOLD_Z_EARLY
    if hours_to_tipoff >= 4:
        return ORACLE_THRESHOLD_Z_MID
    return ORACLE_THRESHOLD_Z_LATE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_oracle_divergence(
    model_spread: float,
    kenpom_home: Optional[float],
    kenpom_away: Optional[float],
    barttorvik_home: Optional[float],
    barttorvik_away: Optional[float],
    hours_to_tipoff: Optional[float] = None,
    oracle_sd: float = ORACLE_SD,
) -> Optional[OracleResult]:
    """
    Compare our model's projected_margin against the KenPom/BartTorvik consensus.

    Parameters
    ----------
    model_spread        analysis.projected_margin (positive = home favoured).
    kenpom_home/away    Raw KenPom AdjEM ratings for each team.
    barttorvik_home/away Raw BartTorvik AdjEM ratings for each team.
    hours_to_tipoff     Hours until game starts; drives threshold selection.
    oracle_sd           Standard deviation for z-score normalisation.

    Returns
    -------
    OracleResult if at least one rating pair is available, else None.
    """
    margins: list[float] = []
    sources: list[str] = []

    if kenpom_home is not None and kenpom_away is not None:
        margins.append(kenpom_home - kenpom_away)
        sources.append("kenpom")

    if barttorvik_home is not None and barttorvik_away is not None:
        margins.append(barttorvik_home - barttorvik_away)
        sources.append("barttorvik")

    if not margins:
        return None

    oracle_spread = sum(margins) / len(margins)
    divergence_points = abs(model_spread - oracle_spread)
    divergence_z = divergence_points / oracle_sd if oracle_sd > 0 else 0.0
    threshold_z = _select_threshold(hours_to_tipoff)

    return OracleResult(
        oracle_spread=oracle_spread,
        model_spread=model_spread,
        divergence_points=divergence_points,
        divergence_z=divergence_z,
        threshold_z=threshold_z,
        flagged=divergence_z >= threshold_z,
        sources=sources,
    )
