"""
WaiverValuationService -- compute add/drop surplus for waiver wire decisions.

Uses CanonicalProjection as data source (feature-flag gated: CANONICAL_PROJECTION_V1).
Falls back to player_board data when canonical projections are unavailable.

TeamContext is built from the current Yahoo roster + CanonicalProjection playing-time data.
Quarantined players (from IdentityResolutionService) are excluded from PA/IP denominators.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from backend.models import CanonicalProjection
from backend.fantasy_baseball.team_context import TeamContext

logger = logging.getLogger(__name__)

# Population fallback playing-time constants (v1 heuristics)
_FALLBACK_PA: float = 450.0
_FALLBACK_IP: float = 100.0

# Population mean normalisation for counting-stat z-score surplus (simple v1 heuristic)
# These are approximate full-season population means used as scale denominators.
_BATTER_POP_MEANS: dict[str, float] = {
    "R": 70.0,
    "HR": 20.0,
    "RBI": 70.0,
    "SB": 12.0,
}

_PITCHER_POP_MEANS: dict[str, float] = {
    "W": 10.0,
    "K": 150.0,
    "SV": 8.0,
}

# Rate categories use raw delta (no population-mean scaling needed at v1).
_BATTER_RATE_CATS: tuple[str, ...] = ("AVG", "OPS")
_PITCHER_RATE_CATS: tuple[str, ...] = ("ERA", "WHIP", "K9")

# For ERA and WHIP lower is better -- flip sign when computing surplus.
_LOWER_IS_BETTER: frozenset[str] = frozenset({"ERA", "WHIP"})


class WaiverValuationService:
    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _latest_projection(self, mlbam_id: int) -> Optional[CanonicalProjection]:
        """Return the most recent CanonicalProjection row for mlbam_id, or None."""
        return (
            self.db.query(CanonicalProjection)
            .filter(CanonicalProjection.player_id == mlbam_id)
            .order_by(CanonicalProjection.projection_date.desc())
            .first()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_team_context(
        self,
        roster_mlbam_ids: list[int],
        quarantined_ids: set[int] | None = None,
        ) -> TeamContext:
        """
        Build TeamContext for a roster from CanonicalProjection data.

        For each MLBAM ID in roster_mlbam_ids:
          - Query CanonicalProjection for the most recent projection (ORDER BY projection_date DESC LIMIT 1)
          - Use projected_pa for batters (player_type == "BATTER"), projected_ip for pitchers
          - If no canonical projection found, use population fallbacks:
              batter PA = 450, pitcher IP = 100 (neutral, cannot distinguish SP/RP without role data)
          - Exclude any player_id that appears in quarantined_ids

        Returns TeamContext assembled via TeamContext.build().
        """
        _quarantined: set[int] = quarantined_ids or set()

        pa_by_player: dict[int, float] = {}
        ip_by_player: dict[int, float] = {}
        resolved_ids: list[int] = []

        for mlbam_id in roster_mlbam_ids:
            if mlbam_id in _quarantined:
                logger.debug("Skipping quarantined player %s in build_team_context", mlbam_id)
                continue

            resolved_ids.append(mlbam_id)

            proj = self._latest_projection(mlbam_id)
            if proj is None:
                # Unknown player type -- use universal PA fallback
                pa_by_player[mlbam_id] = _FALLBACK_PA
                logger.debug(
                    "No canonical projection for player %s; using fallback PA=%.0f",
                    mlbam_id, _FALLBACK_PA,
                )
                continue

            if proj.player_type == "PITCHER":
                ip_val = proj.projected_ip if proj.projected_ip is not None else _FALLBACK_IP
                ip_by_player[mlbam_id] = ip_val
            else:
                # BATTER or any unrecognised type defaults to PA
                pa_val = proj.projected_pa if proj.projected_pa is not None else _FALLBACK_PA
                pa_by_player[mlbam_id] = pa_val

        return TeamContext.build(
            roster_player_ids=resolved_ids,
            projected_pa_by_player=pa_by_player,
            projected_ip_by_player=ip_by_player,
            quarantined_player_ids=_quarantined,
        )

    def add_drop_surplus(
        self,
        add_mlbam_id: int,
        drop_mlbam_id: int,
        team_context: TeamContext,
    ) -> dict:
        """
        Compute the net category surplus of adding `add_player` and dropping `drop_player`.

        Algorithm:
          1. Fetch CanonicalProjection for both players (most recent)
          2. For each tracked category, compute per-category delta:
               delta[cat] = add_player.projected_value - drop_player.projected_value
             For rate categories (AVG, ERA, WHIP, K9), weight by PA/IP share in the new roster
          3. Return summary dict with per-category deltas and an overall surplus score

        Returns:
          {
            "add_player_id": int,
            "drop_player_id": int,
            "category_deltas": {category: float},  # positive = add player better
            "surplus_score": float,                  # weighted sum of deltas
            "data_source": "canonical" | "fallback",
          }

        If canonical projections are missing for either player, returns {"status": "no_data"}.
        """
        add_proj = self._latest_projection(add_mlbam_id)
        drop_proj = self._latest_projection(drop_mlbam_id)

        if add_proj is None or drop_proj is None:
            missing = add_mlbam_id if add_proj is None else drop_mlbam_id
            logger.debug("No canonical projection for player %s; returning no_data", missing)
            return {"status": "no_data"}

        category_deltas: dict[str, float] = {}
        surplus_score: float = 0.0

        # Determine player types
        add_is_pitcher = add_proj.player_type == "PITCHER"
        drop_is_pitcher = drop_proj.player_type == "PITCHER"

        # ---- Batter counting stats ----
        batter_counting: dict[str, tuple[Optional[int], Optional[int]]] = {
            "R":   (add_proj.proj_r,   drop_proj.proj_r),
            "HR":  (add_proj.proj_hr,  drop_proj.proj_hr),
            "RBI": (add_proj.proj_rbi, drop_proj.proj_rbi),
            "SB":  (add_proj.proj_sb,  drop_proj.proj_sb),
        }

        for cat, (add_val, drop_val) in batter_counting.items():
            # Only include if at least one player is a batter
            if add_is_pitcher and drop_is_pitcher:
                continue
            a = float(add_val) if add_val is not None else 0.0
            d = float(drop_val) if drop_val is not None else 0.0
            delta = a - d
            category_deltas[cat] = delta
            pop_mean = _BATTER_POP_MEANS.get(cat, 1.0)
            surplus_score += delta / pop_mean if pop_mean else 0.0

        # ---- Batter rate stats ----
        batter_rates: dict[str, tuple[Optional[float], Optional[float]]] = {
            "AVG": (add_proj.proj_avg, drop_proj.proj_avg),
            "OPS": (add_proj.proj_ops, drop_proj.proj_ops),
        }

        for cat, (add_val, drop_val) in batter_rates.items():
            if add_is_pitcher and drop_is_pitcher:
                continue
            a = float(add_val) if add_val is not None else 0.0
            d = float(drop_val) if drop_val is not None else 0.0
            # Weight by the add player's PA share in the updated roster
            pa_weight = team_context.batter_pa_share(add_mlbam_id)
            delta = (a - d) * (pa_weight if pa_weight > 0 else 1.0)
            category_deltas[cat] = delta
            surplus_score += delta

        # ---- Pitcher counting stats ----
        pitcher_counting: dict[str, tuple[Optional[int], Optional[int]]] = {
            "W":  (add_proj.proj_w,  drop_proj.proj_w),
            "K":  (add_proj.proj_k,  drop_proj.proj_k),
            "SV": (add_proj.proj_sv, drop_proj.proj_sv),
        }

        for cat, (add_val, drop_val) in pitcher_counting.items():
            # Only include if at least one player is a pitcher
            if not add_is_pitcher and not drop_is_pitcher:
                continue
            a = float(add_val) if add_val is not None else 0.0
            d = float(drop_val) if drop_val is not None else 0.0
            delta = a - d
            category_deltas[cat] = delta
            pop_mean = _PITCHER_POP_MEANS.get(cat, 1.0)
            surplus_score += delta / pop_mean if pop_mean else 0.0

        # ---- Pitcher rate stats ----
        pitcher_rates: dict[str, tuple[Optional[float], Optional[float]]] = {
            "ERA":  (add_proj.proj_era,  drop_proj.proj_era),
            "WHIP": (add_proj.proj_whip, drop_proj.proj_whip),
            "K9":   (add_proj.proj_k9,   drop_proj.proj_k9),
        }

        for cat, (add_val, drop_val) in pitcher_rates.items():
            if not add_is_pitcher and not drop_is_pitcher:
                continue
            a = float(add_val) if add_val is not None else 0.0
            d = float(drop_val) if drop_val is not None else 0.0
            ip_weight = team_context.pitcher_ip_share(add_mlbam_id)
            raw_delta = (a - d) * (ip_weight if ip_weight > 0 else 1.0)
            # For ERA/WHIP lower is better -- invert so positive = better
            if cat in _LOWER_IS_BETTER:
                raw_delta = -raw_delta
            category_deltas[cat] = raw_delta
            surplus_score += raw_delta

        return {
            "add_player_id": add_mlbam_id,
            "drop_player_id": drop_mlbam_id,
            "category_deltas": category_deltas,
            "surplus_score": surplus_score,
            "data_source": "canonical",
        }


# ---------------------------------------------------------------------------
# Module-level convenience wrapper
# ---------------------------------------------------------------------------

def build_team_context_for_roster(
    roster_mlbam_ids: list[int],
    db: Session,
    quarantined_ids: set[int] | None = None,
) -> TeamContext:
    """Convenience wrapper for the most common call pattern."""
    svc = WaiverValuationService(db)
    return svc.build_team_context(roster_mlbam_ids, quarantined_ids=quarantined_ids)
