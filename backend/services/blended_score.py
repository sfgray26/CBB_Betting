"""
P28 Blended Score — talent-anchored composite for roster optimization.

Combines three signals into a single forward-looking player value Z-score:
  - talent_z : season true-talent (Statcast xwOBA / xERA, or 30-day fallback)
  - form_z   : recent production (14-day composite_z from player_scores)
  - matchup_z: today's opponent context (from matchup_context, 5-factor model)

Design intent
-------------
The previous roster-optimize score was 14-day-only: a one-week cold streak could
sink an elite player (Soto benched for Soderstrom) because there was no track-
record anchor, and the optimizer could never bench a star for the *right* reason
(facing an ace) because the matchup signal was inert.

The blend is talent-anchored by default (60/20/20): a star stays anchored high
unless BOTH cold AND in a bad matchup. Form is confidence-shrunk toward talent
(Bayesian-flavored) so a tiny-sample slump can't override a long track record.

    final_z = w_talent * talent_z
            + w_form   * form_z_shrunk
            + w_matchup* matchup_z

    form_z_shrunk = talent_z + confidence * (form_z - talent_z)

Pure module. Zero I/O. Caller (the optimize endpoint) fetches all signals and
passes them in. Weights come from config_service.get_threshold so they're
tunable without a redeploy.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from backend.services.config_service import get_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default weights (overridable via threshold_config)
# ---------------------------------------------------------------------------

def _load_weights() -> "BlendedWeights":
    """Load blend weights from config, falling back to the talent-anchored
    60/20/20 default. Keys: optimize.weight.{talent,form,matchup}."""
    return BlendedWeights(
        talent=get_threshold("optimize.weight.talent", default=0.60),
        form=get_threshold("optimize.weight.form", default=0.20),
        matchup=get_threshold("optimize.weight.matchup", default=0.20),
    )


@dataclass(frozen=True)
class BlendedWeights:
    """Blend weights for the three score components. Need not sum to 1.0 —
    they are renormalized at compute time against whichever components are
    actually available for a given player."""
    talent: float = 0.60
    form: float = 0.20
    matchup: float = 0.20


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BlendedScoreResult:
    """Output of compute_blended_score. Carries the final Z plus a full
    component breakdown for explainability (surfaced to the frontend in P4)."""
    final_z: float                       # the blended Z-score (comparable across positions)
    talent_z: Optional[float]            # input talent Z (None if unavailable)
    form_z: Optional[float]              # input raw form Z (pre-shrinkage; None if unavailable)
    form_z_shrunk: Optional[float]       # confidence-shrunk form Z (None if form unavailable)
    matchup_z: Optional[float]           # input matchup Z (None if no game / unavailable)
    confidence: float                    # 14-day sample confidence of the form signal (0..1)
    weights_used: BlendedWeights         # the effective (post-renorm) weights applied
    talent_source: str                   # "statcast" | "score_30d" | "none"
    low_confidence: bool                 # True when form or talent data was thin
    components: dict = field(default_factory=dict)  # structured breakdown for API


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_to_z(p: float) -> float:
    """Convert a 0-100 percentile (from score_0_100) to a standard-normal Z.

    Used to turn the 30-day score_0_100 fallback into a Z comparable to the
    14-day composite_z. Uses the inverse normal CDF (percent point function).

    Clamps to [1, 99] to avoid +/-inf at the extremes.
    """
    try:
        from statistics import NormalDist
    except ImportError:  # py<3.8 safety (shouldn't trigger on 3.11)
        # Linear approximation fallback (Abramowitz & Stegun 26.2.23)
        p = max(1.0, min(99.0, float(p)))
        t = (p / 100.0 - 0.5) / 0.5
        # crude but bounded
        return max(-3.0, min(3.0, t * 3.0))
    p = max(1.0, min(99.0, float(p)))
    return round(NormalDist().inv_cdf(p / 100.0), 4)


def z_to_percentile(z: float) -> float:
    """Convert a Z-score to a 0-100 percentile via the standard-normal CDF.

    Inverse of percentile_to_z. Used to render final_z back to the familiar
    0-100 lineup_score the frontend expects, now absolute (vs the whole league)
    rather than relative (vs the user's roster).
    """
    try:
        from statistics import NormalDist
    except ImportError:
        z = max(-3.0, min(3.0, float(z)))
        return round(50.0 + (z / 3.0) * 50.0, 1)
    z = max(-3.0, min(3.0, float(z)))
    return round(NormalDist().cdf(z) * 100.0, 1)


# ---------------------------------------------------------------------------
# Core blend
# ---------------------------------------------------------------------------

def compute_blended_score(
    *,
    talent_z: Optional[float],
    form_z: Optional[float],
    confidence: float,
    matchup_z: Optional[float],
    matchup_confidence: Optional[float] = None,
    talent_source: str = "none",
    weights: Optional[BlendedWeights] = None,
) -> BlendedScoreResult:
    """Compute the talent-anchored blended Z-score for one player.

    Parameters
    ----------
    talent_z          : season true-talent Z (from Statcast xwOBA/xERA or 30d
                        score fallback). None when no talent signal is available.
    form_z            : 14-day composite_z (recent production). None when the
                        player has no 14-day score row.
    confidence        : 14-day sample confidence (games/window), 0..1. Gates how
                        strongly form_z can pull away from talent_z.
    matchup_z         : today's matchup Z from matchup_context. None when the
                        player has no game today or matchup data is unavailable.
    matchup_confidence: matchup_context confidence (0..1), used to dampen a
                        low-confidence matchup signal. Optional.
    talent_source     : provenance tag for the talent signal ("statcast",
                        "score_30d", "none") — surfaced for explainability.
    weights           : BlendedWeights override (defaults loaded from config).

    Returns
    -------
    BlendedScoreResult with final_z and a full component breakdown.

    Behavior notes
    --------------
    - Dynamic weight renormalization: when a component is None, its weight is
      redistributed to the remaining components so they still sum to 1.0.
    - Confidence-shrinkage of form: form_z_shrunk = talent_z + c*(form_z - talent_z).
      A low-confidence form signal barely moves off the talent anchor.
    - When BOTH talent and form are None, final_z falls back to 0.0 (neutral,
      low-confidence) — the optimizer will treat the player as replacement-level.
    - matchup_z is optionally damped by matchup_confidence when it's low (<0.4).
    """
    if weights is None:
        weights = _load_weights()

    # Cap confidence to [0, 1]
    confidence = max(0.0, min(1.0, float(confidence)))

    # ---- Confidence-shrink the form term toward talent ----
    form_z_shrunk: Optional[float] = None
    if form_z is not None:
        if talent_z is not None:
            form_z_shrunk = talent_z + confidence * (form_z - talent_z)
        else:
            # No talent anchor — use raw form, but flag low confidence
            form_z_shrunk = form_z

    # ---- Dampen matchup by its own confidence if provided and low ----
    matchup_z_eff = matchup_z
    if matchup_z is not None and matchup_confidence is not None:
        if matchup_confidence < 0.4:
            matchup_z_eff = matchup_z * (matchup_confidence / 0.4)

    # ---- Dynamic weight renormalization (only over available components) ----
    w = {
        "talent":  weights.talent  if talent_z          is not None else 0.0,
        "form":    weights.form    if form_z_shrunk     is not None else 0.0,
        "matchup": weights.matchup if matchup_z_eff     is not None else 0.0,
    }
    total_w = sum(w.values())

    if total_w <= 0.0:
        # No usable signal at all — neutral, low-confidence
        return BlendedScoreResult(
            final_z=0.0,
            talent_z=talent_z,
            form_z=form_z,
            form_z_shrunk=form_z_shrunk,
            matchup_z=matchup_z,
            confidence=confidence,
            weights_used=weights,
            talent_source=talent_source,
            low_confidence=True,
            components={"final_z": 0.0, "note": "no_signal"},
        )

    norm = {k: v / total_w for k, v in w.items()}

    final_z = (
        norm["talent"]  * (talent_z          if talent_z          is not None else 0.0)
      + norm["form"]    * (form_z_shrunk     if form_z_shrunk     is not None else 0.0)
      + norm["matchup"] * (matchup_z_eff     if matchup_z_eff     is not None else 0.0)
    )

    # Cap at ±3 to stay consistent with the scoring engine's Z_CAP convention
    final_z = max(-3.0, min(3.0, final_z))

    # ---- Low-confidence flag ----
    # True when the form sample is thin OR talent data was missing entirely.
    low_conf = (confidence < 0.4) or (talent_z is None) or (talent_source == "none" and form_z is None)

    components = {
        "talent_z": talent_z,
        "form_z_raw": form_z,
        "form_z_shrunk": form_z_shrunk,
        "matchup_z": matchup_z_eff,
        "confidence": confidence,
        "matchup_confidence": matchup_confidence,
        "weights_effective": {
            "talent": round(norm["talent"], 4),
            "form": round(norm["form"], 4),
            "matchup": round(norm["matchup"], 4),
        },
        "talent_source": talent_source,
    }

    return BlendedScoreResult(
        final_z=final_z,
        talent_z=talent_z,
        form_z=form_z,
        form_z_shrunk=form_z_shrunk,
        matchup_z=matchup_z,
        confidence=confidence,
        weights_used=weights,
        talent_source=talent_source,
        low_confidence=low_conf,
        components=components,
    )
