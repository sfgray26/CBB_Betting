"""
Model recalibration service.

The model has three calibratable parameters:

    home_advantage  (default 3.09 pts)
        Estimated by comparing model predictions to game results for
        home vs. neutral-site games.  If the model consistently over-
        or under-predicts home teams relative to neutral sites, the
        home_advantage parameter is off.

    sd_multiplier  (default 0.85, used as sqrt(total) * sd_multiplier)
        Estimated from probability calibration.  If the model's predicted
        win probabilities are systematically higher than the actual win
        rates ("over-confidence"), the distribution is too narrow and the
        multiplier should increase.  If under-confident, it decreases.

All changes are:
    - Bounded to prevent over-correction in a single run
    - Logged to the model_parameters table for full audit trail
    - Applied immediately to the next analysis run (analysis.py reads the
      latest ModelParameter values before constructing CBBEdgeModel)

Minimum sample requirement: MIN_BETS_FOR_RECALIBRATION (env var, default 30).
"""

import logging
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session, joinedload

from backend.models import BetLog, Game, Prediction, ModelParameter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Minimum settled bets with linked predictions required before recalibration
_MIN_BETS = int(os.getenv("MIN_BETS_FOR_RECALIBRATION", "30"))

# Maximum correction per recalibration run (prevents over-shooting)
_MAX_HA_ADJ_PER_RUN = 0.50          # ±0.50 pts on home_advantage
_MAX_SD_MULT_ADJ_PER_RUN = 0.03     # ±3% on sd_multiplier

# Safety bounds for each parameter
_HA_MIN, _HA_MAX = 1.5, 5.5
_SD_MULT_MIN, _SD_MULT_MAX = 0.70, 1.10

# Fraction of the measured bias to correct per run (conservatism factor)
_HA_CORRECTION_RATE = 0.25   # Correct 25% of the observed bias per run
_SD_CORRECTION_RATE = 0.50   # Correct 50% of the observed over-confidence


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_settled_records(db: Session, limit: int = 200) -> List[Dict]:
    """
    Return settled bets that have linked predictions and game scores.

    Each record contains the data needed for both margin-bias and
    probability-calibration analysis.
    """
    rows = (
        db.query(BetLog, Prediction, Game)
        .join(Game, BetLog.game_id == Game.id)
        .join(Prediction, BetLog.prediction_id == Prediction.id)
        .filter(
            BetLog.outcome.isnot(None),
            BetLog.outcome != -1,           # exclude pushes
            Game.home_score.isnot(None),
            Game.away_score.isnot(None),
            Prediction.projected_margin.isnot(None),
        )
        .order_by(Game.game_date.desc())
        .limit(limit)
        .all()
    )

    records = []
    for bet, pred, game in rows:
        records.append({
            "predicted_margin": pred.projected_margin,   # home-team perspective
            "actual_margin":    game.home_score - game.away_score,
            "model_prob":       bet.model_prob,
            "outcome":          int(bet.outcome),         # 0 or 1
            "is_neutral":       bool(game.is_neutral),
        })
    return records


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------

def _margin_bias(records: List[Dict]) -> Optional[float]:
    """
    Overall mean prediction error (positive = model over-predicts home team).
    """
    errors = [r["predicted_margin"] - r["actual_margin"] for r in records]
    return sum(errors) / len(errors) if errors else None


def _home_advantage_bias(records: List[Dict]) -> Optional[float]:
    """
    Isolate the home-advantage component of prediction error.

    Split records into neutral-site and home-site games.  The difference
    in mean prediction error between the two groups reflects a systematic
    home_advantage mis-calibration rather than a global rating offset.

    Returns: positive → model over-predicts home teams → reduce home_advantage
             negative → model under-predicts home teams → increase home_advantage
             None    → insufficient data
    """
    home_errors = [
        r["predicted_margin"] - r["actual_margin"]
        for r in records if not r["is_neutral"]
    ]
    neutral_errors = [
        r["predicted_margin"] - r["actual_margin"]
        for r in records if r["is_neutral"]
    ]

    if not home_errors:
        return None

    home_bias    = sum(home_errors) / len(home_errors)
    neutral_bias = sum(neutral_errors) / len(neutral_errors) if neutral_errors else 0.0

    return home_bias - neutral_bias


def _overconfidence(records: List[Dict]) -> Optional[float]:
    """
    Measure probability over-confidence: mean(predicted_prob - actual_win_rate).

    Positive → predictions are too high → SD needs to widen (increase multiplier)
    Negative → predictions are too low  → SD needs to narrow (decrease multiplier)
    None     → insufficient data with valid model_prob values
    """
    valid = [(r["model_prob"], r["outcome"]) for r in records
             if r["model_prob"] is not None and 0.0 < r["model_prob"] < 1.0]

    if len(valid) < 10:
        return None

    mean_prob    = sum(p for p, _ in valid) / len(valid)
    mean_outcome = sum(o for _, o in valid) / len(valid)
    return mean_prob - mean_outcome


def _brier_score(records: List[Dict]) -> Optional[float]:
    """Mean squared error between predicted probability and binary outcome."""
    valid = [(r["model_prob"], r["outcome"]) for r in records
             if r["model_prob"] is not None]
    if not valid:
        return None
    return sum((p - o) ** 2 for p, o in valid) / len(valid)


# ---------------------------------------------------------------------------
# Current parameter loading
# ---------------------------------------------------------------------------

def load_current_params(db: Session) -> Dict[str, float]:
    """
    Load calibrated parameter values from model_parameters table.

    Falls back to environment-variable defaults if no DB overrides exist.
    This is called by analysis.py before constructing CBBEdgeModel so that
    each nightly run uses the most recent calibrated values.
    """
    defaults: Dict[str, float] = {
        "home_advantage": float(os.getenv("HOME_ADVANTAGE", "3.09")),
        "sd_multiplier":  float(os.getenv("SD_MULTIPLIER",  "0.85")),
    }

    result = dict(defaults)
    for name in defaults:
        latest = (
            db.query(ModelParameter)
            .filter(ModelParameter.parameter_name == name)
            .order_by(ModelParameter.effective_date.desc())
            .first()
        )
        if latest and latest.parameter_value is not None:
            result[name] = latest.parameter_value
            logger.debug(
                "Loaded calibrated %s = %.4f (from %s)",
                name, latest.parameter_value, latest.effective_date,
            )

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_recalibration(
    db: Session,
    changed_by: str = "auto",
    min_bets: Optional[int] = None,
    apply_changes: bool = True,
) -> Dict:
    """
    Analyse recent prediction performance and adjust model parameters.

    Parameters adjusted (within bounded limits):
        home_advantage  — corrected for home-team systematic margin bias
        sd_multiplier   — corrected for probability over/under-confidence

    Args:
        db:             SQLAlchemy session.
        changed_by:     Who triggered the run ("auto" or user identifier).
        min_bets:       Override minimum settled bets required.
        apply_changes:  If False, return diagnostics without writing to DB
                        (dry-run mode).

    Returns:
        dict with keys:
            status            "ok" | "insufficient_data" | "no_changes"
            bets_analyzed     int
            parameters_changed  int
            changes           list of {parameter, old, new, reason}
            diagnostics       dict of raw metrics
            timestamp         ISO string
    """
    required = min_bets if min_bets is not None else _MIN_BETS
    records = _fetch_settled_records(db, limit=max(required * 3, 200))

    if len(records) < required:
        logger.info(
            "Recalibration skipped — %d bets available, %d required",
            len(records), required,
        )
        return {
            "status": "insufficient_data",
            "message": (
                f"Need {required} settled bets with linked predictions; "
                f"have {len(records)}."
            ),
            "bets_available": len(records),
            "min_required": required,
            "timestamp": datetime.utcnow().isoformat(),
        }

    current = load_current_params(db)
    changes: List[Dict] = []

    # Collect diagnostics regardless of whether changes are applied
    diag = {
        "bets_analyzed":        len(records),
        "margin_bias":          _margin_bias(records),
        "home_advantage_bias":  _home_advantage_bias(records),
        "overconfidence":       _overconfidence(records),
        "brier_score":          _brier_score(records),
        "current_home_advantage": current["home_advantage"],
        "current_sd_multiplier":  current["sd_multiplier"],
    }
    logger.info(
        "Recalibration diagnostics — margin_bias=%.3f, ha_bias=%.3f, "
        "overconfidence=%.3f, brier=%.4f",
        diag["margin_bias"] or 0,
        diag["home_advantage_bias"] or 0,
        diag["overconfidence"] or 0,
        diag["brier_score"] or 0,
    )

    # ---- 1. Home advantage --------------------------------------------------
    ha_bias = diag["home_advantage_bias"]
    if ha_bias is not None and abs(ha_bias) > 0.5:
        raw_adj = -ha_bias * _HA_CORRECTION_RATE        # opposite sign: over-pred → reduce HA
        adj = max(-_MAX_HA_ADJ_PER_RUN, min(_MAX_HA_ADJ_PER_RUN, raw_adj))
        new_ha = round(current["home_advantage"] + adj, 3)
        new_ha = max(_HA_MIN, min(_HA_MAX, new_ha))

        if abs(new_ha - current["home_advantage"]) > 0.01:
            reason = (
                f"ha_bias={ha_bias:+.3f} (n_home={sum(1 for r in records if not r['is_neutral'])})"
            )
            if apply_changes:
                db.add(ModelParameter(
                    parameter_name="home_advantage",
                    parameter_value=new_ha,
                    parameter_value_json={
                        "old": current["home_advantage"],
                        "new": new_ha,
                        "ha_bias": ha_bias,
                        "bets_analyzed": len(records),
                    },
                    reason=reason,
                    changed_by=changed_by,
                ))
            changes.append({
                "parameter": "home_advantage",
                "old": current["home_advantage"],
                "new": new_ha,
                "reason": reason,
                "applied": apply_changes,
            })
            logger.info(
                "home_advantage: %.3f → %.3f (%s)",
                current["home_advantage"], new_ha, reason,
            )

    # ---- 2. SD multiplier (probability calibration) -------------------------
    overconf = diag["overconfidence"]
    if overconf is not None and abs(overconf) > 0.03:
        raw_adj = overconf * _SD_CORRECTION_RATE        # positive overconf → increase SD
        adj = max(-_MAX_SD_MULT_ADJ_PER_RUN, min(_MAX_SD_MULT_ADJ_PER_RUN, raw_adj))
        new_mult = round(current["sd_multiplier"] + adj, 4)
        new_mult = max(_SD_MULT_MIN, min(_SD_MULT_MAX, new_mult))

        if abs(new_mult - current["sd_multiplier"]) > 0.001:
            n_probs = sum(1 for r in records if r["model_prob"] is not None)
            reason = (
                f"overconfidence={overconf:+.4f} (n={n_probs}), "
                f"brier={diag['brier_score']:.4f}"
            )
            if apply_changes:
                db.add(ModelParameter(
                    parameter_name="sd_multiplier",
                    parameter_value=new_mult,
                    parameter_value_json={
                        "old": current["sd_multiplier"],
                        "new": new_mult,
                        "overconfidence": overconf,
                        "brier_score": diag["brier_score"],
                        "bets_analyzed": len(records),
                    },
                    reason=reason,
                    changed_by=changed_by,
                ))
            changes.append({
                "parameter": "sd_multiplier",
                "old": current["sd_multiplier"],
                "new": new_mult,
                "reason": reason,
                "applied": apply_changes,
            })
            logger.info(
                "sd_multiplier: %.4f → %.4f (%s)",
                current["sd_multiplier"], new_mult, reason,
            )

    if apply_changes and changes:
        db.commit()

    return {
        "status": "ok" if changes else "no_changes",
        "bets_analyzed": len(records),
        "parameters_changed": len([c for c in changes if c["applied"]]),
        "changes": changes,
        "diagnostics": diag,
        "timestamp": datetime.utcnow().isoformat(),
    }
