"""
P18 -- Backtesting Harness.

Pure-computation module (no DB imports, no side effects).
Compares P16 simulation projections (proj_p50) against actual mlb_player_stats
outcomes over a rolling 14-day window to measure forecast accuracy.

Metrics:
  - Per-stat MAE and RMSE (single projection/actual pair, so RMSE == MAE)
  - Composite MAE: mean of non-None per-stat MAEs (hitters: HR/RBI/SB/AVG;
    pitchers: K/ERA/WHIP; two-way: all fields)
  - Regression detection: composite_mae > baseline * 1.20

ADR-004: Never import betting_model or analysis.
"""

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_PATH = "reports/backtesting_baseline.json"

# Regression threshold: 20% above golden baseline
_REGRESSION_THRESHOLD = 1.20

# ---------------------------------------------------------------------------
# Input/Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BacktestInput:
    """One player's projection + actual pair for a single backtest window."""

    bdl_player_id: int
    as_of_date: date             # date projection was made (simulation_results.as_of_date)
    player_type: str             # "hitter" | "pitcher" | "two_way"

    # Projections (from simulation_results p50 columns)
    proj_hr_p50: Optional[float]
    proj_rbi_p50: Optional[float]
    proj_sb_p50: Optional[float]
    proj_avg_p50: Optional[float]
    proj_k_p50: Optional[float]
    proj_era_p50: Optional[float]
    proj_whip_p50: Optional[float]

    # Actuals (aggregated from mlb_player_stats over the backtest window)
    actual_hr: Optional[float]
    actual_rbi: Optional[float]
    actual_sb: Optional[float]
    actual_avg: Optional[float]
    actual_k: Optional[float]
    actual_era: Optional[float]
    actual_whip: Optional[float]

    games_played: int            # number of actual games in the window


@dataclass
class BacktestResult:
    """Per-player forecast accuracy metrics."""

    bdl_player_id: int
    as_of_date: date
    player_type: str
    games_played: int

    # Per-stat MAE and RMSE (None when projection or actual unavailable)
    mae_hr: Optional[float]
    rmse_hr: Optional[float]
    mae_rbi: Optional[float]
    rmse_rbi: Optional[float]
    mae_sb: Optional[float]
    rmse_sb: Optional[float]
    mae_avg: Optional[float]
    rmse_avg: Optional[float]
    mae_k: Optional[float]
    rmse_k: Optional[float]
    mae_era: Optional[float]
    rmse_era: Optional[float]
    mae_whip: Optional[float]
    rmse_whip: Optional[float]

    # Composite accuracy (mean of non-None stat MAEs)
    composite_mae: Optional[float]

    # Direction accuracy placeholder (populated by caller for multi-player sets)
    direction_correct: Optional[bool]


@dataclass
class BacktestSummary:
    """Cohort-level aggregate accuracy report for a single backtest run."""

    window_start: date
    window_end: date
    n_players: int
    n_hitters: int
    n_pitchers: int

    # Aggregate MAE across all players
    mean_composite_mae: Optional[float]

    # Per-stat aggregate MAE
    mean_mae_hr: Optional[float]
    mean_mae_rbi: Optional[float]
    mean_mae_sb: Optional[float]
    mean_mae_avg: Optional[float]
    mean_mae_k: Optional[float]
    mean_mae_era: Optional[float]
    mean_mae_whip: Optional[float]

    # Golden baseline comparison
    baseline_mean_mae: Optional[float]
    regression_detected: bool
    regression_delta: Optional[float]


# ---------------------------------------------------------------------------
# Pure computation functions
# ---------------------------------------------------------------------------


def compute_mae(projected: Optional[float], actual: Optional[float]) -> Optional[float]:
    """
    Mean Absolute Error for a single projection/actual pair.

    Returns None when either value is None (projection unavailable or
    player had no games in the window).
    """
    if projected is None or actual is None:
        return None
    return abs(projected - actual)


def compute_rmse(projected: Optional[float], actual: Optional[float]) -> Optional[float]:
    """
    Root Mean Squared Error for a single projection/actual pair.

    For a single pair: RMSE == |projected - actual|.  Kept as a
    separate function so callers can switch to multi-sample RMSE
    in future without touching call sites.
    """
    if projected is None or actual is None:
        return None
    return abs(projected - actual)


def _mean_non_none(values: list) -> Optional[float]:
    """Return arithmetic mean of non-None items; None if list is empty or all None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def evaluate_player(inp: BacktestInput) -> BacktestResult:
    """
    Compute per-stat MAE/RMSE and composite_mae for one player.

    Composite MAE formula:
      hitter   -> mean([mae_hr, mae_rbi, mae_sb, mae_avg])  -- skip None
      pitcher  -> mean([mae_k, mae_era, mae_whip])           -- skip None
      two_way  -> mean of all non-None stat MAEs

    direction_correct is left as None here; the caller may populate it
    after ranking a multi-player cohort.
    """
    mae_hr   = compute_mae(inp.proj_hr_p50,   inp.actual_hr)
    rmse_hr  = compute_rmse(inp.proj_hr_p50,  inp.actual_hr)
    mae_rbi  = compute_mae(inp.proj_rbi_p50,  inp.actual_rbi)
    rmse_rbi = compute_rmse(inp.proj_rbi_p50, inp.actual_rbi)
    mae_sb   = compute_mae(inp.proj_sb_p50,   inp.actual_sb)
    rmse_sb  = compute_rmse(inp.proj_sb_p50,  inp.actual_sb)
    mae_avg  = compute_mae(inp.proj_avg_p50,  inp.actual_avg)
    rmse_avg = compute_rmse(inp.proj_avg_p50, inp.actual_avg)
    mae_k    = compute_mae(inp.proj_k_p50,    inp.actual_k)
    rmse_k   = compute_rmse(inp.proj_k_p50,   inp.actual_k)
    mae_era  = compute_mae(inp.proj_era_p50,  inp.actual_era)
    rmse_era = compute_rmse(inp.proj_era_p50, inp.actual_era)
    mae_whip = compute_mae(inp.proj_whip_p50, inp.actual_whip)
    rmse_whip = compute_rmse(inp.proj_whip_p50, inp.actual_whip)

    pt = inp.player_type
    if pt == "hitter":
        composite_mae = _mean_non_none([mae_hr, mae_rbi, mae_sb, mae_avg])
    elif pt == "pitcher":
        composite_mae = _mean_non_none([mae_k, mae_era, mae_whip])
    elif pt == "two_way":
        composite_mae = _mean_non_none(
            [mae_hr, mae_rbi, mae_sb, mae_avg, mae_k, mae_era, mae_whip]
        )
    else:
        composite_mae = None

    return BacktestResult(
        bdl_player_id=inp.bdl_player_id,
        as_of_date=inp.as_of_date,
        player_type=inp.player_type,
        games_played=inp.games_played,
        mae_hr=mae_hr,
        rmse_hr=rmse_hr,
        mae_rbi=mae_rbi,
        rmse_rbi=rmse_rbi,
        mae_sb=mae_sb,
        rmse_sb=rmse_sb,
        mae_avg=mae_avg,
        rmse_avg=rmse_avg,
        mae_k=mae_k,
        rmse_k=rmse_k,
        mae_era=mae_era,
        rmse_era=rmse_era,
        mae_whip=mae_whip,
        rmse_whip=rmse_whip,
        composite_mae=composite_mae,
        direction_correct=None,
    )


def evaluate_cohort(inputs: list) -> list:
    """
    Run evaluate_player for every input.

    Players with unknown player_type (not hitter/pitcher/two_way) are silently
    skipped -- they produce no composite_mae and would pollute aggregate stats.
    """
    results = []
    for inp in inputs:
        if inp.player_type not in ("hitter", "pitcher", "two_way"):
            continue
        results.append(evaluate_player(inp))
    return results


def summarize(
    results: list,
    window_start: date,
    window_end: date,
    baseline_mean_mae: Optional[float] = None,
) -> BacktestSummary:
    """
    Aggregate a list of BacktestResult into a BacktestSummary.

    Regression is detected when mean_composite_mae > baseline_mean_mae * 1.20.
    If baseline is None, regression_detected is always False.
    """
    n_players = len(results)
    n_hitters = sum(1 for r in results if r.player_type == "hitter")
    n_pitchers = sum(1 for r in results if r.player_type == "pitcher")

    mean_composite_mae = _mean_non_none([r.composite_mae for r in results])
    mean_mae_hr  = _mean_non_none([r.mae_hr  for r in results])
    mean_mae_rbi = _mean_non_none([r.mae_rbi for r in results])
    mean_mae_sb  = _mean_non_none([r.mae_sb  for r in results])
    mean_mae_avg = _mean_non_none([r.mae_avg for r in results])
    mean_mae_k   = _mean_non_none([r.mae_k   for r in results])
    mean_mae_era = _mean_non_none([r.mae_era for r in results])
    mean_mae_whip = _mean_non_none([r.mae_whip for r in results])

    regression_detected = False
    regression_delta: Optional[float] = None

    if baseline_mean_mae is not None and mean_composite_mae is not None:
        regression_delta = mean_composite_mae - baseline_mean_mae
        regression_detected = mean_composite_mae > baseline_mean_mae * _REGRESSION_THRESHOLD

    return BacktestSummary(
        window_start=window_start,
        window_end=window_end,
        n_players=n_players,
        n_hitters=n_hitters,
        n_pitchers=n_pitchers,
        mean_composite_mae=mean_composite_mae,
        mean_mae_hr=mean_mae_hr,
        mean_mae_rbi=mean_mae_rbi,
        mean_mae_sb=mean_mae_sb,
        mean_mae_avg=mean_mae_avg,
        mean_mae_k=mean_mae_k,
        mean_mae_era=mean_mae_era,
        mean_mae_whip=mean_mae_whip,
        baseline_mean_mae=baseline_mean_mae,
        regression_detected=regression_detected,
        regression_delta=regression_delta,
    )


def load_golden_baseline(baseline_path: str) -> dict:
    """
    Load golden baseline from a JSON file.

    Returns a dict with key "mean_composite_mae" (float).
    Returns {"mean_composite_mae": None} if the file is not found,
    cannot be parsed, or lacks the expected key.

    Uses stdlib json only -- no external dependencies.
    """
    try:
        with open(baseline_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        val = data.get("mean_composite_mae")
        if val is not None and not isinstance(val, (int, float)):
            return {"mean_composite_mae": None}
        return {"mean_composite_mae": float(val) if val is not None else None}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"mean_composite_mae": None}


def save_golden_baseline(summary: BacktestSummary, baseline_path: str) -> None:
    """
    Write the current summary as the new golden baseline (JSON).

    Only writes when regression_detected is False -- we never persist a
    regressed baseline, as that would move the goalposts permanently.

    Creates the parent directory if it does not exist.
    Uses stdlib json only.
    """
    if summary.regression_detected:
        return

    parent = os.path.dirname(baseline_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    payload = {
        "mean_composite_mae": summary.mean_composite_mae,
        "window_start": str(summary.window_start),
        "window_end": str(summary.window_end),
        "n_players": summary.n_players,
    }
    with open(baseline_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
