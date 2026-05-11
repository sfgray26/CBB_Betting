"""
P20 -- Snapshot Engine.

Pure-computation module (no DB imports, no side effects).
All imports are at module top level -- no imports inside functions.

Computes daily pipeline health status from pre-aggregated counts passed in by
the DB orchestrator (_run_snapshot in daily_ingestion.py).

Emits SnapshotResult dataclasses that are persisted to daily_snapshots.

ADR-004: Never import betting_model or analysis.
Layer 1 contract: no I/O, no DB, no logging -- pure deterministic transforms.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


# Snapshot health is for the fantasy decision pipeline. Other successful jobs
# may share the scheduler status dict, but persisting them here makes fantasy
# monitoring look degraded or unrelated.
_FANTASY_PIPELINE_JOBS: tuple[str, ...] = (
    "mlb_game_log",
    "mlb_box_stats",
    "rolling_windows",
    "player_scores",
    "player_momentum",
    "ros_simulation",
    "decision_optimization",
    "backtesting",
    "explainability",
    "snapshot",
    # Sprint 4/5 pipeline jobs added 2026-05-07
    "statcast",
    "savant_ingestion",
    "fangraphs_ros",
    "waiver_scan",
    "opportunity_update",
    "market_signals_update",
    "matchup_context_update",
    "canonical_projection_refresh",
)
_FANTASY_PIPELINE_JOB_SET = set(_FANTASY_PIPELINE_JOBS)


# ---------------------------------------------------------------------------
# Input dataclass -- all fields assembled upstream by the DB orchestrator
# ---------------------------------------------------------------------------

@dataclass
class SnapshotInput:
    """All pre-aggregated data needed to compute the daily snapshot."""

    as_of_date: date
    n_players_scored: int
    n_momentum_records: int
    n_simulation_records: int
    n_decisions: int
    n_explanations: int
    n_backtest_records: int
    mean_composite_mae: Optional[float]
    regression_detected: bool
    top_lineup_player_ids: list        # up to 5 bdl_player_ids by lineup_score desc
    top_waiver_player_ids: list        # up to 3 bdl_player_ids by value_gain desc
    pipeline_jobs_run: list            # list of job name strings run today


# ---------------------------------------------------------------------------
# Output dataclass -- pure result, NOT the ORM model
# ---------------------------------------------------------------------------

@dataclass
class SnapshotResult:
    """Computed snapshot result. Written to daily_snapshots by the orchestrator."""

    as_of_date: date
    n_players_scored: int
    n_momentum_records: int
    n_simulation_records: int
    n_decisions: int
    n_explanations: int
    n_backtest_records: int
    mean_composite_mae: Optional[float]
    regression_detected: bool
    top_lineup_player_ids: list
    top_waiver_player_ids: list
    pipeline_jobs_run: list
    pipeline_health: str               # "HEALTHY" | "DEGRADED" | "FAILED"
    health_reasons: list               # list of ASCII strings explaining any issues
    summary: str                       # one-sentence human-readable headline


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _check_health(inp: SnapshotInput) -> tuple:
    """
    Evaluate pipeline health. Returns (pipeline_health: str, health_reasons: list).

    Rules (in priority order):
    - If n_players_scored == 0: "FAILED", reason "No player scores for date"
    - If regression_detected: "DEGRADED", reason "Backtesting regression detected"
    - If n_decisions == 0 and n_players_scored > 0: "DEGRADED", reason "No decisions generated"
    - If n_explanations == 0 and n_decisions > 0: "DEGRADED", reason "No explanations generated"
    - Otherwise: "HEALTHY", no reasons
    """
    if inp.n_players_scored == 0:
        return ("FAILED", ["No player scores for date"])

    reasons = []

    if inp.regression_detected:
        reasons.append("Backtesting regression detected")

    if inp.n_decisions == 0 and inp.n_players_scored > 0:
        reasons.append("No decisions generated")

    if inp.n_explanations == 0 and inp.n_decisions > 0:
        reasons.append("No explanations generated")

    if reasons:
        return ("DEGRADED", reasons)

    return ("HEALTHY", [])


def _build_summary(inp: SnapshotInput, health: str, reasons: list) -> str:
    """
    One-sentence ASCII summary.
    HEALTHY: "Pipeline healthy: {n} players scored, {n} decisions, {n} explanations."
    DEGRADED: "Pipeline degraded: {n} players scored, {n} decisions. Issues: {reasons}."
    FAILED: "Pipeline FAILED for {date}: no player scores computed."
    """
    if health == "FAILED":
        return "Pipeline FAILED for {}: no player scores computed.".format(inp.as_of_date)

    if health == "DEGRADED":
        issue_text = "; ".join(reasons)
        return (
            "Pipeline degraded: {} players scored, {} decisions. Issues: {}.".format(
                inp.n_players_scored,
                inp.n_decisions,
                issue_text,
            )
        )

    # HEALTHY
    return "Pipeline healthy: {} players scored, {} decisions, {} explanations.".format(
        inp.n_players_scored,
        inp.n_decisions,
        inp.n_explanations,
    )


def _normalize_pipeline_jobs_run(jobs: list) -> list:
    """Return unique fantasy pipeline job names in canonical order."""
    seen = {job for job in jobs if job in _FANTASY_PIPELINE_JOB_SET}
    return [job for job in _FANTASY_PIPELINE_JOBS if job in seen]


def build_snapshot(inp: SnapshotInput) -> SnapshotResult:
    """Compute health, build summary, return SnapshotResult."""
    health, reasons = _check_health(inp)
    summary = _build_summary(inp, health, reasons)
    pipeline_jobs_run = _normalize_pipeline_jobs_run(inp.pipeline_jobs_run)

    return SnapshotResult(
        as_of_date=inp.as_of_date,
        n_players_scored=inp.n_players_scored,
        n_momentum_records=inp.n_momentum_records,
        n_simulation_records=inp.n_simulation_records,
        n_decisions=inp.n_decisions,
        n_explanations=inp.n_explanations,
        n_backtest_records=inp.n_backtest_records,
        mean_composite_mae=inp.mean_composite_mae,
        regression_detected=inp.regression_detected,
        top_lineup_player_ids=inp.top_lineup_player_ids,
        top_waiver_player_ids=inp.top_waiver_player_ids,
        pipeline_jobs_run=pipeline_jobs_run,
        pipeline_health=health,
        health_reasons=reasons,
        summary=summary,
    )
