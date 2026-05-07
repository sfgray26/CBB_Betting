"""
Tests for P20 Snapshot Engine -- pure function tests (no DB).

All tests operate on SnapshotInput / SnapshotResult dataclasses only.
No database, no I/O, no external dependencies.
"""

import pytest
from datetime import date

from backend.services.snapshot_engine import (
    SnapshotInput,
    SnapshotResult,
    _check_health,
    _build_summary,
    build_snapshot,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_input(**overrides) -> SnapshotInput:
    """Return a SnapshotInput with healthy defaults. Override any field via kwargs."""
    defaults = dict(
        as_of_date=date(2026, 4, 5),
        n_players_scored=120,
        n_momentum_records=115,
        n_simulation_records=110,
        n_decisions=22,
        n_explanations=22,
        n_backtest_records=110,
        mean_composite_mae=0.18,
        regression_detected=False,
        top_lineup_player_ids=[1001, 1002, 1003, 1004, 1005],
        top_waiver_player_ids=[2001, 2002, 2003],
        pipeline_jobs_run=[
            "rolling_windows", "player_scores", "player_momentum",
            "ros_simulation", "decision_optimization",
            "backtesting", "explainability",
        ],
    )
    defaults.update(overrides)
    return SnapshotInput(**defaults)


# ---------------------------------------------------------------------------
# 1. Healthy snapshot
# ---------------------------------------------------------------------------

def test_healthy_snapshot():
    inp = _make_input()
    result = build_snapshot(inp)
    assert result.pipeline_health == "HEALTHY"
    assert result.health_reasons == []


# ---------------------------------------------------------------------------
# 2. FAILED when no players scored
# ---------------------------------------------------------------------------

def test_failed_snapshot_no_players():
    inp = _make_input(n_players_scored=0)
    result = build_snapshot(inp)
    assert result.pipeline_health == "FAILED"
    assert len(result.health_reasons) >= 1
    assert "No player scores" in result.health_reasons[0]


# ---------------------------------------------------------------------------
# 3. DEGRADED when regression detected
# ---------------------------------------------------------------------------

def test_degraded_regression():
    inp = _make_input(regression_detected=True)
    result = build_snapshot(inp)
    assert result.pipeline_health == "DEGRADED"
    assert any("regression" in r.lower() for r in result.health_reasons)


# ---------------------------------------------------------------------------
# 4. DEGRADED when no decisions generated but players were scored
# ---------------------------------------------------------------------------

def test_degraded_no_decisions():
    inp = _make_input(n_players_scored=50, n_decisions=0, n_explanations=0)
    result = build_snapshot(inp)
    assert result.pipeline_health == "DEGRADED"
    assert any("No decisions" in r for r in result.health_reasons)


# ---------------------------------------------------------------------------
# 5. DEGRADED when no explanations but decisions exist
# ---------------------------------------------------------------------------

def test_degraded_no_explanations():
    inp = _make_input(n_decisions=10, n_explanations=0)
    result = build_snapshot(inp)
    assert result.pipeline_health == "DEGRADED"
    assert any("No explanations" in r for r in result.health_reasons)


# ---------------------------------------------------------------------------
# 6. HEALTHY summary format
# ---------------------------------------------------------------------------

def test_summary_healthy_format():
    inp = _make_input()
    result = build_snapshot(inp)
    assert result.pipeline_health == "HEALTHY"
    summary_lower = result.summary.lower()
    assert "healthy" in summary_lower
    assert str(inp.n_players_scored) in result.summary
    assert str(inp.n_decisions) in result.summary
    assert str(inp.n_explanations) in result.summary


# ---------------------------------------------------------------------------
# 7. FAILED summary format
# ---------------------------------------------------------------------------

def test_summary_failed_format():
    inp = _make_input(n_players_scored=0)
    result = build_snapshot(inp)
    assert "FAILED" in result.summary


# ---------------------------------------------------------------------------
# 8. DEGRADED summary format
# ---------------------------------------------------------------------------

def test_summary_degraded_format():
    inp = _make_input(regression_detected=True)
    result = build_snapshot(inp)
    assert "degraded" in result.summary.lower()
    assert "regression" in result.summary.lower()


# ---------------------------------------------------------------------------
# 9. build_snapshot returns SnapshotResult dataclass
# ---------------------------------------------------------------------------

def test_build_snapshot_returns_correct_type():
    inp = _make_input()
    result = build_snapshot(inp)
    assert isinstance(result, SnapshotResult)


def test_build_snapshot_filters_non_fantasy_job_names():
    inp = _make_input(pipeline_jobs_run=["mlb_odds", "player_scores", "snapshot", "mlb_odds"])
    result = build_snapshot(inp)
    assert result.pipeline_jobs_run == ["player_scores", "snapshot"]


# ---------------------------------------------------------------------------
# 10. HEALTHY health_reasons is empty list
# ---------------------------------------------------------------------------

def test_health_reasons_empty_when_healthy():
    inp = _make_input()
    result = build_snapshot(inp)
    assert result.pipeline_health == "HEALTHY"
    assert result.health_reasons == []


# ---------------------------------------------------------------------------
# 11. Summary contains only ASCII bytes
# ---------------------------------------------------------------------------

def test_ascii_only_summary():
    for scenario in [
        _make_input(),
        _make_input(n_players_scored=0),
        _make_input(regression_detected=True),
        _make_input(n_decisions=0),
        _make_input(n_explanations=0),
    ]:
        result = build_snapshot(scenario)
        try:
            result.summary.encode("ascii")
        except UnicodeEncodeError as exc:
            pytest.fail(
                "Non-ASCII character in summary for health={}: {}".format(
                    result.pipeline_health, exc
                )
            )


# ---------------------------------------------------------------------------
# Bonus: FAILED takes priority over regression flag
# ---------------------------------------------------------------------------

def test_failed_takes_priority_over_regression():
    inp = _make_input(n_players_scored=0, regression_detected=True)
    result = build_snapshot(inp)
    assert result.pipeline_health == "FAILED"


# ---------------------------------------------------------------------------
# Bonus: multiple DEGRADED reasons accumulate
# ---------------------------------------------------------------------------

def test_multiple_degraded_reasons():
    inp = _make_input(regression_detected=True, n_decisions=0, n_explanations=0)
    result = build_snapshot(inp)
    assert result.pipeline_health == "DEGRADED"
    # regression + no decisions should both appear; no_explanations skipped since n_decisions=0
    assert any("regression" in r.lower() for r in result.health_reasons)
    assert any("decisions" in r.lower() for r in result.health_reasons)
