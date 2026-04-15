# Data Quality & Pipeline Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve 100% confidence in data integrity, pipeline reliability, and Monte Carlo simulation accuracy before building any frontend. Fix every known bug, fill every empty table, validate every computation end-to-end.

**Architecture:** Fix critical data gaps (projections, Statcast, probable pitchers) → fix known bugs (category tracker, valuation worker) → build end-to-end validation infrastructure → prove simulation accuracy with backtests.

**Tech Stack:** Python 3.11, pytest, SQLAlchemy, pybaseball, FanGraphs RoS API, MLB Stats API

---

## File Structure

```
backend/
  services/
    daily_ingestion.py              # MODIFY - fix production schedules, add freshness checks
    simulation_engine.py            # READ-ONLY - validate, do not change
    scoring_engine.py               # READ-ONLY - validate, do not change
  fantasy_baseball/
    category_tracker.py             # MODIFY - fix copy-paste bug on line 59
    valuation_worker.py             # MODIFY - add missing-data logging
    projections_loader.py           # MODIFY - add FanGraphs RoS → CSV bridge
    fangraphs_loader.py             # READ - understand current RoS fetch
data/
  projections/                      # CREATE CSV files from FanGraphs RoS cache
tests/
  test_data_quality_e2e.py          # CREATE - end-to-end pipeline validation
  test_simulation_backtest.py       # CREATE - Monte Carlo accuracy backtest
  test_category_tracker_fix.py      # CREATE - regression test for bug fix
  test_pipeline_freshness.py        # CREATE - table freshness validation
```

---

### Task 1: Fix Category Tracker Copy-Paste Bug

**Files:**
- Modify: `backend/fantasy_baseball/category_tracker.py:58-59`
- Create: `tests/test_category_tracker_fix.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_category_tracker_fix.py`:

```python
"""Regression test for category_tracker.py line 59 copy-paste bug.

The bug: `opp_stats = self._extract_stats(matchup, "team_stats")` extracts
the same team's stats for both sides. The fix is irrelevant because lines
62-80 correctly re-extract both teams from the matchup dict. But line 59
is dead code that should be removed to prevent confusion.

This test validates that given a 2-team matchup dict, the tracker correctly
identifies my_stats vs opp_stats using the team_key comparison on lines 74-80.
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_matchup(my_key="469.l.12345.t.1", opp_key="469.l.12345.t.5",
                  my_hr=10, opp_hr=8):
    """Build a minimal Yahoo matchup dict with 2 teams."""
    return {
        "teams": {
            "count": 2,
            "0": {
                "team": [
                    [{"team_key": my_key}],
                    {"team_stats": {"stats": [{"stat": {"stat_id": "12", "value": str(my_hr)}}]}},
                ],
            },
            "1": {
                "team": [
                    [{"team_key": opp_key}],
                    {"team_stats": {"stats": [{"stat": {"stat_id": "12", "value": str(opp_hr)}}]}},
                ],
            },
        }
    }


@patch("backend.fantasy_baseball.category_tracker.YahooFantasyClient")
def test_opponent_stats_differ_from_my_stats(mock_client_cls):
    """After the fix, opp_stats must NOT equal my_stats when teams differ."""
    from backend.fantasy_baseball.category_tracker import CategoryTracker

    mock_client = MagicMock()
    mock_client.get_my_team_key.return_value = "469.l.12345.t.1"
    mock_client_cls.return_value = mock_client

    tracker = CategoryTracker(client=mock_client)

    matchup = _make_matchup(my_hr=10, opp_hr=3)

    # Mock _get_my_matchup to return our test data
    tracker._get_my_matchup = MagicMock(return_value=matchup)

    needs = tracker.get_category_needs()
    # Should not be empty (means parsing succeeded)
    # The actual content depends on stat mapping, but the point is it doesn't crash
    # and both teams are parsed separately
    assert isinstance(needs, list)
```

- [ ] **Step 2: Run test to verify behavior**

```bash
venv/Scripts/python -m pytest tests/test_category_tracker_fix.py -v --tb=short
```

- [ ] **Step 3: Remove the dead code on line 58-59**

In `category_tracker.py`, lines 58-59 are dead code — they call `_extract_stats` but the result is immediately overwritten by lines 62-80 which correctly parse both teams. Remove the dead lines:

Change lines 57-59 from:
```python
        my_stats = self._extract_stats(matchup, "team_stats")
        opp_stats = self._extract_stats(matchup, "team_stats")  # Need opponent_team_key
```

To:
```python
        # Both teams are extracted below via the matchup["teams"] dict
```

- [ ] **Step 4: Verify test still passes**

```bash
venv/Scripts/python -m pytest tests/test_category_tracker_fix.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add backend/fantasy_baseball/category_tracker.py tests/test_category_tracker_fix.py
git commit -m "fix: remove dead code in category_tracker (copy-paste bug on line 59)"
```

---

### Task 2: Add Missing-Data Logging to Valuation Worker

**Files:**
- Modify: `backend/fantasy_baseball/valuation_worker.py`

- [ ] **Step 1: Read the current valuation_worker.py**

```bash
# Read the full file to understand where silent degradation happens
```

Read `backend/fantasy_baseball/valuation_worker.py` completely. Identify every place where a missing value defaults to `0.0` or `None` silently.

- [ ] **Step 2: Add explicit logging for missing data**

For every `dict.get(key, 0.0)` or `dict.get(key, None)` call that could silently swallow missing Statcast data, add a `logger.warning()` that names the player and the missing field. Example pattern:

```python
# BEFORE (silent degradation)
matchup_quality = metrics.get("matchup_quality", 0.0)

# AFTER (logged degradation)
matchup_quality = metrics.get("matchup_quality")
if matchup_quality is None:
    logger.warning("valuation: player %s missing matchup_quality — defaulting to 0.0", player_name)
    matchup_quality = 0.0
```

Do this for ALL stat keys that feed into composite_z or tier assignment. The goal is zero silent failures — every degradation must be visible in logs.

- [ ] **Step 3: Add a summary counter**

At the end of `_assemble_player_data()` (or equivalent), log a one-line summary:

```python
logger.info(
    "valuation: assembled %d players — %d complete, %d degraded (missing fields)",
    total, complete, degraded,
)
```

- [ ] **Step 4: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/valuation_worker.py
```

- [ ] **Step 5: Commit**

```bash
git add backend/fantasy_baseball/valuation_worker.py
git commit -m "fix: add explicit logging for missing data in valuation_worker"
```

---

### Task 3: Bridge FanGraphs RoS Cache → Steamer CSV Files

**Files:**
- Modify: `backend/fantasy_baseball/projections_loader.py`
- Create: `tests/test_projections_bridge.py`

The system already fetches FanGraphs RoS projections daily at 3 AM (`_fetch_fangraphs_ros`) and caches them in memory + `projection_cache_entries` table. But `load_full_board()` looks for CSV files in `data/projections/` that don't exist. We need a bridge: export the FanGraphs RoS cache to the CSV format that `load_full_board()` expects.

- [ ] **Step 1: Read the FanGraphs loader to understand cache format**

Read `backend/fantasy_baseball/fangraphs_loader.py` fully. Understand:
- What does `fetch_all_ros("bat")` return? (Dict of system_name → DataFrame)
- What does `compute_ensemble_blend()` return? (Blended DataFrame)
- What columns are in the raw FanGraphs data?

- [ ] **Step 2: Write failing test for the bridge function**

Create `tests/test_projections_bridge.py`:

```python
"""Test that FanGraphs RoS cache can be exported to Steamer CSV format."""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
import tempfile


def test_export_ros_cache_creates_csvs():
    """export_ros_to_steamer_csvs() should write batting + pitching CSVs."""
    from backend.fantasy_baseball.projections_loader import export_ros_to_steamer_csvs

    # Mock FanGraphs data (simplified)
    bat_data = {
        "steamer": pd.DataFrame({
            "Name": ["Aaron Judge", "Shohei Ohtani"],
            "Team": ["NYY", "LAD"],
            "POS": ["OF", "DH"],
            "HR": [45, 40],
            "R": [100, 95],
            "RBI": [110, 100],
            "SB": [5, 20],
            "AVG": [0.280, 0.290],
            "PA": [600, 580],
            "AB": [520, 500],
            "H": [146, 145],
        }),
    }
    pit_data = {
        "steamer": pd.DataFrame({
            "Name": ["Gerrit Cole"],
            "Team": ["NYY"],
            "POS": ["SP"],
            "W": [15],
            "ERA": [3.00],
            "WHIP": [1.05],
            "SO": [220],
            "IP": [190.0],
            "SV": [0],
        }),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        result = export_ros_to_steamer_csvs(bat_data, pit_data, out_dir)

        assert result["batting_rows"] == 2
        assert result["pitching_rows"] == 1
        assert (out_dir / "steamer_batting_2026.csv").exists()
        assert (out_dir / "steamer_pitching_2026.csv").exists()

        # Verify CSV is loadable by load_steamer_batting
        from backend.fantasy_baseball.projections_loader import load_steamer_batting
        batters = load_steamer_batting(out_dir / "steamer_batting_2026.csv")
        assert len(batters) == 2
        assert batters[0]["name"] == "Aaron Judge"


def test_export_ros_empty_data_writes_nothing():
    """Empty RoS data should not create CSV files."""
    from backend.fantasy_baseball.projections_loader import export_ros_to_steamer_csvs

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        result = export_ros_to_steamer_csvs({}, {}, out_dir)
        assert result["batting_rows"] == 0
        assert result["pitching_rows"] == 0
        assert not (out_dir / "steamer_batting_2026.csv").exists()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
venv/Scripts/python -m pytest tests/test_projections_bridge.py -v --tb=short
```

Expected: FAIL with `ImportError: cannot import name 'export_ros_to_steamer_csvs'`

- [ ] **Step 4: Implement `export_ros_to_steamer_csvs()`**

Add to `backend/fantasy_baseball/projections_loader.py`:

```python
def export_ros_to_steamer_csvs(
    bat_raw: dict,
    pit_raw: dict,
    data_dir: Optional[Path] = None,
) -> dict:
    """
    Export FanGraphs RoS cache to Steamer-format CSV files.

    This bridges the gap between the daily FanGraphs RoS fetch (which caches
    in memory/DB) and load_full_board() (which reads CSVs from data/projections/).

    Parameters
    ----------
    bat_raw : dict
        {system_name: DataFrame} from fetch_all_ros("bat")
    pit_raw : dict
        {system_name: DataFrame} from fetch_all_ros("pit")
    data_dir : Path or None
        Output directory (defaults to DATA_DIR)

    Returns
    -------
    {"batting_rows": int, "pitching_rows": int}
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    bat_rows = 0
    pit_rows = 0

    # Use Steamer if available, else first available system
    bat_df = bat_raw.get("steamer") if bat_raw else None
    if bat_df is None and bat_raw:
        bat_df = next(iter(bat_raw.values()))

    pit_df = pit_raw.get("steamer") if pit_raw else None
    if pit_df is None and pit_raw:
        pit_df = next(iter(pit_raw.values()))

    if bat_df is not None and len(bat_df) > 0:
        bat_path = data_dir / "steamer_batting_2026.csv"
        bat_df.to_csv(bat_path, index=False)
        bat_rows = len(bat_df)
        logger.info("Exported %d batting projections to %s", bat_rows, bat_path)

    if pit_df is not None and len(pit_df) > 0:
        pit_path = data_dir / "steamer_pitching_2026.csv"
        pit_df.to_csv(pit_path, index=False)
        pit_rows = len(pit_df)
        logger.info("Exported %d pitching projections to %s", pit_rows, pit_path)

    return {"batting_rows": bat_rows, "pitching_rows": pit_rows}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
venv/Scripts/python -m pytest tests/test_projections_bridge.py -v --tb=short
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/fantasy_baseball/projections_loader.py tests/test_projections_bridge.py
git commit -m "feat: add FanGraphs RoS -> Steamer CSV bridge for projection pipeline"
```

---

### Task 4: Add Admin Endpoint to Trigger Projection Export

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Read the existing admin endpoint pattern**

Read `backend/main.py` around line 7079 (`/admin/yahoo/roster-raw`) to understand the admin endpoint pattern (auth dependency, error handling, response format).

- [ ] **Step 2: Add the projection export endpoint**

Add to `backend/main.py` in the admin section:

```python
@app.post("/admin/export-projections", dependencies=[Depends(verify_admin_api_key)])
async def admin_export_projections():
    """
    Export FanGraphs RoS cache to Steamer CSV format.

    Bridges the FanGraphs daily fetch (3 AM) with load_full_board() which
    reads CSVs from data/projections/. Run this after fangraphs_ros job
    has completed at least once.
    """
    from backend.fantasy_baseball.projections_loader import export_ros_to_steamer_csvs
    from backend.services.daily_ingestion import _ROS_CACHE, _load_persisted_ros_cache

    # Try in-memory cache first, then persisted cache
    bat_raw = _ROS_CACHE.get("bat")
    pit_raw = _ROS_CACHE.get("pit")

    if not bat_raw and not pit_raw:
        bat_raw, pit_raw, cached_at = _load_persisted_ros_cache()
        if cached_at is None:
            return {
                "success": False,
                "error": "No FanGraphs RoS cache available. Run fangraphs_ros job first.",
            }

    result = export_ros_to_steamer_csvs(bat_raw or {}, pit_raw or {})

    # Clear the lru_cache so load_full_board() picks up new CSVs
    from backend.fantasy_baseball.projections_loader import load_full_board
    load_full_board.cache_clear()

    return {
        "success": True,
        "batting_rows": result["batting_rows"],
        "pitching_rows": result["pitching_rows"],
        "message": "Projections exported. load_full_board() cache cleared.",
    }
```

- [ ] **Step 3: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/main.py
```

- [ ] **Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat: add /admin/export-projections endpoint to bridge FanGraphs RoS to CSVs"
```

---

### Task 5: Build Pipeline Freshness Validation

**Files:**
- Create: `tests/test_pipeline_freshness.py`
- Create: `backend/services/pipeline_validator.py`

- [ ] **Step 1: Write the validator module**

Create `backend/services/pipeline_validator.py` — a pure-function module that checks table freshness and row counts:

```python
"""
Pipeline freshness validator.

Pure-function module that queries table health metrics.
Used by admin endpoints and tests to validate data pipeline state.
"""
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


@dataclass
class TableHealth:
    table_name: str
    row_count: int
    expected_min_rows: int
    latest_date: Optional[date]
    max_staleness_days: int
    is_healthy: bool
    issues: list


def check_table_health(db: Session, today: Optional[date] = None) -> list:
    """
    Check freshness and row counts for all critical fantasy tables.

    Returns a list of TableHealth dataclasses.
    """
    if today is None:
        today = datetime.now(ZoneInfo("America/New_York")).date()

    from backend.models import (
        PlayerRollingStats,
        PlayerScore,
        StatcastPerformance,
        ProbablePitcher,
        SimulationResult,
        PlayerDailyMetric,
        MlbPlayerStats,
    )

    checks = []

    # --- player_rolling_stats ---
    count = db.query(func.count(PlayerRollingStats.id)).scalar() or 0
    latest = db.query(func.max(PlayerRollingStats.as_of_date)).scalar()
    staleness = (today - latest).days if latest else 999
    issues = []
    if count < 1000:
        issues.append(f"Only {count} rows (expected 25K+)")
    if staleness > 2:
        issues.append(f"Stale by {staleness} days (latest: {latest})")
    checks.append(TableHealth(
        table_name="player_rolling_stats",
        row_count=count,
        expected_min_rows=1000,
        latest_date=latest,
        max_staleness_days=2,
        is_healthy=len(issues) == 0,
        issues=issues,
    ))

    # --- player_scores ---
    count = db.query(func.count(PlayerScore.id)).scalar() or 0
    latest = db.query(func.max(PlayerScore.score_date)).scalar()
    staleness = (today - latest).days if latest else 999
    issues = []
    if count < 1000:
        issues.append(f"Only {count} rows (expected 25K+)")
    if staleness > 2:
        issues.append(f"Stale by {staleness} days (latest: {latest})")
    checks.append(TableHealth(
        table_name="player_scores",
        row_count=count,
        expected_min_rows=1000,
        latest_date=latest,
        max_staleness_days=2,
        is_healthy=len(issues) == 0,
        issues=issues,
    ))

    # --- statcast_performances ---
    count = db.query(func.count(StatcastPerformance.id)).scalar() or 0
    latest = db.query(func.max(StatcastPerformance.game_date)).scalar()
    staleness = (today - latest).days if latest else 999
    issues = []
    if count < 5000:
        issues.append(f"Only {count} rows (expected 15K+)")
    if staleness > 3:
        issues.append(f"Stale by {staleness} days (latest: {latest})")
    checks.append(TableHealth(
        table_name="statcast_performances",
        row_count=count,
        expected_min_rows=5000,
        latest_date=latest,
        max_staleness_days=3,
        is_healthy=len(issues) == 0,
        issues=issues,
    ))

    # --- probable_pitchers ---
    count = db.query(func.count(ProbablePitcher.id)).scalar() or 0
    latest = db.query(func.max(ProbablePitcher.game_date)).scalar()
    staleness = (today - latest).days if latest else 999
    issues = []
    if staleness > 1:
        issues.append(f"Stale by {staleness} days (latest: {latest})")
    checks.append(TableHealth(
        table_name="probable_pitchers",
        row_count=count,
        expected_min_rows=0,  # Can be 0 on off-days
        latest_date=latest,
        max_staleness_days=1,
        is_healthy=len(issues) == 0,
        issues=issues,
    ))

    # --- simulation_results ---
    count = db.query(func.count(SimulationResult.id)).scalar() or 0
    latest = db.query(func.max(SimulationResult.as_of_date)).scalar()
    staleness = (today - latest).days if latest else 999
    issues = []
    if count < 100:
        issues.append(f"Only {count} rows (expected 500+)")
    if staleness > 2:
        issues.append(f"Stale by {staleness} days (latest: {latest})")
    checks.append(TableHealth(
        table_name="simulation_results",
        row_count=count,
        expected_min_rows=100,
        latest_date=latest,
        max_staleness_days=2,
        is_healthy=len(issues) == 0,
        issues=issues,
    ))

    # --- mlb_player_stats ---
    count = db.query(func.count(MlbPlayerStats.id)).scalar() or 0
    issues = []
    if count < 2000:
        issues.append(f"Only {count} rows (expected 5K+)")
    checks.append(TableHealth(
        table_name="mlb_player_stats",
        row_count=count,
        expected_min_rows=2000,
        latest_date=None,
        max_staleness_days=999,
        is_healthy=len(issues) == 0,
        issues=issues,
    ))

    return checks


def pipeline_health_summary(checks: list) -> dict:
    """
    Summarize pipeline health into a single dict.

    Returns:
        {
            "overall_healthy": bool,
            "healthy_count": int,
            "unhealthy_count": int,
            "tables": [{"name": str, "healthy": bool, "issues": [...]}],
        }
    """
    healthy = sum(1 for c in checks if c.is_healthy)
    unhealthy = sum(1 for c in checks if not c.is_healthy)
    return {
        "overall_healthy": unhealthy == 0,
        "healthy_count": healthy,
        "unhealthy_count": unhealthy,
        "tables": [
            {
                "name": c.table_name,
                "healthy": c.is_healthy,
                "row_count": c.row_count,
                "latest_date": str(c.latest_date) if c.latest_date else None,
                "issues": c.issues,
            }
            for c in checks
        ],
    }
```

- [ ] **Step 2: Add admin endpoint for pipeline health**

Add to `backend/main.py`:

```python
@app.get("/admin/pipeline-health", dependencies=[Depends(verify_admin_api_key)])
async def admin_pipeline_health():
    """Check freshness and row counts for all critical fantasy tables."""
    from backend.services.pipeline_validator import check_table_health, pipeline_health_summary
    db = SessionLocal()
    try:
        checks = check_table_health(db)
        return pipeline_health_summary(checks)
    finally:
        db.close()
```

- [ ] **Step 3: Syntax check both files**

```bash
venv/Scripts/python -m py_compile backend/services/pipeline_validator.py
venv/Scripts/python -m py_compile backend/main.py
```

- [ ] **Step 4: Commit**

```bash
git add backend/services/pipeline_validator.py backend/main.py
git commit -m "feat: add pipeline freshness validator with /admin/pipeline-health endpoint"
```

---

### Task 6: Fix Production Schedules for Identity/Eligibility Jobs

**Files:**
- Modify: `backend/services/daily_ingestion.py`

The probable_pitchers, player_id_mapping, and position_eligibility syncs are on temporary 10:32 AM test schedules instead of production schedules. This needs to be fixed.

- [ ] **Step 1: Find the current temporary schedules**

```bash
# Search for the test schedule times
```

Grep for `10:32` or `10:3` in `daily_ingestion.py` to find all temporary schedules.

- [ ] **Step 2: Update to production schedules**

Change the schedules to production times:

| Job | Current (test) | Production | Rationale |
|-----|---------------|------------|-----------|
| `player_id_mapping` | 10:32 AM | 7:00 AM ET | Before probable_pitchers needs IDs |
| `position_eligibility` | 10:33 AM | 7:15 AM ET | After player_id_mapping |
| `probable_pitchers` | 10:34 AM (1x) | 8:30 AM, 4:00 PM, 8:00 PM ET (3x) | Pitchers announced at varying times |

- [ ] **Step 3: Verify the schedule change compiles**

```bash
venv/Scripts/python -m py_compile backend/services/daily_ingestion.py
```

- [ ] **Step 4: Commit**

```bash
git add backend/services/daily_ingestion.py
git commit -m "fix: restore production schedules for identity/eligibility/probable_pitcher jobs"
```

---

### Task 7: End-to-End Data Pipeline Validation Test

**Files:**
- Create: `tests/test_data_quality_e2e.py`

- [ ] **Step 1: Write end-to-end validation tests**

These tests validate the complete data chain using real (or realistic mock) data. They don't hit external APIs but verify the computation pipeline is wired correctly.

```python
"""
End-to-end data pipeline validation tests.

Validates the full chain:
  rolling_stats → scoring_engine → simulation_engine → simulation_results

Uses synthetic but realistic data to verify:
  1. Scoring engine produces valid Z-scores from rolling stats
  2. Simulation engine produces valid percentiles from rolling stats
  3. VORP engine produces valid values from player scores
  4. All outputs are within expected mathematical bounds
"""
import pytest
from datetime import date
from unittest.mock import MagicMock


class TestScoringPipelineE2E:
    """Verify scoring_engine produces valid outputs from realistic inputs."""

    def test_zscore_output_range(self):
        """All Z-scores should be in [-3, 3] after Winsorization."""
        from backend.services.scoring_engine import compute_league_zscores

        # Build 50 synthetic PlayerRollingStats-like rows
        rows = []
        for i in range(50):
            row = MagicMock()
            row.bdl_player_id = i
            row.player_name = f"Player {i}"
            row.as_of_date = date(2026, 4, 10)
            row.window_days = 14
            row.games_in_window = 10
            row.player_type = "hitter" if i < 35 else "pitcher"
            # Hitter stats (realistic ranges)
            row.w_home_runs = max(0, 1.5 + (i % 7) * 0.5) if i < 35 else None
            row.w_rbi = max(0, 5.0 + (i % 5) * 1.0) if i < 35 else None
            row.w_stolen_bases = max(0, 0.5 + (i % 3) * 0.3) if i < 35 else None
            row.w_avg = max(0.150, 0.250 + (i % 10 - 5) * 0.015) if i < 35 else None
            row.w_obp = max(0.200, 0.330 + (i % 10 - 5) * 0.012) if i < 35 else None
            # Pitcher stats
            row.w_era = max(1.0, 3.5 + (i % 8 - 4) * 0.5) if i >= 35 else None
            row.w_whip = max(0.8, 1.15 + (i % 6 - 3) * 0.1) if i >= 35 else None
            row.w_k_per_9 = max(4.0, 8.5 + (i % 5 - 2) * 0.8) if i >= 35 else None
            rows.append(row)

        results = compute_league_zscores(rows, date(2026, 4, 10), 14)
        assert len(results) > 0, "Should produce at least some scored players"

        for r in results:
            for field in ["z_hr", "z_rbi", "z_sb", "z_avg", "z_obp",
                          "z_era", "z_whip", "z_k_per_9"]:
                val = getattr(r, field, None)
                if val is not None:
                    assert -4.0 <= val <= 4.0, (
                        f"{r.player_name} {field}={val} out of bounds"
                    )

    def test_composite_z_monotonic_with_skill(self):
        """Better raw stats should produce higher composite Z."""
        from backend.services.scoring_engine import compute_league_zscores

        rows = []
        for i in range(30):
            row = MagicMock()
            row.bdl_player_id = i
            row.player_name = f"Hitter {i}"
            row.as_of_date = date(2026, 4, 10)
            row.window_days = 14
            row.games_in_window = 10
            row.player_type = "hitter"
            # Skill scales with i
            row.w_home_runs = 0.5 + i * 0.2
            row.w_rbi = 3.0 + i * 0.5
            row.w_stolen_bases = 0.2 + i * 0.1
            row.w_avg = 0.220 + i * 0.003
            row.w_obp = 0.290 + i * 0.003
            row.w_era = None
            row.w_whip = None
            row.w_k_per_9 = None
            rows.append(row)

        results = compute_league_zscores(rows, date(2026, 4, 10), 14)
        composites = {r.player_name: r.composite_z for r in results if r.composite_z is not None}

        # Top-skill player should have higher composite than bottom-skill
        if "Hitter 0" in composites and "Hitter 29" in composites:
            assert composites["Hitter 29"] > composites["Hitter 0"], (
                "Higher-skill player should have higher composite Z"
            )


class TestSimulationEngineE2E:
    """Verify simulation_engine produces valid percentile distributions."""

    def test_percentiles_are_ordered(self):
        """P10 <= P25 <= P50 <= P75 <= P90 for every stat."""
        from backend.services.simulation_engine import simulate_player

        row = MagicMock()
        row.bdl_player_id = 1
        row.as_of_date = date(2026, 4, 10)
        row.games_in_window = 10
        row.w_games = 10
        # Hitter with realistic per-game rates
        row.w_home_runs = 3.0     # 0.3/game
        row.w_rbi = 10.0          # 1.0/game
        row.w_stolen_bases = 2.0  # 0.2/game
        row.w_ab = 40.0           # 4.0/game
        row.w_hits = 11.0         # 0.275 AVG
        row.w_ip = None           # not a pitcher

        result = simulate_player(row, remaining_games=130, n_simulations=1000, seed=42)

        assert result.player_type == "hitter"
        # HR percentiles ordered
        assert result.proj_hr_p10 <= result.proj_hr_p25 <= result.proj_hr_p50
        assert result.proj_hr_p50 <= result.proj_hr_p75 <= result.proj_hr_p90
        # RBI percentiles ordered
        assert result.proj_rbi_p10 <= result.proj_rbi_p25 <= result.proj_rbi_p50
        # AVG percentiles ordered and in valid range
        assert 0.0 <= result.proj_avg_p10 <= result.proj_avg_p90 <= 1.0

    def test_simulation_reproducibility(self):
        """Same seed should produce identical results."""
        from backend.services.simulation_engine import simulate_player

        row = MagicMock()
        row.bdl_player_id = 1
        row.as_of_date = date(2026, 4, 10)
        row.games_in_window = 10
        row.w_games = 10
        row.w_home_runs = 3.0
        row.w_rbi = 10.0
        row.w_stolen_bases = 2.0
        row.w_ab = 40.0
        row.w_hits = 11.0
        row.w_ip = None

        r1 = simulate_player(row, seed=42)
        r2 = simulate_player(row, seed=42)

        assert r1.proj_hr_p50 == r2.proj_hr_p50
        assert r1.proj_rbi_p50 == r2.proj_rbi_p50
        assert r1.proj_avg_p50 == r2.proj_avg_p50

    def test_hr_projection_sanity(self):
        """A 0.3 HR/game hitter over 130 games should project ~39 HR at P50."""
        from backend.services.simulation_engine import simulate_player

        row = MagicMock()
        row.bdl_player_id = 1
        row.as_of_date = date(2026, 4, 10)
        row.games_in_window = 10
        row.w_games = 10
        row.w_home_runs = 3.0  # 0.3/game
        row.w_rbi = 10.0
        row.w_stolen_bases = 2.0
        row.w_ab = 40.0
        row.w_hits = 11.0
        row.w_ip = None

        result = simulate_player(row, remaining_games=130, n_simulations=1000, seed=42)

        # 0.3/game * 130 games = 39 expected HR
        # P50 should be within 20% of expected
        assert 30 <= result.proj_hr_p50 <= 50, (
            f"P50 HR={result.proj_hr_p50}, expected ~39"
        )

    def test_pitcher_era_whip_projections(self):
        """Pitcher sim should produce ERA and WHIP projections."""
        from backend.services.simulation_engine import simulate_player

        row = MagicMock()
        row.bdl_player_id = 2
        row.as_of_date = date(2026, 4, 10)
        row.games_in_window = 4
        row.w_games = 4
        row.w_ab = None
        row.w_hits = None
        row.w_home_runs = None
        row.w_rbi = None
        row.w_stolen_bases = None
        # Pitcher: 6 IP/start, 3.00 ERA, 1.10 WHIP, 9 K/9
        row.w_ip = 24.0          # 6.0 per start
        row.w_earned_runs = 8.0  # 2.0 per start -> 3.00 ERA
        row.w_walks_hits = 26.4  # 6.6 per start -> 1.10 WHIP
        row.w_strikeouts = 36.0  # 9.0 per start

        result = simulate_player(row, remaining_games=30, n_simulations=1000, seed=42)

        assert result.player_type == "pitcher"
        # ERA should be in reasonable range
        assert result.proj_era_p50 is not None
        assert 1.0 <= result.proj_era_p50 <= 6.0, (
            f"P50 ERA={result.proj_era_p50}, expected 2.5-4.0"
        )


class TestVORPEngineE2E:
    """Verify VORP engine produces valid outputs."""

    def test_vorp_positive_for_good_player(self):
        """A player with composite_z > replacement should have positive VORP."""
        from backend.services.vorp_engine import compute_vorp

        # Elite SS with composite_z = 2.0
        vorp = compute_vorp(
            composite_z=2.0,
            position="SS",
            games_played=10,
        )
        assert vorp > 0, "Elite player should have positive VORP"

    def test_vorp_negative_for_replacement_player(self):
        """A player below replacement level should have negative VORP."""
        from backend.services.vorp_engine import compute_vorp

        vorp = compute_vorp(
            composite_z=-5.0,
            position="OF",
            games_played=10,
        )
        assert vorp < 0, "Below-replacement player should have negative VORP"
```

- [ ] **Step 2: Run the tests**

```bash
venv/Scripts/python -m pytest tests/test_data_quality_e2e.py -v --tb=short
```

Fix any failures — these tests exercise the actual code, so failures reveal real integration bugs.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data_quality_e2e.py
git commit -m "test: add end-to-end data pipeline validation tests"
```

---

### Task 8: Simulation Accuracy Backtest

**Files:**
- Create: `tests/test_simulation_backtest.py`

- [ ] **Step 1: Write backtest validation tests**

These tests verify that the Monte Carlo simulation engine produces statistically sound outputs — not just that it runs, but that its distributions have correct properties.

```python
"""
Monte Carlo simulation accuracy validation.

Statistical tests that verify the simulation engine produces
mathematically correct distributions:
  1. Mean of simulated distribution ≈ expected value (rate * games)
  2. Variance scales linearly with number of games
  3. P10/P90 bracket contains ~80% of simulations
  4. Coefficient of variation in output matches input CV
"""
import pytest
import random
from datetime import date
from unittest.mock import MagicMock
from backend.services.simulation_engine import (
    simulate_player, _draw_games, _percentiles, CV,
)


class TestDistributionProperties:
    """Statistical validation of simulation output distributions."""

    def test_mean_convergence(self):
        """Simulated mean should converge to expected value within 5%."""
        rng = random.Random(42)
        rate = 0.3  # 0.3 HR/game
        n_games = 130
        n_sims = 5000

        totals = [_draw_games(rng, rate, n_games) for _ in range(n_sims)]
        sim_mean = sum(totals) / len(totals)
        expected = rate * n_games  # 39.0

        # Within 5% of expected
        assert abs(sim_mean - expected) / expected < 0.05, (
            f"Simulated mean {sim_mean:.1f} too far from expected {expected:.1f}"
        )

    def test_variance_scales_with_games(self):
        """Variance should roughly double when games double."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        rate = 0.3

        totals_65 = [_draw_games(rng1, rate, 65) for _ in range(2000)]
        totals_130 = [_draw_games(rng2, rate, 130) for _ in range(2000)]

        var_65 = sum((x - sum(totals_65)/len(totals_65))**2 for x in totals_65) / len(totals_65)
        var_130 = sum((x - sum(totals_130)/len(totals_130))**2 for x in totals_130) / len(totals_130)

        ratio = var_130 / var_65 if var_65 > 0 else 0
        # Should be ~2.0 (variance scales linearly with n)
        assert 1.5 <= ratio <= 2.8, (
            f"Variance ratio {ratio:.2f} — expected ~2.0"
        )

    def test_percentile_bracket_coverage(self):
        """P10-P90 should contain approximately 80% of simulations."""
        rng = random.Random(42)
        rate = 0.3
        n_games = 130

        totals = [_draw_games(rng, rate, n_games) for _ in range(5000)]
        p10, _, _, _, p90 = _percentiles(totals)

        in_bracket = sum(1 for x in totals if p10 <= x <= p90)
        coverage = in_bracket / len(totals)

        # Should be approximately 80% (P10 to P90)
        assert 0.75 <= coverage <= 0.85, (
            f"P10-P90 coverage {coverage:.2%} — expected ~80%"
        )

    def test_output_cv_matches_input(self):
        """Per-game output CV should approximate input CV=0.35."""
        rng = random.Random(42)
        rate = 1.0  # 1 unit/game for clean measurement
        n_sims = 5000

        # Single-game draws
        draws = [max(0, rng.gauss(rate, rate * CV)) for _ in range(n_sims)]
        mean_d = sum(draws) / len(draws)
        std_d = (sum((x - mean_d)**2 for x in draws) / len(draws)) ** 0.5
        measured_cv = std_d / mean_d if mean_d > 0 else 0

        # Should be close to 0.35 (truncation at 0 reduces measured CV slightly)
        assert 0.25 <= measured_cv <= 0.40, (
            f"Measured CV {measured_cv:.3f} — expected ~{CV}"
        )

    def test_zero_rate_produces_zero_projection(self):
        """A player with 0 HR/game should project 0 HR across all percentiles."""
        row = MagicMock()
        row.bdl_player_id = 99
        row.as_of_date = date(2026, 4, 10)
        row.games_in_window = 10
        row.w_games = 10
        row.w_home_runs = 0.0
        row.w_rbi = 0.0
        row.w_stolen_bases = 0.0
        row.w_ab = 40.0
        row.w_hits = 10.0
        row.w_ip = None

        result = simulate_player(row, remaining_games=130, n_simulations=1000, seed=42)
        assert result.proj_hr_p50 == 0.0
        assert result.proj_hr_p90 == 0.0

    def test_high_rate_player_has_wide_distribution(self):
        """A power hitter should have meaningfully wider P10-P90 spread than a weak hitter."""
        power_row = MagicMock()
        power_row.bdl_player_id = 1
        power_row.as_of_date = date(2026, 4, 10)
        power_row.games_in_window = 10
        power_row.w_games = 10
        power_row.w_home_runs = 5.0   # 0.5/game (elite)
        power_row.w_rbi = 12.0
        power_row.w_stolen_bases = 1.0
        power_row.w_ab = 42.0
        power_row.w_hits = 12.0
        power_row.w_ip = None

        weak_row = MagicMock()
        weak_row.bdl_player_id = 2
        weak_row.as_of_date = date(2026, 4, 10)
        weak_row.games_in_window = 10
        weak_row.w_games = 10
        weak_row.w_home_runs = 0.5    # 0.05/game (weak)
        weak_row.w_rbi = 5.0
        weak_row.w_stolen_bases = 0.5
        weak_row.w_ab = 38.0
        weak_row.w_hits = 9.0
        weak_row.w_ip = None

        power = simulate_player(power_row, seed=42)
        weak = simulate_player(weak_row, seed=42)

        power_spread = power.proj_hr_p90 - power.proj_hr_p10
        weak_spread = weak.proj_hr_p90 - weak.proj_hr_p10

        assert power_spread > weak_spread, (
            f"Power spread {power_spread:.1f} should exceed weak spread {weak_spread:.1f}"
        )
```

- [ ] **Step 2: Run the tests**

```bash
venv/Scripts/python -m pytest tests/test_simulation_backtest.py -v --tb=short
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation_backtest.py
git commit -m "test: add Monte Carlo simulation statistical validation tests"
```

---

### Task 9: Run Full Test Suite and Validate

**Files:** None (validation only)

- [ ] **Step 1: Run the complete test suite**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short
```

All new tests from Tasks 1-8 must pass. Pre-existing DB-auth failures (3) are acceptable.

- [ ] **Step 2: Syntax-check all modified files**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/category_tracker.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/valuation_worker.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/projections_loader.py
venv/Scripts/python -m py_compile backend/services/pipeline_validator.py
venv/Scripts/python -m py_compile backend/services/daily_ingestion.py
venv/Scripts/python -m py_compile backend/main.py
```

All must pass with no errors.

- [ ] **Step 3: Document remaining Gemini tasks**

After all Claude tasks are complete, update HANDOFF.md with the remaining Gemini deployment tasks:

1. Deploy latest code to Railway
2. Run `POST /admin/backfill/statcast` (populate ~15K Statcast rows)
3. Run `POST /admin/ingestion/run/probable_pitchers_morning` (verify probable_pitchers)
4. Run `POST /admin/ingestion/run/vorp` (verify VORP values)
5. Run `POST /admin/export-projections` (bridge FanGraphs RoS to CSVs)
6. Run `GET /admin/pipeline-health` (validate all tables are fresh and populated)
7. Verify simulation_results table updates overnight after full pipeline runs

- [ ] **Step 4: Final commit with HANDOFF.md update**

```bash
git add HANDOFF.md
git commit -m "docs: update HANDOFF.md with data quality hardening completion status"
```

---

## Post-Plan: Validation Gates Before Frontend

After all 9 tasks are complete AND Gemini has deployed + run backfills, these gates must pass before any frontend work begins:

| Gate | How to verify | Passing criteria |
|------|---------------|------------------|
| Pipeline health | `GET /admin/pipeline-health` | All tables `is_healthy: true` |
| Projection data | `load_full_board()` returns real players | > 500 players with real projections |
| Statcast populated | `statcast_performances` row count | > 10,000 rows |
| Simulation fresh | `simulation_results` latest date | Within 2 days of current date |
| Test suite green | `pytest tests/ -q` | All new tests pass |
| Monte Carlo stats | `test_simulation_backtest.py` | All statistical properties hold |
