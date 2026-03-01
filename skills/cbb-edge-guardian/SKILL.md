# Skill: CBB Edge Guardian
Expertise in monitoring, validating, and recalibrating the College Basketball Edge Analyzer model.

## Core Mandates
- **Conservative Bias:** Always prioritize model stability and CLV (Closing Line Value) over volume.
- **Data Integrity:** Verify that database fetches and model runs are current before performing analysis.
- **Audit First:** Always perform a "dry-run" recalibration before applying parameter changes.

## Available Actions

### `check_performance`
Analyze recent betting results and model accuracy.
- **Workflow:** 
  1. Execute `python -c "from backend.main import SessionLocal; from backend.services.performance import calculate_summary_stats; db=SessionLocal(); print(calculate_summary_stats(db)); db.close()"`
  2. Parse ROI, Win Rate, and Mean CLV.
  3. Alert if CLV status is 'WARNING' or 'STOP'.

### `audit_recalibration`
Check if the model's Home Advantage or SD Multiplier needs adjustment.
- **Workflow:**
  1. Execute `python -c "from backend.main import SessionLocal; from backend.services.recalibration import run_recalibration; db=SessionLocal(); print(run_recalibration(db, apply_changes=False)); db.close()"`
  2. Report any suggested bias corrections for `home_advantage` or `sd_multiplier`.

### `verify_code_integrity`
Run the test suite to ensure no regressions were introduced by new builds.
- **Workflow:**
  1. Run `pytest tests/test_betting_model.py -v`
  2. Run `pytest tests/test_portfolio.py -v`
  3. Report any failures immediately.

## Monitoring Strategy (Heartbeat)
- **Daily @ 4:30 AM ET:** Run `check_performance` and `audit_recalibration`. 
- **Trigger:** If `mean_clv < 0.00`, notify the user immediately via the primary channel.
