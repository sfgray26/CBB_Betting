# Data Pipeline Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 4 root-cause bugs causing stale/missing data across the fantasy baseball pipeline: APScheduler misfire silently skipping jobs, probable_pitchers getting 0 records due to missing API hydration, probable_pitchers job status never updating per-variant, and the validation audit not detecting zero-quality statcast shell records.

**Architecture:** All fixes are in `backend/services/daily_ingestion.py` and `backend/admin_endpoints_validation.py`. No schema changes. No new files. Each fix is a targeted 1-5 line change with regression tests. The statcast backfill is an operational step (API call), not a code change.

**Tech Stack:** Python 3.11, FastAPI, APScheduler, SQLAlchemy, pytest

**Root Cause Evidence:**
- **Bug 1 (rolling_windows misfire):** APScheduler default `misfire_grace_time=1` second. If the scheduler loop is even slightly delayed, CronTrigger jobs are silently skipped. Manual trigger produces 866 players / 2533 rows flawlessly.
- **Bug 2 (probable_pitchers 0 records):** MLB Stats API `/schedule?hydrate=probablePitcher` does NOT include `team.abbreviation`. The code reads `team.abbreviation` → gets `""` → `if not team_abbr: continue` skips every record. Adding `team` to hydrate returns abbreviation.
- **Bug 3 (variant job status):** `_sync_probable_pitchers` hardcodes `self._record_job_run("probable_pitchers", ...)` regardless of which variant (morning/afternoon/evening) triggered it.
- **Bug 4 (validation blind spot):** The `/admin/validation-audit` endpoint checks row count and date range for statcast_performances but never checks whether quality metrics (xwoba, exit_velocity, barrel_pct) are actually populated vs all-zero shell records.

---

### Task 1: Fix APScheduler Misfire Grace Time

**Files:**
- Modify: `backend/services/daily_ingestion.py:393-394` (constructor)
- Test: `tests/test_ingestion_orchestrator.py`

The default `misfire_grace_time` for APScheduler is 1 second. If the event loop is busy when a CronTrigger fires, the job is silently dropped. Setting it to 300 seconds (5 minutes) ensures jobs fire even if the scheduler is slightly delayed.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion_orchestrator.py`:

```python
def test_scheduler_misfire_grace_time():
    """APScheduler must tolerate delayed execution up to 5 minutes."""
    from backend.services.daily_ingestion import DailyIngestionOrchestrator
    orch = DailyIngestionOrchestrator()
    # The scheduler should be configured with misfire_grace_time > 1 second
    # to prevent silent job skipping when the event loop is busy.
    config = orch._scheduler._job_defaults
    assert config.get("misfire_grace_time", 1) >= 300, (
        f"misfire_grace_time={config.get('misfire_grace_time', 1)}s is too low; "
        "jobs will be silently skipped if the scheduler loop is even 1s late"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_ingestion_orchestrator.py::test_scheduler_misfire_grace_time -v`
Expected: FAIL — default misfire_grace_time is 1

- [ ] **Step 3: Implement the fix**

In `backend/services/daily_ingestion.py`, change the constructor (line ~394):

```python
# BEFORE:
self._scheduler = AsyncIOScheduler()

# AFTER:
self._scheduler = AsyncIOScheduler(
    job_defaults={"misfire_grace_time": 300},
)
```

This gives every CronTrigger job a 5-minute window to fire after its scheduled time before being considered "misfired" and skipped.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_ingestion_orchestrator.py::test_scheduler_misfire_grace_time -v`
Expected: PASS

- [ ] **Step 5: Syntax check**

Run: `venv/Scripts/python -m py_compile backend/services/daily_ingestion.py`
Expected: No output (clean compile)

- [ ] **Step 6: Commit**

```bash
git add backend/services/daily_ingestion.py tests/test_ingestion_orchestrator.py
git commit -m "fix: set APScheduler misfire_grace_time=300s to prevent silent job skipping"
```

---

### Task 2: Fix Probable Pitchers API Hydration

**Files:**
- Modify: `backend/services/daily_ingestion.py:4362` (hydrate parameter)
- Test: `tests/test_ingestion_orchestrator.py`

The MLB Stats API's `/schedule` endpoint only includes `team.abbreviation` when `team` is in the `hydrate` parameter. Without it, every record is skipped because `team_abbr` is empty.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion_orchestrator.py`:

```python
@pytest.mark.asyncio
async def test_probable_pitchers_hydrate_includes_team():
    """MLB Stats API request must hydrate both probablePitcher and team."""
    from backend.services.daily_ingestion import DailyIngestionOrchestrator
    import requests

    orch = DailyIngestionOrchestrator()
    captured_params = {}

    original_get = requests.get

    def _capture_get(url, params=None, **kwargs):
        if "statsapi.mlb.com" in url:
            captured_params.update(params or {})
        # Return empty response to short-circuit
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"dates": []}
        return mock_resp

    with patch("requests.get", side_effect=_capture_get):
        with patch.object(orch, "_record_job_run"):
            try:
                await orch._sync_probable_pitchers()
            except Exception:
                pass  # Advisory lock may fail without DB

    hydrate_value = captured_params.get("hydrate", "")
    assert "team" in hydrate_value, (
        f"hydrate={hydrate_value!r} is missing 'team'. "
        "MLB Stats API won't return team.abbreviation without it."
    )
    assert "probablePitcher" in hydrate_value, (
        f"hydrate={hydrate_value!r} is missing 'probablePitcher'."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_ingestion_orchestrator.py::test_probable_pitchers_hydrate_includes_team -v`
Expected: FAIL — current hydrate is `"probablePitcher"` only

- [ ] **Step 3: Implement the fix**

In `backend/services/daily_ingestion.py`, find line ~4362:

```python
# BEFORE:
params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher"},

# AFTER:
params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team"},
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_ingestion_orchestrator.py::test_probable_pitchers_hydrate_includes_team -v`
Expected: PASS

- [ ] **Step 5: Syntax check**

Run: `venv/Scripts/python -m py_compile backend/services/daily_ingestion.py`
Expected: No output (clean compile)

- [ ] **Step 6: Commit**

```bash
git add backend/services/daily_ingestion.py tests/test_ingestion_orchestrator.py
git commit -m "fix: add 'team' to MLB Stats API hydrate so probable_pitchers gets abbreviations"
```

---

### Task 3: Fix Probable Pitchers Job Status Recording Per Variant

**Files:**
- Modify: `backend/services/daily_ingestion.py:4302-4483` (_sync_probable_pitchers)
- Test: `tests/test_ingestion_orchestrator.py`

Currently `_sync_probable_pitchers` always calls `self._record_job_run("probable_pitchers", ...)` regardless of which schedule variant triggered it. The morning/afternoon/evening variants always show `None/None` in status.

The fix: add an optional `job_id` parameter that defaults to `"probable_pitchers"` but can be overridden. Wire each scheduled variant to pass its own ID. But this is complex because all three schedule entries point to the same method.

Simpler approach: after the shared `_sync_probable_pitchers` records to `"probable_pitchers"`, also copy that status to all variant keys in `_job_status`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion_orchestrator.py`:

```python
def test_probable_pitchers_status_propagates_to_variants():
    """After probable_pitchers runs, morning/afternoon/evening variants must reflect the status."""
    from backend.services.daily_ingestion import DailyIngestionOrchestrator

    orch = DailyIngestionOrchestrator()
    # Simulate status dict initialization (normally done in start())
    for jid in ["probable_pitchers", "probable_pitchers_morning",
                "probable_pitchers_afternoon", "probable_pitchers_evening"]:
        orch._job_status[jid] = {
            "name": jid, "enabled": True,
            "last_run": None, "last_status": None, "next_run": None,
        }

    # Record a run for "probable_pitchers"
    orch._record_job_run("probable_pitchers", "success", 42)

    # All three variants should now reflect the same status
    for variant in ["probable_pitchers_morning", "probable_pitchers_afternoon",
                    "probable_pitchers_evening"]:
        assert orch._job_status[variant]["last_status"] == "success", (
            f"{variant} status is {orch._job_status[variant]['last_status']!r}, expected 'success'"
        )
        assert orch._job_status[variant]["last_run"] is not None, (
            f"{variant} last_run is None after probable_pitchers ran"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_ingestion_orchestrator.py::test_probable_pitchers_status_propagates_to_variants -v`
Expected: FAIL — variants stay at `None`

- [ ] **Step 3: Implement the fix**

In `backend/services/daily_ingestion.py`, modify `_record_job_run` (line ~747):

```python
# BEFORE:
def _record_job_run(self, job_id: str, status: str, records: int = 0) -> None:
    """Update in-memory job status after a run."""
    self._job_status[job_id] = {
        "name": job_id,
        "enabled": True,
        "last_run": now_et().isoformat(),
        "last_status": status,
        "next_run": self._get_next_run(job_id),
    }

# AFTER:
def _record_job_run(self, job_id: str, status: str, records: int = 0) -> None:
    """Update in-memory job status after a run."""
    run_info = {
        "name": job_id,
        "enabled": True,
        "last_run": now_et().isoformat(),
        "last_status": status,
        "next_run": self._get_next_run(job_id),
    }
    self._job_status[job_id] = run_info

    # Propagate probable_pitchers status to all schedule variants
    if job_id == "probable_pitchers":
        for variant in ("probable_pitchers_morning",
                        "probable_pitchers_afternoon",
                        "probable_pitchers_evening"):
            if variant in self._job_status:
                self._job_status[variant]["last_run"] = run_info["last_run"]
                self._job_status[variant]["last_status"] = run_info["last_status"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_ingestion_orchestrator.py::test_probable_pitchers_status_propagates_to_variants -v`
Expected: PASS

- [ ] **Step 5: Syntax check**

Run: `venv/Scripts/python -m py_compile backend/services/daily_ingestion.py`
Expected: No output (clean compile)

- [ ] **Step 6: Commit**

```bash
git add backend/services/daily_ingestion.py tests/test_ingestion_orchestrator.py
git commit -m "fix: propagate probable_pitchers status to morning/afternoon/evening variants"
```

---

### Task 4: Add Statcast Quality Metrics Check to Validation Audit

**Files:**
- Modify: `backend/admin_endpoints_validation.py:210-267` (Section 4)
- Test: `tests/test_admin_validation_audit.py`

The validation audit checks statcast row count and date range but never detects zero-quality shell records (rows where exit_velocity_avg=0, xwoba=0, xba=0, etc.). This is the exact bug that let 6,255 broken records go undetected.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_admin_validation_audit.py`:

```python
def test_validation_audit_detects_zero_quality_statcast(mock_db_session):
    """Audit must flag statcast rows where all quality metrics are zero."""
    # The actual test depends on the existing test infrastructure in this file.
    # If the file uses a mock_db_session fixture, use it; otherwise create one.
    #
    # The key assertion: when statcast_performances has rows but >50% have
    # all-zero quality metrics, the audit must raise a HIGH finding.
    from backend.admin_endpoints_validation import router
    # We need to verify the SQL query exists in the endpoint code.
    import inspect
    source = inspect.getsource(router.routes[-1].endpoint)  # validation_audit
    assert "exit_velocity_avg" in source or "quality_metrics" in source, (
        "validation_audit does not check statcast quality metrics (xwoba, exit_velocity, etc.)"
    )
```

Note: This test may need adaptation based on the existing test patterns in `tests/test_admin_validation_audit.py`. The goal is to verify the audit code *contains* the quality check. A stronger integration test would mock the DB to return rows with zero metrics and verify the finding is raised.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_admin_validation_audit.py::test_validation_audit_detects_zero_quality_statcast -v`
Expected: FAIL — current code doesn't check quality metrics

- [ ] **Step 3: Implement the fix**

In `backend/admin_endpoints_validation.py`, after the existing statcast row count check (around line 266), add a quality metrics check:

```python
        # Check 4.2: Statcast quality metrics (detect zero-filled shell records)
        # This catches the column-mapping bug where rows are created but
        # exit_velocity_avg, xwoba, xba, barrel_pct are all zero.
        result = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(*) FILTER (WHERE exit_velocity_avg = 0
                    AND xwoba = 0 AND xba = 0 AND xslg = 0
                    AND hard_hit_pct = 0 AND barrel_pct = 0) as zero_quality_rows
            FROM statcast_performances
        """)).fetchone()

        if result.total_rows > 0:
            zero_pct = result.zero_quality_rows / result.total_rows * 100
            if zero_pct > 50:
                add_finding("high", "Data Quality", "statcast_performances",
                    f"{result.zero_quality_rows}/{result.total_rows} rows ({zero_pct:.0f}%) have ALL-ZERO quality metrics "
                    "(exit_velocity, xwoba, xba, barrel_pct all = 0). Likely column mapping bug.",
                    "Run POST /admin/backfill/statcast to re-ingest with corrected column mapping.",
                    "SELECT COUNT(*) FROM statcast_performances WHERE exit_velocity_avg = 0 AND xwoba = 0 AND xba = 0")
            elif zero_pct > 20:
                add_finding("medium", "Data Quality", "statcast_performances",
                    f"{result.zero_quality_rows}/{result.total_rows} rows ({zero_pct:.0f}%) have zero quality metrics.",
                    "Some players may legitimately have zero Statcast data (e.g., pinch runners). "
                    "If >20%, consider re-running backfill.",
                    None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_admin_validation_audit.py::test_validation_audit_detects_zero_quality_statcast -v`
Expected: PASS

- [ ] **Step 5: Syntax check**

Run: `venv/Scripts/python -m py_compile backend/admin_endpoints_validation.py`
Expected: No output (clean compile)

- [ ] **Step 6: Commit**

```bash
git add backend/admin_endpoints_validation.py tests/test_admin_validation_audit.py
git commit -m "fix: add statcast quality metrics check to validation audit"
```

---

### Task 5: Run Full Test Suite and Verify

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `venv/Scripts/python -m pytest tests/ -q --tb=short`
Expected: 1738+ passed, 0 new failures (3 pre-existing DB-auth failures are OK)

- [ ] **Step 2: Verify pipeline health (post-manual-trigger)**

The rolling_windows, player_scores, and ros_simulation jobs were already manually triggered during investigation. Verify the pipeline is healthy:

Run:
```bash
curl -s -H "X-API-Key: $API_KEY" "https://fantasy-app-production-5079.up.railway.app/admin/pipeline-health"
```
Expected: `overall_healthy: true` (or all tables showing fresh dates)

- [ ] **Step 3: Run validation audit**

Run:
```bash
curl -s -H "X-API-Key: $API_KEY" "https://fantasy-app-production-5079.up.railway.app/admin/validation-audit"
```

Verify: No new CRITICAL or HIGH findings. The statcast quality check (Task 4) will start reporting once deployed.

---

### Task 6: Post-Deploy Operational Steps (Gemini Delegation)

These are NOT code changes — they are operational steps to run after deployment:

- [ ] **Step 1: Deploy latest code to Railway**

This includes Tasks 1-4 fixes.

- [ ] **Step 2: Run statcast backfill**

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  "https://fantasy-app-production-5079.up.railway.app/admin/backfill/statcast"
```

Expected: ~15K rows with real quality metrics for March 20 - April 13.

- [ ] **Step 3: Trigger probable_pitchers after deploy**

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  "https://fantasy-app-production-5079.up.railway.app/admin/ingestion/run/probable_pitchers_morning"
```

Expected: Non-zero records (should be ~14-20 records for today's slate).

- [ ] **Step 4: Verify rolling_windows fires at 3 AM automatically**

Check the next morning:
```bash
curl -s -H "X-API-Key: $API_KEY" \
  "https://fantasy-app-production-5079.up.railway.app/admin/ingestion/status" | \
  python -c "import sys,json; j=json.load(sys.stdin)['jobs']['rolling_windows']; print(j['last_run'], j['last_status'])"
```

Expected: Shows today's date with `success` status.

- [ ] **Step 5: Run validation audit and verify statcast quality check works**

```bash
curl -s -H "X-API-Key: $API_KEY" \
  "https://fantasy-app-production-5079.up.railway.app/admin/validation-audit"
```

Expected: If shell records still exist pre-backfill, should show HIGH finding for zero-quality rows. After backfill, should drop to 0 or low percentage.
