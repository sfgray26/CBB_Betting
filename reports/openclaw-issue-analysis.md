# OpenClaw Issue Analysis & Optimization Path

**Date:** 2026-03-06  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Status:** EMAC-044 Follow-up — Operational Audit

---

## Executive Summary

**OpenClaw is FUNCTIONAL but INCOMPLETE.**

The core infrastructure works (O-7 validation passed 4/4), but there are **critical gaps** in:
1. **Memory/Logging** — No `memory/` directory exists; no operational trace
2. **HANDOFF.md Updates** — No mechanism for OpenClaw to write findings
3. **HEARTBEAT.md Execution** — Loops defined but not triggered
4. **Self-Monitoring** — No telemetry on actual runtime performance

---

## Issue Breakdown

### Issue 1: Missing Memory Directory
**Status:** CRITICAL — No operational persistence

**Evidence:**
```powershell
# Glob pattern 'memory/*.md' returned NO MATCHES
# Directory 'memory/' does not exist
```

**Impact:**
- OpenClaw has nowhere to write nightly health check summaries
- No historical record of integrity sweep results
- No trace for debugging operational issues
- Cannot track calibration drift over time

**Required by HEARTBEAT.md:**
> "5. Write summary to `memory/YYYY-MM-DD.md` (today's date)."

**Fix:** Create `memory/` directory + JSON state tracker

---

### Issue 2: No HANDOFF.md Write Mechanism
**Status:** HIGH — Breaks escalation chain

**Evidence:**
- `sentinel.py` returns `summary` dict but never writes to HANDOFF.md
- `analysis.py` runs integrity sweep but doesn't log VOLATILE/ABORT counts to HANDOFF
- No file I/O functions for markdown updates in any OpenClaw-related file

**Required by AGENTS.md (OpenClaw section):**
> "> 20% of BET-tier games return VOLATILE → log `SYSTEM_RISK_ELEVATED` warning"  
> "> 1 ABORT in a single slate → surface in Morning Briefing as priority alert"

**Current State:** Warnings go to `logger.warning()` only — lost in logs

**Fix:** Add `update_handoff()` function to sentinel.py + coordinator

---

### Issue 3: HEARTBEAT.md Loops Not Triggered
**Status:** HIGH — Defined but dormant

**Evidence:**
| Loop | Defined In | Triggered By | Actual Trigger |
|------|------------|--------------|----------------|
| Integrity Sweep | HEARTBEAT.md:9 | `run_nightly_analysis()` Pass 1 | ✅ YES — `analysis.py:942` |
| Nightly Health Check | HEARTBEAT.md:49 | 4:30 AM ET / post-analysis | ❌ NO — not scheduled |
| Weekly Calibration | HEARTBEAT.md:75 | Monday 6 AM ET | ❌ NO — not scheduled |

**The Problem:**
The integrity sweep IS running (inside `analysis.py`), but:
1. No results are persisted
2. No telemetry on execution time, success rate, or verdict distribution
3. Health checks and calibration reviews are NEVER triggered

**Fix:** Add APScheduler jobs in `sentinel.py` or new `scripts/openclaw_scheduler.py`

---

### Issue 4: Integrity Sweep Telemetry Gap
**Status:** MEDIUM — Flying blind on performance

**Current `analysis.py` Implementation:**
```python
# Line 942
_integrity_results = await _integrity_sweep(_sweep_inputs)
# Results used immediately, never logged
```

**Missing Metrics:**
- How many games were checked?
- How long did it take?
- What verdicts were returned (CONFIRMED/CAUTION/VOLATILE/ABORT)?
- Any DDGS rate limit errors?
- Circuit breaker state?

**Required for Optimization:**
Need data to tune `max_concurrent` (currently 8) and identify failure patterns.

**Fix:** Add structured logging to `.openclaw/sweeps/YYYY-MM-DD.jsonl`

---

### Issue 5: No Tiered Escalation Wiring (O-9)
**Status:** MEDIUM — Spec'd but not implemented

**From HANDOFF.md Section 6:**
> "When integrity_verdict contains 'VOLATILE' OR recommended_units >= 1.5 OR tournament_round >= 4 → route to Kimi for second opinion"

**Current State:**
- `coordinator.py` has routing logic
- `analysis.py` does NOT call coordinator — uses legacy `perform_sanity_check()` directly

**Evidence:**
```python
# analysis.py line 617
return perform_sanity_check(home, away, verdict, context)
# ^ Uses scout.py directly, not coordinator.py
```

**Fix:** Replace direct `perform_sanity_check()` calls with coordinator routing

---

## Root Cause Analysis

```
OpenClaw Architecture Gap
├── Infrastructure: ✅ Implemented (coordinator, config, circuit breaker)
├── Execution: ⚠️ Partial (integrity sweep runs, but no telemetry)
├── Persistence: ❌ Missing (no memory/, no HANDOFF writes)
├── Scheduling: ❌ Missing (health checks never triggered)
└── Integration: ⚠️ Partial (analysis.py bypasses coordinator)
```

**Fundamental Issue:** OpenClaw was designed as a **library** (called by analysis.py), not an **autonomous agent** (self-triggering, self-reporting).

---

## Optimization Path

### Phase 1: Infrastructure (Immediate — EMAC-045)

**1.1 Create Memory System**
```bash
mkdir memory/
touch memory/.gitkeep
```

**1.2 Add State Tracker** (`.openclaw/state_tracker.py`)
```python
"""OpenClaw operational state tracking."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

STATE_FILE = ".openclaw/operational-state.json"

def log_sweep_results(slate_date: str, results: Dict[str, str], metadata: Dict[str, Any]):
    """Log integrity sweep results with telemetry."""
    Path(".openclaw/sweeps").mkdir(exist_ok=True)
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "slate_date": slate_date,
        "games_checked": len(results),
        "verdict_counts": _count_verdicts(results),
        "latency_ms": metadata.get("latency_ms"),
        "circuit_breaker_state": metadata.get("circuit_breaker_state"),
        "results": results
    }
    
    with open(f".openclaw/sweeps/{slate_date}.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

def update_health_status(check_type: str, status: str, details: Dict):
    """Update operational state for heartbeat queries."""
    state = _load_state()
    state["last_checks"][check_type] = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "details": details
    }
    _save_state(state)

def _load_state() -> Dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_checks": {}, "version": "1.0"}

def _save_state(state: Dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
```

**1.3 Create HANDOFF.md Updater** (`.openclaw/handoff_integration.py`)
```python
"""Safe HANDOFF.md updates from OpenClaw."""
import re
from datetime import datetime
from pathlib import Path

def update_openclaw_section(
    last_sweep: dict = None,
    alerts: list = None,
    circuit_state: str = None
):
    """Update OpenClaw section in HANDOFF.md without overwriting other content."""
    handoff_path = Path("HANDOFF.md")
    content = handoff_path.read_text()
    
    # Build update block
    update_block = f"""### OpenClaw Status (Auto-Updated {datetime.now().strftime('%Y-%m-%d %H:%M')})

| Metric | Value |
|--------|-------|
| Last Sweep | {last_sweep.get('timestamp', 'N/A') if last_sweep else 'N/A'} |
| Games Checked | {last_sweep.get('games_checked', 0) if last_sweep else 0} |
| Circuit Breaker | {circuit_state or 'UNKNOWN'} |

**Active Alerts:**
"""
    if alerts:
        for alert in alerts:
            update_block += f"- 🚨 {alert}\n"
    else:
        update_block += "- None\n"
    
    # Replace or insert OpenClaw section
    pattern = r'### OpenClaw Status \(Auto-Updated.*?(?=###|\Z)'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, update_block.rstrip(), content, flags=re.DOTALL)
    else:
        # Insert before Section 2 (System Context)
        content = content.replace(
            "## 2. SYSTEM CONTEXT",
            update_block + "\n\n## 2. SYSTEM CONTEXT"
        )
    
    handoff_path.write_text(content)
```

---

### Phase 2: Scheduler (EMAC-046)

**2.1 Create OpenClaw Scheduler** (`scripts/openclaw_scheduler.py`)
```python
"""OpenClaw autonomous execution scheduler."""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

from backend.services.sentinel import run_nightly_health_check
from backend.services.scout import generate_morning_briefing_narrative
from .openclaw.state_tracker import update_health_status

logger = logging.getLogger("openclaw.scheduler")

scheduler = BackgroundScheduler()

def schedule_openclaw_jobs():
    """Schedule all OpenClaw heartbeat jobs."""
    
    # Nightly Health Check — 4:30 AM ET
    scheduler.add_job(
        func=_run_health_check_with_logging,
        trigger=CronTrigger(hour=4, minute=30),
        id="nightly_health_check",
        name="OpenClaw Nightly Health Check",
        replace_existing=True,
    )
    
    # Weekly Calibration Review — Monday 6 AM ET
    scheduler.add_job(
        func=_run_calibration_review,
        trigger=CronTrigger(day_of_week="mon", hour=6, minute=0),
        id="weekly_calibration_review",
        name="OpenClaw Weekly Calibration Review",
        replace_existing=True,
    )
    
    # Integrity Sweep Summary — Daily 7 AM ET (post-analysis)
    scheduler.add_job(
        func=_summarize_integrity_sweeps,
        trigger=CronTrigger(hour=7, minute=0),
        id="integrity_sweep_summary",
        name="OpenClaw Integrity Sweep Summary",
        replace_existing=True,
    )
    
    scheduler.start()
    logger.info("OpenClaw scheduler started with %d jobs", len(scheduler.get_jobs()))

def _run_health_check_with_logging():
    """Wrapper that logs results."""
    try:
        summary = run_nightly_health_check()
        update_health_status("nightly_health_check", "completed", summary)
        
        # Update HANDOFF.md if issues found
        alerts = []
        if summary.get("portfolio", {}).get("status") == "YELLOW":
            alerts.append(f"Portfolio drawdown at {summary['portfolio']['current_drawdown_pct']:.1%}")
        if summary.get("performance", {}).get("status") == "RED":
            alerts.append(f"MAE critical: {summary['performance']['mean_mae']}")
        if summary.get("system", {}).get("status") == "RED":
            alerts.append("Pytest failures detected")
            
        if alerts:
            from .openclaw.handoff_integration import update_openclaw_section
            update_openclaw_section(alerts=alerts)
            
    except Exception as e:
        logger.error("Health check failed: %s", e)
        update_health_status("nightly_health_check", "failed", {"error": str(e)})

def _run_calibration_review():
    """Weekly calibration drift detection."""
    # Implementation TBD — query model_parameters, compare to baselines
    pass

def _summarize_integrity_sweeps():
    """Summarize yesterday's integrity sweep results."""
    # Read .openclaw/sweeps/ files, generate summary
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    schedule_openclaw_jobs()
    
    # Keep running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
```

---

### Phase 3: Integration (EMAC-047)

**3.1 Wire Coordinator to Analysis.py**

Replace lines 604-619 in `analysis.py`:
```python
# OLD: Direct scout.py call
return perform_sanity_check(home, away, verdict, context)

# NEW: Coordinator routing with telemetry
from .openclaw.coordinator import get_coordinator, TaskContext, TaskType

coordinator = get_coordinator()
ctx = TaskContext(
    home_team=home,
    away_team=away,
    recommended_units=game.get('edge', 0),
    tournament_round=game.get('tournament_round')
)

result = await coordinator.route_task(
    task_type=TaskType.INTEGRITY_CHECK,
    context=ctx,
    prompt=build_integrity_prompt(home, away, verdict, context)
)

# Log for telemetry
coordinator.log_usage(result, TaskType.INTEGRITY_CHECK, ctx)

if result.output == "ESCALATE_TO_KIMI":
    # Handle escalation — queue for Kimi review
    _queue_kimi_escalation(game, ctx)
    return "KIMI_ESCALATION_PENDING"

return result.output if result.success else "Sanity check unavailable"
```

**3.2 Add Integrity Sweep Telemetry**

In `_integrity_sweep()` after line 643:
```python
from .openclaw.state_tracker import log_sweep_results

log_sweep_results(
    slate_date=datetime.now().strftime("%Y-%m-%d"),
    results=results_dict,
    metadata={
        "latency_ms": latency_ms,
        "circuit_breaker_state": get_coordinator().circuit_breaker.state,
        "concurrency": 8
    }
)
```

---

## Quick Fixes (Can Do Now)

### Fix A: Create Memory Directory
```powershell
mkdir memory
```

### Fix B: Add OpenClaw Status Section to HANDOFF.md
```markdown
### OpenClaw Status (Auto-Updated)

| Metric | Value |
|--------|-------|
| Last Sweep | Not yet run |
| Games Checked | 0 |
| Circuit Breaker | CLOSED |

**Active Alerts:**
- Scheduler not yet activated (manual runs only)
```

### Fix C: Create Operational State File
```powershell
'{"version": "1.0", "last_checks": {}, "first_run": "2026-03-06"}' | Out-File -FilePath ".openclaw/operational-state.json"
```

---

## Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| Memory directory | ❌ Missing | ✅ Created |
| Sweep telemetry | ❌ None | ✅ JSONL per day |
| HANDOFF updates | ❌ Manual | ✅ Auto on alerts |
| Health checks | ❌ Never run | ✅ Daily at 4:30 AM |
| Coordinator usage | ❌ Bypassed | ✅ Integrated |
| Escalation to Kimi | ❌ Not wired | ✅ O-9 complete |

---

## Recommendation

**Immediate (Today):**
1. Create `memory/` directory
2. Add OpenClaw status section to HANDOFF.md
3. Create `.openclaw/operational-state.json`

**Short-term (This Week):**
1. Implement `state_tracker.py`
2. Implement `handoff_integration.py`
3. Wire telemetry to `_integrity_sweep()`

**Medium-term (Pre-Tournament):**
1. Implement `openclaw_scheduler.py`
2. Wire coordinator to `analysis.py`
3. Complete O-9 tiered escalation

---

**Analyst:** Kimi CLI  
**Confidence:** High — based on direct code inspection  
**Urgency:** Medium — tournament phase needs full autonomy
