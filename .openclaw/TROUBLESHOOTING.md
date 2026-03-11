# OpenClaw Troubleshooting Guide v3.0

**Last Updated:** 2026-03-11  
**Maintainer:** Kimi CLI (OpenClaw Config Owner)  
**Scope:** OpenClaw Lite v3.0 (heuristic-based)

---

## Quick Diagnostics

Run the health check first:

```bash
python .openclaw/health_check.py --verbose
```

---

## Common Issues

### Issue 1: Integrity Checks Returning "Sanity check unavailable"

**Symptoms:**
```
All integrity checks return "Sanity check unavailable"
```

**Root Cause:**
DDGS (DuckDuckGo Search) module not installed or import error.

**Diagnosis:**
```bash
python -c "from duckduckgo_search import DDGS; print('OK')"
```

**Resolution:**
```bash
pip install duckduckgo-search>=5.0
```

---

### Issue 2: High Latency in Integrity Sweep

**Symptoms:**
- Analysis taking >30 seconds for slate of games
- Timeouts in integrity checks

**Root Cause:**
DDGS rate limiting or slow network.

**Diagnosis:**
Check health check performance numbers:
```bash
python .openclaw/health_check.py
```

Look for:
- Heuristic Performance should be <1ms
- If higher, DDGS is the bottleneck

**Resolution:**

1. **Reduce concurrency** (in `backend/services/analysis.py`):
```python
semaphore = asyncio.Semaphore(4)  # Reduce from 8
```

2. **Add delays between batches:**
```python
import asyncio
await asyncio.sleep(1.0)  # Between game batches
```

3. **Check DDGS status:**
```python
python -c "
from duckduckgo_search import DDGS
import time
start = time.time()
with DDGS() as ddgs:
    list(ddgs.text('test', max_results=3))
print(f'DDGS latency: {time.time()-start:.2f}s')
"
```

---

### Issue 3: No High-Stakes Escalations Created

**Symptoms:**
- High-stakes games (≥1.5u) not appearing in escalation queue
- `.openclaw/escalation_queue/` is empty

**Root Cause:**
Missing `game_key` parameter in async calls.

**Diagnosis:**
Check your code:
```python
# WRONG - Won't escalate
await async_perform_sanity_check(
    home_team="Duke",
    away_team="UNC",
    verdict="Bet 2.0u Duke -4",
    search_results="..."
    # Missing game_key!
)

# CORRECT - Will escalate
await async_perform_sanity_check(
    home_team="Duke",
    away_team="UNC",
    verdict="Bet 2.0u Duke -4",
    search_results="...",
    game_key="UNC@Duke"  # Required for escalation
)
```

**Resolution:**
Update all calls to include `game_key`:
```python
game_key = f"{away_team}@{home_team}"
verdict = await async_perform_sanity_check(..., game_key=game_key)
```

---

### Issue 4: Telemetry Showing 0 Checks

**Symptoms:**
```python
checker.get_telemetry()
# Returns: {'total_checks': 0, ...}
```

**Root Cause:**
Using sync method instead of async, or telemetry disabled.

**Diagnosis:**
```python
from backend.services.openclaw_lite import get_openclaw_lite

checker = get_openclaw_lite()
print(checker.telemetry)  # None if disabled

# Check if enabled
checker = get_openclaw_lite(enable_telemetry=True)
```

**Resolution:**
1. Enable telemetry:
```python
checker = get_openclaw_lite(enable_telemetry=True)
```

2. Use async API (telemetry only records on async calls):
```python
# Won't record telemetry
checker._check_integrity_heuristic_sync(...)

# Will record telemetry
await checker.check_integrity(...)
```

---

### Issue 5: Escalation Queue Directory Growing Too Large

**Symptoms:**
- `.openclaw/escalation_queue/` contains thousands of files
- Disk usage concerns

**Root Cause:**
Resolved escalations not cleaned up.

**Resolution:**

1. **Archive old escalations:**
```bash
# Archive escalations older than 30 days
find .openclaw/escalation_queue/ -name "*.json" -mtime +30 -exec mv {} .openclaw/escalation_archive/ \;
```

2. **Auto-cleanup script** (add to cron/scheduler):
```python
# cleanup_escalations.py
from pathlib import Path
from datetime import datetime, timedelta

queue_dir = Path(".openclaw/escalation_queue")
cutoff = datetime.now() - timedelta(days=30)

for f in queue_dir.glob("*.json"):
    # Parse timestamp from filename (format: YYYYMMDD_HHMMSS_...)
    try:
        timestamp_str = f.stem.split('_')[0] + '_' + f.stem.split('_')[1]
        file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        if file_time < cutoff:
            f.unlink()
            print(f"Deleted {f.name}")
    except:
        pass
```

---

### Issue 6: Tests Failing

**Symptoms:**
```bash
pytest tests/test_openclaw_lite.py -v
# Some tests fail
```

**Root Cause:**
Outdated tests or missing dependencies.

**Resolution:**

1. **Run full test suite:**
```bash
python -m pytest tests/test_openclaw_lite.py -v --tb=short
```

2. **Check specific failure:**
```bash
python -m pytest tests/test_openclaw_lite.py::TestClass::test_method -v
```

3. **Update tests if API changed:**
```bash
git diff tests/test_openclaw_lite.py
```

---

## Health Check Reference

### Expected Output

```
============================================================
OpenClaw Health Check v3.0
============================================================

[CHECK] Imports...
   [PASS]: All imports successful

[CHECK] Heuristic Performance...
   [PASS]: Avg latency: 0.05ms (100 checks)

[CHECK] Verdict Accuracy...
   [PASS]: All 4 test cases passed

[CHECK] Escalation Queue...
   [PASS]: Escalation queue functional

[CHECK] Running async checks...
   [PASS] Telemetry: Telemetry tracking: 5 checks
   [PASS] Async Functionality: Async concurrent: 10 checks in 0.002s

============================================================
SUMMARY
============================================================
[PASS] Imports
[PASS] Heuristic Performance
[PASS] Verdict Accuracy
[PASS] Escalation Queue
[PASS] Telemetry
[PASS] Async Functionality
------------------------------------------------------------
Result: 6/6 checks passed
[OK] All systems operational
```

### Interpreting Failures

| Failure | Likely Cause | Fix |
|---------|--------------|-----|
| Imports fail | Missing dependency | `pip install -r requirements.txt` |
| Performance slow | System under load | Run during off-peak |
| Accuracy fails | Logic bug | Check keyword taxonomies |
| Escalation fail | Permission error | Check `.openclaw/` permissions |
| Telemetry fail | Async event loop issue | Restart Python process |
| Async fail | Event loop conflict | Use `asyncio.run()` properly |

---

## Log Locations

| Log | Location | Description |
|-----|----------|-------------|
| Escalation Queue | `.openclaw/escalation_queue/*.json` | Pending high-stakes reviews |
| Telemetry | In-memory (access via `get_telemetry()`) | Performance metrics |
| Health Check | stdout | Diagnostic output |
| Test Results | pytest output | Test suite results |

---

## Emergency Procedures

### Procedure 1: Disable High-Stakes Escalation

If escalation queue is causing issues:

```python
# In your code, skip escalation logic
verdict = await async_perform_sanity_check(
    ...,
    game_key=None  # Won't escalate if None
)
```

### Procedure 2: Purge Escalation Queue

**WARNING:** Destroys all pending escalations!

```bash
rm -rf .openclaw/escalation_queue/*.json
```

### Procedure 3: Reset Singleton

If `get_openclaw_lite()` returns corrupted state:

```python
import backend.services.openclaw_lite as ocl
ocl._lite_instance = None  # Reset singleton

# Get fresh instance
fresh_checker = get_openclaw_lite(enable_telemetry=True)
```

---

## Performance Tuning

### For High-Volume Slates (20+ games)

```python
# Increase concurrency
from backend.services.openclaw_lite import get_openclaw_lite

checker = get_openclaw_lite()
checker._semaphore = asyncio.Semaphore(16)  # Increase from 8

# Add batching
BATCH_SIZE = 10
for i in range(0, len(games), BATCH_SIZE):
    batch = games[i:i+BATCH_SIZE]
    results = await asyncio.gather(*[check(g) for g in batch])
    await asyncio.sleep(0.5)  # Brief pause between batches
```

### For Low-Latency Requirements

```python
# Disable telemetry for pure speed
checker = get_openclaw_lite(enable_telemetry=False)

# Use sync method (no semaphore overhead)
result = checker._check_integrity_heuristic_sync(...)
```

---

## Contact

**OpenClaw Issues:** Kimi CLI (config owner)  
**Core Architecture:** Claude Code (architect)  
**Test Failures:** Check with Kimi CLI first

---

*This guide is updated as new issues are discovered and resolved.*
