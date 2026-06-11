# Production Cleanup — Implementation Report

**Date:** 2026-05-20  
**Agent:** Copilot CLI  
**Status:** ✅ Complete — py_compile clean, 17/17 tests pass

---

## Task 1 — Remove Debug Endpoints from `backend/main.py`

### Removed

Five debug/temporary routers were removed from both their `import` declarations (lines ~55-71)
and their `app.include_router()` call sites (lines ~634-644):

| Router variable | Module | Purpose |
|---|---|---|
| `_db_verify_router` | `backend.admin_db_verify` | Database verification (post-migration) |
| `_yahoo_debug_router` | `backend.admin_yahoo_debug` | Yahoo API raw response debugging |
| `_yahoo_token_router` | `backend.admin_yahoo_token_refresh` | Yahoo OAuth token seeding |
| `_yahoo_parsing_test_router` | `backend.admin_test_yahoo_parsing` | Yahoo response parsing tests |
| `_yahoo_structure_dump_router` | `backend.admin_yahoo_structure_dump` | Yahoo roster structure dumps |

All were mounted at `/test/*` — they are no longer reachable in production.

### Left in place (out of scope)

Per the task instructions, only the five listed routers were removed. The following are still
present and require separate cleanup sessions:

- `_test_router` (`_test_router` — sync job testing)
- `_era_diagnostic_router` (REMOVE AFTER TASK 10 COMPLETE)
- `_validation_audit_router` (REMOVE AFTER TASK 11 COMPLETE)
- `_backfill_ops_whip_router` (REMOVE AFTER TASK 26 COMPLETE)
- `_statcast_diag_router` (REMOVE AFTER STATCAST VALIDATION COMPLETE)
- `_scoring_diag_router` (REMOVE AFTER NSB ROLLOUT VALIDATED)
- `_constraint_migration_router` (REMOVE AFTER YAHOO ID SYNC IS LIVE)

### Verification

```
py_compile backend/main.py → OK
grep for removed router names → 0 results
```

---

## Task 2 — Harden Test Environment

### Problem

WSL test environment could not run pytest because `requests` and `apscheduler` were not installed
in the active Python virtual environment, even though they appear in `requirements.txt`. Root cause:
WSL may use a different Python path or the venv was not created/refreshed correctly.

### Solution

Created two scripts that ensure the venv matches `requirements.txt` exactly:

| Script | Platform | Usage |
|---|---|---|
| `scripts/setup_test_env.ps1` | Windows PowerShell | `.\scripts\setup_test_env.ps1` |
| `scripts/setup_test_env.sh` | WSL / Linux / macOS | `bash scripts/setup_test_env.sh` |

Both scripts:
1. Create `./venv` if it doesn't exist (or `--force` / `-Force` to recreate)
2. Upgrade pip to latest
3. Install all packages from `requirements.txt`
4. Smoke-test six critical imports: `requests`, `apscheduler`, `pytest`, `sqlalchemy`, `pydantic`, `fastapi`
5. Run `tests/test_injury_overlay.py` + `tests/test_fantasy_budget.py` as a final gate
6. Print the standard test run command on success

### WSL-specific notes

In WSL, use:
```bash
# One-time setup (or after any requirements.txt change):
bash scripts/setup_test_env.sh

# If venv is corrupted or uses the wrong Python:
bash scripts/setup_test_env.sh --force

# After setup, run tests:
venv/bin/python -m pytest tests/ -q --tb=short
```

**Never use bare `python` in WSL** — it may resolve to the system Python (2.x or an unrelated 3.x).
Always use `venv/bin/python` after setup.

### Smoke test results (Windows)

```
17 passed in 0.72s
tests/test_injury_overlay.py   3/3 ✅
tests/test_fantasy_budget.py  14/14 ✅
```

---

## Files Modified

| File | Change |
|---|---|
| `backend/main.py` | Removed 5 debug router imports + 5 `app.include_router()` calls |

## Files Created

| File | Purpose |
|---|---|
| `scripts/setup_test_env.ps1` | Windows venv hardening + smoke test |
| `scripts/setup_test_env.sh` | WSL/Linux venv hardening + smoke test |

---

## Hand-off Notes for Claude Code

1. Seven other `REMOVE AFTER` routers remain in `main.py`. Schedule a follow-up cleanup
   session once their associated tasks (10, 11, 26, statcast validation, NSB rollout,
   Yahoo ID sync) are confirmed complete.

2. The admin module files (`backend/admin_db_verify.py`, `backend/admin_yahoo_debug.py`, etc.)
   were **not deleted** from the repo — only their mounts were removed. Once confirmed safe,
   those source files can be deleted in a follow-up to reduce dead code.
