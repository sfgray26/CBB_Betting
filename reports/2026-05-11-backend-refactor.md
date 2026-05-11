# Backend Refactor: CBB Decoupling

Date: 2026-05-11

## Scope

Touched only the requested backend files:

- `backend/main.py`
- `backend/routers/fantasy.py`
- `backend/schedulers/edge_scheduler.py`
- `backend/services/daily_ingestion.py`

## Changes

- Retired `GET /api/tournament/bracket-projection` by changing it to return HTTP `410 Gone` with a clear closed-season message. The route no longer imports or executes tournament bracket modules.
- Disabled startup registration for `tournament_bracket_notifier` in both `backend/main.py` and `backend/schedulers/edge_scheduler.py`.
- Left MLB analysis and fantasy scheduler jobs active.
- Cleaned the fantasy scoreboard week comment so it is explicitly MLB fantasy focused.
- Added sport-boundary documentation to `daily_ingestion.py`, labeling active MLB stages and CBB-legacy stages.
- Labeled `_compute_clv` and `_start_openclaw_monitoring` as CBB-legacy in docstrings.

## Verification

- `venv/Scripts/python -m py_compile ...` could not run because the venv launcher points to missing `C:\Users\sfgra\AppData\Local\Programs\Python\Python311\python.exe`.
- Re-ran py_compile with available `C:\Python314\python.exe` via `python`; all modified files compiled successfully:
  - `python -m py_compile backend/main.py`
  - `python -m py_compile backend/routers/fantasy.py`
  - `python -m py_compile backend/schedulers/edge_scheduler.py`
  - `python -m py_compile backend/services/daily_ingestion.py`
- Confirmed no active scheduler registration remains for `tournament_bracket_notifier` in `backend/main.py` or `backend/schedulers/edge_scheduler.py`.
- Confirmed `backend/routers/fantasy.py` has no CBB/tournament/bracket references.
