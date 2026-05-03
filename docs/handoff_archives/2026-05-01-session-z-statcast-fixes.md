## 0. Session Z — Claude Implementation (DONE, 2026-05-01)

### Fix Z-A/B/C: Three bugs in statcast_loader + fantasy.py (post Session W+X code review)

**Z-A:** Tier priority inversion — `_load_from_db()` ran BEFORE pybaseball/CSV in
`_ensure_loaded()`, but `batters.update(_db_batters)` was called BEFORE those loops,
so pybaseball/CSV overwrote DB data. Fixed: moved both `.update()` calls to AFTER
the pybaseball+CSV loops so DB tier wins (last writer wins).

**Z-B:** Silent `except: pass` on opponent roster fetch in `fantasy.py` suppressed all
errors. Fixed: replaced with `except Exception as exc: logger.warning(...)` so issues
are visible in Railway logs.

**Z-C:** `_load_from_db()` hardcoded `season=2026` as a string literal in SQL. Fixed:
added `season: int = 2026` parameter with SQLAlchemy bind params to prevent SQL injection
and allow future-proofing.

**Files changed:** `backend/fantasy_baseball/statcast_loader.py`, `backend/routers/fantasy.py`
**Test result:** 2475 pass / 3 skip / 0 fail

---

