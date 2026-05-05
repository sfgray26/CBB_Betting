# CLAUDE → CODEX DELEGATION TEMPLATE
## Standardized Handoff for Feature PRs

**Version:** 1.0  
**Date:** 2026-05-05  
**Authority:** Claude Code (Principal Architect) delegates to Codex (Senior Engineer).  
**Rule:** Codex may NOT modify files outside the explicit file list below without Claude approval.

---

## TEMPLATE INSTRUCTIONS FOR CLAUDE

Copy this entire template into a new file: `reports/YYYY-MM-DD-codex-handoff-pr-<number>.md`

Fill in ALL sections marked `[FILL IN]`. Delete the instructional text (like this paragraph) before handing off.

---

# PR [NUMBER]: [TITLE]

## 1. METADATA

| Field | Value |
|-------|-------|
| **PR Number** | `[FILL IN — e.g., 2.1]` |
| **Epic** | `[FILL IN — e.g., Epic 2: Statcast Integration]` |
| **Priority** | `[FILL IN — P0 / P1 / P2]` |
| **Estimated Effort** | `[FILL IN — e.g., 2-4 hours]` |
| **Claimed By** | Codex |
| **Approved By** | Claude Code |
| **Branch** | `stable/cbb-prod` (commit `[FILL IN HASH]`) |

## 2. WHY THIS PR EXISTS

[FILL IN — 2-3 sentences on the business or technical motivation. Link to any design docs in `reports/`.]

Example:
> The `sprint_speed` column in `statcast_batter_metrics` is 100% NULL because no ingestion pipeline populates it. This PR adds a Savant scraper and wires it into the daily ingestion job so the column gets backfilled for the 2026 season.

## 3. SCOPE BOUNDARY

### In Scope (Codex implements)
- [ ] `[FILL IN — e.g., Create backend/ingestion/savant_scraper.py]`
- [ ] `[FILL IN — e.g., Add hook in daily_ingestion.py _refresh_statcast() method]`
- [ ] `[FILL IN — e.g., Unit tests in tests/test_savant_scraper.py]`

### Out of Scope (Do NOT touch)
- `[FILL IN — e.g., Do NOT modify scoring_engine.py — sprint_speed integration is PR 2.5]`
- `[FILL IN — e.g., Do NOT modify StatcastBatterMetrics SQLAlchemy model]`
- `[FILL IN — e.g., Do NOT change the Statcast CSV URL — use the one specified below]`

## 4. FILES TO CREATE / MODIFY

### New Files
```
[FILL IN — e.g., backend/ingestion/savant_scraper.py]
[FILL IN — e.g., tests/test_savant_scraper.py]
```

### Modified Files
```
[FILL IN — e.g., backend/services/daily_ingestion.py — add savant hook after pybaseball refresh]
```

### Files to Read (for context, do not modify)
```
[FILL IN — e.g., backend/services/daily_ingestion.py — read _refresh_statcast() and _refresh_pybaseball()]
[FILL IN — e.g., backend/models.py — read StatcastBatterMetrics columns]
```

## 5. TECHNICAL SPEC

### Function Signature
```python
[FILL IN — exact function signature, types, and return value]

# Example:
def fetch_sprint_speed(year: int = 2026) -> pd.DataFrame:
    """
    Fetch Baseball Savant sprint speed leaderboard.
    Returns DataFrame with columns: mlbam_id, player_name, sprint_speed.
    Returns empty DataFrame on any failure (HTTP, parse, timeout).
    """
```

### Algorithm / Logic
```
[FILL IN — pseudocode or step-by-step logic. Be explicit about edge cases.]

# Example:
1. Construct URL: https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={year}&position=&team=&min=0&csv=true
2. Request with _BROWSER_HEADERS (see pybaseball_loader.py for pattern)
3. If status != 200: log warning, return empty DataFrame
4. Parse CSV with pandas.read_csv(StringIO(response.text))
5. Select only columns: ['player_id', 'player_name', 'sprint_speed']
6. Rename 'player_id' → 'mlbam_id'
7. Return DataFrame
```

### DB Interaction
```sql
[FILL IN — any SQL statements. Use parameter placeholders, never raw f-strings for values.]

-- Example:
UPDATE statcast_batter_metrics
SET sprint_speed = %s
WHERE mlbam_id = %s AND season = %s;
```

### Integration Point
```
[FILL IN — exactly where and how the new code hooks into existing code.]

# Example:
In backend/services/daily_ingestion.py, inside _refresh_statcast():
After the pybaseball call succeeds, call:
    df = fetch_sprint_speed(year=2026)
    if not df.empty:
        for _, row in df.iterrows():
            db.execute(UPDATE_SQL, (row['sprint_speed'], str(row['mlbam_id']), 2026))
        db.commit()
        logger.info("Updated sprint_speed for %d players", len(df))
```

### Feature Flag
```python
[FILL IN — the feature flag name, default, and where to check it.]

# Example:
if get_threshold("feature.sprint_speed_ingestion", default=False):
    df = fetch_sprint_speed()
    ...
```

## 6. ACCEPTANCE CRITERIA

### Code Quality Gates
- [ ] `venv\Scripts\python -m py_compile <new_or_modified_file>` passes for every file
- [ ] `venv\Scripts\python -m pytest tests/ -q --tb=short` — **baseline must remain 2488 pass / 4 skip / 0 fail**
- [ ] New code has ≥80% test coverage
- [ ] No `datetime.utcnow()` introduced
- [ ] Feature flag exists and defaults to `False`
- [ ] DB operations are idempotent (running twice = same result)

### Functional Checks
- [ ] `[FILL IN — e.g., fetch_sprint_speed(2026) returns DataFrame with 3 columns]`
- [ ] `[FILL IN — e.g., fetch_sprint_speed(2099) returns empty DataFrame on 404]`
- [ ] `[FILL IN — e.g., After daily ingestion, SELECT COUNT(*) FROM statcast_batter_metrics WHERE sprint_speed IS NOT NULL > 0]`
- [ ] `[FILL IN — e.g., Malformed CSV is handled gracefully]`

## 7. DEPENDENCIES

### Blocked By
- [ ] `[FILL IN — e.g., PR 1.2 (config_service) must be deployed]`

### Blocks
- [ ] `[FILL IN — e.g., PR 2.3 (validation) requires this scraper to exist]`

## 8. TEST STRATEGY

### Unit Tests (new file)
```python
# [FILL IN — test file path and key test cases]

# Example for tests/test_savant_scraper.py:
def test_fetch_sprint_speed_success(mock_requests):
    """Returns DataFrame with correct columns on 200."""

def test_fetch_sprint_speed_http_error(mock_requests):
    """Returns empty DataFrame on 500."""

def test_fetch_sprint_speed_malformed_csv(mock_requests):
    """Returns empty DataFrame on unparseable CSV."""
```

### Integration Tests (if needed)
```python
# [FILL IN — any tests that hit the real DB or external API]

# Example:
def test_daily_ingestion_updates_sprint_speed(db_session):
    """After running _refresh_statcast(), sprint_speed is populated."""
```

## 9. ROLLBACK PLAN

```
[FILL IN — how to revert if something goes wrong.]

# Example:
1. Disable feature flag: UPDATE feature_flags SET enabled = false WHERE flag_name = 'sprint_speed_ingestion';
2. Revert code: git revert <commit>
3. Clear bad data (if applicable): UPDATE statcast_batter_metrics SET sprint_speed = NULL WHERE season = 2026;
```

## 10. KNOWN GOTCHAS

```
[FILL IN — any pitfalls discovered during design.]

# Example:
- Savant CSV has a BOM (Byte Order Mark) on some days — use encoding='utf-8-sig'
- The 'player_id' column in Savant is integer, but our DB stores mlbam_id as VARCHAR — cast to str
- Do NOT call this during the pybaseball refresh window (7:30 AM ET) to avoid rate limits
```

---

## 11. HOW CODEX REPORTS BACK

When complete, Codex will:

1. **Run the test suite** and paste the pytest output
2. **Write a completion report** to `reports/YYYY-MM-DD-codex-pr-<number>-completion.md` containing:
   - What was implemented
   - What was changed vs. the spec (and why)
   - Test results
   - Any deviations or blockers encountered
3. **Update this file** with checkmarks on all acceptance criteria
4. **Hand off to Claude** for architectural review
5. **Hand off to Gemini** for deploy (if applicable)

---

## 12. CLAUDE SIGN-OFF

> I, Claude Code, approve Codex to implement PR [NUMBER] within the scope defined above.  
> Architecture is frozen. Interfaces are locked. No scope expansion without my approval.

**Signed:** Claude Code  
**Date:** `[FILL IN]`

---

# APPENDIX: EXAMPLE FILLED-IN HANDOFF

Below is a real example for PR 2.1 that Claude can copy-paste and adapt.

---

## EXAMPLE: PR 2.1 — Savant Scraper

### 1. METADATA
| Field | Value |
|-------|-------|
| PR Number | 2.1 |
| Epic | Epic 2: Statcast Integration |
| Priority | P1 |
| Estimated Effort | 3-4 hours |
| Claimed By | Codex |
| Approved By | Claude Code |
| Branch | `stable/cbb-prod` (commit `cf29a64`) |

### 2. WHY
> The `sprint_speed` column exists in `statcast_batter_metrics` but is 100% NULL. We need a Savant scraper to populate it. This is a prerequisite for PR 2.2 (pipeline hook) and PR 2.3 (validation).

### 3. SCOPE
**In Scope:**
- Create `backend/ingestion/savant_scraper.py`
- Create `tests/test_savant_scraper.py`

**Out of Scope:**
- Do NOT modify `daily_ingestion.py` — that's PR 2.2
- Do NOT modify `scoring_engine.py` — that's PR 2.5
- Do NOT create the `feature_flags` entry — that's PR 2.3

### 4. FILES
**New:**
- `backend/ingestion/savant_scraper.py`
- `tests/test_savant_scraper.py`

**Read for context:**
- `backend/ingestion/pybaseball_loader.py` (see `_BROWSER_HEADERS`)
- `backend/fantasy_baseball/statcast_loader.py` (see existing Statcast patterns)

### 5. TECHNICAL SPEC
```python
def fetch_sprint_speed(year: int = 2026) -> pd.DataFrame:
    """
    Fetch Baseball Savant sprint speed leaderboard CSV.
    
    Returns DataFrame with columns:
        - mlbam_id (str)
        - player_name (str)
        - sprint_speed (float)
    
    Returns empty DataFrame on any failure.
    """
```

**Algorithm:**
1. URL: `https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={year}&position=&team=&min=0&csv=true`
2. Headers: Copy `_BROWSER_HEADERS` from `pybaseball_loader.py`
3. On non-200: log warning, return empty DataFrame
4. Parse with `pd.read_csv(io.StringIO(response.text))`
5. Select and rename: `player_id` → `mlbam_id`
6. Return only `['mlbam_id', 'player_name', 'sprint_speed']`

### 6. ACCEPTANCE CRITERIA
- [ ] `py_compile` passes
- [ ] pytest passes (baseline 2488/4/0)
- [ ] Returns correct columns on success
- [ ] Returns empty DataFrame on HTTP error
- [ ] Handles malformed CSV gracefully
- [ ] No `utcnow()`

### 7. DEPENDENCIES
**Blocked by:** None  
**Blocks:** PR 2.2, PR 2.3

### 8. TESTS
```python
# tests/test_savant_scraper.py

def test_fetch_sprint_speed_success(mock_requests):
    ...

def test_fetch_sprint_speed_http_error(mock_requests):
    ...

def test_fetch_sprint_speed_malformed_csv(mock_requests):
    ...
```

### 9. ROLLBACK
```
1. git revert <commit>
2. No DB changes in this PR — nothing to roll back in Postgres
```

### 10. GOTCHAS
- Savant occasionally returns CSV with BOM — use `encoding='utf-8-sig'` when reading response text
- `player_id` is numeric in CSV but our DB uses VARCHAR — cast to `str`
- Some rows may have missing `sprint_speed` — drop NaN before returning

### 11. CODEX REPORTS BACK
(See section 11 in template)

### 12. CLAUDE SIGN-OFF
> Approved. Interfaces locked. No scope expansion without my approval.

---

# END OF TEMPLATE
