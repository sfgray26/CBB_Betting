# Claude Code Task: BDL #2 — Player Search Auto-Heal Unmapped Yahoo Players

## Task ID
**t_b9c63f54** — EV 4.67 — P2 Priority

## Project Context
**CBB Edge** — Fantasy baseball (MLB) lineup optimization platform  
**Stack**: Python 3.11, FastAPI, PostgreSQL, SQLAlchemy  
**Branch to work in**: `agent/claude/bdl-2-player-search`  
**Worktree**: Use your own isolated worktree (Claude Code handles this automatically)

## The Problem

When a player gets called up from the minors, Yahoo assigns them a `player_key`, but our pipeline cannot enrich them because:

1. `player_id_mapping` table does not have the new player yet
2. BDL player search is NOT triggered as a fallback
3. All downstream features return null/empty:
   - Ownership % shows "— owned"
   - Statcast metrics are missing
   - Projections are blank
   - Matchup scores are incomplete

**This suppresses every BDL/Statcast feature for new call-ups.**

## Root Cause

The current flow:
```
Yahoo API → get_roster() → lookup player_id_mapping → ❌ NOT FOUND → skip enrichment
```

The missing step: when lookup fails, trigger BDL player search to find the player and add them to the mapping.

## Files to Modify

### 1. `backend/services/player_search_service.py` (NEW FILE)
Create this service with:
- `search_and_map_player(yahoo_player_key, player_name, team_abbrev)` function
- Calls BDL player search API (use requests, endpoint pattern from other BDL services)
- On match found: inserts into `player_id_mapping` with `source="auto-heal"`
- On no match: logs warning, returns None (graceful degradation)
- `batch_heal_unmapped_players(player_list)` — processes multiple players

### 2. `backend/fantasy_baseball/yahoo_client_resilient.py`
Modify `get_roster()` or roster enrichment flow:
- After parsing Yahoo roster, check each player against `player_id_mapping`
- For unmapped players: trigger `search_and_map_player()` asynchronously
- Do NOT block roster response on search (fire-and-forget or background task)
- Log auto-heal attempts and outcomes

### 3. `backend/models.py`
Add to `player_id_mapping` table or create migration:
- `source` column (VARCHAR): manual, auto-heal, import
- `healed_at` column (DateTime): when auto-heal occurred
- `heal_attempts` column (Integer): count of search attempts

### 4. `backend/services/daily_ingestion.py`
Add to daily ingestion pipeline:
- After ingesting Yahoo rosters, scan for unmapped players
- Call `batch_heal_unmapped_players()`
- Report count of healed players in daily audit log

## Code Patterns (Follow These)

### Pattern 1: Silent Failure Guard
```python
# NEVER silently return empty. Always log + attempt fallback.
player = db.query(PlayerIdMapping).filter_by(yahoo_key=yahoo_key).first()
if not player:
    logger.warning(f"Unmapped Yahoo player: {yahoo_key} ({player_name}, {team})")
    # TRIGGER AUTO-HEAL HERE
    player = search_and_map_player(yahoo_key, player_name, team)
```

### Pattern 2: Graceful Degradation
```python
# Auto-heal should not block the main flow
# Fire-and-forget or background task
asyncio.create_task(search_and_map_player_async(yahoo_key, name, team))
# OR use a queue/threading for sync contexts
```

### Pattern 3: Source Tracking
```python
# Track where mappings came from
player_id_mapping.source = "auto-heal"  # vs "manual", "import"
player_id_mapping.healed_at = datetime.utcnow()
```

## Implementation Steps

1. **Read existing player_id_mapping model** — understand schema and query patterns
2. **Read yahoo_client_resilient.py** — find where roster parsing happens
3. **Read daily_ingestion.py** — find the daily pipeline hook point
4. **Create player_search_service.py** with the auto-heal logic
5. **Wire into yahoo_client_resilient.py** — trigger on unmapped players
6. **Wire into daily_ingestion.py** — batch-heal during daily run
7. **Add migration** for new columns
8. **Write tests**
9. **Run tests**
10. **Commit**

## Tests Required

Create `backend/test_player_search_auto_heal.py`:

1. **test_auto_heal_success** — mock BDL search returning a match, verify mapping created
2. **test_auto_heal_no_match** — mock BDL search returning empty, verify graceful fallback
3. **test_downstream_enrichment** — after auto-heal, verify ownership % is populated
4. **test_batch_heal** — verify multiple unmapped players are processed
5. **test_source_tracking** — verify source="auto-heal" is set
6. **test_no_regression** — verify existing mapped players are untouched

## Acceptance Criteria

- [ ] New call-ups show ownership % within 24 hours of first Yahoo appearance
- [ ] Auto-heal events are logged (warning level) with player_key + name
- [ ] Auto-healed mappings have `source="auto-heal"` and `healed_at` populated
- [ ] No regression in existing player mapping queries
- [ ] Batch-heal runs during daily ingestion without blocking
- [ ] All 6 tests pass
- [ ] Handoff summary written to `.agent-handoffs/COMPLETED_t_b9c63f54.md`

## Key Context from Prior Work

- Ownership bug (showing "— owned") was a **deploy gap** — fixed by deploying latest code to Railway
- This task fixes the **root cause for NEW players** — auto-healing the mapping gap
- BDL #1 (streaming signal, EV 7.11) was just implemented — it depends on player mapping being correct
- BDL #3 (injury overlay, EV 4.08) is blocked until player search works — do this task first

## Commands to Run

```bash
# After implementing, run tests
cd backend
python -m pytest test_player_search_auto_heal.py -v

# Also run existing tests to check for regressions
python -m pytest tests/ -x -q

# If all tests pass, commit
git add -A
git commit -m "feat(bdl-2): player search auto-heal for unmapped Yahoo players

- Add player_search_service.py with auto-heal logic
- Wire auto-heal into yahoo_client_resilient.py roster parsing
- Add batch-heal to daily_ingestion.py pipeline
- Add source/healed_at columns to player_id_mapping
- 6 tests: auto-heal success, no-match fallback, downstream enrichment, batch, source tracking, no regression

All tests passing."
```

## Notes
- BDL = Baseball Data Labs (data provider)
- Use `requests` for API calls (pattern from other services)
- Use `sqlalchemy` for DB operations (pattern from existing models)
- Use `logging` not print statements
- Keep functions under 50 lines where possible
- Add type hints for new functions
