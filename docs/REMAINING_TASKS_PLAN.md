# Remaining Improvement Tasks - Execution Plan

## Status: 4 of 8 Tasks Complete

| Task | Agent | Status |
|------|-------|--------|
| 1. Test Coverage | Codex | ⏳ Retry needed (timed out) |
| 2. Remove Prints | Codex | ✅ Complete |
| 3. Quality Score | Claude | ✅ Complete |
| 4. Stale Data Tracking | Claude | ⏳ In Progress |
| 5. Documentation | Gemini | ✅ Complete |
| 6. Game Context | Claude | ⏳ Pending |
| 7. Performance | Claude | ⏳ Pending |
| 8. Security Audit | Gemini | ✅ Complete |

---

## Task 4: Stale Data Tracking (Next)

**Problem**: `fetched_at=None` and `is_stale=False` hardcoded in 69 places

**Solution Approach**:
```python
# 1. Add to Yahoo client return values
return {
    "players": players,
    "fetched_at": datetime.now(ZoneInfo("America/New_York"))
}

# 2. Compute is_stale at API layer
is_stale = (datetime.now() - fetched_at).total_seconds() > (15 * 60)
```

**Files to Modify**:
- `backend/fantasy_baseball/yahoo_client_resilient.py` - Add fetched_at tracking
- `backend/routers/fantasy.py` - 8 locations need dynamic is_stale computation
- `backend/services/scoreboard_orchestrator.py` - 1 location
- `backend/services/player_mapper.py` - 2 locations

---

## Task 6: Game Context (Pending)

**TODO Location**: `player_mapper.py:112,203`

**Fields to Add**:
- `opponent_team` - From Yahoo matchup data
- `game_time` - First pitch time
- `is_home` - Home/away boolean
- `weather` - Weather conditions

**Implementation**:
```python
# Query Yahoo matchup data
matchup = yahoo_client.get_matchup(player_team)
game_context = {
    "opponent_team": matchup.opponent,
    "game_time": matchup.start_time,
    "is_home": matchup.is_home,
    "weather": get_weather(matchup.venue)  # Optional
}
```

---

## Task 7: Performance Optimization (Pending)

**Target Endpoints**:
- `GET /api/fantasy/roster` 
- `GET /api/fantasy/scoreboard`
- `POST /api/fantasy/optimize-lineup`

**Current Issues**:
- N+1 queries for player data
- No eager loading (`joinedload`)
- Missing DB indexes

**Optimizations**:
```python
# Add to models.py - indexes
Index('idx_player_id_date', 'player_id', 'game_date')
Index('idx_team_date', 'team', 'game_date')

# Add eager loading
from sqlalchemy.orm import joinedload
query = db.query(Player).options(joinedload(Player.stats))

# Cache frequent queries
cache_key = f"roster:{team_key}:{date}"
```

---

## Quick Wins Already Delivered

### ✅ Task 3: Quality Score (Complete)
**Change**: `backend/routers/fantasy.py:2074`
```python
# Before:
quality_score=None,  # TODO: populate from ProbablePitcherSnapshot

# After:
quality_score=_pitcher_quality_map.get(name.lower()) if _fa_is_pitcher else None,
```

Plus added bulk lookup from ProbablePitcherSnapshot.

### ✅ Task 2: Remove Prints (Complete)
**Change**: `backend/models.py:622`
```python
# Before:
print("✅ Database tables created")

# After:
logger.info("Database tables created")
```

### ✅ Tasks 5 & 8: Documentation & Security (Complete)
**Files Created**:
- `docs/TODO.md` - 28 TODOs documented
- `docs/SECURITY_AUDIT.md` - Security findings
- `HERMES.md` - Technical debt section

---

## Recommended Execution Order

### Phase 1: Foundation (Today)
1. **Task 4** - Stale data tracking (affects all endpoints)
2. **Task 1** - Test coverage (Codex retry)

### Phase 2: Features (Tomorrow)
3. **Task 6** - Game context
4. **Task 7** - Performance optimization

---

## PowerShell Commands to Complete Sprint

```powershell
# 1. Create branch for remaining tasks
cd C:\Users\sfgra\repos\Fixed\cbb-edge
git checkout -b agent/claude/remaining-improvements-20260516

# 2. Run Codex task (retry test coverage)
Start-Process codex -ArgumentList '-p', '"Create comprehensive test suite for matchup_engine.py, player_mapper.py, row_projector.py. Target 80%+ coverage. Use pytest with mocks.", '--permission-mode', 'bypassPermissions'

# 3. Task 4: Implement stale data tracking
codex -p "Implement fetched_at tracking and is_stale computation across the codebase. 

Files to modify:
1. backend/fantasy_baseball/yahoo_client_resilient.py - Add fetched_at to return values
2. backend/routers/fantasy.py - Compute is_stale dynamically (15 min threshold)
3. backend/services/scoreboard_orchestrator.py - Pass through timestamps
4. backend/services/player_mapper.py - Track fetch time

Requirements:
- fetched_at should be datetime in America/New_York timezone
- is_stale = True if data is > 15 minutes old
- Non-breaking: handle None fetched_at gracefully (assume fresh)
- Add tests for staleness logic

Run: python -m pytest tests/ -v" --permission-mode bypassPermissions

# 4. Task 6: Game context
codex -p "Implement game context for players (Task 6). 

Add to PlayerCardResponse:
- opponent_team: str (from Yahoo matchup)
- game_time: datetime (first pitch)
- is_home: bool (home/away)

Files:
- backend/services/player_mapper.py - Add game_context lookup
- backend/schemas.py - Add fields to response model
- backend/fantasy_baseball/yahoo_client_resilient.py - Fetch matchup data

Wire up Yahoo matchup data to populate these fields." --permission-mode bypassPermissions

# 5. Task 7: Performance
codex -p "Optimize database queries for high-traffic endpoints.

Target files:
- backend/routers/fantasy.py - get_roster, get_scoreboard, optimize_lineup
- Add joinedload for relationships
- Add missing indexes to models.py
- Add query result caching (TTL 60s)

Measure: Before/after timing with logging." --permission-mode bypassPermissions
```

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Test Coverage | 3 files | 6+ files, 80%+ |
| Print Statements | 20+ | 0 (production) |
| Quality Score | Always None | Populated for pitchers |
| Data Freshness | Unknown | Tracked + stale indicators |
| API Response Time | Variable | <100ms target |
| Security Audit | None | Complete |
| TODO Documentation | None | 28 documented |

---

## Success Criteria

- [ ] All 8 tasks complete
- [ ] Test suite passes
- [ ] No new P1/P2 bugs introduced
- [ ] Performance benchmarks show improvement
- [ ] Documentation updated

---

*Plan created: 2025-05-16*
*Current progress: 50% (4/8 tasks)*
