# P1 Bundle Routing — Fantasy Baseball App
**Date:** 2026-06-11
**Status:** READY FOR PARALLEL EXECUTION
**Deployment Blocker:** daily_availability_overrides table missing (separate investigation)

---

## CONTEXT

### What's Done (P0)
- ✅ Availability guard (DailyAvailabilityOverride admin API, day-off blacklist)
- ✅ Roster constraint awareness (IL slot capacity, FAAB balance checks)
- ✅ IL crisis detection (Dashboard emergency alerts)
- ✅ Win probability context labels (Week N·IN-FLIGHT/PREVIEW badges)
- ✅ 3073 tests passing, zero regressions
- ✅ Branch pushed to stable/cbb-prod (commit 9e74f44)
- ✅ Railway deployment 7637ef33 SUCCESS (smoke failed on DB table)

### Deployment Status
- **BLOCKED:** Production DB missing `daily_availability_overrides` table
- **Root cause:** lifespan() in backend/main.py (lines 205-218) didn't create table during deployment
- **Workaround investigation:** Railway SSH hangs, direct Postgres access times out
- **Next deploy action:** Review lifespan() logic, fix or apply manual migration

---

## P1 TASKS (Estimated 20-25 hours)

### TASK 1: Global Freshness Normalization (~6 hours)
**Agent:** Claude Code (owns backend routes, schemas, models)

**Backend Deliverables:**
1. **FreshnessSeverity enum** in `backend/contracts.py`:
   ```python
   class FreshnessSeverity(str, Enum):
       FRESH = "fresh"
       WARNING = "warning"      # >60 minutes
       CRITICAL = "critical"    # >120 minutes
       UNKNOWN = "unknown"
   ```

2. **FreshnessState model** in `backend/contracts.py`:
   ```python
   class FreshnessState(BaseModel):
       severity: FreshnessSeverity
       minutes_ago: int | None = None
       last_updated: datetime | None = None
       warning_text: str | None = None
       is_clickable: bool = True
   ```

3. **Computation function** in `backend/services/freshness_service.py` (new file):
   ```python
   def compute_freshness_state(last_updated: datetime) -> FreshnessState:
       now = datetime.now(ZoneInfo("America/New_York"))
       minutes_ago = int((now - last_updated).total_seconds() / 60)

       if minutes_ago <= 60:
           severity = FreshnessSeverity.FRESH
           warning_text = None
       elif minutes_ago <= 120:
           severity = FreshnessSeverity.WARNING
           warning_text = "Projections may be stale — click to refresh."
       else:
           severity = FreshnessSeverity.CRITICAL
           warning_text = "Projections are stale — refresh required."

       return FreshnessState(
           severity=severity,
           minutes_ago=minutes_ago,
           last_updated=last_updated,
           warning_text=warning_text,
           is_clickable=True
       )
   ```

4. **Update services** to include freshness:
   - `backend/services/dashboard_service.py` — add `freshness: FreshnessState` to dashboard response
   - `backend/services/waiver_edge_detector.py` — add freshness to waiver targets
   - `backend/fantasy_baseball/daily_lineup_optimizer.py` — add freshness to projection responses
   - Remove 26-hour War Room threshold normalization

5. **API endpoint** in `backend/routers/fantasy.py`:
   ```python
   @router.get("/freshness")
   async def get_global_freshness() -> Dict[str, FreshnessState]:
       """Returns freshness state for all modules."""
       # Implementation: query last_updated timestamps from relevant tables
       return {
           "dashboard": compute_freshness_state(...),
           "war_room": compute_freshness_state(...),
           "waiver": compute_freshness_state(...),
           "roster": compute_freshness_state(...),
           "streaming": compute_freshness_state(...),
       }
   ```

**Verification:**
- `python -m py_compile backend/contracts.py backend/services/freshness_service.py backend/routers/fantasy.py`
- `pytest tests/ -q` — ensure no regressions
- Test endpoint: `GET /api/fantasy/freshness` returns valid JSON

**Frontend (Kimi CLI) — AFTER BACKEND COMPLETE:**
- Create `frontend/src/components/freshness/FreshnessBadge.tsx`
- Apply to War Room, Dashboard, Waiver, Roster, Streaming Station

---

### TASK 2: ETA Expiration Watchdog (~4 hours)
**Agent:** Claude Code (owns backend models, cron jobs)

**Backend Deliverables:**

1. **Update IngestedInjury model** in `backend/models.py`:
   ```python
   class IngestedInjury(Base):
       # ... existing fields ...
       expired_eta = Column(Boolean, default=False)  # NEW: flag when ETA passed
       eta_expiration_checked_at = Column(DateTime, nullable=True)  # NEW
   ```

2. **Create cron job** in `backend/main.py` lifespan():
   ```python
   from backend.services.eta_expiration_watchdog import check_eta_expiration

   @async_app.scheduled_job('cron', hour=6, minute=0, timezone='America/New_York')
   async def eta_expiration_check():
       """Run at 6:00 AM ET daily to check for expired ETAs."""
       await check_eta_expiration()
   ```

3. **Implementation** in `backend/services/eta_expiration_watchdog.py` (new file):
   ```python
   from datetime import datetime
   from zoneinfo import ZoneInfo
   from sqlalchemy.ext.asyncio import AsyncSession

   async def check_eta_expiration(session: AsyncSession):
       """Mark injuries as expired when ETA has passed and status unchanged."""
       now = datetime.now(ZoneInfo("America/New_York"))

       # Query injuries with return_date < now and expired_eta = False
       expired = await session.execute(
             select(IngestedInjury)
             .where(IngestedInjury.return_date < now)
             .where(IngestedInjury.expired_eta == False)
         )
         .scalars().all()

       for injury in expired:
           injury.expired_eta = True
           injury.eta_expiration_checked_at = now
           # Preserve original status/comment

       await session.commit()
   ```

4. **Update frontend contract** in `backend/contracts.py`:
   ```python
   class InjStatus(str, Enum):
       IL = "IL"
       DTD = "DTD"
       OUT = "OUT"
       QUESTIONABLE = "QUESTIONABLE"
       ETA_EXPIRED = "ETA_EXPIRED"  # NEW: "ETA PASSED — STATUS UNKNOWN"
   ```

5. **API response update** in waiver wire / roster endpoints:
   - When `expired_eta = True`, show status as `ETA_EXPIRED`
   - Display warning: "ETA PASSED — STATUS UNKNOWN"

**Verification:**
- `python -m py_compile backend/models.py backend/services/eta_expiration_watchdog.py backend/main.py`
- Run manually: `await check_eta_expiration(session)` — verify Colt Emerson (ETA Jun 9) flagged

**Frontend (Kimi CLI) — AFTER BACKEND COMPLETE:**
- Show red "ETA PASSED — STATUS UNKNOWN" badge on injury cards
- Preserve original ETA text for reference

---

### TASK 3: Predictive Stats Pipeline (~8 hours)
**Agent:** Claude Code (owns backend ingestion, models)

**Backend Deliverables:**

1. **Feature flag** in `backend/services/config_service.py`:
   ```python
   @lru_cache
   def get_predictive_stats_v1_enabled() -> bool:
       return os.getenv("PREDICTIVE_STATS_V1_ENABLED", "false").lower() == "true"
   ```

2. **Update ingestion modules** (all existing, just add gating):
   - `backend/ingestion/savant_ingestion.py` — gate xwOBA, Hard-Hit%, wRC+ behind flag
   - `backend/ingestion/pybaseball_loader.py` — gate FIP, xFIP, SIERA behind flag
   - `backend/ingestion/pitcher_deep_dive.py` — gate predictive pitching metrics

3. **Observability tracking** — add to ingestion output:
   ```python
   class PredictiveStatsIngestResult(BaseModel):
       refresh_timestamp: datetime
       rows_ingested: int
       source: str  # "savant", "fangraphs", "statcast"
       status: str  # "success", "stale", "error"
       error_message: str | None = None
   ```

4. **Update models** in `backend/models.py`:
   ```python
   class PredictiveStatsSnapshot(Base):
       id = Column(Integer, primary_key=True)
       player_id = Column(Integer)  # BDL player ID
       fip = Column(Float, nullable=True)
       x_fip = Column(Float, nullable=True)
       sieras = Column(Float, nullable=True)
       xwoba = Column(Float, nullable=True)
       hard_hit_pct = Column(Float, nullable=True)
       wrc_plus = Column(Float, nullable=True)
       refreshed_at = Column(DateTime, default=_now_et)
       source = Column(String)
       is_stale = Column(Boolean, default=False)
   ```

5. **API integration** in `backend/routers/fantasy.py`:
   ```python
   @router.get("/players/{player_id}/predictive-stats")
   async def get_predictive_stats(player_id: int):
       """Returns FIP, xFIP, SIERA, xwOBA, Hard-Hit%, wRC+ if enabled."""
       if not config_service.get_predictive_stats_v1_enabled():
           raise HTTPException(404, "Predictive stats feature disabled")
       # Return from PredictiveStatsSnapshot
   ```

**Verification:**
- `python -m py_compile backend/services/config_service.py backend/ingestion/*.py backend/models.py backend/routers/fantasy.py`
- Test with `PREDICTIVE_STATS_V1_ENABLED=false` — endpoints should return 404
- Test with `PREDICTIVE_STATS_V1_ENABLED=true` — endpoints return valid data

**Frontend (Kimi CLI) — AFTER BACKEND COMPLETE:**
- Add predictive stats column to player cards (when enabled)
- Show refresh timestamp and staleness indicator

---

### TASK 4: MLB Lineup API Integration (~6 hours)
**Agent:** Claude Code (owns backend pipelines, lineup optimizer)

**Backend Deliverables:**

1. **Lineup cards ingestion** in `backend/ingestion/mlb_lineup_ingestion.py` (new file):
   ```python
   from datetime import datetime
   from zoneinfo import ZoneInfo

   async def fetch_lineup_cards(date: str) -> List[Dict]:
       """Fetch lineup cards from MLB Stats API."""
       url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&gameType=R&hydrate=lineup"
       async with httpx.AsyncClient() as client:
           resp = await client.get(url, timeout=10.0)
           resp.raise_for_status()
           data = resp.json()
           # Parse lineup data, extract starters vs TBD
           return data

   async def update_probable_pitchers_with_lineups(date: str, session: AsyncSession):
       """Resolve 'TBD' opponents using lineup card data."""
       lineups = await fetch_lineup_cards(date)
       # Update ProbablePitcherSnapshot records with actual SP names from lineups
   ```

2. **Add to morning pipeline** in `backend/services/daily_ingestion.py`:
   ```python
   async def run_morning_ingestion():
       """Complete by 10:00 AM ET SLA."""
       # ... existing ingestion ...
       await fetch_lineup_cards(str(target_date))
       await update_probable_pitchers_with_lineups(str(target_date), session)
   ```

3. **Freshness/SLA tracking** in `backend/models.py`:
   ```python
   class LineupCardSnapshot(Base):
       id = Column(Integer, primary_key=True)
       game_date = Column(Date)
       team = Column(String)
       opponent = Column(String)  # Resolved from lineup card
       lineup_data = Column(JSONB)
       fetched_at = Column(DateTime, default=_now_et)
       sla_met = Column(Boolean, default=False)  # True if fetched before 10:00 AM ET
   ```

4. **API endpoint** in `backend/routers/fantasy.py`:
   ```python
   @router.get("/lineup-cards")
   async def get_lineup_cards(date: str = None):
       """Returns lineup cards with freshness and SLA status."""
       if date is None:
           date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
       # Query LineupCardSnapshot and return
   ```

5. **Integration with optimizer** in `backend/fantasy_baseball/daily_lineup_optimizer.py`:
   - Use lineup card data to confirm starters before solving
   - SLA check: if `sla_met = False`, warn about incomplete lineup information

**Verification:**
- `python -m py_compile backend/ingestion/mlb_lineup_ingestion.py backend/services/daily_ingestion.py backend/models.py backend/routers/fantasy.py`
- Test: `GET /api/fantasy/lineup-cards?date=2026-06-11` returns valid lineup data
- Verify SLA: `sla_met = True` for today's cards

**Frontend (Kimi CLI) — AFTER BACKEND COMPLETE:**
- Show lineup card freshness badge in lineup optimizer
- Display SLA status (complete by 10:00 AM ET)

---

## POWER SHELL ROUTING COMMANDS

### DEPLOYMENT INVESTIGATION (Codex)
```powershell
# Review lifespan() to identify why table didn't create
cd C:\Users\sfgra\repos\Fixed\cbb-edge
code backend/main.py
# Search for "daily_availability_overrides" around lines 205-218

# Alternative: Create manual migration script
code scripts/create_daily_availability_overrides.py
```

### P1 TASK 1: Freshness (Claude Code)
```powershell
cd C:\Users\sfgra\repos\Fixed\cbb-edge
gh copilot run --model claude-sonnet-4 --editor-code "Implement Task 1: Global Freshness Normalization. See HERMES_P1_ROUTING.md TASK 1 for full spec. Create FreshnessSeverity enum and FreshnessState in backend/contracts.py. Create freshness_service.py with compute_freshness_state function using datetime.now(ZoneInfo('America/New_York')). Update dashboard_service.py, waiver_edge_detector.py, daily_lineup_optimizer.py to include freshness. Add GET /api/fantasy/freshness endpoint in routers/fantasy.py. Run py_compile on all modified files. Run pytest tests/ -q. No frontend code." --repo ./
```

### P1 TASK 2: ETA Watchdog (Claude Code)
```powershell
cd C:\Users\sfgra\repos\Fixed\cbb-edge
gh copilot run --model claude-sonnet-4 --editor-code "Implement Task 2: ETA Expiration Watchdog. See HERMES_P1_ROUTING.md TASK 2 for full spec. Add expired_eta and eta_expiration_checked_at columns to IngestedInjury model in backend/models.py. Create eta_expiration_watchdog.py with check_eta_expiration function using datetime.now(ZoneInfo('America/New_York')). Add nightly cron job at 6:00 AM ET in backend/main.py lifespan(). Add ETA_EXPIRED enum to InjStatus in backend/contracts.py. Update waiver/roster endpoints to show 'ETA PASSED — STATUS UNKNOWN' when expired_eta=True. Validate Colt Emerson case (ETA Jun 9 passed). Run py_compile." --repo ./
```

### P1 TASK 3: Predictive Stats (Claude Code)
```powershell
cd C:\Users\sfgra\repos\Fixed\cbb-edge
gh copilot run --model claude-sonnet-4 --editor-code "Implement Task 3: Predictive Stats Pipeline. See HERMES_P1_ROUTING.md TASK 3 for full spec. Add PREDICTIVE_STATS_V1_ENABLED feature flag (default false) to config_service.py. Create PredictiveStatsSnapshot model in backend/models.py with fip, x_fip, sieras, xwoba, hard_hit_pct, wrc_plus columns. Gate predictive stats in savant_ingestion.py, pybaseball_loader.py, pitcher_deep_dive.py behind the flag. Add PredictiveStatsIngestResult observability contract. Add GET /api/fantasy/players/{player_id}/predictive-stats endpoint in routers/fantasy.py (404 when flag false). Run py_compile. Test with PREDICTIVE_STATS_V1_ENABLED=false and =true." --repo ./
```

### P1 TASK 4: MLB Lineup API (Claude Code)
```powershell
cd C:\Users\sfgra\repos\Fixed\cbb-edge
gh copilot run --model claude-sonnet-4 --editor-code "Implement Task 4: MLB Lineup API Integration. See HERMES_P1_ROUTING.md TASK 4 for full spec. Create mlb_lineup_ingestion.py with fetch_lineup_cards (MLB Stats API /api/v1/schedule?hydrate=lineup) and update_probable_pitchers_with_lineups. Add to morning_ingestion.py daily pipeline (SLA: 10:00 AM ET). Create LineupCardSnapshot model in backend/models.py with sla_met flag. Add GET /api/fantasy/lineup-cards?date=YYYY-MM-DD endpoint in routers/fantasy.py. Integrate with daily_lineup_optimizer.py to confirm starters using lineup data. Use datetime.now(ZoneInfo('America/New_York')). Run py_compile. Test endpoint returns valid lineup data." --repo ./
```

### FRONTEND COMPONENTS (Kimi CLI) — AFTER ALL BACKEND COMPLETE
```powershell
# Shared FreshnessBadge
kimi "Create frontend/src/components/freshness/FreshnessBadge.tsx. Props: severity ('fresh'|'warning'|'critical'|'unknown'), minutesAgo, warningText, isClickable. Show green badge for fresh, orange for warning >60min, red badge for critical >120min. Click to refresh. Use React 18+ with TypeScript."

# Apply to pages
kimi "Apply FreshnessBadge to War Room header (war-room/page.tsx), Dashboard header (dashboard/_components/dashboard-client.tsx), Waiver wire (war-room/waiver/page.tsx), Roster (war-room/roster/page.tsx), Streaming Station (war-room/streaming/page.tsx). Use GET /api/fantasy/freshness data."

# ETA expiration badges
kimi "Add red 'ETA PASSED — STATUS UNKNOWN' badge to injury cards in waiver and roster when backend returns status='ETA_EXPIRED' or expired_eta=true. Preserve original ETA text for reference."

# Predictive stats columns
kimi "Add predictive stats table to player detail page (when PREDICTIVE_STATS_V1_ENABLED=true): FIP, xFIP, SIERA, xwOBA, Hard-Hit%, wRC+. Show refresh timestamp and staleness indicator. Hide column when feature disabled."

# Lineup card SLA status
kimi "Add lineup card freshness badge to lineup optimizer. Show 'Lineups complete ✓' if sla_met=True, 'Lineups pending ⏳' if sla_met=False. Display last_updated timestamp."
```

---

## COMPLETION CRITERIA

Each task is complete when:
1. All backend files pass `python -m py_compile`
2. Relevant pytest tests pass (no regressions)
3. API endpoint returns valid JSON
4. Code uses `datetime.now(ZoneInfo("America/New_York"))` not `datetime.utcnow()`
5. Changes documented in HANDOFF.md session log

P1 Bundle complete when:
- All 4 tasks verified locally
- Pushed to stable/cbb-prod branch
- Deployment-ready (pending DB table fix resolution)
- Frontend components complete (Kimi CLI)

---

## NOTES

- **Deployment blocker separate:** daily_availability_overrides table issue is orthogonal to P1
- **Frontend after backend:** Kimi CLI should only start after all backend endpoints verified
- **Feature flags:** Predictive stats pipeline defaults to disabled (safe deployment)
- **SLA tracking:** MLB lineup cards include freshness/SLA status for observability