# CBB Edge — Platform Expansion Roadmap
*Generated: 2026-03-19 | Architect: Claude Sonnet 4.6*

---

## Current Status

| Area | Status | Notes |
|------|--------|-------|
| V8 Betting Model | ✅ LOCKED (GUARDIAN) | No changes until April 7 |
| CI/CD Pipeline | ✅ Green | TypeScript clean, pytest passing |
| Mobile UI | ✅ Fixed | Sidebar drawer, bet card UX |
| Railway Deploy | ✅ Working | Backend + Frontend services |
| CLV Tracking | ✅ Stable | Null-safety fix merged |

---

## Phase 0 — Foundation Hardening (Sprint 1, Week 1)
*Prerequisite for all expansion work*

### Checklist
- [ ] Set `RAILWAY_TOKEN` and `NEXT_PUBLIC_API_URL` in GitHub Secrets
- [ ] Confirm Railway DB connection string in production env
- [ ] Add `FANTASY_BASEBALL_API_KEY` to `.env.example`
- [ ] Create `tasks/lessons.md` if not exists
- [ ] Tag current release as `v0.8.0-cbb-stable`

### Agent Delegation — Agent: DevOps-Bot
**Task:** Configure Railway environment variables and verify health checks.
**Files:** `railway.json`, `.env.example`, `.github/workflows/deploy.yml`
**Command:** `gh secret set RAILWAY_TOKEN --body <value>`
**Report:** Confirm `/health` returns 200 on production URL.

---

## Phase 1 — Fantasy Baseball Integration (Sprint 2–3, Weeks 2–4)

### 1.1 UI/UX Vision
- **Draft Board page**: `/fantasy/draft` — ranked player table with ADP, projected stats, injury flags. Filter by position, team, tier.
- **Lineup Optimizer**: `/fantasy/lineup` — drag-drop lineup builder with projected points and stack recommendations.
- **Standings & Waiver**: `/fantasy/standings` — league standings + waiver wire recommendations sorted by FAAB value.
- **User Journey**: User lands on Draft Board → filters by position → clicks player for deep-dive modal (projections, news, ownership%) → adds to lineup optimizer → exports lineup.

### 1.2 Database Schema Updates
```sql
-- New tables (add to backend/models.py)
CREATE TABLE fantasy_players (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) UNIQUE NOT NULL,   -- external ID (MLB, ESPN, etc.)
    name VARCHAR(100) NOT NULL,
    team VARCHAR(50),
    position VARCHAR(20),
    injury_status VARCHAR(20) DEFAULT 'active',
    adp FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE fantasy_projections (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) REFERENCES fantasy_players(player_id),
    projection_date DATE NOT NULL,
    source VARCHAR(30) NOT NULL,             -- 'fangraphs', 'steamer', 'zips'
    projected_hr FLOAT,
    projected_avg FLOAT,
    projected_ops FLOAT,
    projected_sb FLOAT,
    projected_era FLOAT,
    projected_whip FLOAT,
    projected_k9 FLOAT,
    projected_sv FLOAT,
    fantasy_points_est FLOAT,
    UNIQUE(player_id, projection_date, source)
);

CREATE TABLE fantasy_lineups (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    lineup_date DATE NOT NULL,
    platform VARCHAR(30),                    -- 'draftkings', 'fanduel', 'yahoo'
    positions JSONB,                         -- {"P": "player_id", "C": "player_id", ...}
    projected_points FLOAT,
    actual_points FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 1.3 Backend API Surface
```
GET  /api/fantasy/players              # list with filters: position, team, injury_status
GET  /api/fantasy/players/{id}         # player detail + projections + news
GET  /api/fantasy/projections/today    # all projections for today's slate
POST /api/fantasy/lineups              # save optimized lineup
GET  /api/fantasy/lineups/{date}       # retrieve saved lineup
GET  /api/fantasy/waiver               # ranked waiver wire recommendations
```

### 1.4 Sub-Tasks for Agent Delegation

**Agent A — DB Schema Agent**
> Implement `fantasy_players`, `fantasy_projections`, `fantasy_lineups` tables in `backend/models.py`. Add SQLAlchemy ORM classes following the patterns in existing models (e.g., `Game`, `Prediction`). Update `scripts/init_db.py` to create these tables. Add corresponding Pydantic schemas to `backend/schemas.py`. Run `pytest tests/ -v` to confirm no regressions. Report: list of new model class names and schema names created.

**Agent B — Fantasy Service Agent**
> Create `backend/services/fantasy.py`. Implement: `fetch_projections(date: date) -> list[PlayerProjection]` (stub returning mock data initially), `rank_players(projections, scoring_system='standard') -> list[RankedPlayer]`, `optimize_lineup(players, positions, budget=50000) -> Lineup` (basic greedy optimizer). Follow patterns from `backend/services/ratings.py`. Add unit tests in `tests/test_fantasy.py`. Report: function signatures and test coverage %.

**Agent C — Fantasy API Routes Agent**
> Add routes to `backend/main.py` for all 6 fantasy endpoints listed in 1.3. Use `verify_api_key` dependency (same as existing routes). Wire to `backend/services/fantasy.py`. Add Pydantic request/response schemas. Test with `curl -H "X-API-Key: dev-key-insecure" http://localhost:8000/api/fantasy/players`. Report: all 6 endpoints return 200 with mock data.

**Agent D — Fantasy Frontend Agent**
> Create `frontend/app/(dashboard)/fantasy/` directory with:
> - `page.tsx` → redirect to `/fantasy/draft`
> - `draft/page.tsx` → player table with columns: Rank, Player, Pos, Team, ADP, Proj Pts, Status. Fetch from `GET /api/fantasy/projections/today`. Follow patterns from `frontend/app/(dashboard)/today/page.tsx`.
> - `lineup/page.tsx` → lineup builder grid (9 positions for baseball). POST to `/api/fantasy/lineups`.
> - `standings/page.tsx` → placeholder with "Coming Soon" until league sync is built.
> Add nav link in `frontend/components/layout/sidebar.tsx` under a new "Fantasy" section. Report: screenshots or describe what renders on each page.

---

## Phase 2 — Advanced Betting Markets (Sprint 4–5, Weeks 5–7)

### 2.1 UI/UX Vision
- **Moneyline Dashboard**: Extend `/today` page with a Moneyline tab. Show ML odds, implied probability, model probability, edge.
- **Player Props**: New page `/props` — table of player prop bets (points, rebounds, assists over/under) with model edge ratings.
- **User Journey**: User clicks "Props" in sidebar → sees today's available props → sorts by edge → clicks prop to see historical hit rate → adds to bet tracker.

### 2.2 Database Schema Updates
```sql
CREATE TABLE player_props (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id),
    player_name VARCHAR(100) NOT NULL,
    player_team VARCHAR(50),
    prop_type VARCHAR(50) NOT NULL,          -- 'points_over', 'rebounds_under', etc.
    line FLOAT NOT NULL,
    over_odds INTEGER,
    under_odds INTEGER,
    model_probability FLOAT,
    edge_pct FLOAT,
    verdict VARCHAR(20),                     -- 'BET_OVER', 'BET_UNDER', 'PASS'
    result VARCHAR(20),                      -- 'hit', 'miss', 'push', null
    actual_value FLOAT,
    captured_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE moneyline_edges (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id),
    team VARCHAR(100) NOT NULL,
    ml_odds INTEGER NOT NULL,
    model_win_prob FLOAT NOT NULL,
    implied_prob FLOAT NOT NULL,
    edge_pct FLOAT NOT NULL,
    verdict VARCHAR(20),
    result VARCHAR(20),
    captured_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2.3 Sub-Tasks for Agent Delegation

**Agent E — Moneyline Model Agent**
> In `backend/betting_model.py` (READ-ONLY — do NOT modify existing methods). Instead, create `backend/services/moneyline.py` with `analyze_moneyline(game_data, model_win_prob) -> MoneylineEdge`. Use Shin vig removal from `backend/betting_model.py` as reference — import `shin_vig_removal` if it's exported, otherwise re-implement it independently. Add `moneyline_edges` table queries. Add tests in `tests/test_moneyline.py`. Report: function signature, Shin method used (import vs reimplementation), test pass/fail.

**Agent F — Props Scraper Agent**
> Create `backend/services/props.py`. Implement `fetch_props(date: date) -> list[PlayerProp]` that reads from The Odds API player props endpoint (see existing `backend/services/odds.py` for API client patterns). Store results in `player_props` table. Add `GET /api/props/today` route. Add tests with mocked API response. Report: which prop markets are available from the API and which are stubbed.

**Agent G — Props Frontend Agent**
> Create `frontend/app/(dashboard)/props/page.tsx`. Table view: Player | Team | Prop | Line | Odds | Edge | Verdict. Color-code edges (green >3%, yellow 1-3%, gray <1%). Add "Props" nav link in sidebar. Follow patterns from `frontend/app/(dashboard)/today/page.tsx` for data fetching. Report: component renders with mock data.

---

## Phase 3 — Discord Bot (Sprint 6, Week 8)

### 3.1 UI/UX Vision
- `/picks` — posts today's BET verdicts with team, line, odds, edge, Kelly fraction
- `/bankroll` — shows current bankroll, ROI, win rate, drawdown status
- `/performance` — 7-day and 30-day CLV, record, P&L
- `/prop <player>` — look up a specific player's prop edges
- Scheduled daily post at 10 AM ET with picks for the day

### 3.2 Technical Architecture
```
backend/
└── discord_bot_runner.py   # Entry point: python backend/discord_bot_runner.py
backend/services/
└── discord_bot.py          # Bot logic using discord.py (slash commands)
```

```python
# discord_bot_runner.py structure
import discord
from discord.ext import commands, tasks
from backend.services import discord_bot

bot = commands.Bot(command_prefix='/', intents=discord.Intents.default())
bot.add_cog(discord_bot.CBBPicksCog(bot))
bot.run(os.environ['DISCORD_BOT_TOKEN'])
```

### 3.3 Sub-Tasks for Agent Delegation

**Agent H — Discord Bot Agent**
> Create `backend/services/discord_bot.py` and `backend/discord_bot_runner.py`. Use `discord.py>=2.3`. Implement slash commands: `/picks`, `/bankroll`, `/performance`. Each command calls the existing FastAPI backend via `httpx.AsyncClient` (use `INTERNAL_API_URL` env var, default `http://localhost:8000`). Auth with `X-API-Key` header using `API_KEY_USER1` env var. Add `discord.py` and `httpx` to `requirements.txt`. Add `DISCORD_BOT_TOKEN` and `INTERNAL_API_URL` to `.env.example`. Report: commands implemented, sample output format for each.

---

## Phase 4 — Admin Risk Dashboard (Sprint 7, Week 9)

### 4.1 UI/UX Vision
- **Risk Overview**: `/admin/risk` — real-time portfolio exposure, drawdown gauge, circuit breaker status
- **User Management**: `/admin/users` — API key rotation, usage stats, permission levels
- **Model Parameters**: `/admin/model` — read-only view of current V8 weights, SD, thresholds (editable post-April 7)
- **Odds Monitor**: `/admin/odds-monitor` — live feed of line movements, quota status

### 4.2 Sub-Tasks for Agent Delegation

**Agent I — Admin Frontend Agent**
> Create `frontend/app/(dashboard)/admin/` directory with:
> - `risk/page.tsx` — fetch `GET /admin/portfolio/status`, show exposure bar, drawdown gauge (use a Radix Progress component), circuit breaker badge (red/green).
> - `odds-monitor/page.tsx` — fetch `GET /admin/odds-monitor/status`, show last poll time, quota remaining, recent movements table.
> - `model/page.tsx` — fetch `GET /admin/scheduler/status` + hardcode current V8 parameters in a read-only table.
> Add "Admin" section in sidebar (only visible when API key has admin privileges — check via a `GET /api/me` endpoint or local storage flag). Report: pages render with real data from backend.

---

## Phase 5 — OpenClaw Autonomous Ops (Sprint 8+, Week 10+)

### 5.1 Vision
Replace manual `POST /admin/run-analysis` trigger with autonomous multi-agent pipeline:
- **Scheduler Agent**: Watches for game slates, triggers analysis at optimal times
- **Data Agent**: Fetches ratings, injuries, odds in parallel
- **Model Agent**: Runs V8 analysis (read-only during GUARDIAN lock)
- **Risk Agent**: Applies portfolio constraints, generates sized recommendations
- **Notifier Agent**: Posts to Discord, logs to DB, sends alerts

### 5.2 Architecture Pattern (C-Compiler TDD)
```
1. Write integration test describing desired end-to-end behavior
2. Run test → RED (fails as expected)
3. Delegate implementation to specialized agent
4. Agent writes code → runs test → GREEN
5. Architect reviews, merges, updates HANDOFF.md
```

### 5.3 Sub-Tasks for Agent Delegation

**Agent J — OpenClaw Orchestrator Agent**
> Create `backend/services/openclaw.py`. Implement `OpenClawOrchestrator` class with `async run_pipeline(date: date)` that orchestrates: (1) fetch injuries, (2) fetch odds, (3) fetch ratings, (4) run analysis, (5) apply portfolio, (6) post Discord picks. Use `asyncio.gather` for steps 1-3 in parallel. Wire to APScheduler in `backend/main.py` (replace existing nightly job). Add integration test `tests/test_openclaw.py` that mocks all external calls and asserts the pipeline produces at least one `GameAnalysis` object. Report: pipeline stages, async gather groupings, test result.

---

## Immediate Next Actions (This Session)

1. **[NOW]** Finish writing this roadmap to `tasks/todo.md` ✅
2. **[NOW]** Commit and push to `claude/fix-clv-null-safety-fPcKB`
3. **[NEXT SESSION]** Start Phase 0: set Railway secrets, tag v0.8.0-cbb-stable
4. **[NEXT SESSION]** Start Phase 1, Agent A: DB schema for fantasy tables

---

## Risk Flags

| Risk | Mitigation |
|------|-----------|
| GUARDIAN lock on `betting_model.py` | Phase 2 ML model goes in new `services/moneyline.py` |
| The Odds API props endpoint may not cover all markets | Agent F must document available markets in `tasks/lessons.md` |
| Discord bot token exposure | Store in Railway env vars only, never commit |
| Fantasy Baseball data source unclear | Start with Fangraphs CSV import stub; wire live API in Phase 1.5 |
| OpenClaw autonomy risk | Keep human-in-the-loop for April 7 model unlock; autonomous only for data fetching |

---

## Agent Prompts — Copy-Paste Ready

### PROMPT-A: Fantasy DB Schema
```
You are working on the CBB_Betting repository (Python, FastAPI, PostgreSQL, SQLAlchemy 2.0).
Task: Add fantasy baseball database tables.

Files to modify:
- backend/models.py: Add SQLAlchemy ORM classes FantasyPlayer, FantasyProjection, FantasyLineup
- backend/schemas.py: Add corresponding Pydantic schemas
- scripts/init_db.py: Add table creation for the new models

Table specs are in tasks/todo.md under "Phase 1 — 1.2 Database Schema Updates".

Follow existing patterns (look at the Game and Prediction classes in backend/models.py).
Run: pytest tests/ -v --tb=short
Report: list all new class names created, confirm tests still pass.
```

### PROMPT-B: Fantasy Service
```
You are working on the CBB_Betting repository (Python, FastAPI, PostgreSQL, SQLAlchemy 2.0).
Task: Create backend/services/fantasy.py with fantasy baseball player projection service.

Read backend/services/ratings.py for patterns to follow.
Implement:
1. fetch_projections(date: date) -> list[PlayerProjection]  # stub with mock data initially
2. rank_players(projections, scoring_system='standard') -> list[RankedPlayer]
3. optimize_lineup(players, positions, budget=50000) -> Lineup  # basic greedy optimizer

Add tests in tests/test_fantasy.py.
Run: pytest tests/test_fantasy.py -v
Report: all function signatures, test count, pass/fail.
```

### PROMPT-C: Discord Bot
```
You are working on the CBB_Betting repository (Python, FastAPI, discord.py).
Task: Create a Discord bot that surfaces CBB Edge picks.

Create:
1. backend/services/discord_bot.py — slash commands /picks /bankroll /performance
2. backend/discord_bot_runner.py — entry point

Each command calls the FastAPI backend at INTERNAL_API_URL (env var, default http://localhost:8000)
with X-API-Key header using API_KEY_USER1 env var.

Use discord.py>=2.3 and httpx for async HTTP.
Add both to requirements.txt.
Add DISCORD_BOT_TOKEN and INTERNAL_API_URL to .env.example.

For /picks: call GET /api/predictions/today, format as Discord embed showing team, line, odds, edge.
For /bankroll: call GET /admin/portfolio/status, show exposure and drawdown.
For /performance: call GET /api/performance/summary, show 7d ROI and CLV.

Do NOT run the bot (requires token). Instead run: python -c "from backend.services.discord_bot import CBBPicksCog; print('Import OK')"
Report: commands implemented, embed format for /picks.
```
