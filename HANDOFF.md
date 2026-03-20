# OPERATIONAL HANDOFF (EMAC-072)

> Ground truth as of **March 20, 2026**. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Full enhancement plan: `tasks/cbb_enhancement_plan.md` · V9.2 spec: `reports/K12_RECALIBRATION_SPEC_V92.md`
> **Frontend Migration:** `FRONTEND_MIGRATION.md` · Task tracker: `tasks/todo.md`

---

## 0-A. FRONTEND MIGRATION STATUS (March 18, 2026 ~18:00 ET)

**Phase 1 — Core Analytics Pages: ✅ ALL 5 PAGES COMPLETE AND FIXED**

All pages now have correct API field mappings verified against `reports/api_ground_truth.md` (Kimi spec).

| Page | Key Fixes |
|------|-----------|
| `/performance` | `summary.overall.roi`, decimal×100, rolling_windows shape |
| `/clv` | `mean_clv` (was `avg_clv_points`), `distribution{}` object→chart, `top_10_clv` array |
| `/bet-history` | `timestamp` (was `placed_at`), removed `clv_grade` (not in API) |
| `/calibration` | `calibration_buckets` (was `bins`), `bin` field (was `label`), nullable brier_score |
| `/alerts` | Uppercase severity `WARNING/CRITICAL`, added `live_alerts` section |

**lib/types.ts** — all 5 interfaces rewritten from ground truth.
**Sidebar** — `drawdown_pct` fixed (was `current_drawdown_pct`).

**Next step:** OpenClaw validates all 5 pages (7-point checklist). See `FRONTEND_MIGRATION.md`.
**Phase 2 scope:** `/predictions` page, odds ticker, admin panel. Blocked on OpenClaw PASS.

---

## 0. ARCHITECT DECISION (March 20, 2026 — EMAC-073 Update 2)

**Session focus:** Phase 5 complete + Platform Expansion Phase 1 (Fantasy Baseball) complete.

### What was delivered this session (Update 2)

1. **Phase 5 — Selective Polish: COMPLETE**
   - `bracket/error.tsx` + `today/error.tsx` (Next.js App Router error boundaries) — pre-existing
   - `bracket/loading.tsx` + `today/loading.tsx` (Suspense fallbacks) — added and committed

2. **Platform Expansion Phase 1 — Fantasy Baseball: COMPLETE**
   - `scripts/migrate_v7.py` — idempotent migration for `fantasy_draft_sessions`, `fantasy_draft_picks`, `fantasy_lineups` tables
   - `lib/types.ts` — added `DraftSession`, `DraftPick`, `CreateDraftSessionResponse`, `RecordPickResponse`
   - `lib/api.ts` — added `fantasyCreateSession()`, `fantasyRecordPick()`, `fantasyGetSession()`
   - `frontend/app/(dashboard)/fantasy/page.tsx` — full rewrite with two tabs:
     - **Draft Board** tab: read-only player table (preserved)
     - **Live Draft** tab: interactive draft assistant
       - Setup form: draft position, teams, rounds
       - Treemendous snake pick order algorithm (R1-R2 linear, R3+ alternating)
       - Status bar: pick counter, round, current drafter, "my turn in N picks"
       - Available players table with "Mine" (my pick) + "Taken" (opponent pick) buttons
       - My Roster panel: shows all my picks in order
       - Recommendations panel: top 5 recommendations from backend after each pick
       - Session persisted to localStorage (survives page refresh)
   - `fantasy/error.tsx` + `fantasy/loading.tsx` — error boundary and Suspense skeleton

3. **TypeScript: clean** (`tsc --noEmit` — 0 errors)

### Phase Status (updated)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 — Foundation | Partial | Railway secrets need manual set |
| Phase 1-4 — Frontend | ✅ DONE | All analytics, trading, tournament, mobile/PWA |
| Phase 5 — Polish (selective) | ✅ DONE (Mar 20) | error.tsx + loading.tsx on /bracket + /today + /fantasy |
| Platform Expansion Phase 1 | ✅ DONE (Mar 20) | Fantasy Baseball Live Draft Assistant |

### Pending (manual actions)
- **Gemini:** Run `scripts/migrate_v7.py` on Railway to create fantasy tables
- **User:** Push `v0.8.0-cbb-stable` tag to remote: `git push origin v0.8.0-cbb-stable`
- **User:** Set `RAILWAY_TOKEN` in GitHub repo secrets
- **User:** Confirm `NEXT_PUBLIC_API_URL` in Railway frontend environment

---

## 0. ARCHITECT DECISION (March 20, 2026 — EMAC-073)

**Session focus:** Phase 4 completed, Phase 0 foundation hardening, Phase 5 architecture decision.

### What was delivered this session

1. **Phase 4 — Mobile & PWA: COMPLETE**
   - Fixed 6 pages with bare `grid-cols-2` → `grid-cols-1 sm:grid-cols-2` (mobile-first)
   - Confirmed: sidebar drawer, manifest, icons, DataTable overflow-x-auto all already live
   - Tagged release `v0.8.0-cbb-stable`

2. **Phase 0 — Foundation Hardening (partial)**
   - `v0.8.0-cbb-stable` git tag created locally ✅
   - `FANTASY_BASEBALL_API_KEY` not needed — `YAHOO_CLIENT_ID`/`YAHOO_CLIENT_SECRET` already in `.env.example` ✅
   - `RAILWAY_TOKEN` in GitHub Secrets — **requires manual action** (no `gh` CLI available)
   - `NEXT_PUBLIC_API_URL` in Railway frontend env — **requires Railway dashboard** (already in `.env.local.example`)

3. **Phase 5 architecture decision**

   **Decision: Selective Phase 5, do NOT block platform expansion.**

   Rationale:
   - Tournament is live, `/bracket` and `/today` are the highest-traffic pages. A white screen during tournament play = bad UX.
   - Full Streamlit retirement has no urgency (no active users on port 8501 in production).
   - Error boundaries are 30 min of work per page; Streamlit retirement is days.

   **Phase 5 scope reduced to:**
   - [ ] Error boundary wrapper on `/bracket/page.tsx` (tournament — highest crash risk)
   - [ ] Error boundary wrapper on `/today/page.tsx` (daily trading view)
   - [ ] Suspense fallback skeletons on the two pages above
   - Skip: full Streamlit decommission (defer indefinitely)

   After these two boundaries are added: proceed directly to Platform Expansion Phase 1 (Fantasy Baseball).

### Phase Status (updated)

| Phase | Status | Pages |
|-------|--------|-------|
| Phase 0 — Foundation | ✅ DONE (partial) | tag created; Railway secrets need manual set |
| Phase 1 — Core Analytics | ✅ DONE | /performance, /clv, /bet-history, /calibration, /alerts |
| Phase 2 — Trading | ✅ DONE | /today, /live-slate, /odds-monitor |
| Phase 3 — Tournament | ✅ DONE | /bracket |
| Phase 4 — Mobile & PWA | ✅ DONE (Mar 20) | viewport, manifest, drawer, responsive grids |
| Phase 5 — Polish (selective) | ⏳ Next 30 min | error boundaries on /bracket + /today only |
| Platform Expansion Phase 1 | ⏳ After Phase 5 | Fantasy Baseball (DB schema → service → API → frontend) |

---

## 0-A-PREV. ARCHITECT DECISION (March 19, 2026 — EMAC-072)

**Session focus:** Next.js frontend Phase 2 + Phase 3 complete. Tournament fully live on new dashboard.

### What was delivered this session

1. **Phase 2 — Trading pages** (3 new pages, claw-validated, pushed to `claude/fix-clv-null-safety-fPcKB`):
   - `/today` — Today's Bets: BET/CONSIDER/PASS cards, KPI row, 5-min auto-refresh
   - `/live-slate` — All predictions incl. started games, filter tabs, sortable table
   - `/odds-monitor` — Odds monitor health + portfolio status + drawdown gauge
   - New API client methods: `todaysPredictions`, `todaysPredictionsAll`, `oddsMonitorStatus`, `portfolioStatusFull`

2. **Phase 3 — Tournament Bracket** (1 new page, pushed same branch):
   - `/bracket` — Monte Carlo bracket simulator UI (10k sims default, 1k/5k/10k/25k selector)
   - Champion hero card, Final Four grid, upset alerts (seed ≥10 with ≥35% win prob)
   - Advancement probability table with inline progress bars, region filter tabs
   - Wired to `GET /api/tournament/bracket-projection?n_sims=N`

3. **Sidebar activated** — Trading and Tournament sections now live (removed "soon" badge)

4. **Types + API** (`lib/types.ts`, `lib/api.ts`):
   - `GameData`, `PredictionEntry`, `TodaysPredictionsResponse`
   - `OddsMonitorStatus`, `PortfolioStatusFull`
   - `UpsetAlert`, `TeamAdvancement`, `BracketProjection`

### Phase Status (Frontend Migration)

| Phase | Status | Pages |
|-------|--------|-------|
| Phase 0 — Foundation | ✅ DONE | scaffold, auth, layout, design system |
| Phase 1 — Core Analytics | ✅ DONE + claw-validated | /performance, /clv, /bet-history, /calibration, /alerts |
| Phase 2 — Trading | ✅ DONE (this session) | /today, /live-slate, /odds-monitor |
| Phase 3 — Tournament | ✅ DONE (this session) | /bracket |
| Phase 4 — Mobile & PWA | ✅ DONE (Mar 20) | viewport meta, manifest, drawer, responsive grids |
| Phase 5 — Polish & Decommission | ⏳ Selective | error boundaries on /bracket + /today ONLY; skip Streamlit retire |

**GUARDIAN (Mar 18 – Apr 7):** Do NOT touch `betting_model.py`, `analysis.py`, or CBB services.

---

## 0-PREV. ARCHITECT DECISION (March 18, 2026)

**Session focus:** First Four day — confirmed odds pipeline is ready, market_ml population delegated to OpenClaw.

1. **Odds pipeline confirmed ready** — `backend/tournament/fetch_tournament_odds.py` exists and is wired to The Odds API + TeamMapper fuzzy match. Requires `THE_ODDS_API_KEY` (set in Railway, not local). All 64 teams currently have `market_ml: null`.
2. **futures_odds_2026.json confirmed present** — championship/F4/E8 futures from BetMGM/DraftKings (Selection Sunday lines). These are NOT game-level moneylines; `market_ml` needs per-game R64 lines from the live API.
3. **Delegation to OpenClaw** — task is to run `fetch_tournament_odds` on Railway and verify `market_ml` is populated before R64 (Thursday March 19 tip-offs begin).

**GUARDIAN (Mar 18 – Apr 7):** Do NOT touch `betting_model.py`, `analysis.py`, or CBB services. All pre-tournament fixes are COMPLETE — no further changes before Apr 7.

---

## 0-PREV. ARCHITECT DECISION (March 16, 2026)

**Session focus:** March Madness bracket release day — three parallel workstreams completed.

1. **Discord notification pipeline fixed** — morning brief, EOD results, and tournament bracket jobs were all silently logging instead of sending to Discord.
2. **Team mapping hardened** — 29 abbreviated "St" variants (e.g. "Kansas St Wildcats") added to prevent KenPom lookup failures; `test_team_mapping.py` (78 tests) added as regression guard.
3. **Monte Carlo bracket simulator built** — replaces deterministic "always pick the favorite" logic with historically-calibrated stochastic projection. Houston wins ~16% of simulated brackets (not 100%).

**GUARDIAN (Mar 18 – Apr 7):** Do NOT touch `betting_model.py`, `analysis.py`, or CBB services. All pre-tournament fixes are COMPLETE — no further changes before Apr 7.

---

## 1. EXECUTIVE SUMMARY

**Status:** ✅ **TOURNAMENT-READY — ALL SYSTEMS GREEN**

| Subsystem | Status | Notes |
|-----------|--------|-------|
| Discord Morning Brief | ✅ FIXED | Now calls `send_todays_bets()` at 7 AM ET |
| Discord EOD Results | ✅ NEW | Runs at 11 PM ET, posts W/L/P + P&L |
| Tournament Bracket Notifier | ✅ UPGRADED | Monte Carlo projection with upset alerts |
| Bracket Dashboard | ✅ NEW | Page 13 — champion %, F4 probs, full table |
| Team Mapping | ✅ HARDENED | 29 new "St" abbreviation entries, 78 tests |
| Duplicate Bet Cleanup | ✅ NEW | Admin Panel purge tool (deduplicates paper trades) |
| V9.1 Model | ✅ Active | Fatigue + sharp money + conf HCA + recency |
| Haslametrics Scraper | ✅ BUILT | `backend/services/haslametrics.py` — wire in Apr 7 |
| Railway Deploy | ✅ Live | Auto-deploys on push to `main` |

---

## 2. SYSTEM STATUS

### 2.1 Core Infrastructure

| Component | Status | Detail | Last Verified |
|-----------|--------|--------|---------------|
| Railway API | ✅ Healthy | All deps correct, preflight passes | 2026-03-16 |
| Database | ✅ Connected | PostgreSQL operational (365 teams) | 2026-03-16 |
| Scheduler | ✅ 12 jobs | +EOD results @11 PM, +bracket @6 PM | 2026-03-16 |
| Discord | ✅ Working | Morning brief + EOD results now firing | 2026-03-16 |
| Streamlit | ✅ 13 pages | New page 13: Tournament Bracket | 2026-03-16 |
| V9.1 Model | ✅ Active | Fatigue integration live | 2026-03-11 |
| Test suite | ✅ 683/686 pass | 3 pre-existing DB-auth failures | 2026-03-13 |

### 2.2 Model & Feature Components

| Feature | Status | File | Tests |
|---------|--------|------|-------|
| Fatigue Model (K-8) | ✅ LIVE | `backend/services/fatigue.py` | 23 pass |
| OpenClaw Lite (K-9) | ✅ LIVE | `backend/services/openclaw_lite.py` | 18 pass |
| Sharp Money (P1) | ✅ LIVE | `backend/services/sharp_money.py` | 15 pass |
| Conference HCA (P2) | ✅ LIVE | `backend/services/conference_hca.py` | 18 pass |
| Recency Weight (P3) | ✅ LIVE | `backend/services/recency_weight.py` | 20 pass |
| Seed-Spread Scalars (A-26) | ✅ LIVE | `betting_model.py` | 26 pass |
| Team Mapping (Mar 16) | ✅ HARDENED | `services/team_mapping.py` | **78 pass** |
| **Bracket Simulator (Mar 16)** | ✅ **NEW** | `services/bracket_simulator.py` | smoke tested |
| Haslametrics (G-R7) | ✅ BUILT | `backend/services/haslametrics.py` | 12 pass |
| Tournament SD Bump | ✅ LIVE | `betting_model.py` (1.15x neutral) | Active |
| Line Movement Monitor | ✅ LIVE | `odds_monitor.py` | Runs 30m |

### 2.3 Discord Job Schedule (Full)

| Time (ET) | Job | Status |
|-----------|-----|--------|
| 3:00 AM | Nightly analysis + picks | ✅ |
| 4:00 AM | Daily performance snapshot | ✅ |
| 4:30 AM | Performance sentinel | ✅ |
| 5:00 AM | Weekly recalibration (Sun only) | ✅ |
| 7:00 AM | **Morning brief → Discord** | ✅ FIXED |
| Every 30 min | Closing line capture | ✅ |
| Every 2 hr | Outcome updates | ✅ |
| Every 5 min | Odds monitor | ✅ |
| 6:00 PM | Tournament bracket notifier (Mar 14–20) | ✅ NEW |
| **11:00 PM** | **EOD results → Discord** | ✅ **NEW** |

---

## 3. COMPLETED WORK (This Session — March 16, 2026)

### 3.1 Discord Pipeline Fixes

**Root cause:** `_morning_briefing_job()` queried DB, generated narrative, then only called `logger.info()` — never sent to Discord.

**Fixes:**
1. `_morning_briefing_job()` now builds `bet_details` + `summary` from Prediction objects and calls `send_todays_bets()` (wrapped in try/except so Discord failure never kills the log)
2. `_end_of_day_results_job()` — new, at 11 PM ET: queries today's settled `BetLog`, sends W/L/P record + P&L units as a Discord embed
3. `_tournament_bracket_job()` — upgraded from "show First Four games" to running a 5,000-simulation Monte Carlo bracket projection and sending projected champion + Final Four + upset alerts to Discord

### 3.2 Monte Carlo Bracket Simulator

**File:** `backend/services/bracket_simulator.py` (521 lines)

**Algorithm:**
- Historical first-round win rates by seed matchup (1v16: 98.7%, 5v12: 64.7%, 8v9: 50.9%, etc.)
- AdjEM logistic win probability with 1.15x tournament SD bump (wider distribution = more upsets)
- Blending: R64 = 40% historical + 60% model; fades to 0% historical by Final Four
- 10,000 stochastic simulations — each game drawn `rng.random() < p`, NOT `argmax(p)`
- Returns per-team advancement probabilities for all 6 rounds
- Upset alerts: any R64 matchup where underdog has ≥35% win prob
- `_redistribute_into_regions()`: handles missing region data from BallDontLie API

**Sample output (5,000 sims):**
```
Champion: Houston (16.5%)
Final Four: Houston, Kansas, Duke, Alabama
Upset alerts: VCU vs Oklahoma (48%), Iowa vs Arkansas (46%)
```

### 3.3 Tournament Bracket Dashboard

**File:** `dashboard/pages/13_Tournament_Bracket.py`

**Sections:**
1. Champion + Final Four probability metrics (top of page)
2. Upset Alerts — R64 games where model gives underdog ≥35%
3. By-region bracket expanders with round-by-round projected winners
4. Cinderella rankings, futures odds EV calculator, interactive bracket input
5. Full advancement probability table (sortable, all 64 teams)

**API endpoint:** `GET /api/tournament/bracket-projection?n_sims=10000`

### 3.4 Team Mapping Hardening

**Added:** 29 abbreviated "St" variants to `ODDS_TO_KENPOM` (e.g. `"Kansas St Wildcats" -> "Kansas St."`) and added `"Kansas St Wildcats"` + `"Kansas St"` to `_MANUAL_OVERRIDES`.

**Test file:** `tests/test_team_mapping.py` (78 tests, 100% pass):
- All 5 Gemini audit examples
- All 29 abbreviated St forms (parametrized)
- Manual override priority, mascot stripping, dangerous-substring guard, 17-school regression

### 3.5 Duplicate Bet Cleanup

**Endpoint:** `POST /admin/cleanup/duplicate-bets?dry_run=true`

Finds and optionally deletes duplicate paper trade `BetLog` entries (same `game_id` + same calendar day). This was inflating bet counts — 7 bets on "Northwestern -6.5" all from the same game.

**Dashboard:** Admin Panel has a new "Duplicate Bet Cleanup" section with scan → preview → confirm checkbox → delete flow.

---

## 4. UPCOMING DEADLINES

| Date | Event | Status | Action |
|------|-------|--------|--------|
| **Mar 16 (Today)** | Bracket released ~6 PM ET | ✅ Notifier live | Fires automatically |
| **Mar 18** | First Four begins | ⏳ Monitor | Model running — GUARDIAN active |
| **Mar 20** | Fantasy Keeper Deadline | ⚠️ | User action needed |
| **Mar 23 7:30am ET** | Fantasy Draft Day | ⚠️ | Run `12_Live_Draft.py` |
| **Apr 7** | Guardian lifts — V9.2 Phase 2 | 🎯 | Execute Section 5 |

---

## 5. APRIL 7+ MISSION (Post-Guardian)

Execute in order. Run `pytest tests/ -q` before each commit.

**Why the model has been over-conservative:** V9.1 stacks SNR scalar (~0.70) x integrity scalar (~0.85) x fractional Kelly (÷2.0) = effective divisor ~3.37x. MIN_BET_EDGE fix (Phase 1) partially addressed this. Full fix is Phase 2 below.

### 5.1 V9.2 Phase 2 Params — `betting_model.py` / `analysis.py`
- `sd_mult` 1.0 → 0.80
- `ha` 2.419 → 2.85
- `SNR_KELLY_FLOOR` 0.50 → 0.75
- Reference: `reports/K12_RECALIBRATION_SPEC_V92.md`

### 5.2 Wire Haslametrics — `ratings.py`
- Scraper already built at `backend/services/haslametrics.py` (12 tests pass)
- Add `from backend.services.haslametrics import get_haslametrics_ratings` to `ratings.py`
- Assign EvanMiya's former 32.5% weight to Haslametrics in `CBBEdgeModel.weights`
- Reference: `docs/THIRD_RATING_SOURCE.md`

### 5.3 K-14 Pricing Engine Tracking — `analysis.py` + DB migration
- Add `pricing_engine` column to `Prediction` model (values: `"markov"` / `"gaussian"`)
- Write field per-prediction in analysis pipeline
- Reference: `reports/K13_POSSESSION_SIM_AUDIT.md`

### 5.4 Bump Version + Validate
- Set `model_version = 'v9.2'`, run full test suite, confirm BET rate improvement
- Target: BET rate 3% → 8-12%

---

## 6. NEXT CLAUDE SESSION PROMPT (post-Apr 7)

```
CONTEXT: Guardian window lifted. CBB model work resumes. All intelligence is in.

STATE:
- V9.1 is over-conservative (effective Kelly divisor ~3.37x vs intended ~2.0x)
- MIN_BET_EDGE already lowered to 1.8% (Phase 1, pre-tournament)
- Haslametrics scraper already built: backend/services/haslametrics.py (12 tests pass)
- K-11 confirms genuine positive CLV — recalibration is directionally correct
- All Discord jobs now working (morning brief, EOD results, bracket notifier)

MISSION EMAC-071: V9.2 Recalibration + Haslametrics
1. betting_model.py / analysis.py: sd_mult 1.0->0.80, ha 2.419->2.85, SNR_KELLY_FLOOR 0.50->0.75
   Read reports/K12_RECALIBRATION_SPEC_V92.md for exact justification
2. ratings.py: wire backend/services/haslametrics.py as 3rd source (32.5% weight, replaces EvanMiya)
   Read docs/THIRD_RATING_SOURCE.md for integration spec
3. analysis.py + models.py: add pricing_engine field to Prediction, write "markov"/"gaussian" per game
   Read reports/K13_POSSESSION_SIM_AUDIT.md for K-14 spec
4. Bump model_version to 'v9.2'. Run pytest tests/ -q. Confirm BET rate increase.

TARGET: BET rate 3% -> 8-12%. CLV already positive (K-11) -- just need to unblock the bets.
```

---

## 7. KNOWN ISSUES / WATCH LIST

| Issue | Severity | Status |
|-------|----------|--------|
| Negative CLV (-1.76% avg) | Medium | Bet earlier (opener tier); model is betting after sharp money moves lines |
| Pick'em bet win rate (8.3%) | Medium | Audit post-deduplication; may normalize |
| Fantasy Baseball (Yahoo OAuth) | Low | Deferred to post-tournament (Apr 7+) |
| `test_sharp_money.py` NameError | Low | Pre-existing: `Tuple` not imported from `typing` |
| EvanMiya dropped | Info | Intentional; 2-source (KP+BT) mode robust by design |

---

## 8. HIVE WISDOM (Updated March 16)

| Lesson | Source |
|--------|--------|
| KenPom is hard-required — missing team name → immediate PASS, game silently skipped | Team mapping audit |
| "Kansas St Wildcats" (no period) was missing from mapping — could confuse Kansas St. with Kansas (+20 AdjEM gap) | Team mapping fix |
| 29 abbreviated "St" school variants were missing from ODDS_TO_KENPOM | Team mapping fix |
| Discord morning brief was ONLY logging, never posting — check send calls after every job change | Discord audit |
| Monte Carlo bracket: using `argmax(win_prob)` always picks every favorite → add stochastic sampling | Bracket simulator |
| Historical upset rates fade after R64/R32 (survivor bias makes seeds less predictive deeper in tournament) | Bracket simulator |
| Tournament SD bump 1.15x — single-elimination has higher variance than regular season | Bracket simulator |
| Duplicate paper trades inflated bet counts 7x — always check for dedup when bet counts seem high | Duplicate cleanup |
| V9.1 effective Kelly divisor ~3.37x — calibrated params were for ÷2.0 | EMAC-067 |
| CLV > 0 = genuine edge. No amount of tuning fixes CLV < 0 | K-11 |
| Haslametrics uses play-by-play garbage-time filter — cleaner than EvanMiya | G-R7 |
| MIN_BET_EDGE 2.5% was too high given wide CI — 1.8% is the right pre-v9.2 value | K-12 |
| possession_sim: push-aware Kelly is worth keeping; add A/B monitoring not removal | K-13 |
| Bet settlement: use `_resolve_home_away()` — never raw string compare | EMAC-064 |
| Yahoo roster pre-draft returns `players:[]` (empty array) — handle gracefully | EMAC-063 |
| Prediction dedup: `run_tier` NULL causes duplicate rows — use `or_()` filter | EMAC-067 |
| Discord token must be in Railway Variables, not just .env | D-1 |
| Conference HCA: Big Ten 3.6 pts vs SWAC 1.5 pts = significant road differential | P2 |
| Recency weighting: 2x for last 3 days, 1.6x for last week in March | P3 |
| Sharp money detection: steam ≥1.5 pts in <30 min = high confidence signal | P1 |

---

## 9. ENVIRONMENT VARIABLES (Railway)

### Required (All Set)
```
DATABASE_URL=postgresql://...
THE_ODDS_API_KEY=...
KENPOM_API_KEY=...
API_KEY_USER1=...
DISCORD_BOT_TOKEN=...
DISCORD_CHANNEL_ID=1477436117426110615
```

### Optional
```
BALLDONTLIE_API_KEY=...     <- Needed for bracket seed data (tournament_data.py)
BARTTORVIK_USERNAME/PASSWORD (not set -- public CSV works without auth)
EVANMIYA_API_KEY (not set -- intentionally dropped)
```

---

## 10. QUICK REFERENCE

```bash
# Test suite
pytest tests/ -q
pytest tests/test_team_mapping.py -v    # team mapping regression guard

# New endpoints (March 16)
curl -H "X-API-Key: $API_KEY" https://{railway-url}/api/tournament/bracket-projection
curl -X POST -H "X-API-Key: $API_KEY" "https://{railway-url}/admin/cleanup/duplicate-bets?dry_run=true"

# Logs / deploy
railway logs --follow
streamlit run dashboard/app.py
```

---

## 11. HANDOFF PROMPTS

### GEMINI CLI (DevOps Strike Lead) — Run Fantasy DB Migration
```
MISSION: Run migrate_v7.py on Railway to create fantasy baseball tables.

CONTEXT (March 20, 2026 — EMAC-073):
- Platform Expansion Phase 1 (Fantasy Baseball) is complete on the codebase side.
- DB models are defined in backend/models.py: FantasyDraftSession, FantasyDraftPick, FantasyLineup.
- Migration script is ready: scripts/migrate_v7.py
- This is idempotent — safe to run even if tables already exist.

TASK 1 — Run migration on Railway:
  railway run python scripts/migrate_v7.py

Expected output (success):
  [ OK ] Created table fantasy_draft_sessions
  [ OK ] Created table fantasy_draft_picks
  [ OK ] Created table fantasy_lineups
  Verification:
    fantasy_draft_sessions: EXISTS
    fantasy_draft_picks: EXISTS
    fantasy_lineups: EXISTS

If tables already exist: [SKIP] messages are fine — still a SUCCESS.

TASK 2 — Verify draft board API is working:
  curl -s "https://{railway-url}/api/fantasy/draft-board?limit=5" \
       -H "X-API-Key: $API_KEY_USER1" | python3 -m json.tool | head -30
  Expected: JSON with "count" > 0 and "players" array.

TASK 3 — Test draft session creation:
  curl -s -X POST "https://{railway-url}/api/fantasy/draft-session?my_draft_position=7&num_teams=12&num_rounds=23" \
       -H "X-API-Key: $API_KEY_USER1"
  Expected: {"session_key": "...", "message": "Draft session created..."}

ESCALATE TO ARCHITECT if:
- Migration fails with a column conflict or constraint error
- Draft board API returns 500 or empty players list
- Railway deployment health check fails after migration

REPORTING: Confirm table creation in Discord #gemini-ops. Include row counts if any data already exists.
```

### OPENCLAW (Execution Agent) — ACTION REQUIRED BEFORE THURSDAY TIP-OFF
```
MISSION: Populate market_ml in bracket_2026.json with live R64 moneylines

CONTEXT (March 18, 2026 — First Four Day):
- The NCAA tournament bracket is live in data/bracket_2026.json
- All 64 teams currently have market_ml: null — the model runs model-only (no market blend)
- The fetch script is ready: backend/tournament/fetch_tournament_odds.py
- THE_ODDS_API_KEY is set in Railway environment variables
- futures_odds_2026.json has championship futures (NOT per-game lines — do not use for market_ml)

TASK 1 — Run fetch script on Railway (before Thursday March 19 tip-offs):
  railway run python -m backend.tournament.fetch_tournament_odds
  Expected output: "Wrote N market_ml values to data/bracket_2026.json" where N > 0
  If N = 0: check that THE_ODDS_API_KEY is valid and R64 lines have been posted

TASK 2 — Verify result:
  python3 -c "
  import json
  d = json.load(open('data/bracket_2026.json'))
  teams = [t for region in ['east','south','west','midwest'] for t in d[region]]
  populated = [t for t in teams if t.get('market_ml') is not None]
  print(f'{len(populated)}/64 teams have market_ml populated')
  for t in populated:
      print(f'  {t[\"name\"]} (#{t[\"seed\"]}): {t[\"market_ml\"]:+d}')
  "

TASK 3 — If API has no lines yet (lines sometimes post <24h before games):
  Re-run fetch script at 6 AM ET Thursday March 19 (before first tip-off ~12:15 PM)
  Set a reminder or cron: railway run python -m backend.tournament.fetch_tournament_odds

TASK 4 — Monitor First Four results (March 18-19):
  Games: UMBC/Howard vs Michigan (Midwest 16), PrairieView/Lehigh vs Florida (South 16),
         Texas/NC State vs BYU (West 11), Miami OH/SMU vs Tennessee (Midwest 11)
  After each game: update bracket_2026.json winner fields if bracket_simulator uses them

ESCALATE TO ARCHITECT if:
- THE_ODDS_API_KEY returns 401/403 (key expired or quota exhausted)
- No R64 lines appear by Wednesday night March 18 6 PM ET
- market_ml values look wrong (e.g., 1-seeds listed as underdogs)

REPORTING: Post completion to Discord #openclaw-briefs channel. Include count of teams updated.
```

### CLAUDE CODE (Master Architect) — Updated March 19
```
MISSION: Next.js frontend Phase 4 (Mobile & PWA) + ongoing tournament monitoring

SYSTEM STATE AS OF MARCH 19 (R64 Day 1):

FRONTEND (Next.js on branch claude/fix-clv-null-safety-fPcKB):
- Phase 0 ✅ Foundation (scaffold, auth, layout, design system)
- Phase 1 ✅ Core Analytics — /performance, /clv, /bet-history, /calibration, /alerts
- Phase 2 ✅ Trading — /today, /live-slate, /odds-monitor
- Phase 3 ✅ Tournament — /bracket (10k MC sims, champion hero, upset alerts, advancement table)
- Phase 4 ⏳ Mobile & PWA — viewport meta, touch targets ≥44px, install prompt
- Phase 5 ⏳ Polish — error boundaries, suspense fallbacks, retire Streamlit

BACKEND (GUARDIAN active — no model changes until Apr 7):
- All Discord jobs working: morning brief (7 AM), EOD results (11 PM), bracket notifier (6 PM)
- V9.1 model live with tournament SD bump (1.15x neutral site)
- Test suite: 683/686 pass (3 pre-existing DB-auth failures, non-blocking)

POSSIBLE NEXT ACTIONS (priority order):
1. Phase 4 — Mobile viewport + PWA manifest (see frontend/PHASE4_SPEC.md when written)
2. Phase 5 — Error boundaries on every page, loading suspense
3. Apr 7+: V9.2 Phase 2 (sd_mult→0.80, ha→2.85, SNR_KELLY_FLOOR→0.75, Haslametrics)

GUARDIAN: pytest tests/test_team_mapping.py before any team mapping changes.
GUARDIAN: no changes to betting_model.py, analysis.py, or backend/services/* until Apr 7.
```

### OPENCLAW (Execution Agent) — Phase 2 + Phase 3 Validation
```
MISSION: Validate Phase 2 (Trading) and Phase 3 (Tournament) frontend pages

CONTEXT (March 19, 2026 — R64 Day 1):
- Next.js frontend on branch: claude/fix-clv-null-safety-fPcKB
- Phase 2 pages just built: /today, /live-slate, /odds-monitor
- Phase 3 page just built: /bracket
- All pages are live in the sidebar (Trading section + Tournament section, "soon" removed)

VALIDATION CHECKLIST (run on each component file listed below):
Apply this checklist to each file — output PASS or list of issues with line numbers:

1. NULL SAFETY — any .field access on potentially undefined/null without ?. guard
2. EMPTY ARRAY — any .map() without a ?? [] fallback on the source
3. DECIMAL DISPLAY — any API field named roi/win_rate/edge/clv/prob displayed without ×100
4. LOADING STATE — every async section has a loading skeleton or spinner
5. CRASH RISK — toFixed/toString/toLocaleString called on a value that could be undefined
6. Object.entries() called without ?? {} guard on the argument
7. EMPTY STATE — if data is empty array or null, is there a user-visible message?

FILES TO VALIDATE:
- frontend/app/(dashboard)/today/page.tsx
- frontend/app/(dashboard)/live-slate/page.tsx
- frontend/app/(dashboard)/odds-monitor/page.tsx
- frontend/app/(dashboard)/bracket/page.tsx

ALSO CHECK:
- frontend/lib/types.ts — verify PredictionEntry, BracketProjection, OddsMonitorStatus match backend
  Compare against:
    backend/main.py lines 693-808 (bracket endpoint)
    backend/main.py lines 2538-2558 (portfolio + odds monitor endpoints)
    backend/main.py lines 983-1028 (predictions/today endpoint)
    backend/schemas.py (TodaysPredictionsResponse)

REPORTING: List each file, PASS or issues. If issues found, include exact line numbers and
the fix needed. Do not make code changes — report only. Claude Code will apply fixes.
```

### KIMI CLI (Deep Intelligence) — Phase 4 Spec Production
```
MISSION: Produce Phase 4 (Mobile & PWA) implementation spec for Next.js frontend

CONTEXT (March 19, 2026):
- Next.js 15 frontend in frontend/ directory
- Design system: zinc-950 background, JetBrains Mono numbers, amber/sky/emerald/rose signals
- Current layout: sidebar (w-60, fixed left) + header + main content area
- All pages use max-w-7xl or max-w-4xl containers
- TanStack Query v5 for data fetching
- Auth: API key in cookie, direct browser→Railway calls

TASK 1 — Audit current mobile gaps:
Read these files completely:
  frontend/app/(dashboard)/layout.tsx
  frontend/components/layout/sidebar.tsx
  frontend/components/layout/header.tsx
  frontend/app/layout.tsx (root layout — check viewport meta)
  frontend/package.json (check if next-pwa or any PWA dep is present)

TASK 2 — For each gap, produce the exact fix:
Mobile concerns to check:
  a. Viewport meta tag — is <meta name="viewport" content="width=device-width, initial-scale=1"> present?
  b. Sidebar on mobile — fixed w-60 sidebar blocks content on narrow screens. Does layout need a drawer/hamburger?
  c. Touch targets — are all buttons/links at least 44×44px (Tailwind: min-h-[44px] min-w-[44px])?
  d. DataTable horizontal scroll — do tables have overflow-x-auto wrappers on mobile?
  e. KpiCard grid — does grid-cols-4 collapse to grid-cols-2 on sm and grid-cols-1 on xs?

TASK 3 — PWA manifest:
  Determine if frontend/public/manifest.json exists. If not, write the complete content for one.
  App name: "CBB Edge", short_name: "CBBEdge", theme_color: "#09090b", background_color: "#09090b"
  Icons: list the icon sizes needed (192×192, 512×512, maskable 512×512)

TASK 4 — Output spec:
Save to: frontend/PHASE4_SPEC.md
Format: for each issue, provide:
  - Issue description
  - File to modify (exact path)
  - Exact code change (old → new)
  - Tailwind classes or JSX diff

Be exhaustive. Every mobile issue Claude Code needs to fix should have a complete, unambiguous fix spec.
Do not write production code — write the spec only.
```

---

**Document Version:** EMAC-073
**Last Updated:** March 20, 2026
**Status:** All Systems Green | Fantasy Draft Assistant LIVE | GUARDIAN active (no model changes until Apr 7)
**Branch:** `claude/fix-clv-null-safety-fPcKB`
**Pending:**
- **Gemini:** Run `scripts/migrate_v7.py` on Railway (see Section 11 prompt above)
- **Manual (user):** Push `v0.8.0-cbb-stable` tag: `git push origin v0.8.0-cbb-stable`
- **Manual (user):** Set `RAILWAY_TOKEN` in GitHub repo secrets
- **Manual (user):** Confirm `NEXT_PUBLIC_API_URL` in Railway frontend environment
- **Draft day (Mar 23):** Switch to Live Draft tab in `/fantasy`, enter position 7, click "Start Draft"
- **Apr 7:** V9.2 recalibration (read Section 5 of this doc)
