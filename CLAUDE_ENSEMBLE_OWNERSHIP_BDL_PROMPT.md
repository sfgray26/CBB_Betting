# Claude Code: Surgical Ensemble Audit — Yahoo Ownership & BallDontLie Underutilization

## System Directive

You are Claude Code conducting a **targeted forensic audit** on two specific failure modes in a production Fantasy Baseball platform. This is not a broad codebase review. You are investigating:

1. **Why every player shows "— owned" (0% ownership) in production** despite a committed `_enrich_ownership_batch()` fix
2. **Where BallDontLie MLB data could deliver meaningful, high-ROI improvements** that Yahoo/FanGraphs/Statcast cannot

You operate as an Ensemble Conductor with four experts. Each expert must produce a **root cause analysis** (not symptom lists) for the ownership bug, and a **prioritized opportunity matrix** (impact × effort) for BDL integration.

The codebase: FastAPI + PostgreSQL + Next.js + Railway. Production deploy `d319beb` lags local HEAD `20349c5` by 7+ commits.

---

## The Expert Panel

### 1. Technical Architect — Will Larson (Stripe/Uber, *Staff Engineer*)
**Mandate:** Own the ownership bug root cause. Treat this as a production incident post-mortem.
**Methodology:**
- Trace the exact data flow: Yahoo API response → `_enrich_ownership_batch()` → DB → API response → frontend render
- Identify every silent failure path (broad `except Exception`, swallowed logging, missing DB writes)
- Check if the fix in local HEAD (`20349c5`) was ever actually deployed to `d319beb`
- Verify the Yahoo endpoint contract: is `players;player_keys={keys}/ownership` still valid? What does Yahoo return when it fails?
- Check if ownership data is stored in `PositionEligibility.league_rostered_pct` or if it lives only in the in-memory player dict
- Validate the 25-player chunking logic: what happens if Yahoo returns partial data for a chunk?
- Review the ADP fallback path: `_estimate_ownership_from_adp()` — is it being called? Does it return real values?
- Check if the `_enrich_ownership_batch()` call is missing from `get_free_agents()` or `get_roster()` in the deployed code
- Demand to see the actual Yahoo API response shape for the ownership subresource (check `_make_request` or response logs)

**Key question:** "If I add a `print()` in `_enrich_ownership_batch()`, does it show non-zero values? If yes, where do they get lost before the frontend? If no, what's Yahoo actually returning?"

### 2. UI/UX Expert — Brad Frost (Atomic Design)
**Mandate:** Own the frontend ownership display. If the backend sends real data, does the UI show it? If the backend sends 0, does the UI communicate that gracefully?
**Methodology:**
- Trace the ownership field from API response → TypeScript type → component props → render path
- Check if `player.percent_owned` and `player.owned_pct` are both handled (the waiver page checks both; the roster page checks one)
- Verify the ownership formatting logic: does `pct.toFixed(0)` crash on null? Does the "— owned" fallback trigger too aggressively?
- Check the `OwnershipBadge` component: does it gate on `pct > 0` or `pct != null`? Is there a difference?
- Review if the dashboard waiver widget uses the same pipeline as the dedicated waiver page (they showed different values in prior UAT)
- Check if the roster page's `player.ownership_pct` field is populated from the roster API or from a separate call
- Validate that the UI doesn't show misleading ownership during loading states

**Key question:** "If the backend suddenly started sending `percent_owned: 45.0`, would every page show it correctly? Which page would break?"

### 3. Fantasy Baseball Expert — Ron Shandler (Baseball Forecaster)
**Mandate:** Own the BDL opportunity analysis. Where could BDL data meaningfully improve decision-making for a fantasy manager?
**Methodology:**
- Evaluate BDL's MLB injury endpoint (`/mlb/v1/player_injuries`) against Yahoo's injury data. Is BDL more current? More granular? Does it include DTD with estimated return dates?
- Evaluate BDL's game schedule endpoint (`/mlb/v1/games`) for two-start pitcher detection. Does it handle rainouts/postponements better than the current heuristic?
- Evaluate BDL's box stats endpoint (`/mlb/v1/stats`) for live stat validation. Could it cross-check Yahoo's "live" stats to catch sync lag?
- Evaluate BDL's player search (`/mlb/v1/players?search=`) for identity resolution. Could it replace the difflib fuzzy name matching that fails on accented names?
- Check if BDL provides platoon splits, park factors, or weather data that Yahoo/FanGraphs lack
- Prioritize by **manager decision impact**: what BDL integration would change a "start/sit" or "add/drop" decision this week?

**Key question:** "If I had to pick ONE BDL endpoint to integrate today, which one would save the most fantasy matchups this season?"

### 4. Quant Analyst — Dan Szymborski (ZiPS creator, FanGraphs)
**Mandate:** Own the data quality and statistical validity of both the ownership pipeline and any proposed BDL integrations.
**Methodology:**
- Review the ownership algorithm: is `percent_owned` from Yahoo's global player pool or league-specific? If global, is that the right denominator for waiver wire decisions?
- Check if the ADP-based ownership estimation has any statistical validity. ADP and rostered % correlate, but what's the R²? Is the fallback worse than showing "unknown"?
- Evaluate BDL's data quality: what are the missing data rates? Are BDL's counting stats (H, R, HR, K) aligned with official MLB data? What's the latency?
- Check if BDL's injury data could introduce **confirmation bias** — if BDL says "DTD" and Yahoo says "Active," which source wins? What's the merge strategy?
- Review the `get_mlb_stats()` endpoint for lookahead bias: are today's stats being used to project tomorrow before the games are final?
- Calculate the **expected value of information** for each BDL endpoint: EV = (probability of decision change) × (value of correct decision) − (integration cost)

**Key question:** "Is ownership percentage even a useful input for waiver decisions, or is it just vanity data? What does the correlation matrix say?"

---

## Audit Protocol

### Phase 1: Ownership Bug — Forensic Trace
Each expert independently traces the ownership data flow. They must answer:

**For Will (Backend Trace):**
1. Read `backend/fantasy_baseball/yahoo_client_resilient.py` lines 841–900. Does `_enrich_ownership_batch()` actually fetch data? What does Yahoo return?
2. Check if `_enrich_ownership_batch()` is called in `get_roster()` (line ~742) and `get_free_agents()` (line ~837). Are both call sites present in production code?
3. Read the `_make_request` or `_get` method. What happens when Yahoo returns 400/401/404 for the ownership subresource?
4. Check `backend/routers/fantasy.py` — the roster endpoint. Does it pass `percent_owned` through to the response?
5. Check the frontend `types.ts` — is `percent_owned` in the `RosterPlayer` and `WaiverAvailablePlayer` types?
6. Search for `percent_owned` in the DB schema (`backend/models.py`). Is there a column for it? Is `PositionEligibility.league_rostered_pct` the right place?

**For Brad (Frontend Trace):**
1. Read `frontend/app/(dashboard)/war-room/roster/page.tsx` — find where ownership is rendered. Which field does it read?
2. Read `frontend/app/(dashboard)/war-room/waiver/page.tsx` — find `OwnershipBadge`. What are its null/0/undefined handling rules?
3. Read `frontend/app/(dashboard)/dashboard/page.tsx` — find the waiver targets widget. Does it use `percent_owned` or `owned_pct`?
4. Read `frontend/lib/types.ts` — check the `RosterPlayer` and `WaiverTarget` interfaces. Are ownership fields optional or required?
5. Check if there's a mismatch: backend sends `percent_owned`, frontend expects `owned_pct`, or vice versa.

**For Ron (Fantasy Logic):**
1. Is ownership % even the right metric? Should we show `league_rostered_pct` (this league only) vs `percent_owned` (all leagues)?
2. If Yahoo's global ownership is 0% for obscure prospects but they're rostered in 80% of competitive leagues, what's the right number?
3. Does the current waiver scoring algorithm use ownership as a tiebreaker? If not, why display it?

**For Dan (Statistical Validity):**
1. What is the distribution of `percent_owned` values in the DB? Is it all zeros, or mixed?
2. If `_enrich_ownership_batch()` fails silently, does the system have any monitoring/alerts for zero ownership rates?
3. What's the correlation between ownership % and waiver "need score"? If near-zero, is ownership display low-value?

### Phase 2: BDL Opportunity — Impact/Effort Matrix
Each expert independently evaluates BDL endpoints. They must answer:

**Available BDL MLB Endpoints:**
- `GET /mlb/v1/games` — daily schedule + scores (used in `daily_ingestion.py`)
- `GET /mlb/v1/player_injuries` — IL + DTD list with injury types (used in `daily_ingestion.py`)
- `GET /mlb/v1/odds` — sportsbook lines per game (used in `daily_ingestion.py`)
- `GET /mlb/v1/players` — player lookup + search by name (NOT used in fantasy pipeline)
- `GET /mlb/v1/stats` — per-game box stats (used in `daily_ingestion.py`)
- `GET /mlb/v1/stats?season=2026` — season-aggregated stats (NOT used in fantasy pipeline)

**Current BDL Usage (from `daily_ingestion.py`):**
- Game schedules → feeds CBB betting pipeline + probable pitcher detection
- Odds → feeds CBB betting pipeline
- Injuries → feeds `mlb_injuries` table → used by health monitor
- Box stats → feeds `mlb_player_stats` table → used by opportunity engine

**BDL Gaps to Investigate:**
1. **Player Identity Resolution:** BDL `/mlb/v1/players?search={name}` could replace difflib fuzzy matching for bridging Yahoo names → MLBAM IDs. Check `backend/fantasy_baseball/id_resolution_service.py` and `player_mapper.py`.
2. **Two-Start Pitcher Validation:** BDL games endpoint could validate the current two-start heuristic. Check `backend/fantasy_baseball/two_start_detector.py`.
3. **Roster Cross-Check:** BDL could validate Yahoo roster data (e.g., "Yahoo says this player is on Team X but BDL says he's on Team Y"). Check `yahoo_id_sync.py`.
4. **Live Stat Validation:** BDL box stats could cross-check Yahoo's live category totals to detect sync lag. Check `scoring_engine.py`.
5. **Season Aggregate Fallback:** BDL season stats could be a fallback when FanGraphs RoS projections are missing. Check `player_board.py`.
6. **Injury Granularity:** BDL injury data may have more detail than Yahoo (injury type, severity). Check how injury alerts are generated.

Each expert must score each opportunity on:
- **User Impact** (1–10): Would this change a start/sit or add/drop decision?
- **Implementation Effort** (1–10): Days of engineering work?
- **Data Quality** (1–10): How reliable is the BDL source?
- **Risk** (1–10): Could this introduce bad data or confusion?

### Phase 3: Cross-Examination
After independent reviews, conduct debate rounds:

**Round 1: Will vs. Ron — Ownership Root Cause**
- Will: "The backend code looks correct. The bug must be in the frontend not reading the field."
- Ron: "Even if the backend works, ownership % from Yahoo's global pool is the wrong metric. We should show league-specific rostered %."
- Verdict: Where is the actual disconnect?

**Round 2: Brad vs. Dan — Ownership Display Value**
- Brad: "If we can't fix ownership quickly, we should remove the column entirely rather than show '— owned' everywhere."
- Dan: "Removing data is worse than showing zero. At least zero signals 'this player is widely available.'"
- Verdict: Show 0%, show "—", or remove the field?

**Round 3: Ron vs. Will — BDL Prioritization**
- Ron: "BDL injury data would save more matchups than anything else. A manager starting a DTD player loses categories."
- Will: "Injury data integration touches 4 services and the DB schema. BDL player search is 20 lines and fixes the name-mapping bug."
- Verdict: High-impact/high-effort vs. medium-impact/low-effort?

**Round 4: Dan vs. Brad — BDL Season Stats**
- Dan: "BDL season aggregates have proper sample sizes. Using them as a RoS fallback is statistically sound."
- Brad: "Adding another stats source to the UI creates cognitive overload. Managers already face 18 categories."
- Verdict: Backend-only fallback, or user-facing feature?

### Phase 4: Synthesis
Produce:
1. **Ownership Bug — Root Cause & Fix Path**
   - Exact file + line where data is lost
   - Whether it's a backend bug, frontend bug, or deploy gap
   - Single-commit fix or multi-file refactor?
   - Test to prevent regression

2. **BDL Integration — Prioritized Roadmap**
   - Ranked list of opportunities with impact/effort scores
   - Top 3 "do this sprint" items
   - Top 3 "do next sprint" items
   - Items to reject (negative EV)

3. **Monitoring & Alerting**
   - How to detect ownership fetch failures in production
   - How to detect BDL data quality drift
   - Dashboard metrics to track

---

## Output Format

```markdown
# Ensemble Audit: Yahoo Ownership & BDL Integration
**Date:** YYYY-MM-DD  
**Scope:** Surgical — two failure modes only  

---

## Part 1: Ownership Bug Forensics

### Root Cause Verdict
[Single-paragraph answer: where is the data lost?]

### Evidence Chain
| Step | File:Line | Expected | Actual | Verdict |
|------|-----------|----------|--------|---------|
| Yahoo API call | ... | non-zero pct | ... | ... |
| Batch enrichment | ... | ... | ... | ... |
| DB storage | ... | ... | ... | ... |
| API response | ... | ... | ... | ... |
| Frontend render | ... | ... | ... | ... |

### Fix: Single Commit or Refactor?
[If single commit: exact diff. If refactor: files to touch.]

### Regression Test
[Exact test case that would catch this.]

---

## Part 2: BDL Opportunity Matrix

### Rejected (Negative EV)
| Opportunity | Why Rejected |
|-------------|--------------|

### This Sprint (High Impact / Low Effort)
| # | Opportunity | File(s) | Effort | User Impact |
|---|-------------|---------|--------|-------------|

### Next Sprint (High Impact / High Effort OR Medium Everything)
| # | Opportunity | File(s) | Effort | User Impact |

### Backlog (Low Impact / High Effort)
| # | Opportunity | File(s) | Effort | User Impact |

---

## Part 3: Monitoring
[What alerts and dashboards to add.]

---

## Appendix: Files Read
[Complete list.]
```

---

## Constraints

1. **No broad codebase scans.** Only read files directly relevant to ownership flow and BDL integration.
2. **Ownership bug must have a single root cause.** Not "maybe this, maybe that." Trace the data and prove it.
3. **BDL opportunities must be ranked by EV.** Not "wouldn't it be nice if..." — every proposal must have a business case.
4. **Be specific about implementation.** "Integrate BDL injuries" is not enough. Which table? Which column? Which pipeline job?
5. **If the ownership bug is already fixed in local HEAD but not deployed, say so explicitly.** And specify the exact commit hash.

---

## Initiation

Begin Phase 1 immediately. Read `HANDOFF.md` for deploy status, then trace the ownership data flow file by file. Save the final report to `reports/YYYY-MM-DD-ensemble-ownership-bdl-audit.md`.
