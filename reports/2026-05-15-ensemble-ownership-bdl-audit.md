# Ensemble Audit: Yahoo Ownership & BDL Integration
**Date:** 2026-05-15  
**Scope:** Surgical — ownership bug forensics + BDL opportunity matrix  

---

## Part 1: Ownership Bug Forensics

### Root Cause Verdict
Production is still running the wrong backend revision. Current HEAD wires `_enrich_ownership_batch()` into both `get_roster()` and `get_free_agents()` and persists fallback ownership into `PositionEligibility` (`backend/fantasy_baseball/yahoo_client_resilient.py:742-743, 836-887`; `backend/services/daily_ingestion.py:7248-7252, 7323-7335`), and the API/frontend field contracts line up (`backend/routers/fantasy.py:2043-2052, 3191-3238`; `backend/services/player_mapper.py:219-258`; `backend/services/dashboard_service.py:539-545`). But `HANDOFF.md` says Railway is live on `d319beb` while the ownership fix is later, and explicitly notes live production still shows `0% owned` because the deployed backend predates those fixes (`HANDOFF.md:3-5, 238-252`). The frontend dashes are downstream presentation of backend zeros, not the root cause.

### Evidence Chain
| Step | File:Line | What Should Happen | What Actually Happens | Verdict |
|------|-----------|-------------------|----------------------|---------|
| Yahoo API call | `backend/fantasy_baseball/yahoo_client_resilient.py:841-856` | Call Yahoo’s only verified ownership endpoint: `players;player_keys={keys}/ownership` | Current HEAD does exactly this, but production is still on pre-fix `d319beb` per `HANDOFF.md:3-5,252`, so live traffic is not benefiting from this path | FAIL |
| Batch enrichment | `backend/fantasy_baseball/yahoo_client_resilient.py:742-743, 836-887` | `get_roster()` and `get_free_agents()` should mutate each player dict with `percent_owned` | Current code wires the helper into both flows; live production still returns all-zero ownership because the deployed backend predates the fix (`HANDOFF.md:240-252`) | FAIL |
| DB storage | `backend/models.py:1834-1840`; `backend/services/daily_ingestion.py:7248-7252, 7323-7335` | Persist enriched ownership to `PositionEligibility.league_rostered_pct` for fallback reads | The column and sync path exist in current code, but `HANDOFF.md:244-246,252` says that ownership-sync work is also part of the not-yet-live fix set | FAIL |
| API response | `backend/routers/fantasy.py:2043-2052, 3191-3238`; `backend/services/player_mapper.py:219-258`; `backend/services/dashboard_service.py:539-545`; `HANDOFF.md:240` | Waiver JSON should expose `owned_pct`, roster JSON should expose `ownership_pct`, dashboard targets should expose `percent_owned` | Contracts are wired correctly, but live inspection already recorded `ownership_pct: 0.0` for every player in production | FAIL |
| Frontend render | `frontend/app/(dashboard)/war-room/waiver/page.tsx:52-63,156-176`; `frontend/app/(dashboard)/dashboard/page.tsx:289-290`; `frontend/app/(dashboard)/war-room/roster/page.tsx:715-716` | Render the backend value, ideally distinguishing real `0` from unknown | Waiver/dashboard render `— owned` for both `0` and `null`; roster hides ownership unless `> 0`. This masks semantics, but it is reacting to backend zeros rather than creating them | PASS |

### Cross-Examination
**Will vs. Ron — Where is the disconnect?**
Will (backend): The ownership path is wired end-to-end in current code — Yahoo batch fetch (`yahoo_client_resilient.py:841-887`), waiver response `owned_pct` (`fantasy.py:2043-2052`), roster response `ownership_pct` (`player_mapper.py:219-258`), dashboard `percent_owned` (`dashboard_service.py:539-545`).  
Ron (fantasy logic): The frontend is not the source of truth problem; waiver reads both `percent_owned` and `owned_pct` (`waiver/page.tsx:156`), roster reads `ownership_pct` (`roster/page.tsx:715-716`), dashboard reads `percent_owned` (`dashboard/page.tsx:289-290`).  
Verdict: The disconnect is deployment state, not field wiring.

**Brad vs. Dan — Show 0%, show "—", or remove?**
Brad (UX): Current UI intentionally treats `0`, `null`, and `undefined` as the same muted state (`waiver/page.tsx:52-55`; `dashboard/page.tsx:290`).  
Dan (stats): A literal `0.0` is a valid baseball fact and should not be merged with “unknown.”  
Verdict: After the deploy, show `0% owned` for confirmed zero and reserve `— owned` for null/unknown only.

### Fix Plan
1. **Deploy, don’t rewrite.** The root-cause fix is already in source; Railway must move off `d319beb` to current `stable/cbb-prod` (includes `27304f8` and later ownership follow-ups).
2. **Post-deploy smoke check:** hit `/api/fantasy/waiver`, `/api/fantasy/roster`, and the dashboard waiver-targets surface; verify at least one player returns a non-zero ownership field.
3. **Optional hardening (secondary, not root cause):** update `frontend/app/(dashboard)/war-room/waiver/page.tsx:52-63`, `frontend/app/(dashboard)/dashboard/page.tsx:289-290`, and `frontend/app/(dashboard)/war-room/roster/page.tsx:715-716` so `0` is rendered as `0% owned` and only `null`/missing becomes `— owned`.

### Regression Test
- **Existing low-level guard:** `tests/test_yahoo_client_ownership.py::test_ownership_merged_from_batch_call` mocks `client._get()` so the secondary Yahoo ownership batch returns `34.5`, then asserts `players[0]["percent_owned"] == 34.5`.
- **Existing route-level guard:** `tests/test_roster_waiver_enrichment_contract.py::test_waiver_populates_percent_owned_from_ownership_subresource` patches `get_yahoo_client()` to return a free agent with `percent_owned: 87.0`, then asserts `/api/fantasy/waiver` returns `owned_pct == 87.0`.
- **Local verification run:** `.\venv\Scripts\python -m pytest tests\test_yahoo_client_ownership.py tests\test_roster_waiver_enrichment_contract.py tests\test_waiver_integration.py -q --tb=short` → **40 passed**.

---

## Part 2: BDL Opportunity Matrix

### Scoring Criteria
- **User Impact (1–10):** Would this change a start/sit or add/drop decision?
- **Effort (1–10):** Engineering days (10 = very hard)
- **Data Quality (1–10):** How reliable/fresh is BDL for this use case?
- **Risk (1–10):** Could bad data cause wrong decisions? (10 = dangerous)
- **EV Score:** (User Impact × Data Quality) / (Effort × Risk) — higher is better

### Available BDL Endpoints (evaluate each)
1. `GET /mlb/v1/player_injuries` — IL + DTD list
2. `GET /mlb/v1/players?search=` — player name/ID lookup
3. `GET /mlb/v1/games` — daily schedule (already partially used)
4. `GET /mlb/v1/stats` — per-game box stats (already partially used)
5. `GET /mlb/v1/stats?season=2026` — season aggregates (NOT currently used in fantasy)
6. `GET /mlb/v1/odds` — sportsbook lines (used in betting, not fantasy)

### Full Matrix
| # | Opportunity | Endpoint | User Impact | Effort | Data Quality | Risk | EV Score | Current Gap |
|---|-------------|----------|------------|--------|--------------|------|----------|-------------|
| 1 | Add market-implied run environment + win expectancy to streamer and two-start pitcher ranking | `GET /mlb/v1/odds` | 8 | 3 | 8 | 3 | 7.11 | Odds are already ingested into `mlb_odds_snapshot` (`backend/services/daily_ingestion.py:1519-1665`), but no inspected fantasy ranking path consumes them; `TwoStartDetector` rates matchups without market data (`backend/fantasy_baseball/two_start_detector.py:218-235`). |
| 2 | Auto-repair unmapped Yahoo players by querying BDL search when name-only mapping fails | `GET /mlb/v1/players?search=` | 7 | 4 | 8 | 3 | 4.67 | Search is implemented (`backend/services/balldontlie.py:471-509`), but inspected identity flows only use existing DB keys + normalized names (`backend/fantasy_baseball/id_resolution_service.py:85-156`; `backend/fantasy_baseball/yahoo_id_sync.py:61-69,106-135`). |
| 3 | Surface richer injury freshness/return-date risk on waiver + dashboard decisions | `GET /mlb/v1/player_injuries` | 7 | 3 | 7 | 4 | 4.08 | Injuries are ingested (`backend/services/daily_ingestion.py:2111-2168`), but inspected fantasy decision/UI paths still use Yahoo `status` / `injury_note` only (`backend/routers/fantasy.py:2005-2057`; `backend/services/player_mapper.py:227-259`). |
| 4 | Harden same-day off-day / postponed / doubleheader context for lineup and streaming cards | `GET /mlb/v1/games` | 3 | 4 | 7 | 4 | 1.31 | Games already feed odds, game logs, and matchup-context jobs (`backend/services/daily_ingestion.py:1519-1665, 1738-1778, 3972-4005`). Incremental fantasy value exists, but schedule itself is not a unique BDL moat. |
| 5 | User-facing live-form fallback for thin-coverage players (last few games / starts) | `GET /mlb/v1/stats` | 5 | 5 | 7 | 6 | 1.17 | Box stats are already ingested into `mlb_player_stats` (`backend/services/daily_ingestion.py:1892-1945`), but the inspected waiver/roster/dashboard surfaces do not expose a lightweight same-day recency layer from that data. |
| 6 | Backend-only fallback when RoS projections are missing for fringe players | `GET /mlb/v1/stats?season=2026` | 3 | 5 | 6 | 6 | 0.60 | Season-aggregate fetch exists (`backend/services/balldontlie.py:606-669`), but no inspected fantasy call site uses it; Yahoo already supplies season stats for rostered players and FanGraphs/Statcast remain better predictive sources. |

### Rejected (Negative EV)
| Opportunity | Why Rejected |
|-------------|--------------|
| Replace the probable-pitchers pipeline with BDL games | Rejected because `backend/services/daily_ingestion.py:7392-7393` explicitly documents that BDL does **not** expose probable pitcher data; MLB Stats API is still the canonical source here. |
| Make BDL season aggregates a user-facing RoS score | Rejected because it is descriptive, not predictive; it duplicates Yahoo season totals (`backend/services/player_mapper.py:206-220`) and is weaker than FanGraphs RoS + Statcast for forward-looking decisions. |

### This Sprint (EV ≥ 1.5)
| # | Opportunity | Files to Touch | Effort | User Impact | Specific Implementation |
|---|-------------|----------------|--------|-------------|------------------------|
| 1 | Odds-driven pitcher streaming signal | `backend/fantasy_baseball/two_start_detector.py`, `backend/services/waiver_edge_detector.py`, `backend/routers/fantasy.py`, `frontend/lib/types.ts`, `frontend/app/(dashboard)/war-room/waiver/page.tsx` | M | High | Join latest `mlb_odds_snapshot` to upcoming pitcher matchups; derive opponent implied runs + team win odds; fold into `quality_score` and display as a small matchup-risk badge/tool-tip. |
| 2 | BDL player-search auto-heal for new Yahoo players | `backend/fantasy_baseball/yahoo_id_sync.py`, `backend/fantasy_baseball/id_resolution_service.py`, `backend/services/daily_ingestion.py` | M | High | On mapping miss, query `search_mlb_players(name)`, require exact normalized-name + team match, persist `bdl_id` into `player_id_mapping`, then retry downstream enrichment. |
| 3 | Injury-risk overlay in waiver/dashboard | `backend/routers/fantasy.py`, `backend/services/dashboard_service.py`, `frontend/lib/types.ts`, `frontend/app/(dashboard)/war-room/waiver/page.tsx`, `frontend/app/(dashboard)/dashboard/page.tsx` | M | High | Read `IngestedInjury` for candidate players; surface IL/DTD/return-date warnings and discount fragile adds in the recommendation score. |

### Next Sprint (0.8 ≤ EV < 1.5)
| # | Opportunity | Files to Touch | Effort | User Impact | Specific Implementation |
|---|-------------|----------------|--------|-------------|------------------------|
| 4 | BDL schedule anomaly guardrails | `backend/services/daily_ingestion.py`, `backend/services/player_mapper.py`, `frontend/app/(dashboard)/war-room/roster/page.tsx` | M | Medium | Use BDL game status/home-away data to flag off-days, postponements, and doubleheaders directly in roster and matchup cards. |
| 5 | Same-day live-form cards from box stats | `backend/services/player_mapper.py`, `backend/routers/fantasy.py`, `frontend/lib/types.ts`, `frontend/app/(dashboard)/war-room/waiver/page.tsx` | M | Medium | Build a compact “last 3 starts / last 7 games” summary for players whose Statcast/FanGraphs coverage is thin or stale. |

### Backlog (EV < 0.8)
| # | Opportunity | Files to Touch | Effort | User Impact |
|---|-------------|----------------|--------|-------------|
| 6 | Season-aggregate fallback from BDL stats | `backend/fantasy_baseball/player_board.py`, `backend/services/dashboard_service.py` | M | Low |

### Cross-Examination
**Ron vs. Will — BDL injury data vs. BDL player search: which first?**
Ron (fantasy impact): Injury data changes stash/add/drop decisions immediately and is already ingested, so surfacing it is the shortest path to user-visible value.  
Will (implementation): Player search is the higher-leverage plumbing fix because unresolved Yahoo players currently fall back to quarantine or stale mapping logic (`id_resolution_service.py:124-156`; `yahoo_id_sync.py:68-90`), which suppresses every downstream BDL/Statcast feature for new call-ups.  
Verdict: **Player search first, injuries second.** Search fixes the platform blind spot; injury UI is the next-fastest win once IDs are reliable.

**Dan vs. Brad — BDL season stats as RoS fallback: backend-only or user-facing?**
Dan (statistical validity): Season aggregates can be a safety net when projections are missing, but they should never outrank FanGraphs RoS or Statcast-derived forward-looking signals.  
Brad (UX complexity): Exposing another season stat block in the UI adds noise and invites users to mistake descriptive totals for projections.  
Verdict: **Backend-only, if at all.** Do not make BDL season aggregates a user-facing recommendation surface.

---

## Part 3: Monitoring & Alerting

### Ownership Pipeline Health
- Log `ownership_enriched / requested_players` for **both** `get_roster()` and `get_free_agents()` using the same player count already available in `backend/fantasy_baseball/yahoo_client_resilient.py:847-885`.
- Emit a per-source breakdown: `yahoo_batch`, `position_eligibility`, `adp_proxy`, `missing` from waiver and roster handlers (`backend/routers/fantasy.py:2065-2135`, `3018-3238`).
- Add a post-deploy smoke check that fails if `/api/fantasy/waiver` returns 25 players with `owned_pct == 0` or `/api/fantasy/roster` returns all `ownership_pct == 0`.
- Add a dashboard-side canary metric: ratio of rendered ownership badges with real numbers vs muted placeholders.

### BDL Data Quality
- Alert if `bdl_injuries` ingests **0 rows** on a normal game day or if `return_date` parsing failure rate spikes (`backend/services/daily_ingestion.py:2120-2150`).
- Alert if `mlb_odds_snapshot` has active games but no odds rows for the latest 30-minute bucket (`backend/services/daily_ingestion.py:1556-1665`).
- Track `player_id_mapping` / identity quarantine growth after Yahoo sync; a spike means player-search repair is needed (`backend/fantasy_baseball/id_resolution_service.py:124-156`; `backend/fantasy_baseball/yahoo_id_sync.py:68-90`).
- Watch BDL validation warnings from `get_mlb_stats()` / `get_mlb_player_season_stats()` so schema drift is caught before fantasy features consume bad rows (`backend/services/balldontlie.py:563-580, 651-669`).

---

## Synthesis: The 5 Actions That Matter Most This Week

| # | Action | Type | Effort | Impact |
|---|--------|------|--------|--------|
| 1 | Deploy current `stable/cbb-prod` to Railway so the ownership fix actually runs in production | Bug fix | S | Restores real ownership across waiver, roster, and dashboard |
| 2 | Add ownership smoke checks + source-distribution logging after deploy | Monitoring | S | Prevents another silent “all 0% owned” regression |
| 3 | Use BDL odds to improve pitcher streamer / two-start ranking | Feature | M | Highest-EV fantasy use of BDL data already in the warehouse |
| 4 | Auto-heal unmapped Yahoo players with BDL player search | Feature | M | Removes new-player blind spots that suppress multiple downstream enrichments |
| 5 | Surface BDL injury freshness and return dates in waiver/dashboard decisions | Feature | M | Improves stash/drop timing and avoids dead-roster adds |

---

## Appendix: Files Read
- `HANDOFF.md` — lines 1-260, 236-253
- `CLAUDE.md` — lines 1-193
- `backend/fantasy_baseball/yahoo_client_resilient.py` — lines 288-380, 680-760, 799-920, 1180-1285, 1600-1692, 1760-1950, 2240-2295
- `backend/models.py` — lines 1-260, 1828-1842
- `backend/schemas.py` — lines 399-540
- `backend/contracts.py` — lines 341-520
- `backend/routers/fantasy.py` — lines 1-110, 330-350, 1687-2145, 2488-2610, 3018-3260, 5908-5952
- `backend/services/player_mapper.py` — lines 1-60, 150-270
- `backend/services/dashboard_service.py` — lines 50-75, 450-550
- `frontend/lib/types.ts` — lines 220-250, 450-475, 606-630
- `frontend/app/(dashboard)/war-room/waiver/page.tsx` — lines 45-180
- `frontend/app/(dashboard)/war-room/roster/page.tsx` — lines 700-725
- `frontend/app/(dashboard)/dashboard/page.tsx` — lines 264-295
- `backend/services/balldontlie.py` — lines 1-260, 280-690
- `backend/services/daily_ingestion.py` — lines 1519-1665, 1738-1778, 1892-1945, 2111-2168, 3972-4005, 7196-7342, 7382-7510
- `backend/fantasy_baseball/id_resolution_service.py` — lines 1-240
- `backend/fantasy_baseball/two_start_detector.py` — lines 1-320
- `backend/fantasy_baseball/yahoo_id_sync.py` — lines 1-220
- `tests/test_yahoo_client_ownership.py` — lines 1-145
- `tests/test_roster_waiver_enrichment_contract.py` — lines 1-220, 498-572
- `tests/test_waiver_integration.py` — lines 1-140
- Git evidence reviewed: commit metadata for `d319beb` and `27304f8`, plus `27304f8` diff for `backend/fantasy_baseball/yahoo_client_resilient.py`

**Checked but not present:** `backend/fantasy_baseball/player_mapper.py`
