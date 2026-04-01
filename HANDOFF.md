# HANDOFF.md — Fantasy Baseball Platform Master Plan (In-Season 2026 Edition)

> **Date:** April 1, 2026 | **Author:** Claude Code (Master Architect)
> **Risk Level:** ELEVATED — in-season with pre-season CSV fallback still active

---

## Executive Summary

The platform is live for the 2026 MLB fantasy season. The Yahoo API pipeline is functional and data is flowing to all four surfaces (roster, matchup, waiver, lineup optimizer). However, the system is operating with a structural debt: pre-season Steamer CSVs are still the primary projection source while the season is active. Kimi audits K-17 through K-23 collectively expose twelve critical bugs (data corruption, silent failures, broken UI contracts) and one existential architectural gap (stale projections at 100% weight in-season). The immediate mandate is twofold: (1) close all critical bugs — twelve are identified, eight are already fixed — and (2) implement the in-season projection pipeline before pre-season CSVs become materially wrong, which is now. Every remaining open item has a clear owner and a concrete next deliverable. No vague work items exist in this document.

---

## Core Principles

- **Contracts first.** Every layer must publish an explicit schema. No implicit data shapes, no "should be fine" assumptions. If a Yahoo API key name changes, it fails loudly, not silently.
- **Determinism before probability.** Fix the data pipeline before tuning the model. A correct batting average beats a sophisticated model fed corrupt stats.
- **Decisions before automation.** Architect the ensemble blender on paper before writing a scheduler job. Premature automation of a bad design just runs the bad design faster.
- **Intelligence before integration.** The MCMC win probability is only as good as its inputs. Do not add a fourth projection system until the three active sources are clean and fresh.
- **Every layer exists only to make the next layer simpler — never smarter.** Ingestion produces clean rows. Scoring produces a ranked list. The UI displays it. No layer compensates for broken upstream.

---

## Major Issues Acknowledged & Mitigation Plan

| Issue | Source | Severity | Current Mitigation | Owner | Status |
|-------|--------|----------|--------------------|-------|--------|
| Name concat with injury text ("Jason Adam Quadriceps") | K-15 | HIGH | Regex strip in `_parse_player()` | Claude | FIXED S3 |
| `date.today()` UTC bias in Statcast ingestion | K-16 / K-19 | HIGH | ET anchor via `ZoneInfo` in `run_daily_ingestion()` | Claude | FIXED S4 |
| Dashboard timestamp shown in UTC to users | K-17 | HIGH | `datetime.now(ZoneInfo("America/New_York"))` at `dashboard_service.py` lines 206, 827 | Claude | FIXED S3 |
| Stat IDs 57/83/85 unmapped (BB/NSV/OBP blank) | K-14 | HIGH | `_YAHOO_STAT_FALLBACK` dict + `constants.ts` STAT_LABELS | Claude | FIXED S3 |
| Impossible negative stat values (-1 GS) | K-18 | MEDIUM | `_NON_NEGATIVE_STATS` frozenset clamp; NSB explicitly excluded | Claude | FIXED S4 |
| Waiver owned% = 0% for all players | K-20 | CRITICAL | Removed `"out": "metadata"`; lookup changed to `percent_rostered` | Claude | FIXED S4 |
| Playoffs hallucination (Week 2 shows "PLAYOFFS") | K-22 | CRITICAL | `is_playoffs = raw_playoffs and (week >= 20)` guard | Claude | FIXED S4 |
| Async lineup optimizer Pydantic crash (5 validation errors) | K-21 | CRITICAL | Worker fetches Yahoo data at execution time; correct `solve_smart_lineup()` call | Claude | FIXED S3 |
| SQL Syntax Error in Job Queue (`:result::jsonb`) | G-1 | CRITICAL | Changed to `CAST(:result AS jsonb)` in `job_queue_service.py` | Claude | FIXED S6 |
| `apiFetch` converts nested error to `[object Object]` | K-21 | CRITICAL | Parse nested detail; extract readable string | Gemini | PENDING G-5 |
| "Score" column is `smart_score`; mislabelled; can be negative | K-21 | HIGH | Rename to "Smart Score" + tooltip; PROJ falls back to `implied_runs` | Gemini | PENDING G-6/G-7 |
| K/9 value (16.20) appearing in Walks column | K-22 | HIGH | Decimal guard in `_extract_team_stats()`: float BB != int -> reject + warn | Claude | FIXED S5 |
| Missing NSV on matchup (1 save shown as 0) | K-22 | HIGH | Stat ID 83 extraction + fallback mapping audit | Claude + Gemini | PENDING G-8 |
| Stat labels missing: K/BB, GS, NSV in `constants.ts` | K-22 | MEDIUM | Add string + numeric ID mappings | Gemini | PENDING G-8 |
| PROBABLE badge shown yellow/injury (should be green/SP) | K-19 domain | MEDIUM | Map PROBABLE to emerald/STARTING | Gemini | PENDING G-10 |
| Waiver "Key Stats" blank (shows deficit analysis, not stats) | K-20 | HIGH | Added `stats: dict` to `WaiverPlayerOut`; `get_free_agents()` now passes `out=stats,percent_owned`; `_to_waiver_player()` extracts stats | Claude | FIXED S6 |
| Settings page is a read-only JSON dump | K-23 | CRITICAL | Rebuild with Switch/Select/Slider; hide z-score behind presets | Gemini | PENDING G-9 |
| Pre-season CSV at 100% weight while season is live | K-17 | CRITICAL | `fangraphs_loader.py` built with ensemble blend; needs daily_ingestion wiring (100_012 + 100_014) | Claude | IN PROGRESS S6 |
| No FanGraphs RoS download | K-17 / K-19 | CRITICAL | `fangraphs_loader.py` built — 4 systems (ATC/BAT/Steamer/ZiPS), cloudscraper, name normalization | Claude | BUILT S6 — needs daily_ingestion wiring |
| No ensemble blender (ATC/Steamer/ZiPS/THE BAT) | K-17 / K-19 | CRITICAL | Schema: 5 blend columns added to `PlayerDailyMetric` model + SQL migration; `compute_ensemble_blend()` in `fangraphs_loader.py` | Claude | BUILT S6 — needs daily_ingestion job (100_014) |
| No Yahoo ADP/injury polling | K-17 / K-19 | HIGH | Lock 100_013 reserved; 4-hour cadence designed | Claude | OPEN |
| No projection freshness alerting | K-16 / K-17 | HIGH | Lock 100_015 reserved; 1-hour SLA gate designed | Claude | OPEN |
| CSV loader rejects full file on column mismatch | K-16 | MEDIUM | Header validation warning added to both loaders; per-row errors upgraded to WARNING | Claude | FIXED S5 |
| CBB V9.2 recalibration (SNR/integrity scalar stacking) | K-12 | BLOCKED | EMAC-068 — do not touch Kelly math before Apr 7 | Claude | BLOCKED APR 7 |
| OddsAPI to BDL GOAT MLB migration | CLAUDE.md | BLOCKED | Cancel post-tournament; expand `balldontlie.py` | Claude | BLOCKED APR 7 |

---

## In-Season Data Strategy & Architecture

### Single Source of Truth: Dynamic RoS + Statcast Bayesian Ensemble

```
Layer 1: Rest-of-Season Projections (counting stats only)
  Sources: ATC 30% / THE BAT 30% / Steamer 20% / ZiPS 20%
  Fetch:   Daily 3 AM ET via cloudscraper (FanGraphs public, no auth required)
  Lock:    100_012

Layer 2: Statcast Underlying Metrics (trend modifiers ONLY — NOT counting stats)
  Sources: pybaseball (Baseball Savant) — xwOBA, Barrel%, Exit Velocity
  Use:     Boost/penalty multiplier on RoS blend — never blended with HR/RBI directly
  Lock:    100_002 (live; UTC bug fixed S4)

Layer 3: Ensemble Blender
  Schema:  5 new columns on PlayerDailyMetric: blend_hr, blend_rbi, blend_avg, blend_era, blend_whip
  Runs:    Daily 5 AM ET after RoS download completes
  Lock:    100_014

Layer 4: Yahoo ADP + Injury Feed
  Source:  YahooFantasyClient (existing base class, new pagination method)
  Cadence: Every 4 hours
  Lock:    100_013

Layer 5: Projection Freshness Guard
  Alert if: RoS > 12h, Statcast > 6h, Yahoo injuries > 4h
  Gate:    Block lineup optimization if any SLA violated
  Lock:    100_015
```

### Pre-Season CSV: Fallback Only

`projections_loader.py` Steamer CSV is retained as fallback **for players not yet in the ensemble** (call-ups, trades). It must never be blended against RoS projections. The `@lru_cache` on `load_full_board()` is acceptable for this fallback use case only.

### Data Freshness SLAs

| Data Type | Max Age | Current Status |
|-----------|---------|----------------|
| Rest-of-Season Projections | < 12 hours | NOT IMPLEMENTED |
| Statcast Metrics | < 6 hours | UTC bug fixed — otherwise live |
| Yahoo ADP / Injuries | < 4 hours | Not automated |
| Ensemble Projections | < 12 hours | NOT IMPLEMENTED |
| Player Roster / Status | < 1 hour | Live via Yahoo client |

---

## Current Pipeline State

| Job | Lock ID | Cadence | Status | Notes |
|-----|---------|---------|--------|-------|
| `statcast` | 100_002 | Daily 2 AM ET | LIVE | UTC bug fixed (S4) |
| `rolling_z` | 100_003 | Daily 3 AM ET | FRAGILE | Silently skips if < 7 days data; no alert |
| `waiver_scan` | 100_007 | Daily 6 AM ET | LIVE | Owned% fix deployed |
| `mlb_brief` | 100_008 | Daily 7 AM ET | LIVE | |
| `valuation_cache` | 100_011 | On demand + scheduled | LIVE | ARCH-001 Phase 2 |
| `mlb_odds` | 100_001 | Every 30 min | DIRTY | Raw OddsAPI — migrate to BDL Apr 7 |
| `fangraphs_ros` | 100_012 | Daily 3 AM ET | NOT BUILT | Highest priority |
| `yahoo_adp_injury` | 100_013 | Every 4 hours | NOT BUILT | |
| `ensemble_update` | 100_014 | Daily 5 AM ET | NOT BUILT | Depends on 100_012 |
| `projection_freshness_check` | 100_015 | Every 1 hour | NOT BUILT | SLA gate before lineup opt |

---

## Future Roadmap

### Phase 1 — Close All Critical Bugs (Target: Apr 3)

| # | Item | Owner | Deliverable |
|---|------|-------|-------------|
| 1.1 | Fix `apiFetch` error parsing | Gemini G-5 | `api.ts` lines 66-75 — readable error messages |
| 1.2 | Smart Score rename + tooltip + PROJ fallback | Gemini G-6/G-7 | `lineup/page.tsx` — column rename, `implied_runs` fallback |
| 1.3 | Fix stat label mappings (K/BB, GS, NSV, HLD) | Gemini G-8 | `constants.ts` — complete stat label map |
| 1.4 | Settings page rebuild | Gemini G-9 | `settings/page.tsx` — Switch/Select/Slider |
| 1.5 | PROBABLE badge green/STARTING | Gemini G-10 | `status-badge.tsx` — correct color map |
| 1.6 | WaiverPlayerOut add `stats` field | Claude | `schemas.py` + `main.py` `_to_waiver_player()` + `yahoo_client_resilient.py` `out=stats` | **DONE S6** |
| 1.7 | K/9 bleed into Walks column | Claude | `main.py` `_extract_team_stats()` — decimal type guard |
| 1.8 | CSV partial loader | Claude | `projections_loader.py` — column intersection; never reject full file |

### Phase 2 — In-Season Pipeline (Target: Apr 7)

| # | Item | Owner | Deliverable |
|---|------|-------|-------------|
| 2.1 | FanGraphs RoS downloader | Claude | `fangraphs_loader.py` — 4 systems, cloudscraper, ensemble blend | **DONE S6** |
| 2.2 | `PlayerDailyMetric` schema extension | Claude | SQL migration `add_blend_columns.sql` + model columns | **DONE S6** |
| 2.3 | Ensemble blender job (100_014) | Claude | `daily_ingestion.py` — weighted blend logic |
| 2.4 | Yahoo ADP/injury poller (100_013) | Claude | `yahoo_client_resilient.py` — pagination + DB write |
| 2.5 | Projection freshness alerting (100_015) | Claude | `daily_ingestion.py` — SLA gate before lineup opt |
| 2.6 | OddsAPI to BDL GOAT MLB migration | Claude | `balldontlie.py` — `/mlb/v1/` endpoints |
| 2.7 | CBB V9.2 recalibration (EMAC-068 unlocks Apr 7) | Claude | `betting_model.py` — scalar stacking fix per K-12 |

### Phase 3 — Intelligence Upgrades (Target: Apr 21, after 3 weeks live data)

| # | Item | Owner | Deliverable |
|---|------|-------|-------------|
| 3.1 | MCMC empirical calibration (Brier score) | Claude | Script: H2H outcomes vs `win_probability` |
| 3.2 | Matchup pace/projections + remaining games | Claude + Gemini | `/api/fantasy/matchup/pace` + frontend display |
| 3.3 | Discord notification pipeline | Gemini | OAuth + per-type toggles in settings |
| 3.4 | MLB betting module (full build) | Claude | Edge calc + Kelly sizing via BDL |

---

## Delegation Matrix

| Area | Owner | Responsibility | Next Deliverable |
|------|-------|----------------|-----------------|
| Backend Python | Claude | All `.py` files, DB migrations, scheduler jobs | H-1 WaiverPlayerOut stats; Phase 2 pipeline |
| Frontend TypeScript | Gemini | All Next.js files; zero `.py` changes | G-5 through G-10 + deploy |
| Deep research / audit | Kimi | Read-only; output to `reports/K*.md` | K-24 Yahoo player_stats spec; K-25 FanGraphs column map |
| Railway DevOps | Gemini | `railway up`, smoke tests | Deploy after G-5 through G-10 complete |
| CBB recalibration | Claude | K-12 spec to V9.2 | After EMAC-068 unblocks Apr 7 |
| MLB odds migration | Claude | BDL GOAT integration | After subscription swap Apr 7 |

---

## Active Delegation Bundles

### GEMINI -- Frontend Bug Closures (G-5 through G-10)

**Guardrails:** Zero `.py` changes. TypeScript must pass (`cd frontend && npx tsc --noEmit`) after every task.
**Deploy:** After all G-tasks pass TypeScript: `railway up`. Then verify each acceptance criterion live.

| Task | File | Change | Acceptance |
|------|------|--------|------------|
| G-5 | `frontend/lib/api.ts` lines 66-75 | `detail = typeof raw === 'string' ? raw : JSON.stringify(raw)` | Error messages read "422: Field required" not "[object Object]" |
| G-6 | `lineup/page.tsx` line 252 | Header "Score" to "Smart Score" + Radix Tooltip explaining composite score | Column says "Smart Score" with tooltip on hover |
| G-7 | `lineup/page.tsx` lines 283-293 | `const proj = valuationsMap[p.player_id]?.projected_value ?? p.implied_runs ?? null` | PROJ column populated for >80% of roster |
| G-8 | `frontend/lib/constants.ts` | Add: `'K/BB': 'K/BB Ratio', 'GS': 'Games Started', 'NSV': 'Net Saves', '38': 'K/BB Ratio', '62': 'Games Started', '83': 'Net Saves'` (skip duplicates) | Matchup shows "Net Saves" not "83", "Games Started" not "62" |
| G-9 | `settings/page.tsx` | Install shadcn Switch/Select/Slider; replace `JSON.stringify` at line 124; booleans get Switch, intervals get Select (300s="5 min"), thresholds get Slider; z-score hidden behind Aggressive/Balanced/Conservative RadioGroup; Save calls `PATCH /api/fantasy/settings` | No raw JSON visible; controls are interactive |
| G-10 | `status-badge.tsx` | Add `PROBABLE`/`Probable` to green set: `{ color: 'bg-emerald-500/15 text-emerald-400', label: 'STARTING' }` | PROBABLE shows green "STARTING" badge |
| G-L2 | `waiver/page.tsx` line 349 | Change maxOwned default from 90 to 50 | Default max owned filter is 50% |

---

### KIMI -- Research to Unblock Phase 2 Backend

**Guardrails:** Read-only. No code changes. Output to `reports/K24_*.md` and `reports/K25_*.md`.

### KIMI -- Research COMPLETE

**K-24 and K-25 research delivered.** All findings documented in:
- `reports/K24_YAHOO_PLAYER_STATS_SPEC.md` — Yahoo API contract for player stats
- `reports/K25_FANGRAPHS_COLUMN_MAP.md` — FanGraphs RoS column headers

#### K-24 Findings Summary

**Recommendation for `_to_waiver_player()`:** Reuse existing `get_free_agents()` call by adding `out=stats,metadata,percent_owned` parameter. This returns stats in the same API call (no batching needed for 25 players/page).

| Stat | ID | Path |
|------|-----|------|
| HR | 7 | `player[1].player_stats.stats[n].stat.value` |
| RBI | 8 | Same structure |
| R | 6 | Same structure |
| SB | 5 | Same structure |
| AVG | 3 | Same structure |
| ERA | 26 | Same structure |
| WHIP | 27 | Same structure |

**Batch limit:** 25 player keys per call (Yahoo hard limit).

#### K-25 Findings Summary

**Column consistency:** ALL systems (ATC, THE BAT, ZiPS DC, Steamer) use identical column headers.

| Stat | Column | Notes |
|------|--------|-------|
| HR | `HR` | Consistent all systems |
| RBI | `RBI` | Consistent all systems |
| R | `R` | Consistent all systems |
| AVG | `AVG` | Consistent all systems |
| SB | `SB` | Consistent all systems |
| ERA | `ERA` | Consistent all systems |
| WHIP | `WHIP` | Consistent all systems |
| Strikeouts | `SO` | **Not "K" — all systems use "SO"** |
| Innings | `IP` | Format: "182.1" |
| Name | `Name` | Format: "Last, First" |

**Scraping:** All systems require `cloudscraper` (Cloudflare protected).

**URLs:**
- ATC: `projections.aspx?...type=atc`
- THE BAT: `projections.aspx?...type=thebat`
- ZiPS DC: `projections.aspx?...type=zipsdc`
- Steamer RoS: `projections.aspx?...type=steamerr`

---

---

## Immediate Action Items

### Complete This Session (S5)

| # | Item | Status |
|---|------|--------|
| H-2 | K/9 decimal guard in `_extract_team_stats()` | DONE |
| 1.8 | CSV header validation + WARNING logging in both loaders | DONE |

### Pending (Claude -- requires quota or K-24/K-25 first)

| # | Item | Dependency | File | Effort |
|---|------|-----------|------|--------|
| H-1 | WaiverPlayerOut `stats` field | K-24 report | `schemas.py` line 380, `main.py` `_to_waiver_player()` | Medium |
| Phase 2.1 | `fangraphs_loader.py` (new file) | K-25 report | `backend/fantasy_baseball/` | Large |
| Phase 2.2 | `PlayerDailyMetric` schema: 5 blend columns | After 2.1 | Alembic migration | Small |
| Phase 2.3 | Ensemble blender job (lock 100_014) | After 2.2 | `daily_ingestion.py` | Medium |
| Phase 2.4 | Yahoo ADP/injury poller (lock 100_013) | After K-24 | `yahoo_client_resilient.py` | Medium |
| Phase 2.5 | Projection freshness SLA gate (lock 100_015) | After 2.3 | `daily_ingestion.py` | Small |
| M-5 | `rolling_z` silent skip alert | None | `daily_ingestion.py` rolling_z job | Small |


## Advisory Lock Registry

| Lock ID | Job | Status |
|---------|-----|--------|
| 100_001 | mlb_odds (OddsAPI — migrate Apr 7) | LIVE / DIRTY |
| 100_002 | statcast | LIVE |
| 100_003 | rolling_z | LIVE (fragile) |
| 100_004 | cbb_ratings | LIVE |
| 100_005 | clv | LIVE |
| 100_006 | cleanup | LIVE |
| 100_007 | waiver_scan | LIVE |
| 100_008 | mlb_brief | LIVE |
| 100_009 | openclaw_perf | LIVE |
| 100_010 | openclaw_sweep | LIVE |
| 100_011 | valuation_cache | LIVE |
| 100_012 | fangraphs_ros | RESERVED |
| 100_013 | yahoo_adp_injury | RESERVED |
| 100_014 | ensemble_update | RESERVED |
| 100_015 | projection_freshness_check | RESERVED |

**Next available:** 100_016

---

## Hard Stops

| Rule | Reason |
|------|--------|
| Do NOT modify Kelly math in `betting_model.py` | EMAC-068 — blocked until Apr 7 |
| Do NOT call BDL `/ncaab/v1/` endpoints | Subscription cancelled — will 401 |
| Do NOT add OddsAPI dependencies | Phase out; BDL MLB replaces post-Apr 7 |
| Do NOT touch `dashboard/` (Streamlit) | Retired — Next.js is canonical |
| Do NOT use `datetime.utcnow()` for game times | Use `datetime.now(ZoneInfo("America/New_York"))` |
| Do NOT write test files outside `tests/` | Architecture locked |
| Do NOT blend Statcast xwOBA/Barrel% with counting stats | K-19 domain: Statcast = trend modifier only |
| Do NOT weight pre-season CSVs against RoS projections | K-17: pre-season CSV = fallback for missing players only |

---

*Handoff complete. New HANDOFF.md is clean, contract-driven, and ready for execution. All K-15-K-23 issues acknowledged and roadmap simplified.*

---

## Session 4 Summary � Gemini CLI (Ops) � Apr 1, 2026

### G-1: Deployment & Verification
- **Build Fixed:** Resolved UTF-8 encoding issue in \status-badge.tsx\ that was blocking production builds.
- **Deploy:** \
ailway up\ successful.
- **Verification:**
    - Player Names: ? CLEAN (no injury suffixes).
    - Matchup Stats: ? NO NEGATIVES ( GS clamped to 0).
    - Dashboard Timestamp: ? ET ANCHORED (EDT/-04:00).
    - **Async-Optimize Job:** ? CRITICAL FAILURE. Job hangs in \processing\ due to SQL Syntax Error in \job_queue_service.py\ (line 124: \:result::jsonb\). Worker transaction aborts. Verified via live polling and code inspection.

### G-2: Yahoo API Capture
- Captured full roster, free agents sample, and league settings JSON.
- Saved to \
eports/GEMINI_YAHOO_RESPONSES_2026-04-01.md\.
- High-fidelity data confirmed for Claude's parsing logic fixes.

### G-3: UI Re-Audit
- Fresh read-only pass across fantasy pages.
- Documented 10+ UX/UI issues in \
eports/GEMINI_UI_AUDIT2_2026-04-01.md\.
- Key findings: dense tables, confusing column naming (Score vs Proj), and missing interactivity in dashboard flags.
