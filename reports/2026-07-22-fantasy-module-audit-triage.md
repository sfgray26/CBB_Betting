# Fantasy Baseball Module Audit — Triage Report

**Date:** 2026-07-22
**Author:** Kimi CLI (read-only investigation; no production files modified)
**Audience:** Claude Code (implementation decisions), PM (prioritization)
**Source:** PM's module-by-module UX audit of War Room, My Roster, Waiver Wire, Streaming Station, Budget, Weekly Preview + Lineup Optimizer (Team 7 "Lindor Truffles")
**Method:** 4 parallel read-only code investigations + SELECT-only production DB probes + Railway log review. Spot-checked load-bearing claims against source.

---

## 0. P0 ADDENDUM — Yahoo API outage (unfolded during this triage)

**Symptom:** From 2026-07-22 19:08 UTC (container restart) onward, every Yahoo Fantasy API call returns 403 "This application is not authorized to perform this action" → frontend 503s on roster/scoreboard/waiver.

**Verified facts:**
- Token refresh **succeeds** (client ID/secret/refresh token all valid; Railway logs show repeated successful refreshes). Freshly-minted access tokens are then rejected by **all** fantasy endpoints.
- Reproduced locally with identical result (rules out Railway env drift).
- A **brand-new consent grant** (full `--auth` re-authorization) also 403s on first API call. App config at developer.yahoo.com is intact (Fantasy Sports Read/Write) and unchanged — it worked earlier the same day.
- Yahoo's OAuth infrastructure (request_auth → 302, get_token → 200) is alive; only the fantasy API authorization layer rejects.
- **Not caused by this audit session** — all work was read-only; `.env` untouched since 4/29 (until the user's own re-auth attempt).

**Conclusion:** Yahoo-side app-level block or incident. No client-side fix exists. Monitoring + retry re-auth; if it persists >24h, escalate to Yahoo developer support.

**Architectural follow-ups (for Claude/Codex — these turned a Yahoo hiccup into a full outage and will again):**
1. Rotated refresh tokens persist only in container memory — `.env` writes fail silently on Railway (`backend/fantasy_baseball/yahoo_client_resilient.py:297-314`). Every restart rolls back to the stale env-var token. Fix: persist tokens to DB or write back to Railway vars programmatically after refresh.
2. The 403-retry path refreshes the token on **every** 403 (`yahoo_client_resilient.py:~390-399`), so an outage produces a tight refresh hammer loop that can extend a throttle-based block. Add backoff/circuit-open on repeated auth failures.
3. Alerts module has zero fantasy-side hooks — no alert fired for a total Yahoo auth outage. Add one.
4. `CREDENTIALS.md` references `scripts/auth/yahoo_oauth.py` — does not exist. Real entry point: `python -m backend.fantasy_baseball.yahoo_client_resilient --auth`. Also: `run_auth_flow` prints "Refresh token saved to .env" unconditionally even when the write fails (`yahoo_client_resilient.py:2095-2098`).
5. `.env` contains duplicate `YAHOO_ACCESS_TOKEN`/`YAHOO_REFRESH_TOKEN` lines (lines 62-63 and 96-97) — a trap for any tooling that reads last-match.
6. `YAHOO_TEAM_KEY` unset in both local and Railway — forces the fragile `get_my_team_key()` resolution path tied to Finding R2 below.

---

## 1. War Room

### W1 — "LEADING 10-7" headline above "Win Probability 0%" — CONFIRMED (UI conflation, amplified by W3)
- Big score = **current live tally**, computed client-side (`frontend/components/war-room/matchup-header.tsx:15-27`, rendered `:82-98`).
- Win Probability = Monte-Carlo **final-outcome** probability (`backend/fantasy_baseball/mcmc_simulator.py:401-402,419`), rendered `matchup-header.tsx:50,111-128`. The strip's left side says "Projected 5-13" but the Win% half is not labeled "Projected".
- **Fix:** label the strip "Projected Final Win Probability" / add caption under live score. Fixing W3 makes the 0% sane.

### W2 — "K" and "HR" reused for batting and pitching — CONFIRMED
- `frontend/lib/types.ts:417-420`: `HR_B`→"HR", `HR_P`→"HR", `K_B`→"K", `K_P`→"K". Directions are *opposite* across sections. Colors also collide (`:426,429` both `#a855f7`).
- **Fix:** disambiguate (`K(B)`/`K(P)`, `HR`/`HRA`), distinct color for `HR_P`.

### W3 — Pitching "L": ahead 1-4 but labeled BEHIND + "PUNT?" — CONFIRMED, real backend direction bug (all 5 lower-is-better cats)
- Frontend bar is correct (`LOWER_IS_BETTER` includes `'L'`, `types.ts:434`; `category-battlefield.tsx:30-35`). Status/action come from backend sim `win_prob` (`category-battlefield.tsx:46-93`).
- **Root cause:** mid-week "anchor" offsets are **raw Yahoo stats added without inversion**: `backend/routers/fantasy.py:7889-7918` → `mcmc_simulator.py:365-373`; comparison at `mcmc_simulator.py:389-390` assumes higher-is-better for all. `LOWER_IS_BETTER` imported at `mcmc_simulator.py:38` but **never used** (verified). Affects `l, hr_p, k_b, era, whip`; "L" shows it worst because z-scores are tiny and the 1-vs-4 raw offset fully decides the sim. Also a units mismatch: raw stat values added to z-score sums, dominating late-week.
- **Fix:** multiply offset by -1 for cats in `LOWER_IS_BETTER` in `simulate_weekly_matchup()`; longer-term, convert offsets to z-scale.

### W4 — "Proj/Win%" column mixes "51→52", "47%", bare numbers, no legend — CONFIRMED
- `category-battlefield.tsx:139-143`: ratio cats → win%, counting cats → unit-less arrows; header `:344`; legend (`:302-309`) explains statuses, not formats.
- **Fix:** split into PROJ FINAL + WIN% columns, or add legend line.

### W5 — "STALE · unavailable" badge never clears — CONFIRMED (backend bug, 2 layers)
- `backend/routers/fantasy.py:7039-7049`: checks `hasattr(_client, 'circuit')` on the base `YahooFantasyClient` — only `ResilientYahooClient` has `.circuit` → always `None` → severity hard-defaults "warning", `minutes_ago=None`.
- Second layer: `CircuitBreaker.get_stats()` (`circuit_breaker.py:168-178`) has no `last_success_time` key anyway.
- **Fix:** track real last-success on the base client; add `last_success_time` to circuit stats; hide badge when `minutes_ago is None`.

---

## 2. My Roster

### R1 — Permanent "STALE · unavailable" badge — same root cause as W5.

### R2 — Direct-URL load → "vs TBD", all categories "–·–"/"T", no error — CONFIRMED
- `/api/fantasy/matchup` returns **HTTP 200 stubs** (`MatchupTeamOut(team_name="TBD", stats={})`, `fantasy.py:6409-6410`) on three failure paths (`:6481-6486, :6635`), distinguished only by a `message` field.
- Frontend `MatchupStrip` (`war-room/roster/page.tsx:582-674`) **never renders `message`**; empty stats → `?? 'T'` (`:564`) → everything looks tied. Error banner only fires on HTTP errors (`:1386-1392`).
- Sidebar-nav vs direct-nav difference = react-query cache sharing (`queryKey: ['matchup']`, 5-min staleTime, both pages); cold direct load can hit the stub paths (cold Yahoo token, 5-request burst tripping the 5s timeout at `:6471-6480`).
- **Fix:** (a) render `matchup.message`/treat TBD+empty-stats as degraded with retry; (b) return non-200 or `degraded: true` server-side; (c) log loudly when `my_team_key` resolves to "" (sets `YAHOO_TEAM_KEY` — see §0.6).

### R3 — IP-pace flips "16.2/18 AHEAD" → "0.0/18 BEHIND" mid-session — CONFIRMED (not a client race)
- `fantasy.py:9120-9122`: if `my_stats` non-empty but lacks `"IP"` key, `ip_data_available=True` while `ip_accumulated=0.0` (scoreboard fallback skipped). `:9123-9146`: on fetch failure returns 200 with `0.0` + BEHIND pace.
- Frontend `BudgetPanel` (`page.tsx:198-275`) **never reads `ip_data_available`/`ip_as_of`** (both in response, `:9274-9275`). Refetch on window focus + transient Yahoo hiccup → flip.
- **Fix:** honor `ip_data_available` in UI ("IP data unavailable", no verdict); don't mark IP available when key absent.

### R4 — Five IL pitchers in active slots, no warning — PARTIAL
- IL status exists end-to-end (parse `yahoo_client_resilient.py:1950`; normalize `player_mapper.py:140-180`; per-card pills `page.tsx:884-929`; "Injured List · N" group `:1277,1501-1520`).
- Missing: page-level "IL player occupying active slot" warning. Also `yahoo-roster-view.tsx:328` checks `status === 'DL'` but mapper normalizes DL→"IL" — dead condition.
- **Fix:** derived banner for `status==='IL' && current_slot not in {IL, IL60, BN}`; fix the `'DL'` literal.

### R5 — Optimizer benches Soto, role players "Score 100.0" — CONFIRMED (normalization bug + no talent anchor; fix exists, flag OFF)
- Live path: `POST /api/fantasy/roster/optimize` (`fantasy.py:5325`). `daily_lineup_optimizer.py`/`smart_lineup_selector.py`/`category_aware_scorer.py` are **not** in this path.
- Raw score = percentile of **14-day-only** composite_z within cohort (`scoring_engine.py:605-615`); no season/talent anchor; small-sample `confidence` (`:600-601`) never consumed.
- **The bug:** `_normalize_group_scores` (`fantasy.py:5603-5634`) min-max re-normalizes within the ~10 roster players → roster-max guaranteed `100.0`, min `0.0`. A cold/unmapped Soto (fallback floor `0.0`, `:417-419`) → benched by the ILP (`lineup_constraint_solver.py:170-194`). Double relativization; comment at `:5593-5598` acknowledges scores are already percentiles.
- No category-needs awareness in this endpoint at all.
- **P28 fix exists but flag-OFF:** blended talent/form/matchup path (`backend/services/blended_score.py`, wired `fantasy.py:5559-5588`) gated on `optimize.blended_score`, default False (`config_service.py:82-89`); **verified absent from `feature_flags` table in prod DB**. Caveat before enabling: `_resolve_blended_signals` computes talent Z vs the roster's own ~10 players (`fantasy.py:5209-5241`) — second roster-relative normalization bug; fix cohort to league-wide Statcast population first.
- **Fix:** (a) drop roster-level min-max re-normalization, relabel display "Form (14d)"; (b) fix blended cohort, then enable flag; (c) weight form scores by `confidence`.

### R6 — "Score 86.7 (player_scores)" raw field name in UI — CONFIRMED
- Server-side reasoning strings: `fantasy.py:5666` (hitters), `:5811` (pitchers) → rendered verbatim `page.tsx:780`. Previously flagged in `UAT_REPORT_FANTASY_BASEBALL_WEEK_10_2026-05-31.md:28`, never fixed.
- **Fix:** map `score_source` → human labels server-side ("recent form" / "projection" / "blended" / "no data").

### R7 — "Apply All 18 Moves" vs "Weekly Moves: 0/8" — CONFIRMED (UX gap + count inflation)
- "Weekly Moves 0/8" = Yahoo add/drop acquisitions (`fantasy.py:9053-9068`, limit hardcoded 8 at `:9043`; UI `page.tsx:222-239`).
- "18 moves" = slot reassignments via one `set_lineup` call (`fantasy.py:4595-4601`) — never consume acquisitions. Explained nowhere. Worse, `allMoves` (`page.tsx:754-757`) includes no-op reassignments (player already in recommended slot).
- **Fix:** relabel "Weekly Adds (waivers/FA): 0/8"; note "slot changes don't count"; filter `allMoves` to `assigned_slot !== current_slot`.

---

## 3. Waiver Wire

### V1 — Sort toggle → full blocking server round-trip (10s+) — PARTIAL
- Sort IS server-side: `queryKey: ['waiver', sort]` (`waiver/page.tsx:669`) + blocking spinner on `isLoading` (`:696-705`). Position filter is client-side only (`:683-694`).
- Endpoint latency: **6+ sequential uncached Yahoo round-trips** per request (`fantasy.py:1887-1969` + `:2570`).
- **Fix:** client-side sort of the ~25-50 row array (or `placeholderData: keepPreviousData`); parallelize/cache Yahoo calls.

### V2 — "Overall Value" sort is a no-op — CONFIRMED (spot-checked)
- `fantasy.py:2589-2592`: only `percent_owned` gets a branch; `"projected_points"` falls to `else` → sorted by `need_score`, identical to "Match Score". No z_score branch exists.
- **Fix:** `elif sort in ("projected_points","z_score"): sort(key=z_score)` (or client-side sort per V1).

### V3 — `?category=k_b` deep link silently ignored — CONFIRMED
- Link emitted at `preview/page.tsx:280`; waiver page has **no `useSearchParams`** anywhere; backend endpoint has no `category` param (`fantasy.py:1855-1865`).
- **Fix:** read search param (in `<Suspense>`), pre-filter by `category_need_match`, show dismissible "Filtered by: K" chip.

### V4 — No add/claim action on player cards — CONFIRMED
- `PlayerRow` (`:169-321`) and `RecommendationCard` (`:512-611`) have no action. Backend `POST /api/fantasy/waiver/add` exists (`fantasy.py:2695-2699`) but **no frontend code references it** (no `api.ts` wrapper).
- **Fix:** add `addWaiverPlayer` to `endpoints`, Add button + `useMutation` invalidating `['waiver']`/`['budget']`.

### V5 — "SV" vs "NSV"; 2-decimal AVG/OPS — CONFIRMED
- Local label map `WAIVER_CAT_LABELS` with `NSV:'SV'` (`page.tsx:29,46,409`) vs canonical `CATEGORY_LABEL` `NSV:'NSV'` (`types.ts:417-420`). Local `fmtVal` (`:345-351`) does `.toFixed(2)` → "0.24"; convention elsewhere is `.toFixed(3)` sans leading zero (`category-battlefield.tsx:19`, `preview/page.tsx:19`).
- **Fix:** delete local maps; import shared `CATEGORY_LABEL` + one shared `formatStat` helper.

---

## 4. Weekly Preview

### P1 — "Projected Win% 100%" renders while Category Projections table has zero rows — CONFIRMED (case mismatch)
- Backend passes simulator's **lowercase v2 keys** verbatim (`fantasy.py:5081-5091`; keys built `mcmc_simulator.py:392-395`; contract documents lowercase at `contracts.py:559`).
- Frontend filters against **UPPERCASE** codes (`preview/page.tsx:173-174` vs `types.ts:440-441`) → zero rows; overall win% renders unconditionally (`:206-214`).
- Bonus: endpoint ignores the simulator's `category_projections` array (`mcmc_simulator.py:408-416`) which already has uppercase keys + Me/Opp projections — so even after case fix, Me/Opp columns show "—".
- **Fix:** consume `sim["category_projections"]` directly (or `cat.upper()`). **One-line-class fix restoring the whole table.**

### P2 — "projected to lose K (0% win rate)" coexisting with 100% overall — CONFIRMED (expected math, bad presentation)
- Same sim run: per-cat flags at `fantasy.py:5092-5104`; overall = majority of 18 cats (`mcmc_simulator.py:401-402`). Amplified because preview sims against an **empty opponent = league-average baseline** (`fantasy.py:5062-5069`), inflating overall win%.
- **Fix:** after P1 the table explains it; consider simulating vs actual opponent roster or labeling "vs league-average opponent".

### P3 — "Schedule Advantage" 0 games both teams — CONFIRMED (hardcoded)
- `fantasy.py:5113` (and `:5058`): `ScheduleAdvantage(my_games=0, opponent_games=0)` — literal stub. No schedule logic behind it. `_fetch_probable_starts_map` (`:165-190`) already pulls MLB schedule (6h cache) and could derive counts.
- **Fix:** implement real computation or hide the card.

---

## 5. Streaming Station

### S1 — "No 2-start pitchers found" across the whole near-term window — CONFIRMED (structural)
- Route (`fantasy.py:7122-7278`) reads **only** the `probable_pitchers` table, requires ≥2 starts in 8-day window.
- **DB facts (2026-07-22 probe):** 2,584 rows, feed alive (synced 12:31 UTC today), BUT coverage collapses after today: 7/22: 30 rows, 7/23: 7, 7/24: 3, 7/25: 3, 7/26-27: 1 each, 7/28+: 0. All future rows `is_confirmed=True`; **zero inferred rows persisted**. Pitchers with ≥2 starts in window: **0**.
- **Root cause:** `_sync_probable_pitchers` (`daily_ingestion.py:7721-8010`) upserts official MLB probables (announced ~1 day out) + fallback `infer_probable_pitcher_for_team` (`probable_pitcher_fallback.py:176-196`) that only infers on an **exact modulo-5 cadence** — post-All-Star rotation resets make exact matches near-impossible. Feature is structurally guaranteed empty.
- Also: 8 future rows NULL `bdl_player_id` silently excluded; `matchup_context` table has **0 rows ever** (`daily_ingestion.py:4090-4311` never populated); no minimum-coverage alerting (job logs "success" with 0 records, `:7833-7836,7974-7979`).
- "STALE · 7 hr ago" badge: threshold 1h (`streaming-recommendations.tsx:139-141`) vs 3×/day cadence → reads STALE most of every day even when healthy.
- **Fix:** loosen inference to ±1-2 day tolerance or project rotations forward across the window; backfill `bdl_player_id`; add coverage alert; align staleness threshold to cadence; short-term UI: show confirmed 1-start pitchers or coverage notice instead of empty state.

### S2 — Footer "Data Sources: ProbablePitcherSnapshot, StatcastPerformances (Quality_Score)" — CONFIRMED
- String hardcoded `fantasy.py:7277`, rendered verbatim `streaming-recommendations.tsx:282-284`. ("StatcastPerformances" isn't even queried by the route.)
- **Fix:** human-readable labels.

### S3 — Roster "No Game" for every player — CONFIRMED (separate cause: unwired feature)
- `/api/fantasy/roster` enrichment never sets `opponent_team`/`is_home`/`game_time` → `_build_player_game_context` returns None (`player_mapper.py:120-137`, TODO at `:227`) → `GamePill` renders "No Game" always (`page.tsx:163-182`).
- **Fix:** populate from `probable_pitchers` (fresh; has opponent/is_home/game_time_et per team/date) or fix `matchup_context` ingestion.

---

## 6. Budget

### B1 — No FAAB figure; name misleading — CONFIRMED
- `BudgetData` (`types.ts:712-732`) has no FAAB field; `/api/fantasy/budget` never calls `get_faab_balance()`. FAAB **is** fetched by the waiver endpoint (`fantasy.py:1899,2686`) but displayed nowhere.
- **Fix:** add `faab_balance` to budget endpoint + a "FAAB Remaining: $NN" row — better than renaming; if renaming, labels at `sidebar.tsx:73`, `header.tsx:27`.

### B2 — Duplicate "Constraint Budget" heading; "Days in Week: 5" ambiguous — CONFIRMED
- Page header `budget/page.tsx:56-62` + card title `budget-panel.tsx:66-71` (same icon, same text). Value shown is `days_in_week_remaining` (`fantasy.py:9104-9105`).
- **Fix:** drop one heading; rename stat "Days Left in Week".

---

## 7. Cross-cutting

| # | Issue | Verdict | Key locations |
|---|-------|---------|---------------|
| X1 | Raw enum strings in UI (`BUY_LOW`, `HIGH_INJURY_RISK`, `player_scores`, `IL10`, `ProbablePitcherSnapshot`) | CONFIRMED | `waiver/page.tsx:269-277,437-439`; signals from `statcast_loader.py:510,526-562`, `fantasy.py:2253-2254,2557-2558`; `fantasy.py:5666,5811,7277`. **No label-mapping helper exists anywhere in frontend/** |
| X2 | Stat naming/precision inconsistency (SV/NSV, 2 vs 3 decimals) | CONFIRMED | see V5 |
| X3 | Silent fallback to false "tied/TBD/0.0" instead of error states | CONFIRMED | R2, R3, S1 — recurring pattern: 200-stub responses + UI ignoring `message`/availability flags |
| X4 | STALE badge unfixable by refresh | CONFIRMED | W5/R1 |
| X5 | Schedule/probable-pitcher feed gaps breaking 3 modules | CONFIRMED w/ nuance | S1 (structural inference gap), S3 (unwired), P3 (hardcoded) — feed itself is alive and syncing 3×/day |

---

## 8. Recommended implementation order (for Claude)

**Tier 1 — actively false information (logic bugs):**
1. **W3** — lower-is-better offset inversion in `mcmc_simulator.py` (corrupts War Room sim, win%, all BEHIND/PUNT advice; likely drove the "0%" alarm).
2. **R5** — optimizer roster-pool min-max normalization (`fantasy.py:5603-5634`) — one-click high-blast-radius feature benches stars. Do (a) now; (b) enable `optimize.blended_score` only after fixing its talent cohort.
3. **R2/R3** — silent 200-stubs + ignored availability flags (TBD/ties, IP 0.0). Render degraded states honestly.

**Tier 2 — broken features, small fixes:**
4. **P1** — preview table case mismatch (consume `sim["category_projections"]` or `.upper()`).
5. **V2** — waiver "Overall Value" sort branch.
6. **V3** — `?category=` deep link.
7. **S1** — probable-pitcher inference tolerance + coverage alert (this is the largest data-engineering item).

**Tier 3 — cheap polish batch:**
8. W1/W2/W4 labels & legends; V5/X2 shared label+format helpers; X1 `SIGNAL_LABELS` map; R6 score_source labels; R7 relabel + no-op move filter; R4 IL banner + `'DL'` literal; B1 FAAB row + B2 heading/label; S2 footer copy; P3 implement-or-hide Schedule Advantage; V4 waiver Add action (needs Write scope — note Yahoo outage §0).

**Infra (Codex, after outage clears):** §0 items 1-3 (token persistence, 403 backoff, Yahoo auth alert) + set `YAHOO_TEAM_KEY` + dedupe `.env` Yahoo lines.

---

*Investigation was read-only throughout: no production code, schema, or data modified; DB access SELECT-only; Railway access limited to `status`/`variables` (comparison only)/`logs`.*


---

## 9. SECOND-PASS UAT ADDENDUM (2026-07-22, post-Yahoo-recovery deploy)

PM re-ran full UAT after the Yahoo outage cleared and a fix deploy landed. Status reconciliation against §1–8, plus verification of four new items.

### Confirmed resolved (from §1–8)
| Item | Status |
|------|--------|
| Yahoo integration (§0) | RESOLVED — was Yahoo-side; self-healed as predicted (refresh-token chain still valid) |
| V2 waiver "Overall Value" sort no-op | FIXED — genuinely reorders by z-score now |
| R3 IP-pace false zero / Budget IL slots | FIXED — Budget reads 3/3 Full, IP 22.1/18 AHEAD |
| W5/R1 staleness/refresh | PARTIAL — "Refresh Data" now recovers real matchup data; first-load TBD stub (R2) still present |
| R5/R6 optimizer transparency | IMPROVED — per-player score-breakdown disclosure shipped ("14-day rolling Z-score percentile, relative to your roster… recent form only"), confirming the Soto benching was by-design form scoring, not miscalculation |
| Streaming badge (S1 staleness threshold) | FIXED — "LIVE · just now"; footer label fix (S2) holding |
| Dashboard | FIXED — Lineup Gaps / Injury Alerts / Waiver Targets populate; "IL Slots 0/33" concatenation gone |

### Still open (unchanged this deploy)
- **W3 War Room direction bug** — PM re-isolated it exactly as §W3 described: pitching L (proj 1 vs 4 → "BEHIND") and batting K (proj 22 vs 35 → "LOST") both inverted, while My Roster/Waiver (current-stat comparisons) score both correctly. Bug is confined to the sim/anchor path in `mcmc_simulator.py:365-390`. **Now the clear #1 fix.**
- **R2** — direct-load "VS TBD / 0W-0L-18T" stub still renders on cold load (refresh now works as a workaround; the stub itself should become a degraded/error state).
- **P1–P3** — Weekly Preview untouched: empty Category Projections table, 100% vs 0%-K juxtaposition, hardcoded Schedule Advantage 0/0.
- **S1** — Streaming still zero 2-start pitchers everywhere (consistent with the structural inference-gap diagnosis; PM noted the Dashboard's one 2-start pitcher is rostered and correctly excluded — worth one verification pass against a known-thin/known-rich week).
- **V1** — waiver load still 20–30s blocking spinner (serial Yahoo calls; the sort fix kept server round-trips).
- **V4** — still no add/claim action on waiver cards.
- **X1** — raw enum chips (BUY_LOW, HIGH_INJURY_RISK) still unformatted.

### New items from second pass — verified
1. **Score-breakdown tooltip "Unknown source" for bench players — CONFIRMED, one-line fix.**
   Bench assignments get `reasoning=f"Bench: score {…}"` with **no `(score_source)` tag** (`backend/routers/fantasy.py:5873`); starters/pitchers include it (`:5709, :5854`). The legacy tooltip extracts the source via regex on reasoning (`war-room/roster/page.tsx:717-719`) → no match → "Unknown source". Fix: append `({player.get('score_source', 'default')})` to the bench reasoning string.
2. **PREMIUM vs STRONG tier flipping between fetches — NOT a labeling bug.**
   Both labels come from one function keyed on `player.need_score` (`waiver/page.tsx:178-184`: ≥20 PREMIUM, ≥15 STRONG). `need_score` is matchup-relative, recomputed from live scoreboard deficits on every request — a player near 20 legitimately crosses the threshold as the live matchup moves. Optional UX: show the need_score value next to the tier so the flip is explainable.
3. **Player Trends HOT list showing identical "Δ 0.3" — NOT a data bug.**
   DB probe: `player_momentum` fresh (848 rows dated today, 829 distinct `delta_z` values, range -1.88..+1.43). The dashboard rounds to 1 decimal (`dashboard-client.tsx:684,704` — `toFixed(1)`), so 0.28/0.31/0.34 all render "0.3". Cosmetic; suggest 2 decimals.
4. **Dashboard injury alert flags only 1 of 4 IL players — BY DESIGN, labeling gap.**
   `_get_injury_flags` deliberately skips injured players already in IL slots (`dashboard_service.py:876-882`); the 1 flag (Díaz) is the one IL player in an *active* slot — matching Budget's IL 3/3 Full. Behavior correct; section title should say "Injury Actions Needed" (or add "3 already on IL" context) so it isn't misread as total injury count.

### Updated priority order for Claude
1. **W3** — sim direction inversion (L, K_B, ERA, WHIP, HR_P) — isolated, reproducible, contradicts two correct pages.
2. **P1** — Weekly Preview table case mismatch (one-line-class) + P3 implement-or-hide.
3. **Addendum-1** — bench "Unknown source" (one line, `fantasy.py:5873`) — polish on an otherwise shippable transparency feature.
4. **V1** — waiver latency (client-side sort of the ~25–50 row array kills both the spinner and the round-trips).
5. **R2** — convert TBD stub to an honest degraded state (refresh workaround exists but shouldn't be required).
6. Batch: V4 add action, X1 enum labels, Addendum-2/3/4 cosmetic labels, S1 inference tolerance (largest data-engineering item — schedule with coverage alerting).
