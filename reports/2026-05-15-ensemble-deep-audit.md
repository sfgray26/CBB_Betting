# Ensemble Deep Audit

> **Auditor:** GitHub Copilot CLI  
> **Date:** 2026-05-15  
> **Primary scope:** Fantasy Baseball backend, lineup/waiver/matchup surfaces, projection assembly, frontend War Room/Dashboard  
> **Method:** Direct file review only, targeted existing test execution, no speculative line references

---

## Executive Summary

The platform is **feature-rich and much closer to a real operator console than a prototype**, but the ensemble still has several high-leverage integrity breaks where the intended architecture and the shipped runtime have drifted apart.

The most important conclusion is this:

1. **The lineup stack has two different brains.** The tested scarcity-aware solver lives in `daily_lineup_optimizer`, but `/api/fantasy/roster/optimize` still uses a simpler greedy allocator (`backend\routers\fantasy.py:3447-3477`).
2. **The daily odds math is inverted for a common favorite case.** `_implied_runs()` gives fewer runs to a favored home team when `spread_home` is negative (`backend\fantasy_baseball\daily_lineup_optimizer.py:420-436`).
3. **Roster ingestion is fragile to Yahoo shape variance.** `get_roster()` hard-depends on `players.count` (`backend\fantasy_baseball\yahoo_client_resilient.py:693-701`), which can collapse the roster to zero rows if Yahoo sends numeric keys without a count field.
4. **One of the five matchup factors is effectively disabled in production.** `_fetch_pitcher_stats()` always returns `hand=None` (`backend\services\matchup_engine.py:227-267`), so the handedness component described as 35% of the signal never activates.
5. **The projection assembly path has a real fallback hazard.** `_assemble_batter()` calls `_get_live_projection(mlbam_id, "")`, and `_get_live_projection()` falls back to `player_name.ilike("%%")` when `mlbam_id` is `None` (`backend\fantasy_baseball\projection_assembly_service.py:239-245`, `503-530`).
6. **Frontend contract drift is now visible in the UI.** The waiver contract still carries duplicate fields (`position` + `positions`, `percent_owned` + `owned_pct`) and category labeling is duplicated across surfaces (`frontend\lib\types.ts:372-396`, `457-485`; `frontend\app\(dashboard)\war-room\waiver\page.tsx:17-35`, `101-104`; `frontend\app\(dashboard)\war-room\roster\page.tsx:92-97`, `483-486`).

### Severity Snapshot

| Issue | Severity | Why it matters |
|---|---:|---|
| `/api/fantasy/roster/optimize` bypasses scarcity-aware solver | **P1** | Can mis-slot or bench the wrong players despite a better tested solver existing |
| `_implied_runs()` sign bug | **P1** | Pollutes batter and streaming pitcher rankings from the odds layer |
| `get_roster()` depends on `players.count` | **P1** | A Yahoo payload variant can blank roster-dependent features |
| Matchup handedness signal disabled | **P1** | Removes a core matchup differentiator the system claims to use |
| Live projection fallback + SB/NSB mismatch | **P1/P2** | Can select the wrong projection row and can silently ignore live steals overrides |
| Frontend contract + design-token drift | **P2** | Raises UI inconsistency and forces defensive client logic everywhere |
| Import-time global `requests.get` monkey-patch | **P2** | Cross-cutting runtime/test risk beyond pybaseball itself |
| `z_nsv` weight drift in scoring engine | **P3** | Configuration/schema drift; lower immediate risk, higher maintenance risk |

### Test Result

I executed the existing targeted suite for this audit:

- `tests\test_projection_assembly_service.py`
- `tests\test_lineup_optimizer.py`
- `tests\test_scoring_engine.py`
- `tests\test_matchup_engine.py`
- `tests\test_ui_contracts.py`
- `tests\test_roster_optimize_api.py`
- `tests\test_pybaseball_loader.py`

**Result:** `188 passed, 1 skipped`.

That is encouraging, but part of this audit is that several of the most important failure modes are **coverage gaps, not red tests**.

---

## Expert Findings for Will

### W1. The shipped roster optimizer is not the tested roster solver

**Evidence**

- `/api/fantasy/roster/optimize` sorts players by score then greedily assigns the first eligible slot in priority order (`backend\routers\fantasy.py:3447-3477`).
- The dedicated lineup optimizer test suite explicitly documents scarcity-first behavior as necessary, including Castro/Chapman-style multi-eligibility protection and off-day suppression (`tests\test_lineup_optimizer.py:57-90`, `92-115`, `143-177`).
- The API endpoint tests do not meaningfully assert those behaviors; two key tests still use placeholder assertions like `len(of_assignments) >= 0` and `len(sp_assignments) >= 0` (`tests\test_roster_optimize_api.py:295-329`, `331-374`).

**Why this matters**

This is architectural drift, not just an implementation preference. The codebase already contains the more mature scarcity-aware logic, but the public endpoint is still using a weaker algorithm. That means production behavior can be worse than the codebase’s own validated solver behavior.

**Verdict:** **P1**

### W2. Yahoo roster parsing still has a shape-fragility trap

**Evidence**

- `get_roster()` drills through `roster -> "0" -> players`, then sets `count = int(players_raw.get("count", 0))` and iterates `range(count)` (`backend\fantasy_baseball\yahoo_client_resilient.py:693-701`).
- If Yahoo returns numeric keys without `count`, the function returns an empty player list even if players exist under `"0"`, `"1"`, etc.
- The nearby resilient parsing helper `_parse_players_block()` already uses `_iter_block()` for more robust list/dict iteration (`backend\fantasy_baseball\yahoo_client_resilient.py:1595-1614`).
- Existing IL roster tests validate `selected_position` extraction, not `get_roster()` iteration behavior under count-less payloads (`tests\test_il_roster_support.py:16-57`).

**Why this matters**

`get_roster()` is upstream of roster view, matchup simulation fetches, lineup apply sanitization, and optimize flows. A single Yahoo response-shape change can create a “healthy API, empty roster” failure that looks like missing fantasy data rather than a parser defect.

**Verdict:** **P1**

### W3. A smaller but real time-policy drift remains in the fantasy router

**Evidence**

- `_fetch_probable_starts_map()` uses `datetime.utcnow()` for cache age (`backend\routers\fantasy.py:110-145`).
- The rest of the fantasy stack is ET-aware by convention (`backend\routers\fantasy.py:2942`, `3332`; project guidance in `CLAUDE.md`).

**Why this matters**

This is not the biggest bug in the system, but it is exactly the kind of convention drift that later creates hard-to-reproduce cache-age and day-boundary defects.

**Verdict:** **P2**

---

## Expert Findings for Brad

### B1. `_implied_runs()` gives the wrong team the scoring edge when the home favorite is negative

**Evidence**

- The function comment states: `spread_home is negative when home team is favored` (`backend\fantasy_baseball\daily_lineup_optimizer.py:429`).
- The implementation sets `home_runs = (total + spread_home) / 2.0` (`backend\fantasy_baseball\daily_lineup_optimizer.py:431`).
- Example: total `8.0`, spread `-1.5` => home `3.25`, away `4.75`, which contradicts the stated meaning of a favored home team.

**Why this matters**

This bug contaminates the odds layer at the source. Batter ranking consumes implied runs (`backend\fantasy_baseball\daily_lineup_optimizer.py:555-597`), and streaming pitcher logic depends on the same game-level conversion. If the sign is wrong, the downstream recommendations can be directionally wrong while still looking numerically reasonable.

**Verdict:** **P1**

### B2. The matchup engine advertises a handedness factor, but production fetch logic disables it

**Evidence**

- The module docstring says handedness is one of five core factors and assigns it 35% weight (`backend\services\matchup_engine.py:8-13`).
- `_fetch_pitcher_stats()` returns `PitcherStats(..., hand=None, ...)` with an inline note: `hand not tracked yet` (`backend\services\matchup_engine.py:259-267`).
- `_fetch_hitter_splits()` immediately returns `None` when `pitcher_hand is None` (`backend\services\matchup_engine.py:287-291`).
- The test file is explicitly pure-function oriented and does not exercise the I/O fetch path (`tests\test_matchup_engine.py:4-5`).

**Why this matters**

This is effectively a silent feature flag stuck off. The math layer works when fed handedness, but the runtime collection path never supplies it. So users receive matchup scores that are less discriminating than the architecture claims.

**Verdict:** **P1**

### B3. Projection assembly has two separate integrity problems in the live override path

**Evidence**

1. `_assemble_batter()` calls `_get_live_projection(mlbam_id, "")` (`backend\fantasy_baseball\projection_assembly_service.py:238-245`).  
2. `_get_live_projection()` falls back to `PlayerProjection.player_name.ilike(f"%{player_name}%")` when `mlbam_id` is `None` (`backend\fantasy_baseball\projection_assembly_service.py:526-530`). With `player_name == ""`, that becomes `ilike("%%")`, i.e. any row.  
3. The same batter path writes live steals to `board_proj["sb"]` (`backend\fantasy_baseball\projection_assembly_service.py:245`), but `_build_batter_steamer()` and category impacts both consume `nsb` instead (`backend\fantasy_baseball\projection_assembly_service.py:328-333`, `544-552`).
4. The current tests validate ordinary `_get_live_projection()` behavior and a local `sb` merge, but not the empty-string fallback or downstream `nsb` path (`tests\test_projection_assembly_service.py:542-569`, `571-598`).

**Why this matters**

The first defect can attach the wrong live projection to a Yahoo-only player. The second means that even when a live steals override exists, the canonical batter path may ignore it because the downstream model consumes `nsb`, not `sb`.

**Verdict:** empty-name fallback **P1**; `sb`/`nsb` mismatch **P2**

---

## Expert Findings for Ron

### R1. The waiver contract is still duplicated, and the UI is compensating instead of converging

**Evidence**

- `WaiverAvailablePlayer` declares both `position` and optional `positions`, plus both `percent_owned` and optional `owned_pct` (`frontend\lib\types.ts:457-485`).
- The waiver page defensively normalizes both pairs at render time (`frontend\app\(dashboard)\war-room\waiver\page.tsx:101-104`).
- The roster route also contains backend-side dual ownership fallbacks (`backend\routers\fantasy.py:3106-3108`).

**Why this matters**

This is the classic shape-drift smell: the frontend no longer trusts the backend contract to be singular. Every new consumer now has to remember which fallback order to apply.

**Verdict:** **P2**

### R2. Category identity logic is duplicated across three surfaces

**Evidence**

- Canonical category labels live in `frontend\lib\types.ts:372-396`.
- The waiver page redefines a separate `WAIVER_CAT_LABELS` map (`frontend\app\(dashboard)\war-room\waiver\page.tsx:17-35`).
- The roster page has its own local `formatCat()` remapping logic (`frontend\app\(dashboard)\war-room\roster\page.tsx:92-97`) and a second canonical/internal fallback bridge in `CategorySummary` (`frontend\app\(dashboard)\war-room\roster\page.tsx:483-486`).
- The streaming page consumes central maps but still relies on raw category strings from the API (`frontend\app\(dashboard)\war-room\streaming\page.tsx:93-95`, `220`).

**Why this matters**

Category identity is part of the product’s mental model. When multiple pages each carry their own remapping rules, label drift becomes inevitable and fixes stop being one-line changes.

**Verdict:** **P2**

### R3. Design-system discipline is inconsistent across War Room surfaces

**Evidence**

- Streaming page loading/error/empty shells still use hardcoded `bg-black` and `text-rose-400` (`frontend\app\(dashboard)\war-room\streaming\page.tsx:16-39`, `46-51`).
- The main page body uses design tokens (`bg-bg-base`, `text-text-secondary`) immediately after that (`frontend\app\(dashboard)\war-room\streaming\page.tsx:66-75`).
- Dashboard uses `keepPreviousData` and shows `Refreshing…` while stale cards remain on screen (`frontend\app\(dashboard)\dashboard\page.tsx:20-26`, `73-77`).

**Why this matters**

None of this breaks math, but it does affect trust. Users see a system that sometimes looks tokenized and intentional, and sometimes falls back to raw hardcoded colors and potentially stale content without stronger recency context.

**Verdict:** **P2**

---

## Expert Findings for Dan

### D1. The test suite is green, but some of the most important assertions are still placeholders

**Evidence**

- `tests\test_roster_optimize_api.py` includes endpoint tests whose key assertions are effectively no-ops: `assert len(of_assignments) >= 0`, `assert len(sp_assignments) >= 0`, `assert len(rp_assignments) >= 0` (`tests\test_roster_optimize_api.py:327-329`, `371-374`).
- Meanwhile, the dedicated optimizer tests define the stronger intended behaviors around scarcity, off-days, and multi-eligibility (`tests\test_lineup_optimizer.py:57-90`, `143-177`, `229-314`).

**Why this matters**

This is why the suite can be fully green while the public optimize endpoint still carries algorithmic drift. The endpoint tests confirm response existence more than lineup quality.

**Verdict:** **P1 test-gap**

### D2. `pybaseball_loader` globally monkey-patches `requests.get` at import time

**Evidence**

- `_patch_pybaseball_user_agent()` rewrites `requests.get` directly (`backend\fantasy_baseball\pybaseball_loader.py:63-103`).
- The patch is invoked at module import (`backend\fantasy_baseball\pybaseball_loader.py:103`).

**Why this matters**

This is a cross-cutting side effect. Any other code importing `requests` in the same process inherits the altered behavior. It is workable as a tactical patch, but risky as a permanent runtime contract.

**Verdict:** **P2**

### D3. Scoring-engine configuration has started to drift from its actual category set

**Evidence**

- `PITCHER_CATEGORIES` defines `z_era`, `z_whip`, `z_k_per_9`, `z_k_p`, `z_qs` (`backend\services\scoring_engine.py:57-63`).
- `_CATEGORY_WEIGHTS` also includes `z_nsv` (`backend\services\scoring_engine.py:102-113`).

**Why this matters**

This looks non-fatal today because the category is absent from the active iteration set, but it is exactly the kind of stale config that confuses future maintainers and leads to phantom debugging sessions.

**Verdict:** **P3**

---

## Cross-Examination Transcript

**Will:** “Why call the optimize endpoint a P1 if the tests are green?”  
**Dan:** “Because the green tests don’t assert the hard part. The dedicated solver tests prove the desired scarcity behavior, while the endpoint tests mostly assert that the route returns a shape.”

**Brad:** “Is the odds bug definitely real, or just a naming mismatch?”  
**Will:** “It’s real. The comment explicitly says negative spread means home favored, but the formula gives the home team fewer runs in that case.”

**Ron:** “Is frontend contract drift really a core issue, or just cosmetic debt?”  
**Brad:** “Not cosmetic. Duplicate ownership and position fields are already forcing defensive rendering logic. That is contract uncertainty leaking into the UI.”

**Dan:** “How worried should we be about the projection assembly fallback?”  
**Brad:** “Very. `ilike("%%")` on a no-MLBAM player is not a fuzzy search; it is an arbitrary-row risk. That is exactly the kind of bug that produces believable-but-wrong outputs.”

**Will:** “What’s the single architectural theme across all of this?”  
**Ron:** “The codebase often has the ‘right idea’ implemented somewhere, but the live path is still using an older or looser variant.”

---

## Synthesis

This system does **not** read like a broken platform. It reads like a platform that has grown quickly and now has **parallel implementations competing for authority**:

- a tested scarcity-aware lineup brain vs. a shipped greedy endpoint,
- a five-factor matchup model vs. a fetch layer that disables handedness,
- a canonical frontend category map vs. local page remappers,
- a robust block iterator in one Yahoo path vs. a brittle `count`-driven parser in another.

That is why the right remediation sequence is **not** “tune the model.” It is:

1. unify the authoritative runtime path,
2. close the parser/math hazards,
3. collapse duplicate contracts,
4. only then tune weights and UX.

---

## Action Priority Matrix

| Priority | Action | Owner | Evidence | Recommended fix |
|---|---|---|---|---|
| **P1** | Replace `/api/fantasy/roster/optimize` greedy assignment with the tested scarcity-aware solver | Backend / lineup | `backend\routers\fantasy.py:3447-3477`; `tests\test_lineup_optimizer.py:57-90`, `143-177` | Route endpoint through the same solver/assignment logic used by the dedicated optimizer path |
| **P1** | Correct `_implied_runs()` sign handling for negative home spreads | Backend / odds math | `backend\fantasy_baseball\daily_lineup_optimizer.py:420-436` | Add explicit examples/tests for favored home, favored away, pick’em |
| **P1** | Make `get_roster()` iterate numeric-key payloads without relying on `count` | Backend / Yahoo ingestion | `backend\fantasy_baseball\yahoo_client_resilient.py:693-701`, `1595-1614` | Reuse `_iter_block()` or equivalent robust dict/list traversal |
| **P1** | Populate pitcher handedness in matchup context | Backend / matchup | `backend\services\matchup_engine.py:227-267` | Extend probable-pitcher snapshot/lookup to persist and return hand |
| **P1** | Guard `_get_live_projection()` from empty-name wildcard fallback | Backend / projections | `backend\fantasy_baseball\projection_assembly_service.py:239-245`, `503-530` | If `mlbam_id is None` and name is empty, return `None` immediately |
| **P2** | Fix `sb` vs `nsb` live override path | Backend / projections | `backend\fantasy_baseball\projection_assembly_service.py:245`, `328-333`, `544-552` | Normalize live steals override into the same canonical key consumed downstream |
| **P2** | Collapse waiver/player UI contract to one ownership field and one position shape | Backend + frontend | `frontend\lib\types.ts:457-485`; `frontend\app\(dashboard)\war-room\waiver\page.tsx:101-104` | Remove duplicate fields after backend response normalization |
| **P2** | Centralize category label mapping and remove page-local remappers | Frontend | `frontend\lib\types.ts:372-396`; `waiver\page.tsx:17-35`; `roster\page.tsx:92-97` | Export one canonical formatter/helper and consume it everywhere |
| **P2** | Replace import-time global `requests.get` patch with a scoped adapter | Backend / ingestion | `backend\fantasy_baseball\pybaseball_loader.py:63-103` | Prefer local session wrapper or patch/unpatch around pybaseball fetch calls |
| **P3** | Reconcile `z_nsv` weight with the active scoring category set | Backend / scoring | `backend\services\scoring_engine.py:57-63`, `102-113` | Either implement NSV fully or delete dead config |

---

## Test Plan

### Immediate regression tests to add

1. **Odds math**  
   Add a unit test that asserts a negative `spread_home` produces **higher** implied runs for the home team.

2. **Yahoo roster parser**  
   Add a fixture with numeric player keys and **no** `count` field; assert `get_roster()` still returns the full roster.

3. **Endpoint-vs-solver parity**  
   Reproduce the Castro/Chapman scarcity case through `/api/fantasy/roster/optimize`, not just `DailyLineupOptimizer.solve_lineup()`.

4. **Matchup handedness integration**  
   Add an integration test around `_fetch_pitcher_stats()` / `collect_matchup_context()` to ensure pitcher hand flows into splits collection.

5. **Projection empty-name fallback**  
   Assert `_get_live_projection(None, "") is None` and never issues a wildcard name query.

6. **Live steals override canonicalization**  
   Assert that a live steals override changes the same downstream category key the batter assembler and category impacts actually consume.

7. **Frontend contract convergence**  
   Add a contract test ensuring waiver UI consumes one ownership field and one position shape.

8. **Token consistency**  
   Add a lightweight UI lint/assertion pass for hardcoded forbidden shells like `bg-black` on War Room pages.

### Existing tests already giving useful signal

- `tests\test_lineup_optimizer.py` is strong and worth promoting to be the behavioral source of truth for lineup assignment.
- `tests\test_matchup_engine.py` is mathematically solid, but it needs an integration companion.
- `tests\test_projection_assembly_service.py` covers helpers well, but it does not yet close the fallback/canonical-key hazards.

---

## Appendix — Files Read

### Required context files
- `HANDOFF.md`
- `CLAUDE.md`
- `frontend\tailwind.config.ts`
- `frontend\app\globals.css`

### Backend / service / model files read
- `backend\routers\fantasy.py`
- `backend\models.py`
- `backend\fantasy_baseball\yahoo_client_resilient.py`
- `backend\fantasy_baseball\daily_lineup_optimizer.py`
- `backend\fantasy_baseball\smart_lineup_selector.py`
- `backend\fantasy_baseball\projection_assembly_service.py`
- `backend\fantasy_baseball\pybaseball_loader.py`
- `backend\services\waiver_edge_detector.py`
- `backend\services\matchup_engine.py`
- `backend\services\scoring_engine.py`
- `backend\services\daily_ingestion.py`

### Frontend files read
- `frontend\app\(dashboard)\dashboard\page.tsx`
- `frontend\app\(dashboard)\war-room\roster\page.tsx`
- `frontend\app\(dashboard)\war-room\waiver\page.tsx`
- `frontend\app\(dashboard)\war-room\streaming\page.tsx`
- `frontend\lib\types.ts`
- `frontend\lib\api.ts`

### Tests read
- `tests\test_projection_assembly_service.py`
- `tests\test_pybaseball_loader.py`
- `tests\test_daily_ingestion.py`
- `tests\test_scoring_engine.py`
- `tests\test_matchup_engine.py`
- `tests\test_waiver_edge.py`
- `tests\test_ui_contracts.py`
- `tests\test_lineup_optimizer.py`
- `tests\test_roster_optimize_api.py`
- `tests\test_il_roster_support.py`

### Conditional scope notes
- `frontend\app\(dashboard)\war-room\matchup\page.tsx` was not present in the repository at audit time.
- `backend\services\pipeline_scheduler.py` was not present; `backend\services\daily_ingestion.py` was reviewed instead.
- `backend\services\statcast_loader.py` was not present; `backend\fantasy_baseball\pybaseball_loader.py` was reviewed instead.
