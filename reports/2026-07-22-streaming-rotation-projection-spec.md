# SPEC MEMO — Rotation Projection for Streaming Station (2-Start Pitcher Detection)

**Date:** 2026-07-22
**Author:** Kimi CLI (research-only; no code changed)
**For:** Claude Code (implementation decision)
**Refs:** triage `reports/2026-07-22-fantasy-module-audit-triage.md` §S1

---

## 1. Problem

Streaming Station returns "No 2-start pitchers found" for the entire near-term
window. The feed is alive — `probable_pitchers` has 2,584 rows, synced 3×/day
(08:30/16:00/20:00 ET), last sync 2026-07-22 12:31 UTC — but coverage collapses
after ~today because MLB only announces probables ~1–2 days out:

| Date (2026-07) | Rows | With bdl_id |
|---|---|---|
| 22 (today) | 30 | 24 |
| 23 | 7 | 5 |
| 24 | 3 | 3 |
| 25 | 3 | 3 |
| 26 | 1 | 1 |
| 27 | 1 | 1 |
| 28+ | 0 | 0 |

The fallback inference that should fill days +2…+7 persisted **zero rows** —
it only fires on an exact modulo-5 cadence (§3), which matches ~31% of real
rotation turns. With no pitcher holding 2 rows in the 8-day window, the
Streaming route's `len(starts) >= 2` test (`backend/routers/fantasy.py:7267-7268`)
can never pass. The feature is structurally guaranteed empty as designed.

## 2. Current state — code & data inventory

**Table `probable_pitchers`** (`backend/models.py:1882-1924`): `game_date`,
`team`, `opponent`, `is_home`, `pitcher_name`, `bdl_player_id`, `mlbam_id`,
`handedness`, `is_confirmed`, `game_time_et`, `park_factor`, `quality_score`
(−2…+2), `fetched_at`. **UniqueConstraint `(game_date, team)`** (`:1920-1922`) —
one row per team per date (doubleheader implications, §5.1).

**Ingestion** `_sync_probable_pitchers` (`backend/services/daily_ingestion.py:7721-8011`):
- Loops dates 0–7, calls MLB Stats API `schedule?hydrate=probablePitcher,team`
  per date (`:7824-7836`), upserts both sides of every game (`:7860-7979`).
- When the API has no `probablePitcher`, falls back to
  `infer_probable_pitcher_for_team` (`:7886-7896`) and silently skips the team
  entirely when inference returns None (`:7892-7893`).
- Already builds per-pitcher rolling ERA (`mlbam_to_era`, last-10-starts,
  `:7785-7813`) and park-factor quality scores (`:7921-7933`) — **directly
  reusable for projected rows**.
- Already knows each team's **game dates** in the window (the schedule loop) —
  projection never needs a second schedule source.
- Records `official_records`/`inferred_records` counters and
  `_record_job_run("probable_pitchers", …)` (`:7984-7987`) — natural hook for
  coverage telemetry (§6).
- Note: `is_confirmed` is set to `bool(pitcher_data and fullName)` (`:7945`),
  i.e. **any** MLB.com probable = True. Docstring claims True = "officially
  announced (lineup card released)" (`models.py:1893-1895`) — semantics already
  conflated; a `source` column (§4.4) fixes this honestly.

**Fallback** `backend/services/probable_pitcher_fallback.py`:
- `build_recent_starter_candidates` (`:93-173`): last 14 days of
  `mlb_player_stats`, starter heuristic = IP ≥ 4.0 (`parse_innings_pitched`,
  `:48-62`), keeps latest start per (team, player), grouped/sorted per team.
  Carries `bdl_player_id`, `mlbam_id`, `pitcher_name`, `last_start_date`,
  `typical_ip`. **This is the right raw material — keep it.**
- `infer_probable_pitcher_for_team` (`:176-196`): accepts a candidate only if
  `(target_date − last_start).days >= 5 AND % 5 == 0` — the exact-cadence bug.

**Streaming route** `GET /api/fantasy/streaming/recommendations`
(`backend/routers/fantasy.py:7194-7350`): reads only `probable_pitchers` for
`[target, target+7]`; filters `bdl_player_id IS NOT NULL`,
`quality_score IS NOT NULL` (`:7238-7243`); groups by pitcher, needs ≥2 starts
(`:7267-7268`); confidence from `is_confirmed` count (`:7272-7279`); **tier
matrix sends LOW-confidence (0 confirmed) to AVOID** (`:7286-7287`) — see §4.5.

**Related:** `_fetch_probable_starts_map` (`fantasy.py:206-244`) hits the same
MLB schedule endpoint (6h cache) for name→start-counts; `mlb_game_log`
(`models.py:1135-1165`) holds only past/today games, no starter linkage;
`mlb_player_stats` has **no `games_started` column** (`models.py:1219-1278+`) —
starter detection must stay heuristic (IP ≥ 4) unless the ingestion contract
is extended. `two_start_detector.py` is dead code w.r.t. this route (route
re-implements grouping inline) — consolidate or delete when touching this.

## 3. Empirical grounding — real rotation cadence (2026 season)

Days between consecutive starts (starter = IP ≥ 4), `mlb_player_stats`, n≈2,480:

| Gap (days) | Share | Interpretation |
|---|---|---|
| 5 | 29.9% | strict 5-man turn |
| 6 | 43.3% | 5-man + team off-day (modal) |
| 7 | 10.1% | off-day / 6-man turn |
| 8–10 | 5.2% | rainout / skipped turn |
| ≥11 | ~11.5% | IL, demotion, call-up, ASB — not projectable |

**5–7 days covers 83.3% of rotation turns.** The current fallback's acceptance
rule (gap ≥ 5 and ≡ 0 mod 5) matches only gaps of exactly 5 and 10 ≈ **31%** of
turns — and post-All-Star-break rotation resets shift every pitcher's phase,
which is why it produced zero rows this week. Per-pitcher median gap (not a
global constant) is the right estimator: 5-man-strict pitchers median 5,
 pitchers on off-day-heavy teams median 6, 6-man-rotation arms median 6–7.

## 4. Proposed design — per-pitcher rotation projection

### 4.1 Algorithm (replaces the `else` inference branch, `daily_ingestion.py:7886-7896`)

Per team, built **once per sync run** from `mlb_player_stats` (extend
`build_recent_starter_candidates` lookback 14 → 45 days and keep full start
history per pitcher, not just latest):

1. **Rotation set:** the ≤6 most recent distinct starters by `last_start_date`
   (covers 5- and 6-man rotations). **Exclude** any pitcher whose last start is
   > 14 days ago (IL/demotion uncertainty — defer those slots to official
   probables rather than guessing).
2. **Personal cadence:** median gap between the pitcher's last ≤5 starts,
   clamped to [4, 8]; default 6 when < 2 prior gaps (call-up with one start).
3. **Walk the window forward:** each pitcher's `next_start = last_start + cadence`.
   Process the team's future game dates **chronologically** (dates come from the
   existing schedule loop — off-days never appear, see §5.2):
   - candidate = rotation pitcher minimizing `|next_start − D|`
   - accept if within **±2 days**; on accept, advance that pitcher's
     `next_start += cadence` and record the row
   - advancing after assignment is what makes the **second** start in the window
     visible — this is the mechanism that surfaces 2-start pitchers
   - if no pitcher within ±2 days: leave the slot empty (conservative; official
     probables fill it ~24–48h out anyway)

### 4.2 Pseudocode

```
rotation = build_rotation_sets(db, today)            # {team: [PitcherState, ...]}
# PitcherState: bdl_id, mlbam_id, name, last_start, cadence, next_start

for D in dates(0..7):
    for game in mlb_schedule(D):                     # existing loop
        for side in (home, away):
            if game.has_official_probable(side):
                upsert(source="official", is_confirmed=True)      # existing path
                # Reconcile: if the official probable IS one of our rotation
                # arms, re-anchor his next_start = D + cadence (self-correcting)
            else:
                p = nearest(rotation[team], D, tolerance=2)
                if p:
                    upsert(source="projected", is_confirmed=False,
                           quality=era_park_heuristic(p))           # existing §7785-7933
                    p.next_start += p.cadence
```

### 4.3 Why per-pitcher beats sequence-based rotation-slot walking

Sequence walking (slot 1→5 in order) is brittle to mid-season call-ups, skipped
turns, and 6-man transitions — the "order" is unobservable. Per-pitcher
`last_start + median_gap` only assumes *rhythm*, not *order*; the ±2 tolerance
absorbs off-day drift; official-probable reconciliation (4.2) re-anchors any
pitcher the model gets wrong, so errors don't compound across the 3×/day syncs.

### 4.4 Provenance — add `source` column

`ALTER TABLE probable_pitchers ADD COLUMN source VARCHAR(20)` — values
`official` / `projected`; backfill `official` (all existing rows came from the
API). Keeps the already-muddled `is_confirmed` semantics untouched while giving
the route and UI an honest discriminator. Cheap, nullable-safe, no backfill risk.

### 4.5 Route/tier impact (required companion change)

Today 0-confirmed 2-starters → `confidence=LOW` → `AVOID`
(`fantasy.py:7286-7287`) regardless of quality — so projected rows would
surface but all read AVOID. Options (pick one):
- **(a) Recommended:** treat `source='projected'` as its own confidence:
  both-starts-projected → `"PROJECTED"` tier label between AVERAGE and GOOD,
  keep existing `risk_note` copy ("Both starts projected — high variance…",
  `:7305-7306` — already written for exactly this).
- (b) Map projected rows to MEDIUM and let the existing matrix rank them.

The route's existing filters (`bdl_player_id`, `quality_score` non-null) are
satisfied by projected rows automatically (candidates carry BDL IDs; ERA lookup
covers them since they have recent starts).

### 4.6 The 634 NULL-`bdl_player_id` rows

24% of the table is invisible to the route (`fantasy.py:7241`). The job already
tries `mlbam_to_bdl` (`daily_ingestion.py:7913-7915`); gaps = missing
`PlayerIDMapping` entries. Out of scope here, but flag to the
`yahoo_id_sync`/mapping job — fixing it also repairs official-probable coverage.

## 5. Edge cases

### 5.1 Doubleheaders — schema blocker
Two starters, same team, same date; `UniqueConstraint(game_date, team)`
(`models.py:1920-1922`) **cannot store both**, and the upsert
(`daily_ingestion.py:7952-7971`) would overwrite G1's starter with G2's.
Options: (a) relax the constraint to `(game_date, team, mlbam_id)` + include
`mlbam_id` in `index_elements` — preferred, small migration; (b) accept
single-row loss and document. Either way the projection algorithm is unaffected
(assign each game its own nearest pitcher by not advancing cadence between
same-day doubleheader games).

### 5.2 Off-days
Non-issue by construction: rows are only created for dates where the MLB
schedule shows a game. Off-days simply shift `next_start` alignment, absorbed
by the ±2 tolerance and by median-cadence being 6 (off-day-inclusive) for most
pitchers (§3).

### 5.3 IL returns / activations
Last start > 14 days → excluded from the rotation set (4.1.1), so a returning
ace is **not** projected into a slot until he either logs a start or appears in
official probables. Conservative and correct: IL-return timing is precisely
what the model can't know. Same rule covers demotions and post-ASB resets.

### 5.4 Openers / bulk relievers
The IP ≥ 4 heuristic (`probable_pitcher_fallback.py:97`) labels the bulk
reliever — not the opener — as the "starter". For fantasy streaming purposes
the bulk arm is actually the more relevant pitcher, so this mislabeling is
benign; note it, don't fix it here.

### 5.5 Trades / call-ups
One start with new team → cadence defaults to 6, next start projected at ±2 —
acceptable. Traded pitcher's history spans teams; group rotation set by
**current** team (already how `build_recent_starter_candidates` buckets,
`:134-136`).

### 5.6 Rainouts / postponements
Game disappears from the schedule → no row (fine). Makeup game appears on a new
date → projection assigns nearest pitcher there. The 3×/day resync + upsert
self-heals within hours.

## 6. Minimum-coverage alert design

The job currently logs "success" with any record count (silent-empty modes:
`:7833-7836`, `:7892-7893`, `:7974-7979`). Add inline coverage accounting to
`_sync_probable_pitchers` (it already sees every scheduled game):

```
per date D in 0..7:
    expected = 2 * games_scheduled(D)      # both sides of each game
    actual   = rows upserted for D (official + projected)
    coverage = actual / expected
```

| Window | Threshold | Severity | Rationale |
|---|---|---|---|
| D = 0–1 | < 90% | **critical** | official probables published; no excuse for gaps |
| D = 2–3 | < 70% | **warning** | probables trickling in; projection should cover most |
| D = 4–7 | < 50% | **warning** | projection-only territory; some teams unprojectable |

Emit: (1) `_record_job_run` with a `coverage_by_date` payload (telemetry already
exists, `:7987`); (2) on threshold breach, route to the existing alerting path
(`backend/services/alerts.py` / Discord via `discord_notifier`) with a compact
message: `"probable_pitchers coverage 2026-07-24: 3/24 (12%) — inference
pipeline degraded"`. This converts today's silent structural emptiness into a
pageable signal, and doubles as the regression tripwire for the projection work
itself. Also surface `coverage` in the streaming route's `freshness` block
(`fantasy.py:7344-7348`) so the UI can show "projected data" context.

## 7. Validation & rollout

1. **Backtest before enabling** (cheap — all inputs in `mlb_player_stats`):
   replay the algorithm over the last 30 days: for each historical date D,
   project using only data ≤ D−1, compare projected starters vs actual starters
   (IP ≥ 4). Target: ≥ 70% exact-name hit rate on D+2…D+5, ≥ 85% within ±1 day.
   If below target, tune tolerance/cadence clamps before shipping.
2. **Phase 1:** write projected rows with `source='projected'`, no route change
   (rows appear, LOW→AVOID). Verify coverage metrics + eyeball a week.
3. **Phase 2:** route tier change (§4.5) + UI label ("Projected" chip), footer
   `data_sources` copy fix (`fantasy.py:7349` — currently names a table the
   route doesn't even query).
4. **Phase 3:** doubleheader schema migration (§5.1a) if doubleheader weeks
   matter for streaming volume.

## 8. Out of scope / notes

- No change to `quality_score` heuristic needed — projected pitchers have
  recent starts, so `mlbam_to_era` covers them.
- `matchup_context` table (0 rows ever; `daily_ingestion.py:4090-4311`) is a
  separate dead pipeline — noted in triage §S1/S3; not required for this spec.
- Consolidate or delete `backend/fantasy_baseball/two_start_detector.py` when
  touching this area (dead code duplicate of the route's grouping).
