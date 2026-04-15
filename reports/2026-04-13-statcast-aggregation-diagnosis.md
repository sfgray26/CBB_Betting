# Statcast Aggregation & CS Population Diagnosis

**Date:** 2026-04-13
**Author:** Claude (Architect)
**Trigger:** Gemini deployment report — Statcast backfill stalled at 13,376 rows; `cs > 0` count is zero across all rows.
**Status:** Investigation complete. No code change applied — backfill is in-flight.

---

## Symptoms (per Gemini's prod report)

1. **Statcast backfill stalled at 13,376 rows.** Logs show entries like *"Stored 10,332 records for 2026-03-27"* — far above expected MLB-wide daily totals (~600–900 unique batter+pitcher player-game rows).
2. **`statcast_performances.cs > 0` count is zero.** The `/admin/backfill-cs-from-statcast` endpoint runs cleanly but updates 0 rows because the source has no positive CS events.
3. **43.2% zero-metric rate** on `statcast_performances`. Pitcher rows correctly carry zero batting metrics; this fraction matches the proportion of pitcher rows and is largely benign.

---

## Root-Cause Hypotheses (ranked)

### H1. Two-way player overwrite at the upsert (CONFIRMED bug, low blast radius)

`StatcastIngestionAgent.fetch_statcast_day()` (`statcast_ingestion.py:404-440`) issues two separate fetches — one with `player_type='batter'`, one with `player_type='pitcher'` — and `pd.concat`s them. They are then upserted serially via `store_performances()` (`statcast_ingestion.py:580-676`) with `index_elements=['player_id', 'game_date']` and `set_=` containing **every** column.

Effect on a two-way player (Ohtani, Tatis-as-relief, etc.):
- Batter row inserts → `pa, ab, h, bb, sb, cs, xwoba, ...` populated; `ip=0, er=0, k_pit=0, bb_pit=0`.
- Pitcher row hits ON CONFLICT → all batter fields zeroed out (because pitcher branch sets `pa=0, ab=0, h=0, ..., sb=0, cs=0` at lines 513-514) and pitcher fields populated.

**Net:** The most recent fetch wins for the row, regardless of which carries real data. Specifically the `cs` field set on line 514 of the pitcher branch silently zeroes any CS event recorded in the batter row.

This is real but only affects ~2–3 players league-wide. **Not the root cause of the 13,376 / zero-CS symptom.**

### H2. `group_by='name-date'` is not being honored (LIKELY ROOT CAUSE)

The fetch (`statcast_ingestion.py:350-366`) hits `https://baseballsavant.mlb.com/statcast_search/csv` with `group_by='name-date'` and **no** `type` parameter. The expectation (encoded in the comment on line 361) is that omitting `type=details` returns the leaderboard-aggregated CSV (one row per player per game).

Gemini's log line *"Stored 10,332 records for 2026-03-27"* contradicts this. 10,332 rows for a single date is **per-PA / per-pitch density**, not daily aggregation. Likely outcomes:
- Baseball Savant's `/statcast_search/csv` endpoint changed and `group_by` is silently ignored without an accompanying `type=` argument.
- `'all': 'true'` is overriding the grouping intent.
- A different leaderboard endpoint is now required (e.g., `https://baseballsavant.mlb.com/leaderboard/custom`).

**The diagnostic logging at `statcast_ingestion.py:388-392` will reveal the actual columns received.** Gemini should grep prod logs for:
```
"Statcast batter columns present:"
"Statcast pitcher columns present:"
```
If columns include `pitch_type`, `release_speed`, `pfx_x`, `description`, etc. → we are receiving per-pitch data and `group_by` is not working. If columns include `pa`, `abs`, `hits`, `xwoba`, `barrels_per_pa_percent` → we're getting the leaderboard but something else is wrong.

### H3. `caught_stealing_2b` is a per-event indicator, not aggregated

In raw Statcast (`type=details`), `caught_stealing_2b` is a per-pitch boolean/categorical column ("1" or NaN per pitch). The leaderboard endpoint **does not expose** a `caught_stealing_2b` column at all — the corresponding aggregated columns on the leaderboard are typically `cs` or omitted entirely.

If H2 is correct and we **are** receiving per-pitch data, then `_icol(row, 'caught_stealing_2b', ...)` returns 0 or 1 for the **single pitch** represented by that row, not the daily total. Combined with the upsert's "last write wins" behavior, the final stored value for any player on any date is whichever pitch landed last in the iteration order — almost always 0.

This compounds the bug: even if H2 is fixed, we still need to **sum `caught_stealing_2b` across all pitches for a given (player_id, game_date)** in code before upserting.

---

## Proposed Fix Sequence (apply AFTER Gemini's backfill completes)

### Fix 1 — Add a code-level aggregator in `transform_to_performance`

Before (or as part of) constructing `PlayerDailyPerformance` objects, group the DataFrame by `(player_id_or_resolved_name, game_date, _statcast_player_type)` and aggregate:
- **Sum:** `pa, ab, h, doubles, triples, hr, r, rbi, bb, so, hbp, sb, cs, pitches, ip-events, er, k_pit, bb_pit`. For `caught_stealing_2b`/`stolen_base_2b` per-pitch indicators: `sum()` after coercing to numeric (NaN→0).
- **PA-weighted mean (or simple mean if no weight):** `xwoba, xba, xslg, exit_velocity_avg, launch_angle_avg, hard_hit_pct, barrel_pct`.

This makes the pipeline **resilient to the leaderboard-vs-details ambiguity** — it produces correct daily totals regardless of which CSV variant Baseball Savant returns.

### Fix 2 — Split the upsert by player_type to stop two-way player corruption

Convert `store_performances` into two narrower upserts:
- For batter rows: only `set_=` the batter columns plus shared identity fields.
- For pitcher rows: only `set_=` the pitcher columns (`ip, er, k_pit, bb_pit, pitches_pit`).

This requires either (a) splitting `pitches` into `pitches_seen` (batter) and `pitches_thrown` (pitcher), or (b) adding a `_pit` suffix to all pitcher-context Statcast metric columns. Option (b) is cleaner long-term — `xwoba_against`, `barrel_pct_against` are semantically different from the batter version.

For the immediate fix, the simplest defensible change is **(a) plus a separate `pitches_pit` column**, leaving `pitches` to mean "pitches seen in PAs" for batter rows.

### Fix 3 — Verify endpoint behavior with a one-off probe

Before deploying Fix 1/2, run a one-shot fetch against today's date and dump the column list and row count to a report. Compare against the leaderboard custom endpoint. This is a 5-minute Gemini task.

---

## What NOT to do right now

- **Do not modify `statcast_ingestion.py` while Gemini's backfill is running.** Code reload would happen on next deploy and corrupt mid-flight state.
- **Do not delete the existing 13,376 rows.** Even if some are stale, they're better than nothing for the partial derived stats layer (OPS/AVG come from `mlb_player_stats` BDL, not Statcast).
- **Do not add a `type=details` parameter without aggregation.** That makes things worse — guarantees per-pitch data and worsens the overwrite problem.

---

## Coordination Asks

**Gemini:**
1. Grep prod logs for `"Statcast batter columns present:"` and paste the column list into HANDOFF.md. This single piece of evidence resolves H2 vs H3.
2. Stop the in-flight backfill cleanly once it reaches a natural pause point (e.g., end of current date). Confirm in HANDOFF.md.

**User / Architect (next session):**
1. Implement Fix 1 + Fix 2 once Gemini confirms backfill is stopped.
2. Add a regression test that constructs a fake DataFrame with two rows for the same (player_id, game_date) — one batter, one pitcher — and asserts both batting AND pitching fields are populated after `store_performances`.
3. Add a regression test that constructs 10 per-pitch rows with `caught_stealing_2b=1` on three of them and asserts the stored `cs` is 3, not 1.

---

## Why This Justifies the Pause

The user's instruction was *"take your time and ensure quality over speed."* Quality here means:
- Distinguishing the upsert bug (H1, real but small) from the aggregation bug (H2/H3, large and invalidates current data).
- Not overwriting Gemini's running backfill.
- Producing a fix that is verifiable (regression tests above) rather than a guess.

The CS-blocker for NSB scoring remains. But the partial derived stats layer (OPS, WHIP, ERA, AVG, ISO) is unblocked from BDL data and does not depend on Statcast — so we can ship the derived stats helpers independently while this Statcast investigation completes.
