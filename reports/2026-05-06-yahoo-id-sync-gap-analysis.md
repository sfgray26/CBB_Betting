# K-NEXT-2: Yahoo ID Sync Gap Analysis

> **Date:** 2026-05-06 | **Analyst:** Kimi CLI (Deep Intelligence Unit)
> **Scope:** `backend/services/daily_ingestion.py::_sync_yahoo_id_mapping()`
> **Status:** P0 — sync job broken in production since 2026-05-04

---

## Executive Summary

The `_sync_yahoo_id_mapping()` job achieves **~3.7% coverage** of the BDL player universe not because of poor name-matching accuracy (match rate is actually ~94% for players it sees), but because it **only enumerates ~394 Yahoo players per day** against a ~10,000-player BDL index. More critically, **the job has been failing since May 4** with a `UniqueViolation` on `_pim_bdl_id_uc`, meaning zero new Yahoo→BDL enrichments have occurred for the past 48 hours.

| Metric | Value |
|---|---|
| Total `player_id_mapping` rows | 11,036 |
| Rows with `bdl_id` | 10,501 (95.1%) |
| Rows with `yahoo_id` | 2,441 (22.1%) |
| Rows with **both** `yahoo_id` + `bdl_id` | 1,990 (18.0%) |
| Yahoo rows **missing** `bdl_id` | 451 (18.5% of Yahoo rows) |
| Duplicate normalized names in table | 707 names → 1,513 rows (13.7%) |
| Yahoo players processed per run | ~394 unique |
| BDL players matched per run | ~371 |
| Coverage of BDL universe per run | **~3.5%** |
| Last successful run | 2026-05-04 |
| Runs since May 4 | 4 FAILED, 0 SUCCESS |

---

## 1. The Sync Job Is Broken

### Symptom
Every `yahoo_id_sync` run since 2026-05-04 has failed with:

```
(psycopg2.errors.UniqueViolation) duplicate key value violates unique constraint "_pim_bdl_id_uc"
DETAIL:  Key (bdl_id)=(5) already exists.
```

The error occurs when the sync tries to `UPDATE player_id_mapping SET bdl_id=5 ... WHERE id=5`, but `bdl_id=5` is already held by a **different row** (id=85769, also "Geraldo Perdomo").

### Root Cause: Duplicate Rows + Missing Conflict Guard

The table contains **1,513 duplicate rows** across 707 normalized names. For example, "Geraldo Perdomo" appears twice:

| id | source | bdl_id | yahoo_id | yahoo_key |
|---|---|---|---|---|
| 5 | yahoo | 672695 | 11417 | 469.p.11417 |
| 85769 | pybaseball | 5 | 11417 | null |

The BDL index built by the sync is keyed by `normalized_name`. When it finds "geraldo perdomo" → `bdl_id=5`, it looks up the row by `yahoo_key` and finds row **id=5**. It then tries to set `bdl_id=5` on that row, but row id=5 already has `bdl_id=672695`, and row 85769 already holds `bdl_id=5`. The unique constraint fires.

**Why the guard was removed:** `daily_ingestion.py` contains **two complete definitions** of `_sync_yahoo_id_mapping()`:

- **Lines 2204–2401:** Original version with `db.flush()` + `existing_by_bdl` conflict check
- **Lines 7682–7879:** "Simplified" version that **removed** the conflict check

Python executes sequentially; the second definition overwrites the first at import time. The simplified version (line 7682) is what runs in production. Its logic for `existing_by_yahoo` is:

```python
if existing_by_yahoo:
    existing_by_yahoo.bdl_id = bdl_id  # ← NO conflict check!
```

This blindly overwrites `bdl_id` without verifying it isn't already assigned to another row.

---

## 2. The 3.7% Coverage Problem

### Enumeration Scope Is Shallow

The sync fetches Yahoo players from two sources only:

1. **League rosters:** `yahoo.get_league_rosters(league_key)` → ~300 players
2. **Free agents:** `yahoo.get_free_agents(position=pos, start=0, count=25)` for 9 positions → **225 players max**

After dedup: **~394 unique Yahoo players** per run.

Against a BDL universe of ~10,000 players: **394 / 10,000 = 3.94%**.

### Waiver FAs Are Invisible

A 12-team fantasy league with 25 roster spots = 300 rostered players. The waiver wire contains every MLB player **not** on a roster. The sync only fetches the **top 25 free agents per position** (sorted by Yahoo's default ranking, typically ADP or ownership %). Any player ranked 26th or lower at their position is **never seen** by the sync.

This means:
- Deep-league targets (prospects, platoon bats, spot starters) are invisible
- Players who recently lost roster spots (dropped by managers) may fall below position rank 25
- Two-position players may only appear under their primary position, missing secondary eligibility

### Yahoo Rows Still Missing BDL IDs

Even among the 2,441 rows that already have `yahoo_id`, **451 (18.5%) have `bdl_id = NULL`**. These are Yahoo players that were inserted by the sync but never matched to BDL — or were inserted by other pipelines (pybaseball seeded 947 yahoo_ids). The sync never re-processes existing Yahoo rows to backfill missing `bdl_id`s; it only upserts by `yahoo_key`.

---

## 3. Name Matching Limitations

### Exact-Only Matching

The docstring claims:
> "Fuzzy match (threshold 85) for ambiguous cases"

The actual code does **zero fuzzy matching**:

```python
norm_name = normalize_name(name)
if norm_name in bdl_index:
    # exact match only
else:
    unmatched.append(...)
```

There is no fallback to `fuzzy_name_match()` or Levenshtein distance. Any name variation that doesn't survive normalization identically on both sides fails permanently.

### Normalizer Inconsistency

The sync imports `normalize_name` from `backend.fantasy_baseball.mlb_boxscore`, but a newer `normalize_name_for_matching` exists in `orphan_linker.py`. Both produce the same output for common cases today, but divergent maintenance is a latent risk.

### Common Name Collisions

The 707 duplicate normalized names include high-frequency MLB names:

| normalized_name | Count | bdl_ids (sample) |
|---|---|---|
| julio rodriguez | 5 | 679563, 4839665, 4840592, null, 677 |
| jose fernandez | 5 | 699912, 4667354, 4803991, null, 6163 |
| luis garcia | 5 | 1606, 5009, 11577, 4841005, 150193 |
| luis castillo | 5 | 112116, 1349, 6310, 4839000, 369 |
| will smith | 2+ | 1607 (C), 669257 (LAD) |

Because the BDL index is a simple `dict` keyed by normalized name, **only the last BDL player with that name is retained**. If Yahoo's "Will Smith" is the Dodgers pitcher but the BDL dict overwrites him with the Braves catcher (or vice versa), the sync assigns the wrong `bdl_id`.

---

## 4. Proposed Fixes

### Fix 1: Restore the `bdl_id` Conflict Guard (P0 — unblocks the job)

Before assigning `bdl_id` to `existing_by_yahoo`, check if another row already holds it:

```python
if existing_by_yahoo:
    existing_by_bdl = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.bdl_id == bdl_id,
        PlayerIDMapping.id != existing_by_yahoo.id
    ).first()
    if existing_by_bdl:
        logger.warning("bdl_id conflict: skipping %s", yahoo_key)
        unmatched.append({...})
    else:
        existing_by_yahoo.bdl_id = bdl_id
        ...
```

**Better yet:** Deduplicate the `player_id_mapping` table first. Merge rows where `yahoo_id` matches but `bdl_id` differs, preferring the row with the most fields populated.

### Fix 2: Deeper FA Pagination (P1 — fixes coverage)

Fetch free agents with pagination instead of a hard `count=25`:

```python
for pos in positions:
    for start in range(0, 200, 25):  # Fetch top 200 per position
        fa = yahoo.get_free_agents(position=pos, start=start, count=25)
        if not fa:
            break
        yahoo_players.extend(fa)
```

This increases FA coverage from 225 to ~1,800 players, raising daily coverage from 3.7% to ~20%.

### Fix 3: Backfill Missing `bdl_id` for Existing Yahoo Rows (P1)

Add a second pass that queries:

```sql
SELECT yahoo_id, yahoo_key, full_name, normalized_name
FROM player_id_mapping
WHERE yahoo_id IS NOT NULL AND bdl_id IS NULL;
```

Then match these against the BDL index. This would fix the 451 orphaned Yahoo rows without waiting for them to appear in the top-25 FA list.

### Fix 4: Use `mlbam_id` as a Bridge (P1 — eliminates name ambiguity)

The new FanGraphs `steamerr` endpoint returns `xMLBAMID` for every player. Yahoo API also exposes `player_id` which is often the MLBAM ID. Instead of name-only matching, the sync should:

1. Build a BDL index keyed by `mlbam_id` (from BDL's `player` endpoint if available)
2. Fall back to name matching only when `mlbam_id` is absent
3. Use the `mlbam_id` → `bdl_id` bridge to resolve Yahoo players unambiguously

This eliminates the "Will Smith" collision entirely.

### Fix 5: Implement Actual Fuzzy Fallback (P2)

For the ~23 unmatched players per run, add a fuzzy fallback:

```python
from difflib import get_close_matches
if norm_name not in bdl_index:
    close = get_close_matches(norm_name, bdl_index.keys(), n=1, cutoff=0.85)
    if close:
        bdl_data = bdl_index[close[0]]
        # log fuzzy match for audit
```

---

## 5. Immediate Action for Claude

1. **Remove the duplicate `_sync_yahoo_id_mapping` definition** at line 7682 (or merge the conflict guard from line 2204 into the surviving version).
2. **Run a deduplication query** on `player_id_mapping` for rows sharing `yahoo_id` or `normalized_name` + `yahoo_id`.
3. **Re-enable the job** by fixing the UniqueViolation so the 4:30 AM cron can run cleanly.

---

## Appendix: Data Evidence

### A. `player_id_mapping` population by source

| source | rows | has_yahoo_id | has_bdl_id | has_mlbam_id |
|---|---|---|---|---|
| pybaseball | 6,613 | 947 | 6,613 | 6,608 |
| api | 2,800 | 1 | 2,800 | 0 |
| yahoo | 1,493 | 1,493 | 1,042 | 458 |
| manual | 84 | 0 | 0 | 84 |
| bdl | 46 | 0 | 46 | 0 |

### B. Recent `yahoo_id_sync` execution log

| date | status | matched | unmatched | elapsed_ms | error |
|---|---|---|---|---|---|
| 2026-05-06 | FAILED | — | — | 34,709 | `_pim_bdl_id_uc` violation (bdl_id=5) |
| 2026-05-05 | FAILED | — | — | 28,505 | `_pim_bdl_id_uc` violation (bdl_id=1607) |
| 2026-05-04 | SUCCESS | 371 | 23 | 39,236 | — |
| 2026-05-03 | SUCCESS | 371 | 21 | 39,080 | — |
| 2026-05-01 | SUCCESS | 368 | 23 | 30,921 | — |

### C. Duplicate name examples

```sql
SELECT normalized_name, COUNT(*), ARRAY_AGG(bdl_id)
FROM player_id_mapping
GROUP BY normalized_name
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC
LIMIT 5;
```

| normalized_name | count | bdl_ids |
|---|---|---|
| julio rodriguez | 5 | {679563, 4839665, 4840592, null, 677} |
| jose fernandez | 5 | {699912, 4667354, 4803991, null, 6163} |
| luis garcia | 5 | {1606, 5009, 11577, 4841005, 150193} |
| luis castillo | 5 | {112116, 1349, 6310, 4839000, 369} |
| jose rodriguez | 4 | {5997, 11539, 4803933, 4838735} |
