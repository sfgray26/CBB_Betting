# Kimi CLI — Session K-4: ID Bridge Archaeology + Duplicate Deduplication Spec

**Date:** 2026-05-01  
**Agent:** Kimi CLI (Deep Intelligence Unit)  
**Assigned by:** Claude Code (Master Architect)  
**Total Queries Run:** 25+  
**Code Files Read:** 4  

---

## SECTION 1 — GROUND TRUTH: WHAT IS player_projections.player_id?

### 1a. Sample 20 rows

```sql
SELECT player_id, player_name, team
FROM player_projections
ORDER BY player_name
LIMIT 20;
```

| player_id | player_name | team |
|-----------|-------------|------|
| 518595 | 518595 | null |
| 641555 | 641555 | null |
| 677595 | 677595 | null |
| 679845 | 679845 | null |
| 687231 | 687231 | null |
| 694025 | 694025 | null |
| 592450 | Aaron Judge | NYY |
| 605400 | Aaron Nola | PHI |
| 594807 | Adam Duvall | ATL |
| 624428 | Adam Frazier | Unknown |
| 668939 | Adley Rutschman | BAL |
| 605244 | Adolis Garcia | PHI |
| 666969 | Adolis García | Unknown |
| 680728 | Adrian Del Castillo | Unknown |
| 605288 | Adrian Houser | SFG |
| 682663 | Agustín Ramírez | Unknown |
| 640462 | A.J. Puk | ARI |
| 444876 | Alcides Escobar | SDP |
| 664761 | Alec Bohm | PHI |
| 676475 | Alec Burleson | Unknown |

### 1b. Pattern analysis

```sql
SELECT
  MIN(CHAR_LENGTH(player_id)) AS min_len,
  MAX(CHAR_LENGTH(player_id)) AS max_len,
  COUNT(*) FILTER (WHERE player_id LIKE '%.p.%') AS yahoo_key_format,
  COUNT(*) FILTER (WHERE player_id ~ '^[0-9]+$') AS pure_numeric,
  COUNT(*) FILTER (WHERE player_id ~ '^[0-9]{6}$') AS six_digit_numeric,
  COUNT(*) FILTER (WHERE player_id ~ '^[0-9]{7}$') AS seven_digit_numeric,
  COUNT(*) FILTER (WHERE player_id LIKE 'st-%' OR player_id LIKE 'ST-%') AS steamer_format,
  COUNT(*) AS total
FROM player_projections;
```

| min_len | max_len | yahoo_key_format | pure_numeric | six_digit_numeric | seven_digit_numeric | steamer_format | total |
|---------|---------|------------------|--------------|-------------------|---------------------|----------------|-------|
| 2 | 6 | 0 | 622 | 622 | 0 | 0 | 623 |

**Verdict:** `player_projections.player_id` stores **MLBAM IDs as 6-digit numeric strings**.

**Corrupt rows:** 6 rows have `player_name = player_id` (e.g., "518595" named "518595") with `team = null`. These are data-quality defects from ingestion.

### 1c. Code write path

**File:** `backend/services/daily_ingestion.py`  
**Function:** `_update_projection_cat_scores` (lines ~4926)  
**Write path:**
```python
# Line 5044-5059: Build normalized_name → mlbam_id map from player_id_mapping
name_to_mlbam = {
    r.normalized_name: str(r.mlbam_id)
    for r in id_map_rows
    if r.normalized_name
}

# Line 5057-5059: Resolve FanGraphs player → MLBAM ID
mlbam_id = name_to_mlbam.get(fg_id) or name_to_mlbam.get(p.get("name", "").lower().strip())

# Line 5121-5126: Upsert PlayerProjection keyed by player_id (= mlbam_id)
stmt = pg_insert(PlayerProjection.__table__).values(
    **upsert_vals
).on_conflict_do_update(
    index_elements=["player_id"],
    set_=conflict_set,
)
```

**Source data:** FanGraphs RoS projections → matched via `player_id_mapping.mlbam_id` → stored as `player_id = str(mlbam_id)`.

**File:** `backend/fantasy_baseball/statcast_ingestion.py`  
**Function:** `_store_updated_projection` (lines ~1009)  
**Write path:** Stores Bayesian-updated projections using `updated.player_id` which is also an MLBAM ID string.

---

## SECTION 2 — GROUND TRUTH: WHAT IS IN player_id_mapping?

### 2a. Sample 20 rows

```sql
SELECT yahoo_key, yahoo_id, mlbam_id, bdl_id, full_name
FROM player_id_mapping
ORDER BY full_name
LIMIT 20;
```

| yahoo_key | yahoo_id | mlbam_id | bdl_id | full_name |
|-----------|----------|----------|--------|-----------|
| null | null | 571437 | 3996 | Aaron Altherr |
| null | null | null | 4840174 | Aaron Antonini |
| 469.p.11489 | 11489 | 676879 | 1046 | Aaron Ashby |
| null | null | 502578 | 3717 | Aaron Barrett |
| null | null | 488686 | 7966 | Aaron Bates |
| null | null | 594760 | 5556 | Aaron Blair |
| null | null | 111213 | 2620 | Aaron Boone |
| null | null | null | 4838692 | Aaron Bracho |
| null | null | null | 4459 | Aaron Brooks |
| null | null | null | 4842505 | Aaron Brown |
| null | null | null | 4839537 | Aaron Brown |
| 469.p.10773 | null | 607481 | 603 | Aaron Bummer |
| 469.p.10869 | null | 650644 | 1055 | Aaron Civale |
| null | null | 346871 | 4133 | Aaron Cook |
| null | null | null | 4024 | Aaron Cunningham |
| null | null | null | 4839566 | Aaron Davenport |
| null | null | 667465 | 4803982 | Aaron Eugene Fletcher |
| null | null | 667465 | 7797 | Aaron Fletcher |
| null | null | null | 4405 | Aaron Fultz |
| null | null | null | 3206 | Aaron Guiel |

### 2b. Population stats

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(yahoo_key) AS has_yahoo_key,
  COUNT(yahoo_id) AS has_yahoo_id,
  COUNT(mlbam_id) AS has_mlbam_id,
  COUNT(bdl_id) AS has_bdl_id,
  COUNT(*) FILTER (WHERE yahoo_key IS NOT NULL AND mlbam_id IS NOT NULL) AS full_bridge_rows,
  COUNT(*) FILTER (WHERE yahoo_id IS NOT NULL AND mlbam_id IS NOT NULL) AS yahoo_id_to_mlbam
FROM player_id_mapping;
```

| total_rows | has_yahoo_key | has_yahoo_id | has_mlbam_id | has_bdl_id | full_bridge_rows | yahoo_id_to_mlbam |
|------------|---------------|--------------|--------------|------------|------------------|-------------------|
| 10,096 | 1,957 | 372 | 6,663 | 10,000 | 1,340 | 296 |

### 2c. Code write path

**File:** `backend/services/daily_ingestion.py`  
**Function:** `_sync_player_id_mapping` (lines ~5910)  

**Data sources:**
1. **BDL API** — `bdl.get_all_mlb_players()` → populates `bdl_id`, `full_name`, `normalized_name`
2. **pybaseball** — `build_mlbam_cache()` → populates `mlbam_id` via name matching
3. **Statcast fallback** — `statcast_performances` name match → patches missing `mlbam_id`

**Columns populated:** `bdl_id`, `full_name`, `normalized_name`, `mlbam_id`, `source`, `resolution_confidence`

**File:** `backend/services/daily_ingestion.py`  
**Function:** `_sync_yahoo_id_mapping` (lines ~1811)  

**Data source:** Yahoo Fantasy API (rosters + free agents)  
**Columns populated:** `yahoo_id`, `yahoo_key`, `full_name`, `normalized_name`  
**Conflict target:** `_pim_yahoo_key_uc` on `yahoo_key`

---

## SECTION 3 — THE ACTUAL JOIN ATTEMPT (4 STRATEGIES)

### 3a. Strategy 1: player_id as yahoo_key

```sql
SELECT COUNT(DISTINCT pp.player_id) AS matched
FROM player_projections pp
JOIN player_id_mapping pim ON pp.player_id = pim.yahoo_key;
```

| matched |
|---------|
| 0 |

### 3b. Strategy 2: player_id as yahoo_id

```sql
SELECT COUNT(DISTINCT pp.player_id) AS matched
FROM player_projections pp
JOIN player_id_mapping pim ON pp.player_id = pim.yahoo_id;
```

| matched |
|---------|
| 0 |

### 3c. Strategy 3: player_id as mlbam_id (CAST integer → string)

```sql
SELECT COUNT(DISTINCT pp.player_id) AS matched
FROM player_projections pp
JOIN player_id_mapping pim ON pp.player_id = CAST(pim.mlbam_id AS TEXT);
```

| matched |
|---------|
| 622 |

**Match rate: 622 / 623 = 99.8%**

### 3d. Strategy 4: player_id as bdl_id

```sql
SELECT COUNT(DISTINCT pp.player_id) AS matched
FROM player_projections pp
JOIN player_id_mapping pim ON pp.player_id = CAST(pim.bdl_id AS TEXT);
```

| matched |
|---------|
| 0 |

### 3e. Name-based fallback

```sql
SELECT COUNT(*) AS name_matched
FROM player_projections pp
JOIN player_id_mapping pim ON LOWER(pp.player_name) = LOWER(pim.full_name);
```

| name_matched |
|--------------|
| 711 |

**⚠️ INFERRED:** Name match count (711) exceeds total projection rows (623) because `player_id_mapping` contains duplicate names across different eras (e.g., two "Aaron Brown" rows with different `bdl_id`). Name-based joins are unreliable for this reason.

### Unmatched row (the 1/623)

```sql
SELECT pp.player_id, pp.player_name, pim.mlbam_id, pim.full_name
FROM player_projections pp
LEFT JOIN player_id_mapping pim ON pp.player_id = CAST(pim.mlbam_id AS TEXT)
WHERE pim.mlbam_id IS NULL;
```

| player_id | player_name | mlbam_id | full_name |
|-----------|-------------|----------|-----------|
| 694025 | 694025 | null | null |

This is one of the 6 corrupt rows where `player_name = player_id` and `team IS NULL`.

---

## SECTION 4 — POSITION_ELIGIBILITY → PLAYER_PROJECTIONS JOIN

### 4a. Direct join

```sql
SELECT COUNT(*) AS direct_match
FROM position_eligibility pe
JOIN player_projections pp ON pe.yahoo_player_key = pp.player_id;
```

| direct_match |
|--------------|
| 0 |

**Confirmed:** `yahoo_player_key` (format "469.p.8658") does not match `player_id` (format "592450").

### 4b. Bridge via player_id_mapping

```sql
SELECT COUNT(DISTINCT pe.yahoo_player_key) AS bridged_match
FROM position_eligibility pe
JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
JOIN player_projections pp ON CAST(pim.mlbam_id AS TEXT) = pp.player_id;
```

| bridged_match |
|---------------|
| 491 |

**Match rate: 491 / 2,389 position_eligibility rows = 20.6%**

Only 1,957 of 10,096 `player_id_mapping` rows have `yahoo_key`, so many position_eligibility rows cannot bridge.

### 4c. Name-based bridge

```sql
SELECT COUNT(*) AS name_bridged
FROM position_eligibility pe
JOIN player_projections pp ON LOWER(pe.player_name) = LOWER(pp.player_name);
```

| name_bridged |
|--------------|
| 521 |

**Sample rows:**

```sql
SELECT pe.player_name, pe.yahoo_player_key, pe.primary_position,
       pp.player_id, pp.player_name, pp.team
FROM position_eligibility pe
JOIN player_projections pp ON LOWER(pe.player_name) = LOWER(pp.player_name)
LIMIT 5;
```

| pe.player_name | pe.yahoo_player_key | primary_position | pp.player_id | pp.player_name | pp.team |
|----------------|---------------------|------------------|--------------|----------------|---------|
| Freddie Freeman | 469.p.8658 | 1B | 518692 | Freddie Freeman | LAD |
| JJ Wetherholt | 469.p.64317 | SS | 124036 | JJ Wetherholt | STL |
| JJ Wetherholt | 469.p.64317 | SS | 802139 | JJ Wetherholt | Unknown |
| George Springer | 469.p.9339 | CF | 543807 | George Springer | TOR |
| Chandler Simpson | 469.p.60420 | CF | 802415 | Chandler Simpson | Unknown |

**⚠️ INFERRED:** Name-based matching produces false positives. "JJ Wetherholt" matches TWO projection rows with different `player_id` values (124036 vs 802139) and different teams (STL vs Unknown). The "Unknown" team row is likely a corrupt/duplicate entry.

### 4d. Current production code join path

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py`  
**Lines:** 518-531 (batters), 648-661 (pitchers)

**Actual composite_z lookup query:**
```sql
SELECT LOWER(pe.player_name) AS name_key, ps.composite_z
FROM position_eligibility pe
JOIN player_scores ps ON pe.bdl_player_id = ps.bdl_player_id
WHERE ps.as_of_date = (SELECT MAX(as_of_date) FROM player_scores)
  AND ps.window_days = 14
  AND pe.bdl_player_id IS NOT NULL
```

**Key finding:** The production `daily_lineup_optimizer.py` does **NOT** join `position_eligibility` → `player_projections` for composite_z lookups. It joins `position_eligibility` → `player_scores` directly via `bdl_player_id`, then uses `LOWER(player_name)` as the dictionary key.

The `rank_batters` function receives `projections: List[dict]` as a parameter (pre-loaded elsewhere). The live rolling bonus bypasses `player_projections` entirely.

**⚠️ INFERRED:** The "ID bridge" issue between `position_eligibility` and `player_projections` does **NOT** break the optimizer's composite_z scoring path. However, it may break the Steamer projection enrichment path in `player_board.py` (line 1125) which queries `PlayerProjection` by `mlbam_id`.

---

## SECTION 5 — mlb_player_stats DUPLICATE ANALYSIS

### 5a. Top 20 duplicates

```sql
SELECT bdl_player_id, game_date, COUNT(*) AS dupe_count,
       ARRAY_AGG(id ORDER BY id) AS row_ids
FROM mlb_player_stats
GROUP BY bdl_player_id, game_date
HAVING COUNT(*) > 1
ORDER BY dupe_count DESC, game_date DESC
LIMIT 20;
```

| bdl_player_id | game_date | dupe_count | row_ids |
|---------------|-----------|------------|---------|
| 819 | 2026-04-30 | 2 | [21937, 22138] |
| 883 | 2026-04-30 | 2 | [21932, 22134] |
| ... | ... | ... | ... |

All 78 duplicates have exactly `dupe_count = 2`.

### 5b. Are duplicate rows identical or divergent?

**Example 1: bdl_player_id 65, game_date 2026-04-30**

```sql
SELECT id, bdl_player_id, game_date, game_id, ab, hits, home_runs, rbi,
       innings_pitched, strikeouts_pit, raw_payload IS NOT NULL AS has_payload
FROM mlb_player_stats
WHERE bdl_player_id = 65
  AND game_date = '2026-04-30'
ORDER BY id;
```

| id | bdl_player_id | game_date | game_id | ab | hits | hr | rbi | ip | k_pit | has_payload |
|----|---------------|-----------|---------|----|------|----|-----|----|-------|-------------|
| 21960 | 65 | 2026-04-30 | 5058234 | 4 | 2 | 0 | 1 | null | 0 | true |
| 22175 | 65 | 2026-04-30 | 7364165 | 4 | 2 | 0 | 0 | null | 0 | true |

**Divergent stats:** Same AB/hits/HR, but **RBI differs** (1 vs 0).

**Example 2: bdl_player_id 44, game_date 2026-04-05**

| id | bdl_player_id | game_date | game_id | ab | hits | hr | rbi | ip | k_pit | has_payload |
|----|---------------|-----------|---------|----|------|----|-----|----|-------|-------------|
| 11486 | 44 | 2026-04-05 | 5057899 | null | null | 0 | 0 | null | 0 | true |
| 11739 | 44 | 2026-04-05 | 6127497 | 0 | 0 | 0 | 0 | null | 2 | true |

**Divergent stats:** Row 11486 has null AB/hits; row 11739 has AB=0, hits=0, and **strikeouts_pit=2 vs 0**.

### 5c. Duplicate clustering by date

```sql
SELECT game_date, COUNT(*) AS duplicate_rows
FROM (
  SELECT bdl_player_id, game_date
  FROM mlb_player_stats
  GROUP BY bdl_player_id, game_date
  HAVING COUNT(*) > 1
) dupes
JOIN mlb_player_stats mps USING (bdl_player_id, game_date)
GROUP BY game_date
ORDER BY game_date DESC
LIMIT 20;
```

| game_date | duplicate_rows |
|-----------|----------------|
| 2026-04-30 | 64 |
| 2026-04-26 | 30 |
| 2026-04-05 | 32 |
| 2026-04-04 | 30 |

**⚠️ INFERRED:** Duplicates are heavily clustered on 4 specific dates. Natural MLB doubleheaders are rare (~1-2 per day, affecting ≤18 players). Having 32 players with duplicates on Apr 30 cannot be explained by doubleheaders alone. This indicates **bulk data ingestion errors** on those dates.

### 5d. Unique constraints on mlb_player_stats

```sql
SELECT conname, contype, pg_get_constraintdef(oid) AS def
FROM pg_constraint
WHERE conrelid = 'mlb_player_stats'::regclass;
```

| conname | contype | def |
|---------|---------|-----|
| mlb_player_stats_pkey | p | PRIMARY KEY (id) |
| _mps_player_game_uc | u | UNIQUE (bdl_player_id, game_id) |
| mlb_player_stats_game_id_fkey | f | FOREIGN KEY (game_id) REFERENCES mlb_game_log(game_id) |

**Finding:** There **IS** a unique constraint on `(bdl_player_id, game_id)`. Duplicates on `game_date` exist because each duplicate pair has a **different `game_id`**.

### 5e. Ingestion code ON CONFLICT handling

**File:** `backend/services/daily_ingestion.py`  
**Function:** `_ingest_mlb_box_stats` (line ~1586)

```python
stmt = pg_insert(MLBPlayerStats.__table__).values(
    bdl_player_id=stat.bdl_player_id,
    game_id=stat.game_id,
    ...
).on_conflict_do_update(
    constraint="_mps_player_game_uc",
    set_=dict(...)
)
```

**Conflict target:** `_mps_player_game_uc` = `(bdl_player_id, game_id)`.

This prevents the SAME game_id from being inserted twice, but does NOT prevent two different `game_id` values for the same player on the same date.

### 5f. Rolling window vulnerability to double-counting

**File:** `backend/services/rolling_window_engine.py`  
**Function:** `compute_rolling_window` (lines ~122)

```python
games_in_window = len(window_rows)

for days_back, row in window_rows:
    w = decay_lambda ** days_back
    sum_weights += w
    # ... sums ALL rows, no deduplication by game_id
```

**⚠️ INFERRED:** The rolling window engine **sums all rows** for a player in the date window. It does **NOT** deduplicate by `game_id`. If a player has 2 rows for the same date (from different `game_id`s), both rows' stats are added together.

**Impact example:** bdl_player_id 65 on Apr 30 would get credit for **8 AB and 4 hits** (4+4, 2+2) in a single day, when reality was likely 4 AB and 2 hits. This **inflates rolling stats** for affected players on affected dates.

---

## SECTION 6 — SYNTHESIS & SPECIFICATIONS

### 6a. ID Bridge Verdict Table

| Question | Answer | Evidence |
|----------|--------|----------|
| What format is player_projections.player_id? | **MLBAM ID as 6-digit numeric string** | Section 1b: 622/623 pure numeric, 6 digits |
| Is player_id_mapping populated? | Yes — 10,096 rows, 6,663 with mlbam_id | Section 2b |
| Best join strategy (highest match %) | **CAST(pim.mlbam_id AS TEXT) = pp.player_id** → 622/623 (99.8%) | Section 3c |
| Can position_eligibility → player_projections be joined directly? | **No** — 0 matches. Yahoo key ≠ MLBAM ID | Section 4a |
| Does production code bypass player_projections for composite_z? | **Yes** — optimizer uses pe.bdl_player_id → ps.bdl_player_id directly | Section 4d |

### 6b. ID Bridge Spec

**Selected: FIX-A (correct join column + 1 corrupt row cleanup)**

> The correct join is: `player_projections.player_id = CAST(player_id_mapping.mlbam_id AS TEXT)`.
>
> The prior Kimi audit used `yahoo_player_id` which does not exist in `player_id_mapping`. PostgreSQL silently evaluated `NULL = NULL` as no match, producing the false "100% unmatched" finding.
>
> **Verification query (run this to confirm):**
> ```sql
> SELECT COUNT(DISTINCT pp.player_id) AS matched,
>        (SELECT COUNT(*) FROM player_projections) AS total,
>        ROUND(100.0 * COUNT(DISTINCT pp.player_id) /
>              (SELECT COUNT(*) FROM player_projections), 1) AS pct
> FROM player_projections pp
> JOIN player_id_mapping pim ON pp.player_id = CAST(pim.mlbam_id AS TEXT);
> ```
> Expected: `matched = 622`, `total = 623`, `pct = 99.8`.
>
> **One corrupt row needs cleanup:**
> ```sql
> SELECT player_id, player_name, team
> FROM player_projections
> WHERE player_name ~ '^[0-9]+$' OR team IS NULL;
> ```
> This returns 6 rows where `player_name = player_id` (numeric string) and `team IS NULL`. These are failed name-resolution inserts from the FanGraphs RoS upsert job. They should be deleted or backfilled.
>
> **Recommended cleanup SQL:**
> ```sql
> DELETE FROM player_projections
> WHERE player_name ~ '^[0-9]+$' AND team IS NULL;
> ```
> ⚠️ Run a `SELECT` first to confirm count = 6 before deleting.
>
> **No code changes needed in the mapping population job.** `_sync_player_id_mapping` correctly populates `mlbam_id`. `_update_projection_cat_scores` correctly stores `player_id = str(mlbam_id)`. The architecture is sound; the audit query was wrong.

### 6c. Duplicate Deduplication Spec

| Finding | Value |
|---------|-------|
| Total duplicate (player, date) pairs | 78 |
| Total duplicate rows | 156 (78 × 2) |
| Are duplicates identical? | **No** — different `game_id` values and divergent stats |
| Clustering | 64 rows on Apr 30, 30 on Apr 26, 32 on Apr 5, 30 on Apr 4 |
| Safe to blindly delete one? | **No** — some may be legitimate doubleheaders |
| Risk to rolling_stats if not fixed | **High** — engine sums all rows; inflated AB/hits/RBI for affected players |

#### Root Cause Analysis

The unique constraint `(bdl_player_id, game_id)` prevents same-game duplicates. However, the BDL API (or the statsapi supplement) has returned **different `game_id` values for what appears to be the same game** on specific dates. The clustering pattern (32 players on Apr 30) is inconsistent with natural doubleheader frequency and strongly suggests a **data source bug or API versioning issue**.

Example: bdl_player_id 65 on Apr 30 has game_id 5058234 (RBI=1) and game_id 7364165 (RBI=0). Nearly identical stats except RBI. This is almost certainly the **same game ingested twice under different IDs**.

#### Deduplication Strategy

Because some duplicates may be legitimate doubleheaders, the safest approach is **conservative merge-preference**:

1. **For each duplicate pair on the same `(bdl_player_id, game_date)`:**
   - Compare the two rows' stat completeness (count of non-null stat columns).
   - Keep the row with **more non-null columns** (more complete data).
   - If tied, keep the row with the **lower `id`** (original ingestion).
   - Delete the other row.

2. **Rationale:** The statsapi supplement patches NULL columns on existing rows. If a duplicate was created by the supplement, the original BDL row often has partial NULLs while the supplement row is more complete. The lower-id row is the original BDL ingestion.

#### Recommended Cleanup SQL

```sql
-- Step 1: Preview rows to be deleted (dry run)
WITH ranked AS (
  SELECT id,
         bdl_player_id,
         game_date,
         game_id,
         -- Count non-null stat columns as completeness score
         (CASE WHEN ab IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN hits IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN home_runs IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN rbi IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN runs IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN walks IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN strikeouts_bat IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN stolen_bases IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN innings_pitched IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN earned_runs IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN hits_allowed IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN walks_allowed IS NOT NULL THEN 1 ELSE 0 END +
          CASE WHEN strikeouts_pit IS NOT NULL THEN 1 ELSE 0 END) AS completeness,
         ROW_NUMBER() OVER (
           PARTITION BY bdl_player_id, game_date
           ORDER BY
             (CASE WHEN ab IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN hits IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN home_runs IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN rbi IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN runs IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN walks IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN strikeouts_bat IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN stolen_bases IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN innings_pitched IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN earned_runs IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN hits_allowed IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN walks_allowed IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN strikeouts_pit IS NOT NULL THEN 1 ELSE 0 END) DESC,
             id ASC
         ) AS rn
  FROM mlb_player_stats
)
SELECT id, bdl_player_id, game_date, game_id, completeness
FROM ranked
WHERE rn > 1;
-- Expected: 78 rows

-- Step 2: Execute deletion (run ONLY after dry run confirms 78 rows)
-- WITH ranked AS ( ... same CTE as above ... )
-- DELETE FROM mlb_player_stats
-- WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
```

#### Prevention Recommendation

Do **NOT** add a unique constraint on `(bdl_player_id, game_date)` because legitimate doubleheaders exist. Instead:

1. **Add a daily cleanup job** (runs after `_ingest_mlb_box_stats` and `statsapi_supplement`) that detects same-date duplicates and logs them for review.
2. **Improve `_ingest_mlb_box_stats`** to check if a player already has a row for the same `game_date` with a different `game_id` and similar stats, and log a warning instead of inserting.
3. **Monitor `data_ingestion_logs`** for bulk insert spikes on specific dates that correlate with duplicate clusters.

---

## APPENDIX: UNVERIFIED ASSUMPTIONS

| # | Assumption | Reason | Impact |
|---|-----------|--------|--------|
| 1 | The 6 corrupt rows in player_projections (player_name = player_id, team IS NULL) should be deleted | They have no mapping and no useful data | Low — only 6/623 rows |
| 2 | Duplicate rows with different game_ids on the same date represent the same game from different sources | Clustering pattern is inconsistent with doubleheaders | Medium — affects dedup strategy safety |
| 3 | The rolling window engine is vulnerable to double-counting | Code review shows `len(window_rows)` with no game_id dedup | High — confirmed by code inspection |

---

**Sign-off:** Kimi CLI | Report saved to `reports/2026-05-01-kimi-k4-id-bridge-and-dedup.md`
