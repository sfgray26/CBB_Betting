## K-35 Savant Column Verification — May 2, 2026

**Auditor:** Kimi CLI  
**Purpose:** Validate exact column names and data formats before Claude implements K-34 fixes  
**Method:** Live HTTP fetches to `baseballsavant.mlb.com` + direct DB queries

---

### Question 1 — ERA Column Name in Savant Pitcher Leaderboard

**URL pattern tested:**
```
https://baseballsavant.mlb.com/leaderboard/custom?year=2026&type=pitcher&filter=&min=0&selections=...&csv=true
```

**Findings:**

| Test | Selections | Rows | ERA Populated? |
|------|-----------|------|----------------|
| A | Full mix (expected + traditional) | 537 | 0 / 537 |
| B | Traditional stats only (`w,l,qs,ip,era,whip,sv,h,hr,k`) | 537 | 0 / 537 |
| C | Traditional stats only, `min=30` | 104 | 0 / 104 |
| D | Minimal (`pa,era`) | 537 | 0 / 537 |

**Exact column header returned:** `era`

**Column header variants NOT present:** `p_era`, `earned_run_avg`, `earned_run_average`

**Conclusion:** The column name **IS correctly `era`** in the Savant CSV header. However, **ALL 537 pitchers have empty values for ERA** (and for all traditional counting stats: `w`, `l`, `qs`, `ip`, `whip`, `sv`, `h`, `hr`, `k`).

**Root cause:** The Baseball Savant Custom Leaderboard endpoint for `year=2026&type=pitcher` returns expected statistics and quality metrics (`xwoba`, `xera`, `barrel_batted_rate`, `hard_hit_percent`, `exit_velocity_avg`, `k_percent`, `bb_percent`, `k_9`, `whiff_percent`) but **does NOT populate traditional box-score statistics** (`era`, `whip`, `w`, `l`, `ip`, etc.) for the 2026 season through this endpoint.

**Implication for fix:** Changing the column name from `"era"` to something else will NOT fix the issue. The fix must either:
1. Fetch ERA from a separate data source (e.g., FanGraphs/pybaseball, or the `statcast_performances` table where `er` and `ip` exist)
2. Accept that `xera_diff` cannot be computed from this endpoint and remove it from the DB-tier signal pipeline
3. Compute ERA on-the-fly from `statcast_performances` (`SUM(er) / SUM(ip) * 9`) and join it into `_load_from_db()`

---

### Question 2 — `team` Column in Savant `statcast_search` Endpoint

**URL pattern tested:**
```
https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R|&hfSea=2026|&player_type=batter&game_date_gt=2026-04-30&game_date_lt=2026-05-02&group_by=name-date&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc
```

**Full header list returned (76 columns):**
```
pitches, player_id, player_name, game_date, game_pk, total_pitches, pitch_percent,
ba, iso, babip, slg, woba, xwoba, xba, hits, abs, launch_speed, launch_angle,
spin_rate, velocity, effective_speed, whiffs, swings, takes, eff_min_vel,
release_extension, pos3_int_start_distance, pos4_int_start_distance,
pos5_int_start_distance, pos6_int_start_distance, pos7_int_start_distance,
pos8_int_start_distance, pos9_int_start_distance, pitcher_run_exp, run_exp,
bat_speed, swing_length, pa, bip, singles, doubles, triples, hrs, so, k_percent,
bb, bb_percent, api_break_z_with_gravity, api_break_z_induced, api_break_x_arm,
api_break_x_batter_in, hyper_speed, bbdist, hardhit_percent,
barrels_per_bbe_percent, barrels_per_pa_percent, release_pos_z, release_pos_x,
plate_x, plate_z, obp, barrels_total, batter_run_value_per_100, xobp, xslg,
pitcher_run_value_per_100, xbadiff, xobpdiff, xslgdiff, wobadiff,
swing_miss_percent, arm_angle, attack_angle, attack_direction, swing_path_tilt,
rate_ideal_attack_angle, intercept_ball_minus_batter_pos_x_inches,
intercept_ball_minus_batter_pos_y_inches
```

**Column variants checked:**
| Variant | Present? |
|---------|----------|
| `team` | ❌ NO |
| `home_team` | ❌ NO |
| `bat_team` | ❌ NO |
| `pitcher_team` | ❌ NO |

**Conclusion:** The `statcast_search/csv` endpoint **does not return any team column** for the 2026 season. The `team` field in `statcast_performances` will always be empty string when ingested from this endpoint.

**Implication for fix:** The `team` column in `statcast_performances` is effectively a no-op for data quality. Consider either:
1. Removing it from the model/ingestion to reduce confusion
2. Populating it via a post-processing join on `player_id_mapping` → `mlbam_id` → team lookup

---

### Question 3 — Name Format Mismatch in DB Tables

**Query executed:**
```sql
SELECT player_name FROM statcast_performances LIMIT 10;
SELECT player_name FROM statcast_batter_metrics LIMIT 10;
```

**Results:**

| Table | Sample Names | Format |
|-------|-------------|--------|
| `statcast_performances` | `'McCutchen, Andrew'`, `'Dingler, Dillon'`, `'France, J.P.'`, `'Wood, James'`, `'Patrick, Chad'` | **"Last, First"** |
| `statcast_batter_metrics` | `'Andrés Giménez'`, `'Oswald Peraza'`, `'Gabriel Moreno'`, `'Masyn Winn'`, `'José Fermín'` | **"First Last"** |

**Confirmation:** The K-34 finding is **100% accurate**.

- `statcast_performances` stores `player_name` exactly as returned by the Savant `statcast_search/csv` endpoint: **"Last, First"**.
- `statcast_batter_metrics` stores `player_name` after conversion by `savant_ingestion._parse_batter_row()`: **"First Last"**.

**The `_load_from_db()` join can never match:**
```sql
LEFT JOIN (
    SELECT LOWER(player_name) AS lname, AVG(woba) AS avg_woba
    FROM statcast_performances
    WHERE woba > 0
    GROUP BY LOWER(player_name)
) sp_agg ON sp_agg.lname = LOWER(sbm.player_name)
```
- `sp_agg.lname` = `'mccutchen, andrew'`
- `LOWER(sbm.player_name)` = `'andrés giménez'`
- These will never match.

**Implication for fix:** The join must be rewritten. Options in order of preference:
1. **Join on `mlbam_id`** (best): Add `mlbam_id` to `statcast_batter_metrics` (it already exists in the Savant CSV as `player_id`). Then join `statcast_performances.player_id` (which is also the mlbam_id string) on `statcast_batter_metrics.mlbam_id::text`.
2. **Normalize names in the subquery**: Parse `"Last, First"` → `"First Last"` inside the `avg_woba` subquery before lowercasing.
3. **Store normalized names in `statcast_performances`**: Add a `normalized_name` column or trigger that converts on insert.

---

### Bonus Finding — Traditional Batter Stats Also Partially Empty

**URL tested:** `.../leaderboard/custom?year=2026&type=batter&...&selections=pa,ab,h,hr,r,rbi,sb,batting_avg,slg_percent,on_base_plus_slg&csv=true`

| Column | Populated? | Rate |
|--------|-----------|------|
| `batting_avg` | ✅ Yes | 459 / 459 |
| `ab` | ✅ Yes | 459 / 459 |
| `pa` | ✅ Yes | 459 / 459 |
| `h` | ❌ **No** | 0 / 459 |
| `hr` | ❌ **No** | 0 / 459 |
| `r` | ❌ **No** | 0 / 459 |
| `rbi` | ❌ **No** | 0 / 459 |
| `sb` | ❌ **No** | 0 / 459 |

**Implication:** The Savant custom leaderboard returns `batting_avg` and `ab` but NOT the raw counting stats (`h`, `hr`, `r`, `rbi`, `sb`) for 2026. The `_parse_batter_row` code maps these columns, but they will always be `None` in the database. This is less critical than the ERA issue because the batter regression signal relies on `woba` (which IS populated in `statcast_performances`), not on traditional counting stats.

---

### Summary for Claude Code

| Question | Answer | Fix Direction |
|----------|--------|---------------|
| 1. ERA column name | Column IS named `era`; all values are empty | **Not a column-name bug** — endpoint simply doesn't return ERA for 2026. Compute ERA from `statcast_performances` (`SUM(er)/SUM(ip)*9`) or fetch from pybaseball/FanGraphs. |
| 2. `team` in statcast_search | **Absent entirely** — no `team`/`home_team`/`bat_team` in CSV | `team` field in `statcast_performances` will always be empty. Accept this or populate via post-join. |
| 3. Name format mismatch | **Confirmed 100%** — "Last, First" vs "First Last" | Rewrite `_load_from_db()` join. Recommended: join on `mlbam_id` instead of `player_name`. |

---

*Verified by Kimi CLI on 2026-05-02 against live Baseball Savant endpoints and production PostgreSQL database (`railway`).*
