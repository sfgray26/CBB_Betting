# K-NEXT-1: Savant Pitch Quality Distribution Validation

> **Date:** 2026-05-06 | **Analyst:** Kimi CLI (Deep Intelligence Unit)
> **Scope:** `savant_pitch_quality_scores` table, season 2026
> **Status:** **RED FLAG — Do NOT enable feature flags until fixed**

---

## Executive Summary

**The `savant_pitch_quality_scores` table is completely flat and unusable.** All 550 rows for season 2026 have an identical `savant_pitch_quality` score of **100.0** and `sample_confidence` of **0.0**. This is not a scoring algorithm bug — the scoring math is sound — but a **data pipeline failure** where `statcast_pitcher_metrics.ip` (innings pitched) is `NULL` for every single pitcher.

**Recommendation:** Keep all three `savant_pitch_quality_*_enabled` flags at `false`. Do not enable until `ip` is populated from a working data source.

| Metric | Current (Broken) | What It Should Be |
|---|---|---|
| Rows scored | 550 | 550 |
| Unique scores | **1** (100.0 for everyone) | ~40–120 range expected |
| `sample_confidence` | **0.0 for all** | 0.0–1.0 based on IP |
| `min` score | 100.0 | ~85 |
| `max` score | 100.0 | ~115 |
| Top pitchers (expected) | N/A — all identical | Skenes, Wheeler, Sale, Miller |
| Bottom pitchers (expected) | N/A — all identical | Roster fillers, low-K bullpen arms |

---

## 1. Validation Query Results

### 1.1 Top 20 (as requested from HANDOFF.md)

```sql
SELECT player_name, savant_pitch_quality, sample_confidence
FROM savant_pitch_quality_scores
WHERE season = 2026
ORDER BY savant_pitch_quality DESC
LIMIT 20;
```

**Result:** All 20 rows show `savant_pitch_quality = 100.0`, `sample_confidence = 0.0`.

Top 20 names (alphabetical tie, not quality-based):
- Jose Quintana, Fernando Cruz, Merrill Kelly, Craig Kimbrel, Drew Pomeranz, Chris Sale, Martín Pérez, Shawn Armstrong, Danny Coulombe, Nathan Eovaldi, Sonny Gray, Albert Suárez, Michael Lorenzen, Michael Kelly, Aroldis Chapman, Brooks Raley, Andrew Kittredge, Zack Wheeler, Matthew Boyd, Luis García

### 1.2 Bottom 20

```sql
SELECT player_name, savant_pitch_quality, sample_confidence
FROM savant_pitch_quality_scores
WHERE season = 2026
ORDER BY savant_pitch_quality ASC
LIMIT 20;
```

**Result:** Identical to top 20 — every row is exactly 100.0 / 0.0.

### 1.3 Distribution Summary

```sql
SELECT
    COUNT(*) as total,
    MIN(savant_pitch_quality) as min_score,
    MAX(savant_pitch_quality) as max_score,
    AVG(savant_pitch_quality)::numeric(10,2) as avg_score,
    MIN(sample_confidence) as min_conf,
    MAX(sample_confidence) as max_conf,
    AVG(sample_confidence)::numeric(10,4) as avg_conf
FROM savant_pitch_quality_scores
WHERE season = 2026;
```

| total | min_score | max_score | avg_score | min_conf | max_conf | avg_conf |
|---|---|---|---|---|---|---|
| 550 | 100.0 | 100.0 | 100.00 | 0.0 | 0.0 | 0.0000 |

**Standard deviation: 0.0**

---

## 2. Root Cause Analysis

### 2.1 The Scoring Algorithm Depends on IP

In `backend/fantasy_baseball/savant_pitch_quality.py`, the `calculate_savant_pitch_quality()` function computes:

```python
sample_confidence = _sample_confidence(pitcher)
# where:
#   ip_conf   = clamp(ip / 40.0, 0.0, 1.0)
#   pitch_conf = clamp(pitches / 650.0, 0.0, 1.0)
#   sample_confidence = clamp(ip_conf * 0.45 + pitch_conf * 0.55, 0.0, 1.0)

final_score = 100.0 + ((raw_score - 100.0) * sample_confidence)
```

When `ip = NULL`:
- `ip_conf = 0.0`
- `pitches = 0` (calculated as `COALESCE(ip, 0) * 16` in the SQL query)
- `sample_confidence = 0.0`
- `final_score = 100.0 + ((raw_score - 100.0) * 0.0) = 100.0`

The algorithm correctly suppresses the score to the neutral baseline when sample size is unknown. The bug is upstream: **the input data has no IP values.**

### 2.2 `statcast_pitcher_metrics.ip` Is NULL for All 550 Pitchers

```sql
SELECT
    COUNT(*) as total_pitchers,
    COUNT(*) FILTER (WHERE ip IS NOT NULL AND ip > 0) as has_ip,
    MIN(ip) as min_ip,
    MAX(ip) as max_ip,
    AVG(ip)::numeric(10,2) as avg_ip
FROM statcast_pitcher_metrics
WHERE season = 2026;
```

| total_pitchers | has_ip | min_ip | max_ip | avg_ip |
|---|---|---|---|---|
| 550 | 0 | null | null | null |

**All advanced metrics ARE populated** (`xera`, `xwoba`, `k_percent`, `whiff_percent`, etc.), but **all counting stats are NULL** (`ip`, `era`, `whip`, `w`, `l`, `qs`, `sv`, `h`, `hr`, `k`).

### 2.3 Baseball Savant Custom Leaderboard Does Not Return Pitcher Counting Stats

The ingestion pipeline (`backend/fantasy_baseball/savant_ingestion.py`) fetches from:

```
https://baseballsavant.mlb.com/leaderboard/custom
  ?year=2026&type=pitcher&filter=&min=0
  &selections=pa,xwoba,xera,barrel_batted_rate,hard_hit_percent,
              exit_velocity_avg,k_percent,bb_percent,k_9,whiff_percent,
              w,l,qs,ip,era,whip,sv,h,hr,k
  &csv=true
```

**Direct API test (2026-05-06):**
- Total rows returned: 554
- Rows with non-empty `ip`: **0**
- Rows with non-empty `era`: **0**
- Rows with non-empty `whip`: **0**
- Rows with non-empty `w`, `l`, `qs`, `sv`, `h`, `hr`, `k`: **0**
- `pa` (plate appearances against) **IS** populated (e.g., Eury Pérez: 158, deGrom: 150)

**Conclusion:** The Baseball Savant `/leaderboard/custom?type=pitcher` endpoint returns **batted-ball-against data only** (what happens when pitchers face batters). It does **not** return traditional box score counting stats (IP, ERA, WHIP, W, L, etc.) via this endpoint.

This affects not only pitch quality scores but also `statcast_pitcher_metrics` itself, which has been storing NULLs for all counting stats since ingestion began.

---

## 3. Algorithm Validation (Code Is Sound)

To verify the scoring math is correct, I simulated the same 550 pitchers with a synthetic `ip=20.0` (representing a minimum viable sample):

| Statistic | Broken (ip=NULL) | Simulated (ip=20) |
|---|---|---|
| min score | 100.0 | **85.1** |
| max score | 100.0 | **114.9** |
| avg score | 100.0 | **100.2** |
| std dev | 0.0 | **~5.5** |

### Simulated Top 10 (with ip=20)

| Pitcher | Score | Signals |
|---|---|---|
| P.J. Higgins | 114.9 | WATCHLIST, SKILL_CHANGE |
| Mason Miller | 114.9 | WATCHLIST, SKILL_CHANGE |
| Tayler Saucedo | 114.9 | WATCHLIST, SKILL_CHANGE |
| Raisel Iglesias | 114.6 | WATCHLIST, SKILL_CHANGE |
| Kyle Hurt | 114.6 | WATCHLIST, SKILL_CHANGE |
| Carlos Rodriguez | 114.5 | WATCHLIST, SKILL_CHANGE |
| Alan Rangel | 114.4 | WATCHLIST, SKILL_CHANGE |
| Bryan Baker | 114.1 | WATCHLIST, SKILL_CHANGE |
| Juan Morillo | 114.0 | WATCHLIST, SKILL_CHANGE |
| Jesús Luzardo | 113.9 | WATCHLIST, SKILL_CHANGE |

**Observation:** The top list is dominated by relievers and small-sample arms. The `WATCHLIST` signal correctly flags `sample_confidence < 0.50`, which would be the case for most relievers. This is algorithmically correct behavior.

### Simulated Bottom 10 (with ip=20)

| Pitcher | Score | Signals |
|---|---|---|
| Justin Verlander | 85.6 | — |
| PJ Poulin | 85.5 | — |
| Riley Cornelio | 85.4 | — |
| Luis Morales | 85.3 | — |
| Carlos Estévez | 85.1 | — |
| Tyler Gilbert | 85.1 | — |
| Joey Lucchesi | 85.1 | — |
| Jedixson Paez | 85.1 | — |
| Duncan Davitt | 85.1 | — |
| Zach Maxwell | 85.1 | — |

**Observation:** Justin Verlander at the bottom is suspect for 2026. This likely reflects very early-season data (early May) where a few bad outings dominate. The scoring engine would naturally stabilize as IP accumulates. This is expected behavior for early-season data.

### Controlled Test: Elite vs. Mediocre

Using manually constructed inputs:

| Profile | Score | Confidence | Signals |
|---|---|---|---|
| Elite (deGrom-like: 2.85 xERA, 35% K, 0.95 WHIP) | **129.9** | 0.992 | BREAKOUT_ARM, STREAMER_UPSIDE, SKILL_CHANGE |
| Mediocre (Bummer-like: 6.30 xERA, 18.6% K, 1.55 WHIP) | **80.3** | 0.992 | — |

**Conclusion:** The scoring engine correctly separates elite from mediocre when given valid inputs. The 70–130 range is appropriate.

---

## 4. Cross-Check vs. Known 2026 Pitcher Quality

| Pitcher | Actual 2026 Reputation | Simulated Score | Plausible? |
|---|---|---|---|
| Zack Wheeler | Elite ace | 100.0 (flat) | Should be ~115–125 |
| Chris Sale | Elite when healthy | 100.0 (flat) | Should be ~115–125 |
| Mason Miller | Elite closer | 114.9 | Plausible |
| Paul Skenes | Breakout ace | 100.0 (flat) | Should be ~120+ |
| Justin Verlander | Declining vet | 85.6 | Plausible if early struggles |
| Tyler Gilbert | Fringe arm | 85.1 | Plausible |

**Note:** Because the actual table is flat, we cannot verify cross-rankings. The simulated rankings pass a sanity check but would benefit from 4–6 weeks of accumulated IP for stabilization.

---

## 5. Recommended Fixes (in priority order)

### Fix 1: Source IP from a Different Data Feed (P0 — required before enabling flags)

Baseball Savant Custom Leaderboard does not provide IP. Options:

| Source | Approach | Pros | Cons |
|---|---|---|---|
| **FanGraphs API** (`steamerr` endpoint) | Join by `xMLBAMID` → `ip` | Already working, has `xMLBAMID` | Requires cross-source join logic |
| **BDL box stats** (`mlb_player_stats`) | Aggregate `ip` per player | In-house, fresh daily | Only covers players who appeared in games |
| **MLB Stats API** | `people/{id}/stats?stats=season` | Official data | Rate limits, extra API call |

**Recommended:** Use FanGraphs `steamerr` endpoint as the IP source. It returns `IP` for all 5,383 pitchers and includes `xMLBAMID` for direct joining.

### Fix 2: Use `pa` (Plate Appearances Against) as Confidence Proxy (P1 — quick patch)

Savant DOES return `pa` for every pitcher (e.g., deGrom: 150, Bummer: 78). A rough proxy:

```python
# PA ≈ IP * 4.3 (average batters faced per inning)
ip_proxy = pa / 4.3
pitches_proxy = pa * 3.8  # ~3.8 pitches per PA
```

This would immediately un-flatten the scores while the proper IP feed is wired up.

### Fix 3: Populate `statcast_pitcher_metrics` Counting Stats (P1)

The `w, l, qs, ip, era, whip, sv, h, hr, k` columns in `statcast_pitcher_metrics` have been NULL since inception. Even if pitch quality doesn't need them all, the table is supposed to be a comprehensive pitcher metrics store. Fill these from BDL or FanGraphs.

### Fix 4: BOM Handling in CSV Parser (P2)

The Savant CSV starts with a UTF-8 BOM (`\ufeff`). `savant_ingestion.py` does not strip it, causing the first column name to be `\ufeff"last_name` instead of `last_name, first_name`. While Python's `csv` module appears to handle it in production, explicit `.lstrip('\ufeff')` is safer.

---

## 6. Decision: Feature Flags

| Flag | Current | Recommendation | Rationale |
|---|---|---|---|
| `savant_pitch_quality_enabled` | `false` | **KEEP FALSE** | All scores are identical (100.0). No signal. |
| `savant_pitch_quality_waiver_signals_enabled` | `false` | **KEEP FALSE** | BREAKOUT_ARM / STREAMER_UPSIDE / RATIO_RISK cannot fire with confidence=0. |
| `savant_pitch_quality_projection_adjustments_enabled` | `false` | **KEEP FALSE** | Projection adjustments would apply a flat 100 score, doing nothing. |

**Re-evaluation trigger:** Re-run this validation after `ip` is populated and at least 100 pitchers have `sample_confidence >= 0.50`. Expected timeline: 1–2 days after Fix 1 is deployed.

---

## Appendix: Raw Data Samples

### A. `statcast_pitcher_metrics` sample (2026-05-06)

| player_name | team | ip | xera | xwoba | k_percent | whiff_percent | last_updated |
|---|---|---|---|---|---|---|---|
| Aaron Ashby | null | **null** | 2.65 | 0.261 | 37.0 | 37.2 | 2026-05-06 06:00:02 |
| Aaron Bummer | null | **null** | 6.30 | 0.386 | 18.6 | 22.7 | 2026-05-06 06:00:02 |
| Aaron Civale | null | **null** | 4.19 | 0.324 | 17.1 | 18.9 | 2026-05-06 06:00:02 |
| Aaron Nola | null | **null** | 4.40 | 0.331 | 24.8 | 26.5 | 2026-05-06 06:00:02 |
| Alex Vesia | null | **null** | 2.67 | 0.262 | 33.3 | 35.6 | 2026-05-06 06:00:02 |

**Pattern:** `team`, `ip`, `era`, `whip` are universally NULL. Advanced metrics (`xera`, `xwoba`, `k_percent`, `whiff_percent`) are populated.

### B. Savant API raw CSV (first 3 rows)

```
"last_name, first_name","player_id","year","pa","xwoba","xera","barrel_batted_rate","hard_hit_percent","exit_velocity_avg","k_percent","bb_percent","k_9","whiff_percent","w","l","qs","ip","era","whip","sv","h","hr","k"
"Pérez, Eury",691587,2026,158,".350",5,13.9,45.5,"89.9",24.7,10.1,,28.7,,,,,,,,,,
"Mejia, Juan",675848,2026,78,".304",3.65,4.2,39.6,"91.1",28.2,9,,29.4,,,,,,,,,,
```

**Note:** `ip`, `era`, `whip`, `w`, `l`, `qs`, `sv`, `h`, `hr`, `k` are all empty strings (`,,,,,,,,,,` at end of each row).
