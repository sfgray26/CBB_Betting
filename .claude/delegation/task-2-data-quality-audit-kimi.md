# Task 2: Data Quality Audit (Kimi CLI)

**Priority:** P0 — Trust foundation for all recommendations
**Assigned:** Kimi CLI (Deep Intelligence Unit)
**Escalation:** Claude Code if data anomalies found
**Timebox:** 3 hours

---

## Mission

Comprehensive audit of production data quality. Bad projections = bad recommendations = lost H2H matchups.

---

## Database Connection

**Railway Production DB:**
```
postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway
```

**PowerShell setup:**
```powershell
$env:DATABASE_URL = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
```

---

## Audit Queries

### 1. NULL Value Analysis (player_projections)

```sql
-- Check for NULL cat_scores
SELECT
  COUNT(*) as total_players,
  COUNT(cat_scores) as has_cat_scores,
  COUNT(*) - COUNT(cat_scores) as null_cat_scores,
  ROUND(COUNT(cat_scores)::NUMERIC / COUNT(*) * 100, 1) as coverage_pct
FROM player_projections;

-- Sample 20 players with NULL or empty cat_scores
SELECT player_name, player_type, cat_scores, updated_at
FROM player_projections
WHERE cat_scores IS NULL OR jsonb_array_length(cat_scores) = 0
ORDER BY updated_at DESC NULLS LAST
LIMIT 20;

-- Check for NULL critical fields
SELECT
  COUNT(*) FILTER (WHERE z_score IS NULL) as null_z_score,
  COUNT(*) FILTER (WHERE adp IS NULL) as null_adp,
  COUNT(*) FILTER (WHERE player_type IS NULL) as null_player_type
FROM player_projections;
```

### 2. cat_scores Distribution Analysis

```sql
-- Distribution of cat_scores non-zero percentage
WITH cat_check AS (
  SELECT
    player_name,
    player_type,
    CASE
      WHEN cat_scores IS NULL THEN 0
      WHEN jsonb_array_length(cat_scores) = 0 THEN 0
      ELSE CAST(
        SELECT COUNT(*) FILTER (WHERE value != '0')
        FROM jsonb_each_text(cat_scores)
      AS INTEGER)
    END as nonzero_categories
  FROM player_projections
)
SELECT
  player_type,
  COUNT(*) as total,
  SUM(CASE WHEN nonzero_categories = 0 THEN 1 ELSE 0 END) as all_zero,
  SUM(CASE WHEN nonzero_categories > 0 THEN 1 ELSE 0 END) as has_signal,
  ROUND(AVG(nonzero_categories)::NUMERIC, 2) as avg_nonzero_cats,
  ROUND(
    SUM(CASE WHEN nonzero_categories > 0 THEN 1 ELSE 0 END)::NUMERIC /
    COUNT(*) * 100, 1
  ) as signal_pct
FROM cat_check
GROUP BY player_type;
```

### 3. Statcast Freshness Check

```sql
-- How recent is Statcast data?
SELECT
  COUNT(*) as total_performances,
  MAX(game_date) as latest_game,
  CURRENT_DATE - MAX(game_date) as days_stale,
  COUNT(*) FILTER (WHERE game_date >= CURRENT_DATE - INTERVAL '7 days') as last_7_days,
  COUNT(*) FILTER (WHERE game_date >= CURRENT_DATE - INTERVAL '30 days') as last_30_days
FROM statcast_performances;

-- Sample of recent Statcast records
SELECT
  player_name,
  game_date,
  xwoba,
  woba,
  xwoba - woba as xwoba_diff,
  barrel_rate,
  hard_hit_percent
FROM statcast_performances
WHERE game_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY game_date DESC, player_name
LIMIT 20;
```

### 4. Rolling Window Accuracy Check

```sql
-- Compare player_projections.cat_scores to player_scores z-scores
-- These should be correlated but not identical (projections vs. actuals)
WITH proj_sample AS (
  SELECT
    player_name,
    cat_scores->>'hr' as proj_hr_z,
    cat_scores->>'r' as proj_r_z,
    cat_scores->>'rbi' as proj_rbi_z
  FROM player_projections
  WHERE cat_scores IS NOT NULL
  ORDER BY RANDOM()
  LIMIT 20
),
score_sample AS (
  SELECT
    player_name,
    z_score->>'hr' as actual_hr_z,
    z_score->>'r' as actual_r_z,
    z_score->>'rbi' as actual_rbi_z
  FROM player_scores
  WHERE updated_at >= CURRENT_DATE - INTERVAL '7 days'
  ORDER BY RANDOM()
  LIMIT 20
)
SELECT
  COALESCE(p.player_name, s.player_name) as player_name,
  p.proj_hr_z,
  s.actual_hr_z,
  CASE WHEN p.proj_hr_z IS NOT NULL AND s.actual_hr_z IS NOT NULL
       THEN ABS(p.proj_hr_z::NUMERIC - s.actual_hr_z::NUMERIC)
       ELSE NULL
  END as hr_z_diff
FROM proj_sample p
FULL OUTER JOIN score_sample s ON p.player_name = s.player_name;
```

### 5. External Benchmark Comparison (Manual Step)

**Research Task:** Compare our top 20 hitters by z_score to ESPN/MLB.com rankings

```sql
-- Get our top 20 hitters
SELECT
  player_name,
  player_type,
  z_score,
  cat_scores->>'hr' as hr_z,
  cat_scores->>'r' as r_z,
  cat_scores->>'ops' as ops_z
FROM player_projections
WHERE player_type = 'hitter'
ORDER BY z_score DESC
LIMIT 20;
```

**Action:** Manually check:
1. ESPN Fantasy Baseball Rankings (top 100 batters)
2. Baseball Reference WAR leaders
3. How many of our top 20 appear in external top 20?
4. Any obvious outliers (e.g., we rank #1 but ESPN ranks #80)?

---

## Deliverable

Create file: `reports/2026-05-03-data-quality-audit.md`

```markdown
# Data Quality Audit — May 3, 2026

## Executive Summary
- [Overall assessment: HEALTHY / CONCERNS / CRITICAL ISSUES]

## 1. NULL Value Analysis
- player_projections total: [X]
- cat_scores coverage: [X%]
- Critical NULLs: [List any fields with >5% NULL]

## 2. cat_scores Distribution
- Hitters with signal: [X%]
- Pitchers with signal: [X%]
- All-zero players: [X]
- Average non-zero categories: [X]

## 3. Statcast Freshness
- Latest game: [DATE]
- Days stale: [X]
- Last 7 days: [X records]
- Last 30 days: [X records]

## 4. Rolling Window Accuracy
- Sample z-score diffs: [min/avg/max]
- Correlation check: [PROJECTIONS MATCH ACTUALS / DIVERGENCE FOUND]

## 5. External Benchmark Comparison
- Our top 20 vs ESPN top 20 overlap: [X/20]
- Notable outliers: [List any players ranked very differently]

## Critical Findings
- [Any anomalies requiring immediate attention]
- [Recommendations for fixing data quality issues]

## Recommendations
- [Prioritized list of fixes]
```

---

## Success Criteria

- [ ] All 5 audit queries executed with results recorded
- [ ] External benchmark comparison completed (manual + documented)
- [ ] Report saved to `reports/2026-05-03-data-quality-audit.md`
- [ ] Any critical findings flagged for escalation

---

## Escalation Triggers

**Escalate to Claude Code immediately if:**
1. cat_scores coverage < 80% — **P0 data issue**
2. Statcast data stale > 7 days — **pipeline broken**
3. Zero external benchmark overlap (< 5/20 match ESPN) — **projection methodology broken**
4. >50% players have all-zero cat_scores — **K-33 regression detected**

---

## Research Sub-Task (If Time Permits)

**External Comparison Research:**
Use web search to find:
1. ESPN Fantasy Baseball 2026 Rankings (top 100 batters, top 50 pitchers)
2. CBS Sports / MLB.com alternative rankings
3. Steamer 2026 projections (official source)
4. Compare our z-score rankings to external sources

Document methodology and findings in the report.

---

## Reporting Format

After completion, create a GitHub issue with title:
```
[Data Quality] Production Audit — May 3, 2026
```

Body: Paste the contents of `reports/2026-05-03-data-quality-audit.md`

Tag @claude-code (me) for review.
