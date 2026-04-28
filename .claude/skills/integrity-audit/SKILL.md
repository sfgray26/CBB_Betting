---
name: integrity-audit
description: Performs comprehensive database and pipeline integrity audits to detect data contamination and logic regressions.
---

# Integrity Audit Skill

This skill codifies the "Brutal Truth" audits to ensure the production database and pipeline stay healthy, particularly focusing on identity mapping and data quality.

## When to Use
Activate this skill when:
- Running a pre-deployment data check.
- Diagnosing "zero value" bugs in waiver or lineup recommendations.
- Verifying the success of a major data ingestion cycle (Savant, BDL, Yahoo).
- Troubleshooting recurring `TypeError` (timezone) or `AttributeError`.

## Core Workflow

### 1. Identity Layer Audit
Verify the completeness of the player identity cross-reference table:
```sql
SELECT 
    COUNT(*) as total_players,
    COUNT(yahoo_id) as mapped_yahoo,
    COUNT(mlbam_id) as mapped_mlbam,
    COUNT(bdl_id) as mapped_bdl
FROM player_id_mapping;
```

### 2. Contamination Audit
Detect placeholder or "stub" data that poisons the z-score pool:
```sql
-- Count default batter rows
SELECT COUNT(*) FROM player_projections 
WHERE hr=15 AND avg=0.25 AND rbi=65;

-- Count default pitcher rows
SELECT COUNT(*) FROM player_projections 
WHERE era=4.0 AND whip=1.3 AND w=0;
```

### 3. Pipeline Freshness Audit
Identify stalled or failing background jobs:
```sql
SELECT job_name, status, started_at, completed_at, error_message
FROM data_ingestion_logs
ORDER BY started_at DESC LIMIT 20;
```

### 4. Integrity Scalars Verification
Audit the `cat_scores` table for logical consistency:
- Batter `avg` should be between 0.0 and 1.0.
- Pitcher `era` should be between 0.0 and 20.0.
- `z_score_total` should not be NULL for active players.

## Guidelines
- **Brutal Truth:** Report the exact failure rate. Do not sugarcoat "mostly working" data.
- **Timezone Safety:** Always check for `offset-naive` vs `offset-aware` mismatches in datetime comparisons.
- **Fail-Fast:** If contamination > 5% of the pool, recommend a `HARD GATE` (Kelly multiplier = 0.0x).

## Examples
"Run a brutal truth audit on the current production state."
-> *Action:* Execute the 4-step audit sequence and update Section 16.5 of `HANDOFF.md`.
