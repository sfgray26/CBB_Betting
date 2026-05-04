## K-34 FINDINGS — Downstream Consumption Audit (2026-04-28)

> **Full report:** `reports/2026-04-28-downstream-consumption-audit.md`  
> **Auditor:** Kimi CLI | **Scope:** `scarcity_rank` + `quality_score` downstream readers

### Q1–Q3: `scarcity_rank` Consumers

| Consumer | Queries `position_eligibility`? | Uses `scarcity_rank`? | Current Scarcity Logic |
|----------|-------------------------------|----------------------|------------------------|
| `waiver_edge_detector.py` | ❌ No | ❌ No | Hardcoded `_POS_GROUP` dict for drop pairing |
| `daily_lineup_optimizer.py` | ❌ No | ❌ No | Hardcoded `_DEFAULT_BATTER_SLOTS` order |
| `lineup_constraint_solver.py` | ❌ No | ❌ No | Static `SLOT_CONFIG` with internal `scarcity_rank` column |

**Gap:** None of the three consumers read `position_eligibility`. `scarcity_rank` must be wired into each scorer before it affects user-facing output.

### Q3: `quality_score` Consumer — `two_start_detector.py`

- **Reads `quality_score`?** ✅ Yes — SELECTs it from `probable_pitchers` and surfaces it in `MatchupRating` / `TwoStartOpportunity` dataclasses.
- **Downstream impact of null:** All pitchers get `0.0` fallback → `"GOOD"` rating for everyone. `EXCELLENT`/`AVOID` buckets unreachable.
- **Schema mismatch:** `MatchupRatingSchema` documents `quality_score` as `-2.0 to +2.0` (`backend/schemas.py:722`), but Session H heuristic emits `0.0–1.0`.

### Q4: Waiver Recommendations Endpoint (`GET /api/fantasy/waiver/recommendations`)

- **Queries `probable_pitchers`?** ❌ No.
- **Response includes `quality_score`?** ❌ No — `WaiverPlayerOut` and `RosterMoveRecommendation` schemas lack the field.
- **Gap:** Pitcher FA recommendations do not include matchup quality context.

### Q5: DB Schema Verification

```text
position_eligibility.scarcity_rank       integer          nullable
position_eligibility.league_rostered_pct  double precision nullable
probable_pitchers.quality_score           double precision nullable
```

All three columns exist and are nullable. No `NOT NULL` constraints.

### K-34 Recommendations for Session H

| Priority | Action |
|----------|--------|
| P0 | Implement `scarcity_rank` in `_sync_position_eligibility` (already in Session H scope) |
| P0 | Implement `quality_score` in `_sync_probable_pitchers` (already in Session H scope) |
| P1 | **Fix `MatchupRatingSchema` docstring** — `-2.0 to +2.0` → `0.0 to 1.0` to match heuristic |
| P2 | Wire `scarcity_rank` into `waiver_edge_detector.py` need-score multiplier |
| P2 | Wire `scarcity_rank` into `daily_lineup_optimizer.py` slot ordering |
| P2 | Wire `scarcity_rank` into `lineup_constraint_solver.py` objective bonus |
| P3 | Add `matchup_quality` to `WaiverPlayerOut` / `RosterMoveRecommendation` for pitcher FAs |

---

*Section added by Kimi CLI v1.17.0 | Downstream consumption audit complete | Full report: `reports/2026-04-28-downstream-consumption-audit.md`*


----

