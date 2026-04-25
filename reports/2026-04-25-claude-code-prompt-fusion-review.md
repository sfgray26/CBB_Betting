# Optimized Prompt for Claude Code: Fusion Engine Architectural Review

**Date:** 2026-04-25  
**From:** Kimi CLI (Remediation Complete)  
**To:** Claude Code (Architectural Review Required)  
**Context:** See `HANDOFF.md` section 16.11 for full background on Gemini violations and Kimi fixes.

---

## THE SITUATION

Gemini violated AGENTS.md and created `backend/fantasy_baseball/fusion_engine.py` (579 lines) — a Bayesian projection fusion engine implementing the exact mathematical framework from Kimi's K-29 research. Gemini also integrated it into `player_board.py` and modified `savant_ingestion.py`.

**Kimi has fixed the integration bugs** (90/90 tests pass, Savant ingestion verified live). But `fusion_engine.py` was created without your architectural approval and needs your review.

---

## WHAT KIMI FIXED (DON'T REPEAT)

1. ✅ `_extract_steamer_data()` pitcher validation — no longer rejects ERA=4.00 + K/9=8.5
2. ✅ `get_or_create_projection()` preserves pre-computed `cat_scores` from DB (avoids single-player `compute_cat_scores()` producing all zeros)
3. ✅ Savant ingestion uses correct `/leaderboard/custom` endpoint with proper `selections` parameter
4. ✅ CSV parser strips UTF-8 BOM and maps Custom Leaderboard column names (`barrel_batted_rate`, `batting_avg`, `slg_percent`, etc.)
5. ✅ Added `_savant_float()` / `_savant_int()` helpers for leading-dot values (`.000`)

---

## WHAT NEEDS YOUR ATTENTION

### Task 1: Review `fusion_engine.py` Architecture (P0)

**File:** `backend/fantasy_baseball/fusion_engine.py`  
**Scope:** Approve, modify, or reject the module.

**What's there:**
- `StabilizationPoints` class — PA/IP thresholds per metric (Carleton 2007)
- `PopulationPrior` class — league-average baselines for rookies
- `marcel_update()` — core Empirical Bayes primitive
- `fuse_batter_projection()` / `fuse_pitcher_projection()` — four-state logic
- `_calculate_batter_cat_scores()` / `_calculate_pitcher_cat_scores()` — 1-100 scale (currently dead code in integration)
- `_safe_get()` / `_safe_num()` — null-safety helpers (well-designed)

**Key design decisions to make:**
1. Is the four-state logic (Steamer+Statcast / Steamer-only / Statcast-only / Neither) the right architecture?
2. Should `_calculate_*_cat_scores()` be deleted (currently dead code) or refactored to match the app's z-score scale?
3. Should the module stay in `fantasy_baseball/` or move to `services/`?

### Task 2: Fix `_convert_fusion_proj_to_board_format()` (P1)

**File:** `backend/fantasy_baseball/player_board.py` (lines ~1335)  
**Problem:** The counting stat heuristics are placeholder-quality:

```python
# Current (placeholder)
"w": round(fusion_proj.get("era", 4.50) * -2 + 20),  # Rough wins from ERA
"r": round(fusion_proj.get("ops", 0.730) * pa * 0.15),  # Rough runs from OPS
"rbi": round(fusion_proj.get("slg", 0.410) * pa * 0.18),  # Rough RBI from SLG
```

**What it needs:**
- `hr` = `hr_per_pa * ROS_PA` (already correct)
- `nsb` = `sb_per_pa * ROS_PA` (already correct)
- `r`, `rbi`, `tb`, `k_bat` need proper formulas based on fused rate stats
- `w`, `l`, `qs`, `nsv` for pitchers need proper formulas

The `cat_scores_builder.py` expects these exact keys:
- Batter: `r`, `h`, `hr`, `rbi`, `k_bat`, `tb`, `avg`, `ops`, `nsb`
- Pitcher: `w`, `l`, `hr_pit`, `k_pit`, `era`, `whip`, `k9`, `qs`, `nsv`

### Task 3: Integrate Fusion with `cat_scores_builder.py` (P1)

**File:** `backend/fantasy_baseball/player_board.py` (lines ~1182)  
**Current state:** Kimi added a fallback that uses pre-computed DB cat_scores when available, and fusion engine cat_scores otherwise.

**The problem:** Fusion engine cat_scores are 1-100 scale. Pre-computed DB cat_scores are z-scores. These are on different scales and can't be compared directly.

**Options:**
1. **Delete `_calculate_*_cat_scores()`** and always use pre-computed DB z-scores (when available). For players without DB rows, accept z_score=0 until a periodic backfill runs.
2. **Scale fusion cat_scores to z-score range** by dividing by a constant (~80) so they're roughly comparable.
3. **Run `compute_cat_scores()` with the full player pool** cached in memory or Redis.

**Recommended:** Option 1 for now (simplest, no scale mismatch). Option 3 is the long-term correct solution.

### Task 4: Production Savant Ingestion (P2)

**Command to run:**
```bash
railway ssh python -c "
from backend.models import get_db
from backend.fantasy_baseball.savant_ingestion import run_savant_ingestion
db = next(get_db())
result = run_savant_ingestion(db)
print(result)
"
```

**What to verify:**
- `statcast_batter_metrics` table has ~445 rows
- `statcast_pitcher_metrics` table has ~507 rows
- `xwoba`, `barrel_percent`, `era`, `whip`, `k_9` columns are non-null for most rows

---

## ACCEPTANCE CRITERIA

- [ ] `fusion_engine.py` reviewed and approved (or replacement designed)
- [ ] `_convert_fusion_proj_to_board_format()` produces realistic counting stats
- [ ] All proxy players use a consistent z-score scale (no mixing 1-100 with z-scores)
- [ ] `pytest tests/test_player_board_fusion.py` passes (25/25)
- [ ] `pytest tests/test_cat_scores_backfill.py` passes (12/12)
- [ ] `pytest tests/test_waiver_edge.py` passes (14/14)
- [ ] Savant ingestion runs successfully in production
- [ ] No `AGENTS.md` violations by Gemini in future sessions

---

## KEY FILES

| File | Purpose | State |
|------|---------|-------|
| `backend/fantasy_baseball/fusion_engine.py` | Bayesian fusion engine | Needs architectural review |
| `backend/fantasy_baseball/player_board.py` | Integration point | Kimi-fixed, needs Claude polish |
| `backend/fantasy_baseball/savant_ingestion.py` | Savant CSV ingestion | Kimi-fixed, verified live |
| `backend/services/cat_scores_builder.py` | Z-score computation | Unchanged, integration gap exists |
| `tests/test_player_board_fusion.py` | Fusion integration tests | 25/25 pass |
| `reports/2026-04-24-mathematical-framework-steamer-statcast-fusion.md` | Math framework | Reference for stabilization constants |

---

## CONSTRAINTS

1. **Do NOT delete `fusion_engine.py` without review.** The math is correct; the integration needs work.
2. **Minimal changes to existing test contracts.** The 25 fusion tests define the expected behavior.
3. **No new dependencies.** Use existing `statistics`, `dataclasses`, etc.
4. **Preserve `datetime.now(ZoneInfo("America/New_York"))` pattern.** No `utcnow()`.
5. **Update HANDOFF.md** with your decisions.
