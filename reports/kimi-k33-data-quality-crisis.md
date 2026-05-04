## K-33 FINDINGS (Kimi — 2026-04-28, RESOLVED by Session H)

See full report: `reports/2026-04-28-data-quality-null-audit.md`

**Key numbers (pre-Session H state):**
- `player_rolling_stats.w_runs/w_tb/w_qs`: 85% null → **backfill scripts created (ff7b5a6)**
- `player_scores.z_r/z_h/z_tb/z_k_b/z_ops/z_k_p/z_qs`: 85% null → **backfill scripts created (ff7b5a6)**
- `position_eligibility.scarcity_rank`: 100% null → **logic implemented (ff7b5a6); populates on next daily run**
- `probable_pitchers.quality_score`: 100% null → **heuristic implemented (ff7b5a6); populates on next sync**
- `mlb_player_stats.bdl_stat_id`: 100% null → **removed from model + code (ff7b5a6); Gemini drops column**
- `statcast_performances`: ✅ 0 nulls (unchanged)
- `player_projections.cat_scores`: ✅ 0 nulls (unchanged)

**Self-healing (no action):** `player_daily_metrics.z_score_total` 100% null because season <30 days old. Resolves automatically ~May 25.

---

*Last updated: 2026-04-29 — Session M complete. HEAD: 858f2fb. Test suite: 2454 pass / 3 skip / 0 fail. /admin/diagnostics/field-coverage deployed. Session N = read field counts after deploy.*

---

<!-- ARCHIVED SESSIONS BELOW — DO NOT EDIT -->

---

