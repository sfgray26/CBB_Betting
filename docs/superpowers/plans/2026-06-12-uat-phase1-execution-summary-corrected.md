# UAT Phase 1 Execution Summary — Corrected

**Date:** 2026-06-12
**Status:** Audits complete, corrected plans ready
**Total Effort:** 13 hours (2 days) for Task 1.2 only
**Baseball IQ Impact:** 6.5/10 → 7.5/10 (with Task 1.2 only)

---

## 📊 Phase 1 Task Status After Audits

| Task | Impact | Effort | Status | Finding | Recommendation |
|------|--------|--------|--------|---------|----------------|
| **1.3 Two-Start Pitcher** | HIGH | 0h | ✅ **AUDITED** | Feature exists, fully wired | Debug logging to identify name matching failures |
| **1.1 Momentum Fix** | HIGH | TBD | ⏸️ **PAUSED** | Plan misdiagnosed root cause | Run diagnostic queries first (see audit) |
| **1.2 Confidence Intervals** | HIGH | 13h | ✅ **READY** | Largely valid, breaking change risk fixed | Implement with backward compatibility |

---

## 🎯 Corrected Phase 1 Goals

**Primary Goal:** Fix the ONE clean implementation task (Task 1.2: Confidence Intervals)

**Success Criteria:**
- Confidence intervals displayed on all projections ✅
- High-variance warnings (CI > 30%) show warning badge ✅
- Zero breaking changes (projection field kept) ✅

**Baseball IQ Target:** 7.5/10 (up from 6.5/10)

---

## 📋 Phase 1 Execution Plan (Corrected)

### Day 1-2: Task 1.2 (Confidence Intervals) — ONLY TASK TO IMPLEMENT

**Why only this task?**
- Task 1.3 already exists (just needs debug logging)
- Task 1.1 requires diagnostic queries first (plan was wrong)
- Task 1.2 is the cleanest, highest-impact fix

**Deliverables:**
- Percentile fields added to player_projections table
- CI badges displaying with variance warnings
- High-variance alert banner
- Zero breaking changes (backward compat)

---

## 🚀 Execution Command (Single Task)

### Terminal 1: Claude Code — Confidence Intervals

```powershell
cd C:\Users\sfgra\repos\Fixed/cbb-edge
gh copilot run --model claude-sonnet-4 --editor-code "Implement Task 1.2: Confidence Intervals on Projections (Corrected Plan)

Follow plan: docs/superpowers/plans/2026-06-12-uat-phase1-ci-corrected.md

CRITICAL: Do NOT rename the 'projection' field — this will break the frontend.

Steps:
1. Add percentile fields to PlayerProjection model in backend/models.py:
   - projection_p05, projection_p25, projection_p50, projection_p75, projection_p95
   - ci_variance, ci_sample_size
   - Keep existing 'projection' field for backward compatibility

2. Create migration: alembic revision --autogenerate -m 'add projection percentiles and confidence intervals'
   Apply: alembic upgrade head

3. Implement compute_projection_percentiles() in backend/services/projection_engine.py
   - Use SimulationResult percentile fields (p10, p50, p90)
   - Compute ci_variance = (p95 - p05) / p50
   - Return dict with p05, p25, p50, p75, p95, ci_variance, ci_sample_size

4. Update projection upsert logic to populate percentile fields
   - Keep projection = projection_p50 (no change to existing field)
   - Add projection_p05, projection_p25, projection_p50, projection_p75, projection_p95
   - Add ci_variance, ci_sample_size

5. Update backend/schemas.py WaiverPlayerOut class
   - Add projection_p05, projection_p50, projection_p95, ci_variance, ci_sample_size, ci_explanation
   - Keep projection field (no change)

6. Update backend/routers/fantasy.py waiver_recommendations endpoint
   - Query percentile fields from player_projections table
   - Build ci_explanation string
   - Pass to WaiverPlayerOut constructor

7. Update frontend/lib/types.ts WaiverAvailablePlayer interface
   - Add projection_p05, projection_p50, projection_p95, ci_variance, ci_sample_size, ci_explanation
   - Keep projection field (no change)

8. Add CI display in frontend/app/(dashboard)/war-room/waiver/page.tsx
   - Show '({p05}-{p95})' below need_score
   - Add '⚠ High Variance' badge if ci_variance > 0.3
   - Add high-variance warning banner if >30% have high variance

9. Test: verify percentiles returned for all projections
10. Test: verify CI > 0.3 shows warning badge
11. Test: verify backward compat (frontend still reads projection field)
12. Commit: feat: add confidence intervals and variance warnings to projections (backward compatible)

Note: Use existing SimulationResult table — Monte Carlo already runs daily at 6 AM ET.
Do NOT rename projection field — keep it for backward compatibility." --repo ./
```

---

## 📋 Post-Phase 1: Next Steps

### After Task 1.2 Complete

**1. Run Diagnostic Queries for Task 1.1**
Follow `docs/superpowers/audits/2026-06-12-duran-cold-diagnostic.md`:
```sql
-- Get Duran's momentum data
SELECT * FROM player_momentum WHERE bdl_player_id = <DURAN_ID> ORDER BY as_of_date DESC LIMIT 5;

-- Get cohort distribution
SELECT COUNT(*), AVG(delta_z), STDDEV(delta_z) FROM player_momentum WHERE as_of_date = '2026-06-11' AND player_type = 'hitter';

-- Get player_scores
SELECT * FROM player_scores WHERE bdl_player_id = <DURAN_ID> AND window_days IN (14, 30);
```

**2. Add Debug Logging for Task 1.3**
Follow `docs/superpowers/audits/2026-06-12-two-start-wiring-audit.md`:
```python
# In _populate_starts_this_week() in backend/routers/fantasy.py
logger.debug("Player %s: positions=%s, starts=%d", _name, _fa.get("positions"), _starts)
if _starts == 0 and starts_map:
    logger.debug("Fuzzy match for %s: best=%s, ratio=%.2f", _name, _best, SequenceMatcher(None, _name, _best).ratio())
```

**3. Decide Next Actions Based on Diagnostics**
- If Duran's surge was AFTER 14d window closed → stale data → add near-real-time computation
- If Duran's surge was INSIDE 14d window but algorithm still says COLD → algorithm bug → implement EMA weighting
- If two-start badges not showing due to name matching → improve matching logic

---

## ✅ Verification Checklist

### After Task 1.2 (Confidence Intervals)

- [ ] Percentile fields exist in player_projections (p05, p25, p50, p75, p95)
- [ ] CI badges display: "1.3 pts (0.8-1.9 pts)"
- [ ] High variance badge: "⚠ High Variance" for CI > 0.3
- [ ] High-variance alert banner displays when >30% have high CI
- [ ] Tooltips explain CI meaning
- [ ] Frontend still reads projection field (backward compat)
- [ ] Migration applied successfully
- [ ] Zero breaking changes

### After Diagnostic Queries (Task 1.1)

- [ ] Duran's momentum data retrieved
- [ ] Cohort distribution computed
- [ ] Duran's z_score_delta computed
- [ ] Surge timing identified (inside or outside 14d window)
- [ ] Root cause determined (stale data vs algorithm bug)

### After Debug Logging (Task 1.3)

- [ ] Logging added to _populate_starts_this_week()
- [ ] Name matching failures logged
- [ ] Fuzzy match attempts logged
- [ ] Logs analyzed to identify failures

---

## 📈 Expected Outcomes

### After Task 1.2 Only

**Data Quality Improvements:**
- **Percentile coverage:** 100% of projections have p05, p50, p95
- **CI width distribution:** 60% have CI < 0.2, 30% have CI 0.2-0.3, 10% have CI > 0.3

**User Experience Improvements:**
- **Trust:** Managers understand projection reliability
- **Decision-making:** Variance flags prevent overconfident picks

**Baseball IQ Improvements:**
- **Current:** 6.5/10 (good discovery tool, weak decision engine)
- **After Task 1.2:** 7.5/10 (+1.0 for transparency)

### After Diagnostic Queries (Task 1.1)

**Understanding:**
- Confirm whether Duran's COLD signal is due to stale data or algorithm bug
- Determine if EMA weighting is needed (algorithm bug) or near-real-time computation (stale data)

### After Debug Logging (Task 1.3)

**Visibility:**
- Identify why two-start badges aren't showing (if they aren't)
- Fix name matching failures if root cause

---

## 🔄 Post-Phase 1: Phase 2 Planning

After completing Task 1.2 + diagnostics + debug logging:

**If diagnostics reveal stale data (Task 1.1):**
- Implement near-real-time momentum computation (12 PM ET run)
- Add "Data from X AM" timestamp

**If diagnostics reveal algorithm bug (Task 1.1):**
- Implement EMA weighting (requires 3-day window addition)
- Add trend reversal detection

**If debug logging reveals name matching failures (Task 1.3):**
- Improve fuzzy matching threshold (0.90 → 0.85)
- Add BDL player_id mapping (more reliable than names)

**Phase 2: Advanced Metrics** (Week 2-3, after Phase 1 root causes resolved)
- Pitcher role tagging (Closer vs Setup vs LOOGY)
- Playing-time context (batting order, vs LHH/RHH)
- Advanced stats integration (xwOBA, K-BB%, FIP, Barrel%)

**Phase 3: UX & Tools** (Week 4)
- Trade calculator with category impact
- Streaming optimizer (1-week pickups)
- Positional scarcity view
- Injury update feeds

---

## 📞 Audit Reports

**Task 1.3 Audit:** `docs/superpowers/audits/2026-06-12-two-start-wiring-audit.md`
- Finding: Feature exists, fully wired
- Root causes: Name matching failure, probable pitchers not announced, cache staleness
- Actions: Add debug logging, reduce cache TTL, add "Missing Probable" badge

**Task 1.1 Diagnostic:** `docs/superpowers/audits/2026-06-12-duran-cold-diagnostic.md`
- Finding: Plan misdiagnosed root cause
- Plan error: Assumed deprecated hardcoded thresholds, actual code uses cohort-relative percentiles
- Plan error: Inconsistent EMA formula (header vs docstring)
- Plan error: 3-day window doesn't exist (pipeline only does 7, 14, 30)
- Actions: Run diagnostic queries to determine root cause (stale data vs algorithm bug)

**Task 1.2 Plan:** `docs/superpowers/plans/2026-06-12-uat-phase1-ci-corrected.md`
- Finding: Largely valid, but breaking change risk
- Breaking change risk: Renaming projection → projection_p50 would break frontend
- Correction: Keep projection field, add percentile fields alongside
- Status: Ready for implementation

---

## 🚨 Key Lessons Learned

1. **Audit before planning** — The original plans assumed features didn't exist. Audit revealed they already did.
2. **Verify code paths** — Plan described deprecated hardcoded thresholds, actual code uses cohort-relative percentiles.
3. **Check schema constraints** — Plan proposed 3-day window, but pipeline only supports 7, 14, 30.
4. **Avoid breaking changes** — Renaming projection field would silently break frontend. Add new fields instead.
5. **Use diagnostic queries** — Don't redesign algorithms until you understand why they misclassify specific players.

---

**Ready to execute Task 1.2 (Confidence Intervals)?** Run the terminal command above.