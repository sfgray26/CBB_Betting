# UAT Phase 1 Execution Summary — Critical Data Fixes

**Date:** 2026-06-12
**Status:** Planning complete, ready for implementation
**Total Effort:** 40 hours (3-5 days)
**Baseball IQ Impact:** 6.5/10 → 8.5/10

---

## 📊 Phase 1 Task Summary

| Task | Impact | Effort | Priority | Status | Agent |
|------|--------|--------|----------|--------|-------|
| **1.1: Fix HOT/COLD Classification** | HIGH | 14h | P0 | Ready | Claude Code |
| **1.2: Add Confidence Intervals** | HIGH | 13h | P0 | Ready | Claude Code |
| **1.3: Two-Start Pitcher Identifier** | HIGH | 13h | P0 | Ready | Claude Code |

---

## 🎯 Phase 1 Goals

**Primary Goal:** Fix critical recommendation logic failures that destroy user trust

**Success Criteria:**
- Jarren Duran signal = HOT (not COLD) ✅
- Confidence intervals displayed on all projections ✅
- Two-start pitchers flagged with start count badge ✅

**Baseball IQ Target:** 8.5/10 (up from 6.5/10)

---

## 📋 Implementation Order

### Day 1-2: Task 1.3 (Two-Start Pitcher Identifier)
**Why first?** Lightest effort, immediate user value, independent of other tasks

**Deliverables:**
- PlayerSchedule table created
- MLB GameDay API integration working
- Start badges displaying in frontend

### Day 3-4: Task 1.2 (Confidence Intervals)
**Why second?** Builds on existing Monte Carlo simulation, no migration needed

**Deliverables:**
- Percentile fields added to player_projections
- CI badges displaying with variance warnings
- High-variance alert banner

### Day 5-7: Task 1.1 (HOT/COLD Classification)
**Why last?** Most complex, requires new EMA computation, depends on stable projections

**Deliverables:**
- EMA-weighted momentum implemented
- Trend reversal detection working
- Trend indicators displaying in frontend

---

## 🚀 Execution Commands

### Terminal 1: Claude Code — Two-Start Pitcher Identifier

```powershell
cd C:\Users\sfgra\repos\Fixed\cbb-edge
gh copilot run --model claude-sonnet-4 --editor-code "Implement Task 1.3: Two-Start Pitcher Identifier

Follow plan: docs/superpowers/plans/2026-06-12-uat-phase1-two-start-pitcher.md

Steps:
1. Create PlayerSchedule model in backend/models.py
2. Create migration: alembic revision --autogenerate -m 'add player_schedule table'
3. Implement backend/services/schedule_fetcher.py with MLB GameDay API query
4. Add _compute_player_schedule() cron job in backend/services/daily_ingestion.py
5. Update backend/contracts.py WaiverPlayerOut schema with schedule fields
6. Update backend/routers/fantasy.py waiver_recommendations endpoint
7. Add start badge display in frontend/components/waiver/player-row.tsx
8. Test: run waiver wire endpoint, verify 2-start pitchers flagged
9. Commit: feat: add two-start pitcher identifier with MLB schedule integration

Note: Use existing MLB GameDay client in backend/fantasy_baseball/mlb_gameday_client.py" --repo ./
```

### Terminal 2: Claude Code — Confidence Intervals

```powershell
cd C:\Users\sfgra\repos\Fixed\cbb-edge
gh copilot run --model claude-sonnet-4 --editor-code "Implement Task 1.2: Confidence Intervals on Projections

Follow plan: docs/superpowers/plans/2026-06-12-uat-phase1-confidence-intervals.md

Steps:
1. Add percentile fields to PlayerProjection model in backend/models.py
2. Create migration: alembic revision --autogenerate -m 'add projection percentiles'
3. Update backend/services/projection_engine.py to compute percentiles from simulation results
4. Update backend/contracts.py WaiverPlayerOut schema with CI fields
5. Update frontend/components/waiver/player-row.tsx to display CI badges
6. Add high-variance warning banner in frontend/components/waiver/waiver-wire.tsx
7. Test: verify percentiles returned for all projections
8. Test: verify CI > 0.3 shows warning badge
9. Commit: feat: add confidence intervals and variance warnings to projections

Note: Use existing simulation_results table — Monte Carlo already runs daily at 6 AM ET" --repo ./
```

### Terminal 3: Claude Code — HOT/COLD Classification Fix

```powershell
cd C:\Users\sfgra\repos\Fixed\cbb-edge
gh copilot run --model claude-sonnet-4 --editor-code "Implement Task 1.1: Fix HOT/COLD Classification with EMA

Follow plan: docs/superpowers/plans/2026-06-12-uat-phase1-momentum-fix.md

Steps:
1. Add trend_direction, trend_strength, ema_14d fields to PlayerMomentum in backend/models.py
2. Create migration: alembic revision --autogenerate -m 'add momentum trend fields'
3. Implement compute_ema_weighted_momentum() in backend/services/momentum_engine.py
4. Update _compute_player_momentum() in backend/services/daily_ingestion.py to use EMA
5. Update frontend/lib/types.ts to add trend indicators
6. Add trend badges (↑ → ↓) in frontend/components/waiver/player-row.tsx
7. Add trend reversal tooltip with explanation
8. Test: verify Jarren Duran signal = HOT (not COLD) when surge in last 3 days
9. Test: verify trend reversal warning displays
10. Commit: feat: fix HOT/COLD classification with EMA and trend reversal detection

Note: This replaces the existing 14d vs 30d momentum algorithm. Add deprecation warning for old logic." --repo ./
```

---

## ✅ Verification Checklist

### After Task 1.3 (Two-Start Pitcher)
- [ ] PlayerSchedule table exists
- [ ] MLB GameDay API queried successfully
- [ ] Start badges display: "🗓️ 2 Starts" for 2-start pitchers
- [ ] Adjusted projection shows: "2.4 pts (1.6 x 2 starts)"
- [ ] No badge for position players
- [ ] Migration applied successfully

### After Task 1.2 (Confidence Intervals)
- [ ] Percentile fields exist in player_projections (p05, p25, p50, p75, p95)
- [ ] CI badges display: "1.3 pts (0.8-1.9 pts)"
- [ ] High variance badge: "⚠ High Variance" for CI > 0.3
- [ ] High-variance alert banner displays when >30% recommendations have high CI
- [ ] Tooltips explain CI meaning
- [ ] Migration applied successfully

### After Task 1.1 (HOT/COLD Classification)
- [ ] Trend fields exist in player_momentum (trend_direction, trend_strength, ema_14d)
- [ ] EMA computation implemented: 0.5 * 3d + 0.5 * 11d
- [ ] Trend reversal detection: recent_trend > 0.3 upgrades to HOT
- [ ] Trend indicators display: ↑ (green), → (gray), ↓ (red)
- [ ] Jarren Duran test case: signal = HOT (not COLD)
- [ ] Migration applied successfully

---

## 📈 Expected Outcomes

### Data Quality Improvements
- **Trend reversal accuracy:** >85% of players with recent_trend > 0.3 show HOT in next 7 days
- **False positive rate:** <10% of upgrades to HOT are not sustained
- **Two-start coverage:** >95% of pitchers have start counts for next 7 days
- **CI width distribution:** 60% have CI < 0.2, 30% have CI 0.2-0.3, 10% have CI > 0.3

### User Experience Improvements
- **Trust:** Managers understand WHY a player is recommended
- **Decision-making:** Variance flags prevent overconfident picks
- **Weekly strategy:** Two-start badges identify high-volume pickups

### Baseball IQ Improvements
- **Current:** 6.5/10 (good discovery tool, weak decision engine)
- **After Phase 1:** 8.5/10 (strong discovery + data-driven decisions)

---

## 🔄 Post-Phase 1: Phase 2 Planning

After completing Phase 1, move to:

**Phase 2: Advanced Metrics** (Week 2-3)
- Pitcher role tagging (Closer vs Setup vs LOOGY)
- Playing-time context (batting order, vs LHH/RHH)
- Advanced stats integration (xwOBA, K-BB%, FIP, Barrel%)

**Phase 3: UX & Tools** (Week 4)
- Trade calculator with category impact
- Streaming optimizer (1-week pickups)
- Positional scarcity view
- Injury update feeds

---

## 📞 Support

For questions during implementation:
- Refer to individual task plan documents in `docs/superpowers/plans/`
- Check UAT report for detailed context: UAT Analysis (2026-06-12)
- Use AGENTS.md for agent swimlanes

---

**Ready to execute Phase 1?** Run the 3 terminal commands above in parallel.