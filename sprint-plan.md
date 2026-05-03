# 2-Week Sprint Plan: Get Solid & Fast
## Lead Developer: Claude Code | Start Date: May 1, 2026

### Week 1: Stabilize Core Fantasy Features (May 1-7)

#### Monday May 4: Verify Core Features Work
- [ ] Test lineup optimization endpoint (call `/api/fantasy/roster/optimize`)
- [ ] Check waiver recommendations quality (compare to ESPN rankings)
- [ ] Verify daily lineups are generating
- [ ] Test MCMC win_prob_gain (check Railway logs)

#### Tuesday May 5: Data Quality Audit
- [ ] Verify rolling window calculations (K-33 fixes)
- [ ] Check player_projections table for NULL values
- [ ] Validate statcast_performances freshness
- [ ] Compare scarcity index to external rankings

#### Wednesday May 6: Deploy W+X+Y Bundle
- [ ] Deploy Session W (opponent roster wiring)
- [ ] Deploy Session X (statcast DB tier)
- [ ] Deploy Session Y (regression tests)
- [ ] Smoke test all fantasy endpoints

#### Thursday May 7: Close Gaps
- [ ] Fix any data quality issues found
- [ ] Add error monitoring (alerts on failures)
- [ ] Performance tuning (lineup optimizer speed)

#### Friday May 8: Documentation + Handoff
- [ ] Document current feature state
- [ ] Create runbook for common issues
- [ ] Update HANDOFF.md with Week 1 progress

### Week 2: MLB Betting + Production Hardening (May 8-14)

#### Monday May 11: MLB Betting Analysis
- [ ] Complete mlb_analysis.py DB-tier integration
- [ ] Add edge calculation verification
- [ ] Test runline projections accuracy

#### Tuesday May 12: Production Monitoring
- [ ] Add alerts for data quality failures
- [ ] Set up uptime monitoring
- [ ] Create incident response runbook

#### Wednesday May 13: Performance
- [ ] Optimize lineup optimizer speed
- [ ] Add caching where needed
- [ ] Load testing

#### Thursday May 14: Final Verification
- [ ] End-to-end test all fantasy features
- [ ] Validate MLB betting pipeline
- [ ] Production readiness check

### Success Criteria
- All fantasy features tested and documented
- Zero data quality gaps (K-33 resolved)
- MLB betting analysis production-ready
- Production monitoring in place
