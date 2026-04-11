# OpenClaw Autonomy v4.0 — Quick Reference

> **Owner:** Kimi CLI (Deep Intelligence Unit)  
> **Purpose:** One-page summary for all agents  
> **Documents:** This is a summary — see full specs for implementation details

---

## Documents Delivered

| Document | Purpose | Owner | Status |
|----------|---------|-------|--------|
| `OPENCLAW_AUTONOMY_SPEC_v4.md` | Full 6-agent architecture | Kimi CLI | ✅ Complete |
| `OPENCLAW_AUTONOMY_SPEC_v4_PHASE1_NOW.md` | Corrected timeline (Phase 1 starts NOW) | Kimi CLI | ✅ Complete |
| `OPENCLAW_AUTONOMY_SPEC_v4_MLB_ADDENDUM.md` | MLB betting model requirements | Kimi CLI | ✅ Complete |
| `KIMI_DESIGN_PHASE2_OPENCLAW.md` | Phase 2: Learning Engine + Roadmap Maintainer | Kimi CLI | ✅ Complete |
| `KIMI_RESEARCH_MLB_OPENCLAW_PATTERNS.md` | MLB-specific vulnerability patterns | Kimi CLI | ✅ Complete |
| `KIMI_OPENCLAW_QUICKREF.md` | This file — executive summary | Kimi CLI | ✅ Complete |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPENCLAW AUTONOMOUS SYSTEM v4.0                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  SCHEDULER                                                                  │
│  ├── Every 2h: Performance Monitor (CLV decay, win rate trends)            │
│  ├── Daily 4am: Pattern Detector (CBB/MLB vulnerabilities)                 │
│  ├── Weekly Mon 6am: Roadmap Maintainer (auto-prioritize)                  │
│  └── Daily 3am: Self-Improvement (⚠️ Post-Apr 7 only)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  AGENTS                                                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                │
│  │ PERFORMANCE     │ │ PATTERN         │ │ LEARNING        │                │
│  │ MONITOR         │ │ DETECTOR        │ │ ENGINE          │                │
│  │ (Phase 1)       │ │ (Phase 1)       │ │ (Phase 2)       │                │
│  │                 │ │                 │ │                 │                │
│  │ • CLV tracking  │ │ • CBB: Conf/seed│ │ • Generate      │                │
│  │ • Decay alerts  │ │ • MLB: Pitcher  │ │   proposals     │                │
│  │ • Health reports│ │   fatigue       │ │ • ROI estimate  │                │
│  │                 │ │ • Weather       │ │ • A/B design    │                │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘                │
│           │                   │                   │                         │
│           └───────────────────┴───────────────────┘                         │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                │
│  │ ROADMAP         │ │ SELF-IMPROVE    │ │ NOTIFIER        │                │
│  │ MAINTAINER      │ │ AGENT           │ │ AGENT           │                │
│  │ (Phase 2)       │ │ (Phase 3-4)     │ │ (All Phases)    │                │
│  │                 │ │                 │ │                 │                │
│  │ • Auto-update   │ │ • ⚠️ Post-Apr 7 │ │ • Discord       │                │
│  │   ROADMAP.md    │ │ • Auto-apply    │ │   alerts        │                │
│  │ • Prioritize    │ │ • Rollback      │ │ • Health        │                │
│  │ • Archive stale │ │ • Safety checks │ │   summaries     │                │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase-by-Phase Timeline

| Phase | Dates | What | Guardian Status |
|-------|-------|------|-----------------|
| **Phase 0** | ✅ Mar 24 | Design complete | N/A |
| **Phase 1** | 🚀 Mar 24-31 | Performance Monitor + Pattern Detector (CBB + MLB) | ✅ **CAN BUILD NOW** — read-only |
| **Phase 2** | Mar 31-Apr 7 | Learning Engine + Roadmap Maintainer | ✅ **CAN BUILD NOW** — proposals only |
| **Phase 3** | Apr 1-7 | Self-improvement framework (disabled mode) | ✅ **CAN BUILD NOW** — flag stays off |
| **Phase 4** | 🚫 Apr 8+ | **Self-improvement ACTIVATION** | ⚠️ **WAIT** — modifies frozen files |

---

## What Claude Builds Now (Phase 1)

### Week 1 (Mar 24-31)

**Day 1-2: Performance Monitor**
```python
backend/services/openclaw/
├── __init__.py
├── performance_monitor.py     # NEW
└── tests/
    └── test_performance_monitor.py
```

**Day 3-4: Pattern Detector**
```python
backend/services/openclaw/
└── pattern_detector.py        # NEW (CBB patterns)
```

**Day 5-7: MLB Patterns + Database**
```python
backend/services/openclaw/
├── mlb_patterns.py            # NEW (12 MLB patterns)
└── database.py                # Migration helper

scripts/migrate_openclaw_v4.py # NEW (4 tables)
```

### Database Tables (4 New)

```sql
openclaw_performance_metrics  # Time-series metrics
openclaw_vulnerabilities      # Detected patterns  
openclaw_proposals           # Improvement ideas
openclaw_ab_tests            # A/B experiments
```

---

## MLB Betting Model (Parallel Track)

**Critical Path:** Must be operational by Apr 7

| Component | Status | Owner | Days |
|-----------|--------|-------|------|
| FanGraphs scraper | ❌ Not built | Claude | 2-3 |
| Baseball-Reference scraper | ❌ Not built | Claude | 2-3 |
| Starting pitcher fetcher | ❌ Not built | Claude | 1 |
| Bullpen stats calculator | ❌ Not built | Claude | 1 |
| Runline projection model | ❌ Not built | Claude | 3-4 |
| **Total** | | | **9-11 days** |

**Buffer:** 3-4 days before Apr 7 deadline

---

## Key Decisions

### What CAN Be Built Now (Guardian-Safe)
✅ NEW infrastructure (monitoring tables, jobs)
✅ READ-ONLY analysis (querying bet_log, predictions)
✅ Discord alerting (notifications)
✅ Proposal generation (markdown/DB, not Python files)
✅ A/B test framework (measurement, not implementation)

### What MUST Wait Until Apr 8
🚫 Auto-recalibration of betting_model.py
🚫 Auto-adjustment of WEIGHT_KENPOM, etc.
🚫 Self-modifying Python code

### The Gate
```python
# In Self-Improvement Agent
if datetime.now() > datetime(2026, 4, 7):
    enable_auto_implementation()
else:
    proposals_only_mode()  # Human review required
```

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | CLV decay detection | <4 hours from degradation → alert |
| Phase 1 | Pattern detection accuracy | >80% validated by manual review |
| Phase 2 | Proposal quality | >70% accepted by human review |
| Phase 2 | Roadmap update latency | <1 hour after pattern detection |
| Phase 4 | Autonomous fix success | >95% (rollbacks vs deployments) |
| Phase 4 | Human oversight reduction | 80% fewer manual reviews |

---

## Risk: MLB Model Timeline

**Scenario A (Best):** MLB model ready Apr 1-2
- 5 days buffer
- Time for overlap testing
- Smooth CBB → MLB transition

**Scenario B (Tight):** MLB model ready Apr 5-6
- 1-2 days buffer
- Minimal overlap testing
- Risk of gaps in betting

**Scenario C (Bad):** MLB model not ready Apr 7
- No betting picks for 1-2 weeks
- Lost revenue during early MLB season
- **Mitigation:** Use simplified model (pitcher ERA + park factor only)

---

## Immediate Next Steps

### For Claude (Today)
1. ✅ Waiver wire fixes (in progress)
2. 🚀 **NEW:** Start OpenClaw Phase 1 (Performance Monitor)
3. 📋 Scope MLB data sources (FanGraphs API vs scraper)

### For Kimi (This Week)
1. ✅ Delivered all design specs
2. 🔄 Monitor Claude's implementation
3. 📝 Prepare Phase 2 detailed specs (if needed)

### For Gemini (This Week)
1. Run `migrate_v8_post_draft.py` on Railway (unblocks EPIC-2)

---

## Communication Protocol

**Weekly Sync:** Claude + Kimi + OpenClaw (every Monday 9 AM)
- Review progress
- Unblock issues
- Adjust priorities

**Daily Updates:** In HANDOFF.md
- Claude: Waiver + MLB model progress
- Kimi: Design review notes, research findings

**Emergency:** Discord #openclaw-escalations
- System failures
- Architecture blockers
- Scope changes

---

## Document Map

```
reports/
├── OPENCLAW_AUTONOMY_SPEC_v4.md              # Full architecture
├── OPENCLAW_AUTONOMY_SPEC_v4_PHASE1_NOW.md   # Corrected timeline
├── OPENCLAW_AUTONOMY_SPEC_v4_MLB_ADDENDUM.md # MLB requirements
├── KIMI_DESIGN_PHASE2_OPENCLAW.md            # Phase 2 design
├── KIMI_RESEARCH_MLB_OPENCLAW_PATTERNS.md    # MLB patterns
└── KIMI_OPENCLAW_QUICKREF.md                 # This summary
```

---

**Version:** QUICKREF-v1  
**Status:** All design docs delivered — Ready for implementation  
**Next Action:** Claude begins Phase 1 implementation
