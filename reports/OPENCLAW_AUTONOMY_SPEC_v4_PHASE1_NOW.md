# OpenClaw Autonomy — Phase 1: IMPLEMENT NOW (Not Post-Apr 7)

> **Status:** CORRECTION — Phase 1 can start IMMEDIATELY  
> **Owner:** Kimi CLI / Claude Code  
> **Timeline:** Start NOW (March 24), not April 7  
> **Rationale:** Guardian freeze only blocks CBB model modifications, not monitoring infrastructure

---

## The Guardian Freeze — What It Actually Blocks

**FROZEN (Cannot Touch Until Apr 7):**
- `backend/betting_model.py` — Risk math, Kelly sizing
- `backend/services/analysis.py` — CBB nightly analysis pipeline
- Any changes to CBB model logic

**NOT FROZEN (Can Build NOW):**
- ✅ NEW monitoring infrastructure (reads from DB, doesn't modify models)
- ✅ Pattern detection (analyzes outcomes, generates reports)
- ✅ Proposal generation (writes to markdown/DB, doesn't change Python)
- ✅ A/B test framework (measures, doesn't implement)
- ✅ Discord alerting (notifications only)

**The Only Thing That Waits:**
- ❌ Self-Improvement Agent auto-modifying Python code (post-Apr 7)

---

## Phase 1: Build NOW (March 24 - April 7)

### Week 1 (Mar 24-31): Foundation

**Performance Monitor Agent**
```python
# backend/services/openclaw/performance_monitor.py
# NEW FILE — doesn't touch frozen code

class PerformanceMonitor:
    """
    Read-only performance tracking.
    Queries bet_log, predictions, closing_lines tables.
    Generates alerts. Does NOT modify models.
    """
    
    def check_clv_decay(self) -> DecayReport:
        # Reads from DB only
        # Alerts if degradation detected
        # Does NOT modify betting_model.py
        pass
```

**Database Schema (NEW tables)**
```sql
-- New tables — doesn't modify existing CBB schema
CREATE TABLE openclaw_performance_metrics (...)
CREATE TABLE openclaw_vulnerabilities (...)
CREATE TABLE openclaw_proposals (...)
```

**Pattern Detector Agent**
```python
# backend/services/openclaw/pattern_detector.py
# NEW FILE — read-only analysis

class PatternDetector:
    """
    Analyzes historical outcomes.
    Identifies CBB-specific patterns (conference, seed, HCA).
    Generates vulnerability reports.
    """
```

**Discord Integration**
- Alerts for alpha decay
- Pattern detection notifications
- Daily health summaries

### Week 2 (Apr 1-7): Intelligence Layer

**Learning Engine**
```python
# backend/services/openclaw/learning_engine.py
# Generates improvement proposals
# Writes to DB and ROADMAP.md (not Python files)

class LearningEngine:
    def generate_proposals(self) -> List[Proposal]:
        # Proposes changes
        # Does NOT implement them
        # Human (or post-Apr 7 agent) reviews
```

**Roadmap Maintainer**
```python
# Auto-updates ROADMAP.md with findings
# Prioritizes proposals
# No code changes
```

**A/B Test Framework**
```python
# Designs experiments
# Tracks control vs treatment
# Measures outcomes
# Does NOT auto-implement winners
```

---

## What Works Immediately (Mar 24)

| Feature | Status | Value |
|---------|--------|-------|
| **CLV decay alerts** | ✅ Operational in 2 days | Know within 4 hours if edge degrades |
| **Pattern detection** | ✅ Operational in 3 days | Find CBB vulnerabilities before tournament |
| **Daily health reports** | ✅ Operational in 2 days | Morning Discord brief with model health |
| **Escalation queue** | ✅ Already built | High-stakes games flagged for review |
| **Proposal generation** | ✅ Operational in 5 days | Auto-ranked improvement list |

**All of this is READ-ONLY monitoring.** No Guardian violation.

---

## What Waits Until Apr 7 (Post-Guardian)

| Feature | Wait Reason |
|---------|-------------|
| **Auto-recalibration** | Modifies `betting_model.py` parameters |
| **Auto-weight adjustments** | Changes `WEIGHT_KENPOM` etc. |
| **Self-implemented code changes** | Any automatic Python file modification |

**The Gate:** Self-Improvement Agent has a flag:
```python
if datetime.now() > GUARDIAN_LIFT_DATE:
    # Enable auto-implementation
else:
    # Proposals only — human review required
```

---

## Immediate Implementation Plan

### Claude Code Tasks (This Week)

**Day 1-2: Performance Monitor**
```bash
# 1. Create database migration
python scripts/migrate_openclaw_v4.py

# 2. Build PerformanceMonitor
backend/services/openclaw/performance_monitor.py

# 3. Add scheduler job (every 2 hours)
main.py: scheduler.add_job(...)

# 4. Test
pytest tests/test_openclaw_performance.py -v
```

**Day 3-4: Pattern Detector**
```bash
# 1. Build PatternDetector
backend/services/openclaw/pattern_detector.py

# 2. Add CBB-specific patterns (conference, seed, HCA)
# 3. Daily 4 AM job
# 4. Test
pytest tests/test_openclaw_patterns.py -v
```

**Day 5-7: Learning Engine + Roadmap**
```bash
# 1. Build LearningEngine
backend/services/openclaw/learning_engine.py

# 2. Build RoadmapMaintainer
backend/services/openclaw/roadmap_maintainer.py

# 3. Weekly Monday 6 AM job
# 4. Discord integration
# 5. Test
pytest tests/test_openclaw_learning.py -v
```

### Kimi CLI Tasks (Parallel)

- Review Claude's implementation
- Design MLB-specific patterns (for when CBB ends)
- Prepare Phase 2 specification

---

## Value During Final CBB Weeks

Even with only 2 weeks of CBB left, this provides value:

| Scenario | Without OpenClaw | With OpenClaw |
|----------|------------------|---------------|
| Model degrades Mar 28 | Notice Apr 5 (too late) | Alert within 4 hours |
| Conference bias found | Gut feel | Statistical proof |
| Improvement needed | Manual brainstorm | Auto-prioritized list |
| Tournament edge | Static model | Continuously monitored |

**Plus:** Foundation is ready for MLB transition.

---

## MLB Transition Benefit

Building Phase 1 NOW means:
- ✅ Performance monitoring ready for MLB (Apr 7)
- ✅ Pattern detection learns CBB → can learn MLB patterns
- ✅ Proposal system generates MLB-specific improvements
- ✅ Only the "auto-implement" flag stays off until post-Apr 7

**No wasted time.** Continuous monitoring from Day 1 of MLB.

---

## Corrected Timeline

| Phase | Original | Corrected | Status |
|-------|----------|-----------|--------|
| **Phase 1: Foundation** | Post-Apr 7 | **START NOW** | Monitoring + detection |
| **Phase 2: Intelligence** | Post-Apr 7 | **Apr 1-7** | Learning + roadmap |
| **Phase 3: Autonomy Setup** | Apr 1-7 | **Apr 1-7** | Framework (disabled) |
| **Phase 4: Activation** | Post-Apr 7 | **Apr 8** | Enable auto-implement |

**Key Change:** Phase 1 starts March 24, not April 7.

---

## Action Items

**Immediate (Today):**
1. **Kimi CLI:** Deliver this correction to HANDOFF.md
2. **Claude Code:** Begin Phase 1 implementation (Performance Monitor)
3. **Gemini CLI:** Verify Railway resources for new tables/jobs

**This Week:**
- Performance Monitor operational
- Pattern Detector operational  
- First CLV decay alerts firing
- First pattern detections reported

**Result:** By April 7, full monitoring infrastructure is LIVE and has been running for 2 weeks.

---

**Document Version:** PHASE1-NOW-v1  
**Last Updated:** March 24, 2026  
**Status:** CORRECTION — Phase 1 implementation starts immediately
