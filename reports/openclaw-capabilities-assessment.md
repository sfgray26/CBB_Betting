# OpenClaw Capabilities Assessment & Strategic Leverage Plan

**Assessment Date:** 2026-03-16  
**Assessor:** Prompt Engineer / Quant Trading Systems Architect  
**Subject:** OpenClaw (Integrity Execution Unit) — qwen2.5:3b via Ollama  
**Context:** CBB Edge Analyzer V9 Production System

---

## 1. OpenClaw Current State Analysis

### 1.1 Core Capabilities Inventory

| Capability | Current Implementation | Utilization | Strategic Value |
|------------|----------------------|-------------|-----------------|
| **Real-time Web Search (DDGS)** | `analysis.py:_ddgs_and_check_sync()` via DuckDuckGo | HIGH — integrity sweep on all BET-tier games | CRITICAL — unique async research at scale |
| **LLM Sanity Checks** | `scout.py:perform_sanity_check()` with qwen2.5:3b | HIGH — verdict generation (CONFIRMED/CAUTION/VOLATILE/ABORT) | HIGH — cheap local inference ($0 cost) |
| **Async Batch Processing** | `analysis.py:_integrity_sweep()` with asyncio.Semaphore(8) | HIGH — concurrent game processing | HIGH — 8x throughput vs sequential |
| **Circuit Breaker Pattern** | `coordinator.py:CircuitBreaker` class | MEDIUM — failover to Kimi on failure | MEDIUM — resilience without human intervention |
| **Cost Tracking** | `coordinator.py` logs to `token-usage.jsonl` | LOW — basic latency/tokens logging | MEDIUM — operational visibility |
| **Tiered Escalation** | Rules: >=1.5u, Elite 8+, VOLATILE → Kimi | NOT YET WIRED — architecture ready | VERY HIGH — highest value opportunity |

### 1.2 Unique Value Proposition

**What OpenClaw does that NO OTHER AGENT can:**

1. **Sub-$0.01 cost per integrity check** (local qwen2.5:3b vs Kimi's API costs)
2. **Sub-2s latency per check** (local inference + DDGS vs Kimi's 10-30s)
3. **Unlimited throughput** (local hardware limited vs API rate limits)
4. **24/7 availability** (no API quota exhaustion)
5. **Parallel web search** (8 concurrent DDGS streams — Kimi cannot match this economically)

**Cost Comparison (per 100 games):**
| Agent | Cost per Check | 100 Games | Notes |
|-------|---------------|-----------|-------|
| OpenClaw | $0.0001 | $0.01 | Local qwen + DDGS |
| Kimi (full) | $0.05-0.10 | $5-10 | API + context window |
| Claude | N/A (no search) | — | No native web search |
| Gemini | N/A (no search) | — | No native web search |

**Conclusion:** OpenClaw is the ONLY agent capable of cost-effective, high-throughput real-time intelligence gathering.

---

## 2. Current Gaps & Underutilization

### 2.1 Gap Analysis

| Gap | Impact | Root Cause |
|-----|--------|------------|
| **Tiered escalation NOT wired** | HIGH — Elite 8+ games not getting Kimi second opinion | Implementation pending tournament |
| **No historical integrity DB** | MEDIUM — cannot trend VOLATILE rate over time | No persistence layer designed |
| **No pre-game line movement monitoring** | MEDIUM — sharp money detection is reactive | No polling loop implemented |
| **No cross-game correlation detection** | MEDIUM — cannot detect slate-wide risks | Analysis is per-game only |
| **Limited DDGS query sophistication** | LOW — basic `team injury lineup` query | No LLM query optimization |
| **No structured news extraction** | MEDIUM — parsing raw text, not entities | No NER/parsing pipeline |

### 2.2 Operational Blind Spots

1. **No visibility into DDGS result quality** — Are we getting relevant news? No metrics.
2. **No A/B testing of prompts** — `perform_sanity_check()` prompt is static.
3. **No feedback loop** — Correct/incorrect verdicts not tracked against outcomes.
4. **No automated alerting** — >20% VOLATILE should page someone (it only logs).

---

## 3. Strategic Leverage Opportunities

### 3.1 Immediate Wins (Week 1 — Pre-Tournament)

#### Opportunity 1: Pre-Tournament Intelligence Dashboard
**What:** Daily automated report on all 68 tournament teams
**How:** 
- Batch run OpenClaw on all tournament teams March 16-17
- Query: `{team} injury suspension news March 2026`
- Aggregate: injury risk heatmap by region
**Value:** Enter tournament with baseline health status for all teams
**Cost:** ~$0.50 (500 teams × $0.001)
**Implementation:** 2-hour script in `scripts/pre_tournament_intel.py`

#### Opportunity 2: Verdict Quality Feedback Loop
**What:** Track integrity verdict accuracy against actual game outcomes
**How:**
- Add `integrity_verdict` to `BetLog` table
- After games settle, compare VOLATILE/CAUTION games vs control
- Weekly report: "VOLATILE games had 15% more upsets than CONFIRMED"
**Value:** Data-driven calibration of scalar values (currently 0.5/0.75 are estimates)
**Implementation:** Migration + analytics query

#### Opportunity 3: Automated Query Optimization
**What:** Use LLM to generate better DDGS queries
**How:**
- Current: `"Duke {opponent} injury suspension lineup {date}"`
- Optimized: `"Duke basketball injury report March 2026" OR "Duke Blue Devils news" -highlights`
- A/B test: original vs optimized query → measure result relevance
**Value:** Higher signal-to-noise in web results
**Implementation:** 50-line addition to `scout.py`

### 3.2 Tournament-Phase Enhancements (March 18-April 7)

#### Opportunity 4: Wire Tiered Escalation to Kimi
**What:** Elite 8+ games get automatic Kimi second opinion
**Current State:** Rules defined in `coordinator.py`, NOT connected to pipeline
**Implementation:**
```python
# In analysis.py, after _integrity_sweep():
if context.recommended_units >= 1.5 or context.tournament_round >= 4:
    verdict = await coordinator.route_to_kimi(task, context, prompt)
else:
    verdict = await coordinator.route_to_local(task, context, prompt)
```
**Value:** High-stakes games get Deep Intelligence without manual trigger
**Cost:** Only ~12 games (Elite 8+), minimal Kimi API spend

#### Opportunity 5: Real-Time Line Movement Alerting
**What:** Monitor sharp line moves within 2h of tip
**How:**
- Poll Odds API every 15 min for games in T-2h window
- If spread moves >2 points against our pick → OpenClaw re-check
- Auto-ABORT if new red flags found
**Value:** Exit bad bets before tipoff; ~5-10% ROI improvement
**Implementation:** `scripts/line_movement_monitor.py` (APScheduler job)

#### Opportunity 6: Cross-Game Correlation Detection
**What:** Detect slate-wide risks (e.g., same conference, same upset pattern)
**How:**
- After integrity sweep, analyze all verdicts
- If 3+ games in same conference are VOLATILE → flag `SYSTEMIC_RISK`
- Alert: "ACC games showing elevated uncertainty — consider slate reduction"
**Value:** Portfolio-level risk management (currently per-game only)

#### Opportunity 7: Live In-Game Monitoring (Sweet 16+)
**What:** Monitor for mid-game injury news
**How:**
- During games, poll Twitter/DDGS every 5 min for injury keywords
- If star player injury detected → alert for live betting opportunity
**Value:** Second-half line moves based on real-time info
**Note:** Requires live betting integration (future phase)

### 3.3 Long-Term Capabilities (Post-Season)

#### Opportunity 8: Off-Season Integrity Model Training
**What:** Train custom classifier on integrity check corpus
**How:**
- Export all (query, context, verdict) tuples from season
- Fine-tune small model (DistilBERT) on "BET vs PASS" labels
- Replace qwen2.5:3b with 10x faster classifier for 90% of checks
- Reserve qwen for edge cases
**Value:** Sub-100ms checks, even lower cost
**Timeline:** May-July 2026

#### Opportunity 9: Multi-Source Intelligence Fusion
**What:** Combine DDGS + Twitter/X + Reddit + ESPN API
**How:**
- DDGS: Official news
- Twitter: Beat reporter accounts
- Reddit: r/CollegeBasketball injury threads
- ESPN API: Official injury reports
- Fusion layer: Weighted credibility scoring
**Value:** Higher recall on breaking news
**Complexity:** HIGH — requires new scrapers

---

## 4. Implementation Priorities

### Phase 1A: Regular Season Enhancement (Now — March 15)
| Priority | Task | Owner | Effort | Impact |
|----------|------|-------|--------|--------|
| P0 | O-6 Integrity Spot-Check | OpenClaw | 1h | Verify production health |
| P1 | A-29 Include CONSIDER-tier | OpenClaw | 2h | 3x training data for calibration |
| P2 | Performance baseline | OpenClaw | 2h | Establish latency/success metrics |
| P3 | DDGS query optimization | OpenClaw | 4h | A/B test query variants |

### Phase 1B: Pre-Tournament (March 16-17)
| Priority | Task | Owner | Effort | Impact |
|----------|------|-------|--------|--------|
| P0 | Wire tiered escalation (Opp #4) | Claude | 4h | CRITICAL for Elite 8+ |
| P1 | Pre-tournament intel batch (Opp #1) | OpenClaw | 2h | Risk baseline |
| P2 | Verdict feedback loop schema (Opp #2) | Gemini | 3h | Long-term calibration |

### Phase 2: Tournament Phase (March 18-April 7)
| Priority | Task | Trigger | Effort | Impact |
|----------|------|---------|--------|--------|
| P0 | Line movement monitor (Opp #5) | Daily at T-2h | 4h | Real-time risk mgmt |
| P1 | Cross-game correlation (Opp #6) | After each sweep | 3h | Portfolio awareness |
| P2 | Query optimization A/B (Opp #3) | Background | 2h | Signal quality |

### Phase 3: Post-Season (April+)
| Priority | Task | Timeline | Effort |
|----------|------|----------|--------|
| P1 | Custom classifier (Opp #8) | May-July | 2 weeks |
| P2 | Multi-source fusion (Opp #9) | Summer | 3 weeks |

---

## 5. Resource Requirements

### Compute (OpenClaw/Ollama Host)
| Phase | qwen2.5:3b Load | DDGS Requests/Day | Est. Cost/Month |
|-------|-----------------|-------------------|-----------------|
| Current | ~50 checks/day | ~50 | $0 (local) |
| Tournament | ~300 checks/day | ~300 | $0 (local) |
| With line monitoring | ~500 checks/day | ~1,000 | $0 (local) |
| Multi-source | ~500 checks/day | ~2,000 | $0 (local) |

**Conclusion:** Local Ollama deployment is cost-effective at all scales.

### Kimi API (Tiered Escalation Only)
| Scenario | Games to Kimi | Cost/Check | Total/Tournament |
|----------|---------------|------------|------------------|
| Elite 8+ only | ~12 games | $0.05 | $0.60 |
| Sweet 16+ + high stakes | ~32 games | $0.05 | $1.60 |
| All high-stakes (>=1.5u) | ~50 games | $0.05 | $2.50 |

**Conclusion:** Even aggressive escalation is <$5/tournament.

---

## 6. OpenClaw Role Evolution

### Current Role: "Integrity Execution Unit"
- Reactive: triggered by BET-tier games only
- Scoped: per-game sanity checks
- Output: verdict string

### Proposed Evolution: "Real-Time Intelligence Layer"

```
┌─────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE HIERARCHY                    │
├─────────────────────────────────────────────────────────────┤
│  TIER 1: OpenClaw (Real-Time)                                │
│  ├── ALL games: DDGS + qwen2.5:3b (~$0/game)                │
│  ├── Sweet 16+: Line movement monitoring                    │
│  └── Output: CONFIRMED/CAUTION/VOLATILE/ABORT               │
├─────────────────────────────────────────────────────────────┤
│  TIER 2: Kimi Escalation (Strategic)                         │
│  ├── Elite 8+ games (automatic)                             │
│  ├── >=1.5u recommended size (automatic)                    │
│  └── VOLATILE + high stakes (OpenClaw triggers)             │
├─────────────────────────────────────────────────────────────┤
│  TIER 3: Human Review (Tactical)                             │
│  ├── Kimi RED FLAG scenarios                                │
│  └── >20% VOLATILE slate (systemic risk)                    │
└─────────────────────────────────────────────────────────────┘
```

### New Responsibilities

1. **Pre-Tournament Baseline** (March 16-17)
   - Batch process all 68 teams
   - Generate injury risk heatmap
   - Flag teams requiring monitoring

2. **Tournament Real-Time** (March 18+)
   - Continuous sweep on all BET-tier games
   - Line movement monitoring (T-2h)
   - Cross-game correlation alerting
   - Automatic Kimi escalation

3. **Post-Game Learning**
   - Log outcomes vs verdicts
   - Weekly calibration reports
   - Query effectiveness A/B tests

4. **Off-Season Model Training**
   - Corpus curation
   - Fine-tuning data prep
   - Evaluation pipeline

---

## 7. Recommendations for Claude (Architect)

### Immediate Actions (This Week)

1. **Add OpenClaw to HEARTBEAT.md daily checks**
   ```yaml
   # HEARTBEAT.md addition
   - Check OpenClaw Ollama health: `ollama ps | grep qwen`
   - Verify integrity sweep ran: grep "integrity sweep" logs
   - Check VOLATILE rate: alert if >30% of slate
   ```

2. **Create `scripts/openclaw_baseline.py`**
   - Pre-tournament batch job
   - Run March 16 9 PM ET (after BallDontLie bracket ready)
   - Output: `data/pre_tournament_intel.json`

3. **Wire tiered escalation NOW** (before tournament)
   - Connect `coordinator.py` rules to `analysis.py`
   - Test with mock context
   - Document escalation triggers in AGENTS.md

### Technical Debt to Address

| Issue | Priority | Fix |
|-------|----------|-----|
| No integrity verdict persistence | HIGH | Add `integrity_verdict` to `BetLog` table |
| No DDGS result caching | MEDIUM | Cache results for 1h (same game re-checks) |
| Static scout prompt | MEDIUM | A/B test framework for prompt variants |
| No query timeout handling | LOW | Add 5s timeout to DDGS calls |

### Success Metrics to Track

1. **Operational**
   - Integrity sweep latency (target: <30s for 8 concurrent)
   - DDGS success rate (target: >95%)
   - qwen2.5:3b inference time (target: <500ms)

2. **Quality**
   - VOLATILE rate vs upset rate correlation
   - CAUTION games edge retention vs control
   - Kimi escalation appropriateness (human review sample)

3. **Cost**
   - Checks per dollar (target: >10,000/$1)
   - Kimi escalation rate (target: <10% of checks)

---

## 8. Summary: OpenClaw Strategic Value

**What OpenClaw enables that transforms the system:**

| Capability | Before OpenClaw | With OpenClaw |
|------------|-----------------|---------------|
| Real-time intelligence | Manual (human) | Automated, <$0.01/game |
| Scale | 5-10 games/day | Unlimited |
| Latency | Hours (human research) | Seconds |
| Coverage | BET-tier only | All tiers + line movement |
| Escalation | Manual trigger | Automatic tiered routing |
| Cost per tournament | $100-500 (human time) | <$5 |

**Bottom Line:** OpenClaw is not just an "Integrity Officer" — it's a **real-time intelligence infrastructure** that enables the system to operate at tournament scale (67 games) with the same diligence as a single-game manual review.

**The key insight:** We've underutilized OpenClaw by limiting it to BET-tier reactive checks. The strategic opportunity is **proactive intelligence** — continuous monitoring, baseline establishment, and predictive alerting.

---

*Assessment complete. Recommend immediate implementation of Pre-Tournament Baseline (Opp #1) and Tiered Escalation Wiring (Opp #4) before March 18 tournament start.*
