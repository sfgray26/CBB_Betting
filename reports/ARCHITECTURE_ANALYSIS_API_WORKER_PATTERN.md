# Architecture Analysis: API-Worker Pattern with Decision Contracts

**Date:** March 31, 2026  
**Subject:** Assessment of proposed architectural pivot for Fantasy Baseball application  
**Analyst:** Claude Code (Master Architect)  
**Status:** Strategic evaluation — NOT a commitment to implement

---

## Executive Summary

Your proposed architecture represents a **fundamental paradigm shift** from a synchronous, monolithic FastAPI application to an **event-driven, contract-based system** with clear separation between API (request/response) and Worker (async processing) concerns.

### Verdict
**The proposed architecture is sound and would likely solve your persistent UI issues**, but the migration cost is significant (2-3 months of focused work). The pattern aligns with modern best practices for data-intensive applications but may be overkill for your current scale.

---

## Part 1: Understanding Your Current Pain Points

Based on the HANDOFF.md audit, your UI issues stem from **architectural coupling**, not code quality:

| Issue Category | Root Cause | Current Symptom |
|----------------|------------|-----------------|
| **Data freshness** | Synchronous API calls to Yahoo + Statcast on request | 500 timeouts, stale projections |
| **Type mismatches** | Frontend assumptions vs backend reality | Runtime crashes, null guards everywhere |
| **Blocking operations** | Heavy computation (MCMC, optimization) in request path | Request timeouts, SSE drops |
| **State management** | UI polls for changes rather than being pushed | Race conditions, stale displays |
| **Error propagation** | Raw Pydantic errors leak to frontend | User confusion, panic debugging |

### The Fundamental Problem
Your current architecture is **synchronous-first** with async bolted on. Heavy operations (Yahoo roster fetch, Statcast ingestion, lineup optimization) happen **during the HTTP request**, leading to:
- 30s+ response times
- Connection drops (Railway/nginx timeouts)
- Partial failures that corrupt UI state
- Cascading retries that hammer external APIs

---

## Part 2: Proposed Architecture Analysis

### Your Design (As Presented)

```
┌─────────────────────────────────────────┐
│ API (apps/api)                          │
│ - POST /lineup/today                    │
│ - POST /waiver/recommendations          │
│ - GET /players/:id/valuation            │
│ - GET /decisions/pending                │
└──────────────┬──────────────────────────┘
               │ Decision Contracts
               ▼
┌─────────────────────────────────────────┐
│ Worker (apps/worker)                    │
│ - Data sync (every 15 min)              │
│ - Valuation generation                  │
│ - Decision processing                   │
│ - Alert triggers                        │
└─────────────────────────────────────────┘
```

### Decision Contracts (Your Design)

```
LineupOptimizationRequest → Immutable input contract
PlayerValuationReport     → Probabilistic output with uncertainty
ExecutionDecision         → Recommendation with alternatives + safety controls
```

---

## Part 3: Benefits of the Proposed Approach

### 3.1 Immediate UI Improvements

| Current | Proposed | Impact |
|---------|----------|--------|
| 8s+ roster load times | <200ms (cached) | Snappy UI |
| Real-time computation | Pre-computed valuations | Instant lineup suggestions |
| Polling for updates | WebSocket/SSE push | Live updates without refresh |
| Blocking optimization requests | Async job queue | Submit → Get job ID → Poll status |
| Raw error dumps | Structured error contracts | Graceful degradation |

### 3.2 Architectural Benefits

#### A. **True Separation of Concerns**
```
Current:  FastAPI route → Yahoo API → Statcast → Optimize → Response
Proposed: FastAPI route → Cache lookup → Response (Worker updates cache separately)
```

#### B. **Immutable Decision Contracts**
Your `ExecutionDecision` pattern enables:
- **Audit trails**: Every recommendation stored with context
- **A/B testing**: Compare decision strategies over time
- **Rollback**: Revert to previous day's valuations if new ones are buggy
- **Regulatory compliance**: Immutable record of system recommendations

#### C. **Worker Benefits**
The separate worker process enables:
- **Circuit breaker isolation**: Worker retries don't affect API availability
- **Rate limiting**: Respect Yahoo's 10,000 req/day limit without impacting users
- **Batching**: Collect multiple valuation requests, process once
- **Backpressure handling**: Queue fills up → shed low-priority work

### 3.3 Specific Benefits for Your Fantasy Baseball Use Cases

| Feature | Current Pattern | Proposed Pattern |
|---------|----------------|------------------|
| **Daily Lineup** | Request-time optimization (slow) | Pre-computed at 6am + on-demand for changes |
| **Waiver Wire** | Full scan on page load | Incremental updates as valuations change |
| **Player Valuation** | Real-time calculation | Background refresh + stale-while-revalidate |
| **Matchup Preview** | MCMC on request (5-10s) | Cached simulation + delta updates |
| **Decision Tracking** | File-based (decisions.jsonl) | Proper database with queryable history |

---

## Part 4: Risks and Trade-offs

### 4.1 Infrastructure Complexity

| Aspect | Current | Proposed |
|--------|---------|----------|
| Processes | 1 (FastAPI) | 2+ (API + Worker + Message Queue) |
| Database | PostgreSQL | PostgreSQL + (Redis/RMQ for queue) |
| Deployment | Railway single container | Railway multi-service or stay single with threading |
| Monitoring | One service | Distributed tracing needed |

**Risk**: Your current Railway deployment is optimized for single-container apps. Adding a worker requires either:
1. **Same container**: Use APScheduler (what you have now) — limited scaling
2. **Separate service**: Additional $5-15/month + networking complexity
3. **Queue service**: Redis/CloudAMQ adds another dependency

### 4.2 Data Consistency Challenges

```
Scenario: User requests lineup at 6:01 AM
- Worker is mid-update (6:00 AM batch job)
- Cache has partial new data, partial old data
- Lineup recommendation uses inconsistent state
```

**Mitigation needed**: Versioned snapshots, atomic updates, or read-replicas

### 4.3 Migration Cost Breakdown

| Component | Effort | Notes |
|-----------|--------|-------|
| Database schema migration | 1-2 weeks | Add decision tables, valuation cache, job queue |
| Worker framework | 1 week | Celery/RQ setup or custom async worker |
| API refactoring | 2-3 weeks | Convert sync endpoints to async job submission |
| Frontend changes | 2 weeks | Job status polling, optimistic UI, error boundaries |
| Testing | 1-2 weeks | Integration tests for async flows |
| **Total** | **7-10 weeks** | Full-time focused effort |

### 4.4 Opportunity Cost

While migrating:
- No new features
- Dual maintenance of old + new systems
- Risk of introducing bugs in working (but slow) code

---

## Part 5: Hybrid Alternative (Recommended)

Given your context, I recommend a **gradual migration** rather than a rewrite:

### Phase 1: Decision Contracts (2 weeks)
Implement the contract pattern within your existing FastAPI structure:

```python
# New: backend/contracts.py
class LineupOptimizationRequest(BaseModel):
    league_config: LeagueConfig
    player_pool: List[PlayerEligibility]
    risk_tolerance: RiskTolerance  # conservative | balanced | aggressive
    
class ExecutionDecision(BaseModel):
    decision_id: UUID
    timestamp: datetime
    recommendation: str
    reasoning: List[str]
    alternatives: List[Alternative]
    safety_controls: SafetyControls
    confidence_interval: Tuple[float, float]  # uncertainty quantification
```

### Phase 2: Async Job Queue (2 weeks)
Add Celery/RQ for heavy operations only:
- Lineup optimization → background job
- Yahoo roster sync → background job
- Statcast ingestion → background job

API returns immediately: `{"job_id": "...", "status": "queued"}`

### Phase 3: Valuation Cache (1 week)
Pre-compute player valuations hourly:
```python
# Worker task (every 15 min)
@celery.task
def refresh_player_valuations():
    for player in active_players:
        valuation = compute_valuation(player)
        cache.set(f"valuation:{player.id}", valuation, ttl=3600)
```

### Phase 4: UI Improvements (2 weeks)
- Job status polling
- Optimistic updates
- Better error boundaries

**Total: 7 weeks vs 10 weeks, with working system throughout**

---

## Part 6: Decision Framework

### Choose the Full Rewrite IF:
- [ ] You're experiencing daily production incidents
- [ ] You have 2-3 months of dedicated development time
- [ ] You plan to scale to multiple users/leagues
- [ ] The current system is truly unworkable (not just slow)

### Choose the Hybrid Approach IF:
- [ ] You need incremental improvements
- [ ] You want to keep shipping features
- [ ] You're the primary/only user
- [ ] Current issues are annoyances, not blockers

### Stay With Current Architecture IF:
- [ ] Issues can be fixed with caching layer only
- [ ] No time for major refactoring
- [ ] System works "well enough" for current season

---

## Part 7: My Recommendation

**Proceed with the Hybrid Approach (Phases 1-2)** for these reasons:

1. **Your UAT fixes from HANDOFF.md are working** — the system is functional, just slow
2. **MLB season just started** — major architectural changes mid-season are risky
3. **You have limited dev bandwidth** — you're effectively a solo developer with agent support
4. **The contracts pattern is valuable regardless** — it improves code clarity immediately

### Immediate Actions (This Week)

```python
# 1. Define contracts in backend/contracts.py
# 2. Refactor one endpoint as proof-of-concept
# 3. Add Redis to Railway (free tier)
# 4. Move lineup optimization to background job
```

### Success Criteria
- Lineup page loads in <2s (currently 8s+)
- No more 500 errors on Yahoo API timeouts
- Clear audit trail of all lineup decisions

---

## Appendix A: Decision Contract Schema (Proposed)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum

class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class LeagueConfig(BaseModel):
    """Immutable configuration for a fantasy league."""
    league_key: str
    scoring_type: str  # "headone_cat" | "headone_points"
    categories: List[str]
    roster_positions: List[str]
    max_moves_per_week: Optional[int]
    
class PlayerEligibility(BaseModel):
    """A player's eligibility for a specific lineup slot."""
    player_id: str
    name: str
    positions: List[str]
    eligible_positions: List[str]
    is_starting_pitcher: bool
    opponent: Optional[str]
    game_time: Optional[datetime]

class LineupOptimizationRequest(BaseModel):
    """Request contract for lineup optimization."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    league_config: LeagueConfig
    player_pool: List[PlayerEligibility]
    risk_tolerance: RiskTolerance = RiskTolerance.BALANCED
    constraints: Optional[dict] = None  # user-defined constraints

class UncertaintyRange(BaseModel):
    """Probabilistic range for a projection."""
    point_estimate: float
    lower_95: float
    upper_95: float
    std_dev: float

class PlayerValuation(BaseModel):
    """Probabilistic valuation with uncertainty quantification."""
    player_id: str
    category_z_scores: dict[str, UncertaintyRange]
    overall_value: UncertaintyRange
    confidence: float  # 0-1 based on data quality
    factors: List[str]  # why this valuation

class Alternative(BaseModel):
    """An alternative lineup choice."""
    description: str
    expected_value_delta: float
    risk_profile: str

class SafetyControl(BaseModel):
    """Safety mechanism to prevent poor decisions."""
    check_type: str  # "weather" | "injury" | "pitcher_quality" | "rest"
    status: str  # "pass" | "warning" | "block"
    message: Optional[str]

class ExecutionDecision(BaseModel):
    """Immutable recommendation with full audit trail."""
    decision_id: str
    request_id: str
    timestamp: datetime
    
    # The recommendation
    recommended_lineup: List[PlayerEligibility]
    benched_players: List[PlayerEligibility]
    
    # Reasoning
    primary_reasoning: List[str]
    category_impact: dict[str, float]
    
    # Alternatives and safety
    alternatives: List[Alternative]
    safety_controls: List[SafetyControl]
    
    # Uncertainty
    confidence_score: float
    expected_outcome_range: Tuple[float, float]
    
    # Audit
    model_version: str
    data_timestamp: datetime
```

---

## Appendix B: Implementation Sketch

### Worker Loop (Simplified)

```python
# backend/worker.py
import asyncio
from datetime import datetime, timedelta

class FantasyWorker:
    def __init__(self):
        self.running = False
        self.last_sync = None
        
    async def run(self):
        self.running = True
        while self.running:
            try:
                # Every 15 minutes
                if not self.last_sync or datetime.now() - self.last_sync > timedelta(minutes=15):
                    await self.sync_data()
                    
                # Process queued jobs
                await self.process_jobs()
                
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(60)
    
    async def sync_data(self):
        """Background data synchronization."""
        async with circuit_breaker("yahoo"):
            rosters = await fetch_all_rosters()
            
        async with circuit_breaker("statcast"):
            projections = await fetch_projections()
            
        # Pre-compute valuations
        valuations = compute_valuations(rosters, projections)
        await cache.set("valuations", valuations, ttl=900)
        self.last_sync = datetime.now()
        
    async def process_jobs(self):
        """Process queued optimization requests."""
        job = await queue.pop("lineup_optimization")
        if job:
            result = optimize_lineup(job.request)
            await cache.set(f"job:{job.id}", result)
            await notify_client(job.user_id, job.id)
```

---

## Conclusion

Your proposed architecture is **architecturally sound** and would solve your UI issues. However, the full implementation is a 2-3 month project.

**I recommend the Hybrid Approach**: implement Decision Contracts immediately (improves code quality), add async job queue for heavy operations (solves timeout issues), and gradually migrate toward your vision over the season.

The key insight: **you don't need to rebuild everything to get the benefits**. The contracts pattern alone will improve your system's clarity and testability, even within the current FastAPI structure.

---

*Report prepared by Claude Code (Master Architect)*  
*Questions? Escalate via HANDOFF.md with "ARCHITECTURE-2026-03-31" reference*
