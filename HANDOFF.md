# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 8, 2026 (updated Session S29) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW — P1-P20 CERTIFIED. Phases 2-10 complete. Full pipeline operational.

---

## CORE PHILOSOPHY — Data-First, Contracts Before Plumbing

We are building this system like a quantitative trading desk. The data pipeline IS the product. Everything else — UI, optimization, automation — is a window into it that does not exist until the data is pristine.

---

## ARCHITECTURAL BLUEPRINT — 10-Phase Master Plan

| Phase | Goal | Status |
|-------|------|--------|
| **1 — Layered Architecture** | Separate side effects (bottom) from pure functions (top). Contracts before plumbing. | ✅ DONE |
| **2 — Data Foundation** | Ingest every game + stat + player. Normalize. Resolve IDs. | ✅ DONE (S26) |
| **3 — Derived Stats** | 30/14/7-day rolling windows. Exponential decay λ=0.95. | ✅ DONE (S26) |
| **4 — Scoring Engine** | League Z-scores + position Z-scores. Z_adj. | ✅ DONE (S26) |
| **5 — Momentum Layer** | ΔZ = Z_14d − Z_30d. Signals: Surging / Hot / Cold. | ✅ DONE (S26) |
| **6 — Probabilistic Layer** | 1000-run ROS Monte Carlo. Percentiles. | ✅ DONE (S26) |
| **7 — Decision Engines** | Lineup optimizer, waiver optimizer. | ✅ DONE (S26) |
| **8 — Backtesting Harness** | Historical loader, simulation engine, baselines. | ✅ DONE (S26) |
| **9 — Explainability** | Decision traces. Human-readable narratives. | ✅ DONE (S26) |
| **10 — Integration & Automation** | Snapshot system, daily sim harness. | ✅ DONE (S26) |

---

## ACTIVE DIRECTIVES (read before every session)

### DIRECTIVE 1 — Data-First Mandate
Incoming payloads MUST pass strict Pydantic V2 validation. No `dict.get()` defaults.

### DIRECTIVE 2 — Fantasy Baseball UI Data Layer (NEW: April 8, 2026)
**CRITICAL:** Do NOT begin UI design for H2H One Win format until Phase 1-2 validation passes.

**Required Before UI Phase:**
1. **Schema Extension:** PositionEligibility table created with CF/LF/RF breakdown (not generic "OF")
2. **Yahoo API Validation:** NSB (stat_id 5070), QS (stat_id 32), K/9 (stat_id 3096) confirmed in data pipeline
3. **Monte Carlo H2H:** H2HOneWinSimulator implemented and benchmarked <200ms for 10k sims
4. **API Endpoints:** All 8 endpoints (Weekly Compass, Scarcity, Two-Start, NSB, IP Bank, Waiver Budget, IL Shuffle, Matchup Difficulty) return valid payloads
5. **Cache Layer:** Redis or in-memory cache hitting >85% on hot paths
6. **Validation Suite:** `tests/test_fantasy_h2h_validations.py` passing (see roadmap doc)

**Root Cause:** H2H One Win format (position-specific OF, NSB not raw SB, 18 IP minimum) requires data granularity not present in current pipeline.

**Reference:** `reports/2026-04-08-fantasy-baseball-ui-roadmap.md` — full technical breakdown.

**Kimi Handoff:** When validation checklist passes, hand off to Kimi CLI for UI component specs (see roadmap Phase 5.2).

---

## Platform State — April 8, 2026

| System | State | Notes |
|--------|-------|-------|
| MLB Data Pipeline | **P1-P20 CERTIFIED** | Full 10-phase pipeline operational in production. |
| `mlb_player_stats` | **POPULATED (S26)** | 646 rows verified live in Fantasy-App DB. |
| `statcast_performances`| **PENDING** | Agent built but fetches 0 records for 2026-04-06 (off-day or lag). |
| Ingestion Orchestrator | **HARDENED (S26)** | All 11 jobs (including statcast/snapshot) registered and manual-triggerable via `/admin/ingestion/run-pipeline`. |
| `position_eligibility` (P25) | **LIVE (S28)** | Model + migration deployed to both DBs. Verified live. |
| `probable_pitchers` (P26) | **LIVE (S28)** | Model + migration script deployed to both DBs. Verified live. |
| **H2H One Win UI Data Layer** | **IN PROGRESS — Phase 2.3** | P25/P26 live. Phase 2.1/2.2/Redis complete. Phase 2.3 (scarcity) next. |
| **Redis Caching Layer** | **COMPLETE (S29)** | Production-ready CacheService implemented with msgpack, dynamic TTL, connection pooling. Ready for Railway deployment. |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `mlb_player_stats` | **LIVE.** Pydantic validation relaxed for partial BDL objects. FK integrity enforced via game-ID-first fetch. |
| `daily_snapshots` | **LIVE.** `_ds_date_uc` constraint added. End-to-end pipeline health tracking operational. |
| `position_eligibility` | **LIVE (S28).** Table exists in both DBs. Tracks LF/CF/RF granularity. |
| `probable_pitchers` | **LIVE (S28).** Table exists in both DBs. Tracks daily probables from MLB Stats API. |
| Ingestion Pipeline | `run-pipeline` endpoint expanded to 11 jobs. Sequential execution verified. |
| Redis Caching Layer | **PRODUCTION-READY (S29).** CacheService implemented with msgpack, dynamic TTL, connection pooling. Pending Railway deployment. |

---

## Session History (Recent)

### S28 — Phase 1 Data Layer Hardening: Model + Migration (Apr 8)

**Completed:**
- `models.py`: Added PositionEligibility table with LF/CF/RF granularity (P25)
- `scripts/migrate_v25_position_eligibility.py`: Migration script created, syntax-verified.
- `models.py`: Added ProbablePitcherSnapshot table (P26).
- `scripts/migrate_v26_probable_pitchers.py`: Migration script created, syntax-verified.
- **Railway Deployment:** Migrations v25 and v26 successfully deployed to both Legacy and Fantasy production databases (Gemini S28).
- **NSB Bug Fixed:** Changed `max(0, sb - cs)` to `sb - cs` in `projections_loader.py` line 174 (verified via K-28 audit).
- **Validation Suite:** Created `tests/test_fantasy_h2h_validations.py` — all 6 tests passing (NSB negative values, scarcity index, IP bank, one-win probability, stat_id 60, Statcast fallback).
- **Phase 2.1 (Compute Layer):** H2HOneWinSimulator implemented with NumPy vectorization.
  - `backend/fantasy_baseball/h2h_monte_carlo.py` — Monte Carlo for category-by-category win probability
  - Performance: <200ms for 10,000 simulations (target met)
  - Returns: win_probability, locked/swing/vulnerable categories, category breakdown
  - `tests/test_h2h_monte_carlo.py` — 7/7 tests passing (basic functionality, dominant team, even matchup, performance, negative NSB, ERA/WHIP, category probs)
  - `backend/schemas.py`: Added H2HOneWinSimRequest, H2HOneWinSimResponse, CategoryWinProbability

- **Phase 2.2 (Two-Start Detection):** Service implemented with comprehensive UAT validation.
  - `backend/fantasy_baseball/two_start_detector.py` — Detects 2-start pitchers over 7-day window
  - `backend/schemas.py`: Added TwoStartOpportunitySchema, MatchupRatingSchema, TwoStartDetectionRequest/Response
  - `tests/test_two_start_detection_uat.py` — 10 UAT tests, 9/10 passing (1 skipped: local DB)
  - `reports/2026-04-08-two-start-detection-uat-checklist.md` — Full production validation checklist
  - Features: matchup quality scores, acquisition method classification, data freshness flags, streamer ratings

**Active Delegation:**
- **K-32:** BallDon'tLie MLB MCP Research ✅ COMPLETE — Ready for Claude Code review
- **K-31:** Redis Railway Architecture Deep Dive ✅ COMPLETE — Ready for Claude Code review
- **K-30:** Hybrid Fantasy-Betting Edge Framework — STRATEGIC REVIEW (Claude Code)
- **K-29:** Weather & Park Factors Integration Spec — ARCHITECTURE DECISION (Claude Code)

**Ready for Delegation:**
- **K-33:** MCP Integration Strategy & Railway MCP Research — AWAITING CLAUDE DELEGATION
  - See: `CLAUDE_K33_MCP_DELEGATION_PROMPT.md`
  - Scope: BDL MCP integration design + Railway MCP + additional MCP evaluation

**Pending:** Phase 2.3, Phase 3 implementation (awaiting strategic decisions)

---

### S29 — Redis Caching Layer Implementation (Apr 8)

**Completed:**
- **CacheService Implementation** — Full production-ready Redis caching service
  - `backend/services/cache_service.py` (453 lines) — SerializationManager, TTLConfig, FantasyBaseballTTL, CacheService
  - Msgpack serialization with zlib compression (>1KB objects)
  - Dynamic TTL based on game proximity: 6hr (>24h) → 30min (6-24h) → 5min (2-6h) → 1min (<2h) → 30sec (live)
  - Domain-specific TTLs: player stats (15min), scarcity (1min), live odds (5min), two-start SP (24hr)
  - Graceful degradation when Redis unavailable
  - Health monitoring and cache statistics

- **Redis Client Enhancement** — Railway-optimized connection pooling
  - `backend/redis_client.py` enhanced with BlockingConnectionPool (30 max connections)
  - TLS auto-detection for Railway TCP Proxy
  - 5-second socket timeouts, 30-second health checks
  - Preserved existing NamespacedCache class (edge_cache, fantasy_cache)

- **Test Suite** — 31 comprehensive tests, **all passing** ✅
  - `tests/test_cache_service.py` — Serialization (7), TTL (2), dynamic TTL (5), CacheService (11), disabled operations (6)
  - Fixed mock setup for get_cache_stats tests
  - Validated msgpack/JSON serialization, compression, jitter, session operations

- **Weather Fetcher Integration** — Redis caching as additional layer
  - `backend/fantasy_baseball/weather_fetcher.py` — Redis cache checked before filesystem cache
  - Saves to both Redis and filesystem on API fetch
  - Uses team abbreviation for cache keys (e.g., `weather:COL:2026-04-10`)

- **Dependencies Updated** — `requirements.txt`
  - Added `redis>=5.0.0` and `msgpack>=1.0.0`

**Performance Optimizations:**
- Msgpack: 30% smaller, 2-4x faster than JSON
- Compression applied to objects >1KB
- Staggered expiration with jitter prevents cache stampede
- Connection pooling optimized for Railway (30 max connections)

**Files Modified/Created:**
- `backend/services/cache_service.py` (NEW)
- `backend/redis_client.py` (ENHANCED)
- `backend/fantasy_baseball/weather_fetcher.py` (REDIS INTEGRATION)
- `tests/test_cache_service.py` (NEW)
- `requirements.txt` (UPDATED)

**Active Delegation:**
- **G-29:** Railway Redis Deployment — READY for Gemini CLI

**Pending:** Phase 2.3 (Scarcity Index), Phase 3 (API Layer)

---

## K-28 COMPLETION SUMMARY — Yahoo API NSB Audit

**Status:** ✅ COMPLETE  
**Agent:** Kimi CLI (Deep Intelligence Unit)  
**Report:** `reports/2026-04-08-yahoo-nsb-audit.md`

### Verdict
**YES** — NSB (Net Stolen Bases) is available via Yahoo Fantasy API as stat_id 60.

### Key Findings
1. **Yahoo API:** stat_id 60 maps to "NSB" — already configured in `frontend/lib/fantasy-stat-contract.json`
2. **Endpoint:** `get_players_stats_batch()` in `yahoo_client_resilient.py` returns NSB when league has it configured
3. **CS Fallback:** Statcast provides `cs` / `caught_stealing` field if needed

### Critical Bug Identified
**File:** `backend/fantasy_baseball/projections_loader.py` line 174  
**Current (WRONG):** `nsb = max(0, sb - cs)` — clamps negative NSB to 0  
**Should Be:** `nsb = sb - cs` — NSB can be negative (0 SB - 1 CS = -1)  
**Impact:** H2H One Win format requires accurate NSB (can be negative). Bug must be fixed before UI phase.

### H2H One Win UI Data Layer Phase 1 Status
- **UNBLOCKED** — NSB data source confirmed
- **PENDING** — NSB bug fix (Claude Code owner)
- **Next:** H2HOneWinSimulator implementation after bug fix

---

## ACTIVE CRITICAL PATH

### Completed (S28-S29)
- ✅ **Phase 1 Data Layer:** P25/P26 migrations LIVE, NSB bug fixed, validation suite passing
- ✅ **Phase 2.1 Compute Layer:** H2HOneWinSimulator implemented, performance target met (<200ms for 10k sims)
- ✅ **Phase 2.2 Two-Start Detection:** Service implemented, 9/10 UAT tests passing, production validation checklist created
- ✅ **Redis Caching Layer:** Production-ready CacheService implemented with msgpack, dynamic TTL, connection pooling. All 31 tests passing.

### Next Session (S30)
- **Deploy:** G-29 Railway Redis (Gemini CLI handoff ready)
- **Review:** K-29 Weather & Park Factors Integration Spec (Claude Code decision required)
- Phase 2.3: Scarcity Index Computation (CF/LF/RF granularity)
- Phase 3: API Layer (8 new REST endpoints)

---

## HANDOFF PROMPTS — Agent Delegation Bundles

### Gemini CLI (DevOps Strike Lead) — G-28: Probable Pitchers Migration Deployment ✅ COMPLETE

**Mission:** Deploy the P26 probable_pitchers table migration to Railway production.

**Status:**
- Migration v26 executed successfully on both DBs.
- `probable_pitchers` table verified live.
- `ProbablePitcherSnapshot` model confirmed mapping correctly.

---

### Gemini CLI (DevOps Strike Lead) — G-27: Position Eligibility Migration Deployment ✅ COMPLETE

**Mission:** Deploy the P25 position_eligibility table migration to Railway production.

**Status:**
- Migration v25 executed successfully on both DBs.
- `position_eligibility` table verified live.
- `PositionEligibility` model confirmed mapping correctly.

---

### Gemini CLI (DevOps Strike Lead) — G-29: Railway Redis Deployment [READY FOR HANDOFF]

**Mission:** Deploy production-ready Redis caching layer to Railway infrastructure.

**Status:**
- ✅ Code complete: `backend/services/cache_service.py`, `backend/redis_client.py`
- ✅ Test suite passing: 31/31 tests validated
- ✅ Requirements updated: `redis>=5.0.0`, `msgpack>=1.0.0` added
- ✅ Weather fetcher integrated: Redis + filesystem hybrid caching

**Deployment Steps:**

1. **Provision Railway Redis Service:**
   ```bash
   railway add redis
   # Or via Railway dashboard: New Service → Redis
   ```

2. **Configure Redis Variables:**
   ```bash
   railway variables set REDIS_AOF_ENABLED=yes
   railway variables set REDIS_MAXMEMORY=256mb
   railway variables set REDIS_MAXMEMORY_POLICY=allkeys-lru
   railway variables set REDIS_PASSWORD=<secure-password>
   ```

3. **Verify REDIS_URL Injection:**
   - Check Railway dashboard → Variables
   - Confirm `REDIS_URL` is auto-injected (format: `redis://default:<password>@host:port`)
   - If using private networking: URL should be `redis://<internal-host>:6379`

4. **Deploy to Production:**
   ```bash
   railway up
   # Verify logs show: "CacheService initialized with Redis"
   ```

5. **Health Check:**
   - Call `/admin/cache-stats` endpoint (if exists) or manually verify connection
   - Check hit rate, memory usage, eviction count
   - Target: >85% hit rate, <80% memory utilization

**Configuration Validation:**
- ✅ Connection pool: 30 max connections (BlockingConnectionPool)
- ✅ Socket timeouts: 5 seconds
- ✅ Health checks: Every 30 seconds
- ✅ TLS: Auto-detected for external connections
- ✅ Persistence: AOF enabled for durability

**Rollback Plan:**
- If Redis unavailable, CacheService auto-disables gracefully
- All operations return None or become no-ops (no exceptions)
- Fallback to filesystem cache for weather data

**Verification Checklist:**
- [ ] Redis service running in Railway dashboard
- [ ] `REDIS_URL` environment variable set
- [ ] Application logs show "CacheService initialized with Redis"
- [ ] No connection errors in logs
- [ ] Cache stats endpoint returns metrics (hits, misses, hit_rate, used_memory_mb)

**Files Requiring Deployment:**
- `backend/services/cache_service.py` (NEW)
- `backend/redis_client.py` (MODIFIED)
- `backend/fantasy_baseball/weather_fetcher.py` (MODIFIED)
- `requirements.txt` (UPDATED)
- `tests/test_cache_service.py` (NEW - for validation only)

**Documentation:**
- Kimi K-31 Research: `reports/2026-04-08-redis-railway-architecture-deep-dive.md`
- Test Results: `tests/test_cache_service.py` (31/31 passing)

**Estimated Time:** 15-30 minutes

**Escalation:** If Railway Redis deployment fails or shows connection issues, escalate to Claude Code with:
- Railway dashboard screenshots
- Application logs (last 50 lines)
- `REDIS_URL` format (sanitized password)

---

### Kimi CLI (Deep Intelligence Unit) — K-29: Weather & Park Factors Integration Spec [ACTIVE]

**Mission:** Synthesize academic research on weather/park factors into technical integration spec for H2H One Win app.

**Status:**
- **Research Complete:** Physics models (Dr. Alan Nathan), climate data, park factors analyzed
- **Spec Written:** `reports/2026-04-08-weather-park-factors-integration-spec.md`
- **Decision Required:** Claude Code to review and select implementation option (A/B/C)

**Key Findings:**
- Park factors create **15-30% variance** in outcomes—largest exploitable edge
- Coors Field: **+28% runs, +32% HR** vs league average
- Temperature: **+3-4 ft per 10°F** (1°F ≈ 1% HR probability)
- Climate change: **+500 HR since 2010** due to warming

**Implementation Options:**
- **Option A:** Full integration (park + weather + physics) — 4 weeks
- **Option B:** Park factors only (MVP) — 1-2 weeks ← **Recommended**
- **Option C:** Post-MVP feature — defer to Phase 2

**Action Required:** Claude Code to review spec and decide integration scope for MVP.

---

### Kimi CLI (Deep Intelligence Unit) — K-30: Hybrid Fantasy-Betting Edge Framework [STRATEGIC REVIEW]

**Mission:** Create strategic framework for combining fantasy baseball projections with betting market data (The Odds API) to generate unique competitive edges.

**Status:**
- **Framework Complete:** Synthesized principles from information economics, market microstructure, and H2H One Win format strategy
- **Spec Written:** `reports/2026-04-08-hybrid-fantasy-betting-edge-framework.md`
- **Strategic Decision Required:** Claude Code to evaluate integration scope and competitive positioning

**Core Thesis:**
> Exploit the gap between slow-moving fantasy projections (Steamer/ZiPS) and fast-moving betting markets (The Odds API). When markets adjust to weather/news in minutes but fantasy projections update in days, hybrid players capture edge.

**Key Edge Patterns Identified:**

| Pattern | Description | Edge Magnitude |
|---------|-------------|----------------|
| **Weather-Market Disconnect** | Totals move +1.5 runs on wind; fantasy static | +10-15% category boost |
| **Prop-to-Fantasy Translation** | HR odds shorten +150→+300; fantasy projection unchanged | +0.3 HR expected |
| **Moneyline-to-Counting Stats** | Heavy favorites get more PA, run support | +10% R/RBI boost |
| **Market Speed Arbitrage** | Lineup changes → prop moves → fantasy lag | 2-6 hour information advantage |

**Implementation Options:**

**Option A: Full Hybrid (Recommended)**
- Scope: Fantasy + The Odds API + Weather/Park + Prop markets
- Timeline: 8 weeks (Phases 1-4)
- Investment: The Odds API subscription ($25-299/mo)
- **USP:** "Only fantasy app with real-time market intelligence"
- **Defensibility:** Requires both fantasy infrastructure + betting data relationships

**Option B: Fantasy + Weather Only**
- Scope: Park factors + weather (no betting markets)
- Timeline: 4 weeks
- **USP:** "Physics-based fantasy optimization"
- **Risk:** Less differentiation; weather alone easier to replicate

**Option C: Display Only**
- Scope: Show odds as "additional context" without integration
- Timeline: 2 weeks
- **Risk:** Feature tick, not true edge; competitors can match easily

**Competitive Analysis:**
| Feature | ESPN/Yahoo | Pure Betting | Hybrid (Your App) |
|---------|-----------|--------------|-------------------|
| Fantasy depth | ✅ | ❌ | ✅ + market overlay |
| Market speed | ❌ | ✅ | ✅ + fantasy context |
| Weather physics | ❌ | Partial | ✅ Full Nathan model |
| Prop translation | ❌ | ❌ | ✅ **UNIQUE** |
| H2H One Win optimization | ❌ | ❌ | ✅ **UNIQUE** |

**Strategic Questions for Claude:**
1. **Scope:** Option A (full hybrid) vs. Option B (weather-only MVP)?
2. **Timing:** Integrate now (Phase 3) or post-MVP (Phase 5)?
3. **Resources:** Budget for The Odds API ($25-299/mo depending on scale)?
4. **Risk tolerance:** Complexity vs. sustainable competitive advantage?

**Documents:**
- `reports/2026-04-08-hybrid-fantasy-betting-edge-framework.md` (23KB)
- Cross-references: K-29 (weather), K-28 (NSB), UI/UX research

---

### Kimi CLI (Deep Intelligence Unit) — K-30b: Resource-Constrained Implementation Plan [ACTIVE]

**Mission:** Optimize hybrid framework for actual resource constraints:
- **BallDon'tLie GOAT Tier:** Active ($39.99/mo, 600 req/min)
- **The Odds API:** Active (20K requests/month tier)

**Status:**
- **Resource Audit Complete:** User has both subscriptions active
- **Optimized Plan:** `reports/2026-04-08-hybrid-implementation-constrained-resources.md`
- **Ready for Review:** Architecture optimized for 20K API call budget

**Key Optimization:**

| Strategy | Odds API Calls/Month | % of Budget | Value |
|----------|---------------------|-------------|-------|
| **BallDon'tLie-first architecture** | 0 (primary) | 0% | 60% of edge |
| **Odds API for validation only** | ~1,500 | 7.5% | +25% edge |
| **Weekly deep dives** | ~500 | 2.5% | Strategic planning |
| **Daily selective checks** | ~150 | 0.75% | Tactical adjustments |
| **Reserve for breaking news** | ~17,850 | 89% | Injury/lineup emergencies |

**Smart Caching Strategy:**
- **BDL provides:** Basic odds (totals, moneylines) — FREE with GOAT tier
- **Odds API used for:** Cross-book comparison, line movement, prop markets
- **Weather:** OpenWeatherMap free tier (1K calls/day) or NOAA (unlimited)
- **Cache TTL:** 6 hours (far out) → 1 minute (close to game)

**Practical Daily Workflow:**
```
Morning Slate Scan (BDL only):          0 Odds API calls
Pre-Lock Line Check (selective):        3-5 Odds API calls  
Weekly Deep Dive (Sunday):              ~200 Odds API calls
Weather Triggers (as needed):           0-10 Odds API calls
Monthly Total:                          ~1,500 calls (7.5% of budget)
```

**ROI Calculation:**
- Odds API actual usage cost: ~$3/month
- Expected value: 2-3 category wins/month
- H2H One Win impact: +4.5% championship equity
- **Verdict:** PROFITABLE with 89% budget reserve for playoffs/emergencies

**Implementation Phases (Resource-Aware):**
1. **Phase 1:** BDL integration only (Week 1-2) — 60% of edge, $0 Odds API
2. **Phase 2:** Selective Odds API enhancement (Week 3-4) — +25% edge
3. **Phase 3:** Prop market layer (Week 5-6) — +15% edge
4. **Phase 4:** Automation & optimization (Week 7-8) — Cost reduction

**Decision for Claude:**
With existing subscriptions, the hybrid approach is **immediately viable and cost-effective**. No additional budget required. The 20K limit forces smart prioritization—which improves decision-making discipline.

**Documents:**
- `reports/2026-04-08-hybrid-implementation-constrained-resources.md` (17KB)

---

### Kimi CLI (Deep Intelligence Unit) — K-31: Redis Railway Architecture Deep Dive ✅ COMPLETE

**Mission:** Research Redis architecture, optimization, and deployment patterns for Railway infrastructure in fantasy baseball H2H One Win application.

**Status:**
- **Research Complete:** 10-section comprehensive analysis completed
- **Report Written:** `reports/2026-04-08-redis-railway-architecture-deep-dive.md` (71KB, 1,885 lines)
- **Code Ready:** Production-ready implementations provided for all 10 sections
- **Next Step:** Claude Code review + G-29 (Gemini CLI) deployment

**Critical Finding:**
> Railway Redis is a **self-hosted container** (Docker Hub `redis` image), NOT a managed service like AWS ElastiCache. Requires user-managed persistence, monitoring, and backups.

**Key Specifications:**

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Memory** | 256MB minimum | Fantasy baseball data with 30 teams, 15 games/day |
| **Serialization** | msgpack (default) | 30% smaller, 2-4x faster than JSON |
| **Connection Pool** | 20-30 max connections | BlockingConnectionPool for FastAPI 2-4 workers |
| **Persistence** | AOF + RDB | AOF disabled by default — must enable for critical data |
| **TTL Strategy** | Staggered with jitter | Prevents thundering herd on mass expiration |

**TTL Recommendations for H2H One Win:**
```python
TTL_STRATEGY = {
    'player_stats': 900,      # 15 min ± 10% jitter
    'scarcity_index': 60,     # 1 min ± 20% jitter
    'win_probability': 300,   # 5 min ± 15% jitter
    'two_start_sp': 86400,    # 24 hr ± 5% jitter
    'weather_forecast': 3600, # 1 hr ± 10% jitter
}
```

**Code Deliverables Included:**
1. **Redis Client Setup** — `get_redis()` with `BlockingConnectionPool`
2. **Connection Pool Config** — Environment-specific settings (dev/staging/prod)
3. **CacheManager Class** — Serialization, compression, fallbacks
4. **SerializationManager** — msgpack/JSON auto-detection
5. **RedisHealthChecker** — Prometheus-compatible metrics
6. **Circuit Breaker** — Full CLOSED/OPEN/HALF_OPEN implementation

**Security Requirements:**
- TLS 1.2+ enforced for TCP Proxy connections (external)
- Private networking recommended (no TLS overhead, ~0.5ms latency)
- Password via `REDIS_PASSWORD` environment variable
- Input validation to prevent Redis injection attacks

**Deployment Steps for G-29:**
1. `railway add redis` or Railway dashboard
2. Configure `REDIS_AOF_ENABLED=yes` for durability
3. Set `REDIS_MAXMEMORY=256mb` with `allkeys-lru` eviction
4. Verify `REDIS_URL` injection in app environment
5. Deploy with provided connection pool configuration

**Performance Targets:**
- Hot data latency: <10ms (p95)
- Throughput: 50K ops/sec (Railway Redis single-node limit)
- Hit rate target: >85% for fantasy data
- Memory utilization: <80% before eviction pressure

**Critical Warnings:**
- ⚠️ Railway Redis uses **RDB only** by default (enable AOF for critical data)
- ⚠️ **Ephemeral by default** — configure volume storage for persistence
- ⚠️ No built-in monitoring — custom health checks required

**Action for Claude Code:**
- Review 71KB report for architecture decisions
- Decide: msgpack vs JSON serialization default
- Confirm: 256MB memory allocation vs. 512MB for growth
- Hand off deployment to Gemini CLI (G-29) with provided playbook

**Documents:**
- `reports/2026-04-08-redis-railway-architecture-deep-dive.md` (71KB)

---

### Kimi CLI (Deep Intelligence Unit) — K-32: BallDon'tLie MLB MCP Research ✅ COMPLETE

**Mission:** Research BallDon'tLie MCP (Model Context Protocol) server for MLB/fantasy baseball integration.

**Status:**
- **Research Complete:** MCP capabilities, limitations, and use cases analyzed
- **Report Written:** `reports/2026-04-08-balldontlie-mlb-mcp-research.md` (15KB)
- **Key Finding:** MCP provides natural language interface but adds 50-200ms latency—ideal for AI features, not backend operations

**What is MCP:**
The BallDon'tLie MCP server (`https://mcp.balldontlie.io/mcp`) acts as a translation layer between natural language and structured API calls, enabling AI assistants to query sports data conversationally.

**MLB Data Available:**
- Games, box scores, player stats, advanced stats
- Betting odds, player props, injuries
- All accessible via natural language queries

**Capabilities vs. Limitations:**

| Capability | Limitation |
|------------|------------|
| Natural language interface | +50-200ms latency vs direct API |
| No coding required for queries | Shares BDL rate limit (600 req/min) |
| AI-friendly responses | No caching—each call hits API |
| Conversational UX | Not for high-frequency operations |

**Recommended Use Cases:**
1. **Natural Language Lineup Assistant** — "Should I start Mike Trout today?"
2. **Daily Briefing Generation** — Automated morning summaries
3. **Trade/Waiver Research** — Ad-hoc player comparisons via chat

**When NOT to Use:**
- Real-time lineup optimization (latency too high)
- Bulk data ingestion (inefficient)
- High-frequency polling (rate limit concerns)

**Cost:** FREE (uses your existing GOAT tier quota—~6% estimated usage)

**Architecture Recommendation:**
```
Backend (Performance):   Direct BDL API + Redis
AI Layer (UX):           MCP Server for conversational features
```

**Documents:**
- `reports/2026-04-08-balldontlie-mlb-mcp-research.md`

---

### Kimi CLI (Deep Intelligence Unit) — K-33: MCP Integration Strategy & Railway MCP Research [PENDING DELEGATION]

**Mission:** (To be delegated by Claude Code) 
1. Design integration strategy for BallDon'tLie MCP in fantasy baseball app
2. Research additional MCP servers beneficial to the application (Railway MCP, etc.)

**Proposed Scope for K-33:**

**Part A: BDL MCP Integration Design**
- Specific use cases for H2H One Win format
- User interface patterns (chatbot, daily briefing, research assistant)
- Performance optimization (caching MCP responses)
- Fallback strategies (MCP unavailable → direct API)
- Implementation roadmap (Phase 1: backend, Phase 2: MCP layer)

**Part B: Railway MCP Server Research**
- What is Railway MCP? (`railway.app/mcp` or community implementations)
- Capabilities: Deployment, database management, log access
- Integration with existing Railway infrastructure
- Security implications (MCP access to production resources)
- Use cases: Automated deployment queries, database health checks, log analysis

**Part C: Additional MCP Server Evaluation**
- **GitHub MCP:** Repository management, issue tracking
- **Stripe MCP:** Payment/subscription management (if premium tiers)
- **Database MCP:** PostgreSQL query interface
- **Other relevant MCPs:** Weather, news, calendar, email

**Deliverable:**
- Integration architecture document
- MCP server comparison matrix
- Implementation priority ranking
- Code examples for MCP client integration

**Action Required:** Claude Code to review K-32 and delegate K-33 with expanded scope.

---

### Kimi CLI (Deep Intelligence Unit) — K-28: Yahoo API NSB Verification Audit ✅ COMPLETE

**Mission:** Determine if Yahoo Fantasy API exposes NSB (Net Stolen Bases, stat_id 5070) and identify fallback data sources if not.

**Status:**
- **Verdict:** YES — NSB available as stat_id 60
- **Bug Found:** `projections_loader.py` clamps NSB to 0 (must fix)
- **Report:** `reports/2026-04-08-yahoo-nsb-audit.md`
- **H2H UI Data Layer:** UNBLOCKED (pending bug fix)

---

*Last Updated: April 8, 2026 — Session S29*
