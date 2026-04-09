# Delegation Bundle — K-31: Railway Redis Optimization Research

**Agent:** Kimi CLI (Deep Intelligence Unit)
**Date:** April 8, 2026
**Priority:** HIGH — Blocks Redis implementation for H2H One Win MVP

---

## Mission

Deep research on **optimal Redis implementation patterns specifically for Railway infrastructure** to ensure production-grade caching for the MLB Platform.

**Goal:** Provide Claude Code with architecture recommendations, connection patterns, and Railway-specific best practices before code implementation begins.

---

## Research Requirements

### 1. Railway Redis Architecture Analysis

**Key Questions:**
- What is Railway's Redis infrastructure? (Managed Redis? Shared instance? Dedicated?)
- What are Railway-specific Redis connection patterns?
- Does Railway use Redis Sentinel or Redis Cluster? (Impacts connection strategy)
- What are Railway's Redis persistence settings? (RDB/AOF? Durability guarantees?)
- Are there Railway-specific Redis limitations or gotchas?

**Documentation to Review:**
- Railway Redis documentation: https://docs.railway.app/reference/redis
- Railway pricing/limits for Redis plans
- Railway community discussions on Redis best practices

### 2. Connection Pooling Strategy

**Research:**
- How should we configure Redis connection pools for Railway?
  - Max connections per dyno?
  - Connection timeout settings?
  - Retry strategy for transient failures?
  - Health check patterns?

**Code Context:**
- We're using `redis-py` (Python redis>=5.0.0)
- FastAPI async/sync considerations
- Multiple dynos (potential concurrent cache access)

**Deliverable:**
```python
# Example: Recommended connection pool configuration
redis.ConnectionPool(
    host=host,
    port=port,
    password=password,
    max_connections=??,      # What's optimal for Railway?
    socket_timeout=??,       # Railway-specific latency?
    socket_connect_timeout=??,
    retry_on_timeout=??,
    health_check_interval=??  # Railway Redis health check frequency?
)
```

### 3. Cache Key Architecture

**Research:**
- Optimal cache key naming strategy for Railway Redis
- Should we use namespacing? (e.g., `odds:mlb:game_123` vs `odds:game_123`)
- Hash tags for multi-key operations? (e.g., `{mlb}:odds:game_123`)
- Key expiration strategy (TTL granular vs batch expiration)

**Use Cases:**
- Odds data (per game, per market, per bookmaker)
- Weather data (per stadium, per date)
- Session data (per user, per league)
- Probable pitchers (per team, per date)

**Deliverable:**
Recommended cache key hierarchy with Railway-specific considerations:
```python
# Example recommendation
CACHE_KEY_PATTERNS = {
    'odds': 'odds:mlb:{sport}:{game_id}:{market}',  # ???
    'weather': 'weather:{stadium_id}:{date}',         # ???
    'session': 'session:{user_id}:{league_id}',      # ???
}
```

### 4. Memory Optimization for Railway Redis 256MB

**Research:**
- How much memory does typical cached object consume?
- What's the optimal serialization format? (JSON, msgpack, pickle?)
- Compression strategy? (zlib for large objects?)
- Memory fragmentation concerns on Railway?

**Use Case Estimates:**
- Odds API response: ~2-5KB per game (15 games × 3 markets = 45-225KB)
- Weather forecast: ~1KB per stadium (30 stadiums = 30KB)
- Session data: ~10-50KB per user (100 users = 1-5MB)
- Probable pitchers: ~500 bytes per team (30 teams = 15KB)

**Deliverable:**
Memory optimization recommendations:
```python
# Example: Serialization strategy
SERIALIZATION = {
    'odds': 'json',           # or msgpack? or pickle?
    'weather': 'json',        # or compress?
    'session': 'pickle',      # or json?
    'compression_threshold': 1024  # bytes
}
```

### 5. TTL Strategy Refinement

**Research:**
- Are Railway Redis TTL policies reliable? (Exact expiration vs approximate?)
- Should we use lazy expiration (check on read) or rely on Redis TTL?
- What's the impact of TTL on Railway Redis memory usage?

**Current Strategy (from K-30b):**
| Time to Game | TTL | Rationale |
|--------------|-----|-----------|
| >24 hours | 6 hours | Line won't move much |
| 6-24 hours | 30 min | Approaching game time |
| 2-6 hours | 5 min | Pre-lineup lock |
| <2 hours | 1 min | Live betting |

**Deliverable:**
Refined TTL strategy with Railway-specific recommendations:
```python
TTL_STRATEGY = {
    'odds:far_out': ??,      # Is 6 hours optimal for Railway?
    'odds:approaching': ??,  # Is 30 min too aggressive?
    'odds:close': ??,        # Should we go shorter?
    'odds:live': ??,         # Real-time considerations?
    'weather': ??,           # Weather forecast update frequency?
    'session': ??            # User session timeout?
}
```

### 6. Failure Modes & Resilience

**Research:**
- What happens when Railway Redis is down? (Deployment? Maintenance?)
- How should the app degrade gracefully?
- Should we implement fallback to in-memory cache?
- Circuit breaker pattern for Redis?

**Code Context:**
- FastAPI app with multiple background jobs
- Daily ingestion pipeline
- H2H One Win UI (user-facing)

**Deliverable:**
Resilience pattern recommendations:
```python
# Example: Fallback strategy
if redis_unavailable:
    # Option A: Return None (no cache)
    # Option B: Fallback to in-memory (LRU cache)
    # Option C: Error and retry
```

### 7. Monitoring & Observability

**Research:**
- How to monitor Redis health on Railway?
- Key metrics to track (hit rate, memory usage, connection count)
- Railway-specific monitoring tools?
- Alerting thresholds?

**Deliverable:**
Monitoring strategy:
```python
# Metrics to track
METRICS = {
    'cache_hit_rate': ??,           # Target >50%
    'memory_usage_pct': ??,         # Alert at 80%?
    'connection_count': ??,         # Alert at ???
    'avg_response_time_ms': ??,     # Alert at ???
    'evicted_keys_per_hour': ??     # Alert at ???
}
```

### 8. Railway Deployment Considerations

**Research:**
- How to deploy Redis changes without downtime?
- Railway Redis migration strategy (if we ever upgrade plans)
- Environment variable management for REDIS_URL
- Local development setup (Docker Compose vs Railway link)

**Deliverable:**
Deployment playbook:
```bash
# Step-by-step deployment commands
railway add redis --name "mlb-platform-cache"
railway link  # Connect to existing project?
railway variable set REDIS_URL "..."
railway up  # Deploy without downtime?
```

### 9. Performance Benchmarks

**Research:**
- What are realistic Redis latency expectations on Railway?
- Throughput limits for 256MB plan? (ops/sec)
- Comparison to in-memory caching (LRU)

**Deliverable:**
Performance targets:
```python
PERFORMANCE_TARGETS = {
    'cache_get_latency_p50_ms': ??,
    'cache_get_latency_p99_ms': ??,
    'cache_set_latency_p50_ms': ??,
    'max_ops_per_second': ??,
    'hit_rate_target': 0.50  # 50%
}
```

### 10. Security Best Practices

**Research:**
- How does Railway secure Redis connections? (TLS?)
- Redis password management on Railway
- Network isolation (private vs public)
- Injection vulnerabilities (cache key poisoning)

**Deliverable:**
Security checklist:
```python
SECURITY_CHECKLIST = [
    'TLS enabled for Redis connection?',
    'REDIS_URL stored in Railway env vars (not hardcoded)?',
    'Cache key sanitization (prevent injection)?',
    'Connection string validation?',
    'Rate limiting on cache operations?'
]
```

---

## Deliverables

1. **Railway Redis Architecture Report** (10-15 pages)
   - Infrastructure analysis
   - Connection pooling strategy
   - Cache key architecture
   - Memory optimization
   - TTL strategy refinement
   - Failure modes & resilience
   - Monitoring & observability
   - Deployment considerations
   - Performance benchmarks
   - Security best practices

2. **Code Recommendations** (pseudo-code)
   - Connection pool configuration
   - Cache key patterns
   - Serialization strategy
   - TTL constants
   - Fallback patterns

3. **Railway-Specific Gotchas** (checklist)
   - Common pitfalls to avoid
   - Railway limitations/constraints
   - Deployment gotchas

4. **Implementation Roadmap** for Claude Code
   - Step-by-step implementation guide
   - Testing strategy
   - Deployment sequence

---

## Constraints

**Timebox:** 3-4 hours research (comprehensive)

**Output Format:** Markdown report + code snippets

**Audience:** Claude Code (Principal Architect) - needs actionable technical recommendations

**Railway-Specific Focus:** This isn't generic Redis research - must be tailored to Railway's infrastructure, limitations, and best practices.

---

## Escalation

If you discover:
- **Railway Redis has critical limitations** → Flag as "BLOCKER: [issue]"
- **256MB plan is insufficient** → Recommend plan upgrade
- **Railway Redis is unreliable** → Recommend alternative (self-hosted?)
- **Complex setup required** → Simplify recommendations for MVP

---

## Related Research

- K-30b: Hybrid Implementation Plan (OddsAPI + BDL integration)
- K-29: Weather & Park Factors Integration Spec
- G-29: Railway Redis Deployment (Gemini CLI - pending your research)

---

**Assigned to:** Kimi CLI (Deep Intelligence Unit)
**Review required:** Claude Code (will implement based on research)
**Sign-off:** User (approve implementation approach)

*Last Updated: April 8, 2026 — Session S29*
