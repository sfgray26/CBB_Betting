# Kimi CLI Delegation Bundle — K-39 through K-43

> **Delegation Date:** April 10, 2026 10:45 AM EDT  
> **Delegating Agent:** Kimi CLI (Self-Delegation)  
> **Status:** READY FOR EXECUTION

---

## 🎯 DELEGATION OVERVIEW

**Context:** Tasks 1-11 (Data Quality Remediation) are complete. The platform now has clean data but needs **infrastructure hardening** before scaling. These 5 research tasks address operational concerns that become critical at production scale.

**Priority Order:**
1. K-39: Database Index Optimization (performance)
2. K-40: API Rate Limiting Strategy (reliability)
3. K-41: Testing Strategy Gap Analysis (quality)
4. K-42: Security Audit (protection)
5. K-43: Backup & Disaster Recovery (resilience)

**Expected Timeline:** 1.5-2 hours per task

---

## K-39: Database Index Optimization Analysis

### Mission
Analyze current database schema and query patterns to identify missing indexes, redundant indexes, and optimization opportunities for production-scale performance.

### Background
The platform has grown organically with ~20,000 player records and multiple tables. After data quality fixes, query performance becomes critical. Poor indexing can cause queries to slow from milliseconds to seconds as data grows.

### Research Questions (Answer All)

#### 1. Current Index Inventory
- What indexes exist on each table? (List all)
- Which indexes are unique constraints vs performance indexes?
- Are there composite indexes? What columns and in what order?
- Are there partial indexes or expression indexes?

#### 2. Query Pattern Analysis
For each major table (`mlb_player_stats`, `position_eligibility`, `player_id_mapping`, `probable_pitchers`):
- What are the most common WHERE clauses?
- What are the most common JOIN conditions?
- What are the most common ORDER BY columns?
- What are the most common GROUP BY columns?

#### 3. Missing Index Identification
Based on query patterns:
- Which columns are frequently filtered but not indexed?
- Which multi-column queries would benefit from composite indexes?
- Which foreign keys lack indexes? (critical for JOIN performance)
- Which columns have low cardinality (good for partial indexes)?

#### 4. Index Overlap & Redundancy
- Are there indexes that are subsets of other indexes? (e.g., idx_a + idx_a_b)
- Can single-column indexes be merged into composite indexes?
- Are there unused indexes consuming write overhead?

#### 5. Write vs Read Trade-offs
- Which tables are write-heavy vs read-heavy?
- What's the INSERT/UPDATE frequency per table?
- Which indexes provide minimal benefit but significant write cost?

#### 6. PostgreSQL-Specific Optimizations
- Would BRIN indexes help for time-series data?
- Would GIN indexes help for JSON/array fields?
- Are there opportunities for covering indexes (INCLUDE clause)?
- What about index-only scans for common queries?

### Deliverable

**File:** `reports/2026-04-10-database-index-optimization.md`

**Required Sections:**
1. **Executive Summary** (current state + top 3 recommendations)
2. **Index Inventory** (table-by-table listing)
3. **Query Pattern Analysis** (most common operations per table)
4. **Missing Indexes** (prioritized list with justification)
5. **Redundancy Report** (indexes to potentially drop)
6. **Implementation Script** (SQL CREATE INDEX statements)
7. **Performance Testing Plan** (how to measure improvement)

### Acceptance Criteria
- [ ] All tables in `backend/models.py` analyzed
- [ ] At least 5 missing indexes identified with query examples
- [ ] Foreign key indexes explicitly verified
- [ ] Write vs read trade-offs documented
- [ ] Ready-to-run SQL migration script provided
- [ ] Risk assessment for each index (write overhead vs query benefit)

### Dependencies
- Blocks: None (can run immediately)
- Related: Task 1-11 completion (clean data needed for analysis)

---

## K-40: API Rate Limiting Strategy Research

### Mission
Document rate limits for all external APIs and design comprehensive rate limiting, circuit breaker, and fallback strategies for production reliability.

### Background
The platform relies on 4+ external APIs (Yahoo Fantasy, BallDontLie, MLB Stats, Statcast). During high-traffic periods, aggressive polling can trigger rate limits causing cascading failures. We need a unified strategy.

### Research Questions (Answer All)

#### 1. Rate Limit Discovery
For each API (Yahoo, BDL, MLB Stats, Statcast, The Odds API):
- What are the documented rate limits? (requests per minute/hour/day)
- What are the actual observed limits? (may differ from docs)
- What happens when limits are exceeded? (429, 503, IP ban?)
- How long do rate limit windows last?
- Are limits per API key, per IP, or per user?

#### 2. Current Implementation Audit
- Where does rate limiting exist today? (search for `sleep`, `retry`, `throttle`)
- Are there hardcoded delays? (e.g., `time.sleep(0.1)`)
- Is there any rate limit tracking/monitoring?
- What happens when APIs fail? (fallback, retry, or crash?)

#### 3. Circuit Breaker Pattern Design
- Which APIs need circuit breakers? (all external?)
- What threshold triggers circuit breaker open? (5 errors in 60 seconds?)
- How long before half-open state? (30 seconds? exponential backoff?)
- What happens during open state? (fallback data, cached data, or skip?)

#### 4. Token Bucket vs Sliding Window
- Which rate limiting algorithm is best for each API?
- Token bucket: good for bursts
- Sliding window: good for strict limits
- Hybrid approach: token bucket per API, sliding window global?

#### 5. Request Prioritization
- Which API calls are critical vs nice-to-have?
- Example: Live lineup changes > Historical stat updates
- How to queue and prioritize when near limits?

#### 6. Distributed Rate Limiting
- Multiple Railway instances may share API keys
- How to coordinate rate limits across instances?
- Redis-based rate limit counter?
- Or per-instance limits (divide global limit by instance count)?

### Deliverable

**File:** `reports/2026-04-10-api-rate-limiting-strategy.md`

**Required Sections:**
1. **Executive Summary** (current risks + recommended strategy)
2. **Rate Limit Matrix** (table: API, limit, window, penalty, reset time)
3. **Circuit Breaker Design** (state machine diagram + thresholds)
4. **Implementation Guide** (pseudocode for rate limiter)
5. **Monitoring & Alerting** (how to track rate limit health)
6. **Fallback Strategies** (what to do when APIs are unavailable)
7. **Configuration Template** (env vars for rate limits)

### Acceptance Criteria
- [ ] All 4+ external APIs documented with rate limits
- [ ] Current code audited for existing rate limiting
- [ ] Circuit breaker thresholds specified for each API
- [ ] At least 2 fallback strategies documented per API
- [ ] Ready-to-implement Python pseudocode provided
- [ ] Monitoring metrics defined (what to track, alert thresholds)

### Dependencies
- Blocks: None
- Related: K-37 (MLB API Comparison) - coordinate on rate limit findings

---

## K-41: Testing Strategy Gap Analysis

### Mission
Analyze current test suite (88 test files) to identify coverage gaps, testing anti-patterns, and create a roadmap for achieving comprehensive test coverage before production scale.

### Background
The project has 88 test files but lacks systematic coverage analysis. Critical paths (data ingestion, API endpoints, database operations) may be undertested. We need to identify what's missing before scaling.

### Research Questions (Answer All)

#### 1. Test Inventory & Categorization
- What tests exist? Categorize by type (unit, integration, e2e, contract)
- Which modules have the most tests? Which have none?
- What's the ratio of happy-path vs error-case tests?
- Are there duplicate tests across files?

#### 2. Critical Path Coverage
For each critical system:
- Data ingestion pipeline (`daily_ingestion.py`)
- API endpoints (`routers/`)
- Database operations (`models.py` interactions)
- External API clients (`yahoo_client_resilient.py`, `balldontlie.py`)
- Fantasy optimization (`daily_lineup_optimizer.py`)

What percentage of code paths are tested? (Use `pytest-cov` if available)

#### 3. Error Handling Coverage
- Which exceptions are tested? Which are not?
- Are database rollback scenarios tested?
- Are API failure scenarios tested (timeouts, 5xx errors)?
- Are validation error cases tested?

#### 4. Integration Test Gaps
- Which external services are mocked vs tested with real calls?
- Are database integration tests isolated (transactions rolled back)?
- Are there tests that depend on external state (real API calls, specific dates)?
- What's the test execution time? (fast unit tests vs slow integration tests)

#### 5. Test Data Strategy
- How is test data created? (fixtures, factories, hardcoded?)
- Are tests deterministic (same results every run)?
- Is there test data that could become stale (player names, team rosters)?

#### 6. CI/CD Integration
- Are tests run automatically on commit/deployment?
- What's the test failure policy? (block deploy or warn?)
- Are there flaky tests that sometimes fail?

### Deliverable

**File:** `reports/2026-04-10-testing-strategy-gap-analysis.md`

**Required Sections:**
1. **Executive Summary** (coverage % + top 3 gaps)
2. **Test Inventory** (table: module, # tests, type, coverage %)
3. **Critical Path Analysis** (which systems need more testing)
4. **Gap Matrix** (table: what to test, priority, effort estimate)
5. **Testing Best Practices Guide** (for new code)
6. **Implementation Roadmap** (phased approach to fill gaps)
7. **Test Data Strategy** (recommendations for fixtures/factories)

### Acceptance Criteria
- [ ] All 88 test files catalogued and categorized
- [ ] Coverage analysis for 6+ critical systems
- [ ] At least 10 specific test gaps identified with priority
- [ ] Identification of any flaky or non-deterministic tests
- [ ] Ready-to-implement test plan for top 3 gaps
- [ ] Estimated effort for achieving 80% coverage

### Dependencies
- Blocks: None
- Related: Code changes from Tasks 1-11 need corresponding tests

---

## K-42: Security Audit - API Layer

### Mission
Conduct security audit of API layer focusing on authentication, authorization, input validation, and sensitive data handling. Identify vulnerabilities before production deployment.

### Background
Fantasy sports platforms handle OAuth tokens, personal data, and financial-adjacent information (betting history). Security incidents could expose user data or compromise Yahoo accounts.

### Research Questions (Answer All)

#### 1. Authentication Audit
- How is Yahoo OAuth implemented? (token storage, refresh, expiry)
- Are tokens encrypted at rest? (database storage)
- Are tokens transmitted securely? (HTTPS only?)
- What happens when tokens expire mid-session?
- Is there session management (logout, session expiry)?

#### 2. Authorization & Access Control
- Are API endpoints protected? (who can call what)
- Is there role-based access control? (admin vs user)
- Can users access other users' data? (horizontal privilege escalation)
- Are there admin endpoints that need extra protection?

#### 3. Input Validation
- Are all API inputs validated? (Pydantic schemas)
- Is there SQL injection protection? (parameterized queries)
- Is there XSS protection? (output encoding for user content)
- Are file uploads restricted? (if applicable)
- Is there rate limiting on authentication endpoints? (brute force protection)

#### 4. Sensitive Data Handling
- What sensitive data is stored? (tokens, passwords, PII)
- Is sensitive data logged? (should never log API keys/tokens)
- Are error messages informative but not revealing? (no stack traces to users)
- Is there PII in URLs? ( Yahoo IDs in query params)

#### 5. Dependency Security
- Are dependencies up to date? (`requirements.txt` audit)
- Any known CVEs in dependencies? (security vulnerabilities)
- Are dev/test dependencies separated from production?

#### 6. Infrastructure Security
- Are environment variables properly secured? (Railway secrets)
- Is DEBUG mode disabled in production?
- Are CORS policies restrictive? (not `*`)
- Is there request size limiting? (prevent DoS)

### Deliverable

**File:** `reports/2026-04-10-security-audit-api-layer.md`

**Required Sections:**
1. **Executive Summary** (risk level + top 3 vulnerabilities)
2. **Authentication Review** (OAuth implementation assessment)
3. **Authorization Matrix** (who can access what)
4. **Input Validation Audit** (what's validated, what's missing)
5. **Sensitive Data Inventory** (what's stored, how it's protected)
6. **Vulnerability Findings** (ranked by severity: Critical/High/Medium/Low)
7. **Remediation Roadmap** (priority order for fixes)
8. **Security Best Practices Guide** (for ongoing development)

### Acceptance Criteria
- [ ] All API endpoints in `backend/routers/` analyzed
- [ ] OAuth token handling fully reviewed
- [ ] At least 5 security findings documented with severity
- [ ] SQL injection and XSS vectors explicitly checked
- [ ] Dependency audit completed
- [ ] Ready-to-implement fixes for Critical/High issues

### Dependencies
- Blocks: None (can run immediately)
- Warning: May identify issues requiring immediate fixes

---

## K-43: Backup & Disaster Recovery Strategy

### Mission
Design comprehensive backup and disaster recovery strategy for PostgreSQL database, including point-in-time recovery, data retention policies, and runbook procedures.

### Background
The platform now has critical data (player mappings, stats, position eligibility) that would be time-consuming to recreate. A data loss event without backups could set the project back weeks. Railway provides automated backups but we need a documented strategy.

### Research Questions (Answer All)

#### 1. Current State Assessment
- What backups exist today? (Railway automated backups?)
- How often do backups run? (hourly, daily?)
- How long are backups retained? (7 days, 30 days?)
- Can we access backup files directly? (for local restore testing)
- Is point-in-time recovery (PITR) available?

#### 2. Data Criticality Classification
Classify each table by criticality:
- **Critical**: Data loss = project failure (player_id_mapping, position_eligibility)
- **Important**: Can be regenerated but time-consuming (mlb_player_stats, probable_pitchers)
- **Reproducible**: Can be regenerated from external sources (game logs, odds)

#### 3. Backup Strategy Design
- Full backups: how often? (daily?)
- Incremental backups: needed or not?
- Transaction log archiving for PITR?
- Cross-region replication? (disaster recovery)
- Local development data snapshots?

#### 4. Retention Policy
- Production backups: how long? (30 days? 90 days?)
- Test environment backups: how long?
- Compliance requirements? (GDPR, data privacy)
- Cost implications of retention?

#### 5. Recovery Procedures
- How to restore from backup? (step-by-step)
- How long does restoration take? (RTO - Recovery Time Objective)
- How much data loss is acceptable? (RPO - Recovery Point Objective)
- How to test backups? (regular restore drills)
- How to recover from specific scenarios? (accidental DELETE, corrupted data, full outage)

#### 6. Monitoring & Alerting
- How to verify backups are running successfully?
- How to alert on backup failures?
- How to monitor backup storage usage?
- How to test backup integrity?

### Deliverable

**File:** `reports/2026-04-10-backup-disaster-recovery-strategy.md`

**Required Sections:**
1. **Executive Summary** (current risk level + strategy overview)
2. **Data Criticality Matrix** (table: table, criticality, regeneration effort)
3. **Backup Strategy** (schedule, retention, storage location)
4. **Recovery Runbook** (step-by-step procedures for 3+ scenarios)
5. **RTO/RPO Definitions** (time objectives for recovery)
6. **Monitoring Setup** (how to verify backups, alert on failures)
7. **Testing Plan** (how to validate backups work)
8. **Implementation Checklist** (action items for Claude/Gemini)

### Acceptance Criteria
- [ ] All tables classified by criticality
- [ ] Backup schedule and retention policy defined
- [ ] At least 3 disaster scenarios with recovery procedures
- [ ] RTO and RPO explicitly stated
- [ ] Monitoring and alerting strategy documented
- [ ] Implementation checklist ready for execution

### Dependencies
- Blocks: None
- Requires: Railway access for backup verification

---

## 🔄 COORDINATION NOTES

### Parallel Execution
All 5 tasks can run in parallel (no dependencies between them).

### Deliverable Format
Each report should follow standard format:
- Executive Summary (for quick decision making)
- Detailed Analysis (for implementation)
- Action Items (checklist format for easy tracking)
- Code Examples (where applicable)

### Integration with Existing Work
- K-39 (Indexes): Should consider query patterns from Task 11 validation
- K-40 (Rate Limiting): Should reference K-37 (API Comparison) findings
- K-41 (Testing): Should review any tests created during Tasks 1-11
- K-42 (Security): Should review auth changes from Yahoo integration work
- K-43 (Backups): Should consider data volume from current DB state

---

## 📋 SUCCESS CRITERIA

When all 5 tasks complete:
- [ ] Database has optimized indexes for query performance
- [ ] API rate limiting strategy prevents cascading failures
- [ ] Testing gaps are identified with remediation plan
- [ ] Security vulnerabilities are documented with fixes
- [ ] Disaster recovery runbook exists and is tested

**Result:** Platform is production-hardened and ready for scale.

---

*End of Delegation Bundle — Ready for Kimi CLI Execution*
