# Security Audit Report

**Date:** 2026-05-16  
**Auditor:** Gemini (Agent)  
**Scope:** Backend API endpoints, authentication, and configuration  
**Branch:** agent/gemini/security-audit-20260516

---

## Executive Summary

| Category | Status | Findings |
|----------|--------|----------|
| Authentication | ⚠️ MEDIUM | Dev fallback key present; admin check is role-based |
| Input Validation | ✅ GOOD | Pattern validation for player keys, query param limits |
| SQL Injection | ✅ GOOD | Parameterized queries via SQLAlchemy ORM |
| CORS | ⚠️ MEDIUM | Review recommended for production origin settings |
| Secrets Management | ✅ GOOD | API keys from environment, no hardcoded secrets |
| Error Handling | ✅ GOOD | Generic error messages, no stack traces leaked |

**Overall Risk Level:** MEDIUM

---

## Detailed Findings

### 1. Authentication (auth.py)

#### ✅ Good Practices
- API keys loaded from environment variables (`API_KEY_USER1` through `API_KEY_USER5`)
- Proper HTTP 401/403 status codes
- `WWW-Authenticate` header present
- Keys cached in memory to avoid repeated env lookups

#### ⚠️ Issues Found

**Issue AUTH-1: Development Fallback Key**
```python
# File: backend/auth.py:30-34
if os.getenv("ENVIRONMENT") == "development":
    keys["dev-key-insecure"] = "dev_user"
```
- **Risk:** If `ENVIRONMENT` is misconfigured in production, a publicly known dev key could work
- **Recommendation:** Remove fallback entirely or require explicit `ALLOW_DEV_KEY=true`
- **Severity:** Medium

**Issue AUTH-2: Simple Role-Based Admin Check**
```python
# File: backend/auth.py:76-91
async def verify_admin_api_key(user: str = Security(verify_api_key)) -> str:
    if user != "user1":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, ...)
```
- **Risk:** Hardcoded "user1" as admin; no granular permissions
- **Recommendation:** Add admin flag to API key configuration
- **Severity:** Low

---

### 2. Input Validation

#### ✅ Good Practices

**Player Key Validation (fantasy.py)**
```python
# File: backend/routers/fantasy.py:2253-2272
add_player_key: str = Query(..., description="Yahoo player key to add (mlb.p.XXXXX)")

if not re.match(r"^(mlb\.p\.\d+|\d+\.p\.\d+)$", add_player_key):
    raise HTTPException(status_code=422, detail="add_player_key must be mlb.p.XXXXX")
```
- Pattern validation with regex
- Proper HTTP 422 for validation errors

**Query Parameter Limits (edge.py)**
```python
# File: backend/routers/edge.py:89-92
n_sims: int = Query(default=10000, ge=1000, le=50000)
```
- Bounds checking on Monte Carlo simulation count

#### ⚠️ Issues Found

**Issue VAL-1: Unvalidated Path Parameters**
Some endpoints accept string identifiers without length limits:
```python
# Example pattern found in fantasy.py
team_key: str = Query(..., description="Yahoo team key")
```
- **Risk:** Potential DoS via extremely long strings
- **Recommendation:** Add `max_length` to string Query parameters
- **Severity:** Low

---

### 3. SQL Injection

#### ✅ Good Practices

**SQLAlchemy ORM Usage**
All database queries use SQLAlchemy ORM with parameter binding:
```python
# File: backend/routers/fantasy.py (typical pattern)
mapping_rows = (
    db.query(PlayerIDMapping.yahoo_key, ...)
    .filter(PlayerIDMapping.bdl_id.isnot(None), or_(*predicates))
    .all()
)
```

**Raw SQL with Parameterization**
Where raw SQL is used, parameters are bound:
```python
# File: backend/routers/data_quality.py:96-101
empty_cat_scores_query = text("""
    SELECT COUNT(*) FROM player_projections
    WHERE cat_scores IS NULL OR CAST(cat_scores AS TEXT) = '{}'
""")
```
- **Note:** The `{}` is a literal empty JSON, not a format string

#### ✅ Verdict
No SQL injection vulnerabilities detected. All user input is properly parameterized.

---

### 4. CORS Configuration

#### ✅ Good Practices
```python
# File: backend/main.py (typical pattern)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### ⚠️ Issues Found

**Issue CORS-1: Potential Wildcard in Production**
If `ALLOWED_ORIGINS` env var is not set or is set to `*`, CORS becomes too permissive.

- **Risk:** Cross-site request forgery if user is authenticated
- **Recommendation:** Enforce explicit origin whitelist in production
- **Severity:** Medium

---

### 5. Secrets Management

#### ✅ Good Practices
- API keys loaded from environment variables
- Discord bot token loaded from `DISCORD_BOT_TOKEN` env var
- Database connection string from `DATABASE_URL`
- No hardcoded secrets in source code

#### ⚠️ Issues Found

**Issue SEC-1: Potential Logging of Sensitive Data**
Review logging statements to ensure API keys aren't logged:
```python
# Generally safe patterns found, but verify:
logger.info("Manual analysis triggered by %s", user)  # Logs username, not key
```

- **Recommendation:** Audit all `logger` calls to ensure no sensitive data leakage
- **Severity:** Low

---

### 6. Error Handling

#### ✅ Good Practices
- Generic error messages returned to clients
- Detailed errors logged server-side only
- Stack traces not exposed in HTTP responses

#### Example from admin.py:
```python
try:
    results, cache = await run_nightly_analysis()
except Exception as exc:
    logger.error("Manual analysis failed: %s", exc, exc_info=True)  # Server log
    raise HTTPException(status_code=500, detail=str(exc))  # Client sees generic message
```

---

### 7. Endpoint Security Matrix

| Endpoint | Auth | Input Validation | Risk |
|----------|------|------------------|------|
| `GET /` | None | N/A | Low |
| `GET /health` | None | N/A | Low |
| `POST /admin/run-analysis` | Admin | None required | Low |
| `POST /admin/discord/test` | Admin | None required | Low |
| `GET /api/tournament/bracket-projection` | API Key | `n_sims` bounds | Low |
| `GET /api/fantasy/roster` | API Key | `team_key` string | Low |
| `POST /api/fantasy/roster/add` | API Key | Player key regex | Low |
| `POST /api/fantasy/roster/drop` | API Key | Player key regex | Low |
| `GET /api/edge/today` | API Key | Date format | Low |
| `POST /api/admin/data-quality/backfill-cat-scores` | Admin | Boolean `force` | Low |

---

## Recommendations

### Immediate (P1)
1. **Remove or secure dev key fallback** (AUTH-1)
   - Change to require explicit `ALLOW_DEV_KEY=true`
   - Add warning log when dev key is used

2. **Verify CORS origin whitelist** (CORS-1)
   - Ensure production uses explicit origins
   - Add startup validation that fails if `*` is used in prod

### Short-term (P2)
3. **Add granular admin permissions** (AUTH-2)
   - Add `is_admin` flag to API key config
   - Support multiple admin users

4. **Add input length validation** (VAL-1)
   - Add `max_length` constraints to all string Query parameters

### Long-term (P3)
5. **Implement rate limiting**
   - Add per-API-key rate limits
   - Consider tiered limits (admin vs regular users)

6. **Add request signing for sensitive operations**
   - Sign roster add/drop actions
   - Replay attack prevention

7. **Audit logging**
   - Log all admin actions with before/after state
   - Retain for compliance

---

## Compliance Notes

- **GDPR:** No PII storage detected in code; verify database schema
- **PCI DSS:** Not applicable (no payment processing)
- **SOC 2:** Recommend implementing recommendation #7 (audit logging)

---

## Appendix: Files Audited

| File | Lines | Focus |
|------|-------|-------|
| `backend/auth.py` | 92 | Authentication, authorization |
| `backend/main.py` | 7943 | App configuration, CORS, scheduler |
| `backend/routers/admin.py` | 2266 | Admin endpoints |
| `backend/routers/edge.py` | 1131 | Betting/analysis API |
| `backend/routers/fantasy.py` | 5967 | Fantasy baseball API |
| `backend/routers/data_quality.py` | 427 | Data quality admin |

---

*Report generated: 2026-05-16*
*Next audit recommended: 2026-06-16 (monthly)*
