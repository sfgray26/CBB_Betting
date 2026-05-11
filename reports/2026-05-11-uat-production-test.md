# Production UAT Report — CBB Edge Fantasy Baseball Platform
**Date:** 2026-05-11  
**Environment:** Railway Production (`fantasy-app-production-5079.up.railway.app`)  
**Tester:** Automated UAT Runner  
**Auth Header:** `Authorization: Bearer test-key`

---

## Summary

| Endpoint | Status | Response Time | Result |
|----------|--------|--------------|--------|
| `GET /health` | **200** | 547ms | ✅ PASS |
| `GET /health/pipeline` | **503** | 349ms | ❌ FAIL |
| `GET /health/db` | **200** | 198ms | ⚠️ PARTIAL (table errors) |
| `GET /api/fantasy/lineup/current` | **401** | 229ms | ❌ FAIL |
| `GET /api/fantasy/budget` | **200** | 1873ms | ✅ PASS |
| `GET /api/fantasy/matchup/current` | **404** | 345ms | ❌ FAIL |
| `GET /api/dashboard` | **401** | 276ms | ❌ FAIL |
| `GET /api/fantasy/decisions` | **401** | 394ms | ❌ FAIL |

**Score: 2/8 endpoints fully passing** | **1 partial** | **5 failing**

---

## Detailed Results

### 1. `GET /health` — ✅ PASS
- **Status:** `200 OK`
- **Response Time:** 547 ms
- **Response Body:**
  ```json
  {
    "status": "healthy",
    "database": "connected",
    "scheduler": "running"
  }
  ```
- **Validation:** JSON structure matches expected schema (`status`, `database`, `scheduler` fields present). All values positive.
- **Notes:** None.

---

### 2. `GET /health/pipeline` — ❌ FAIL
- **Status:** `503 Service Unavailable`
- **Response Time:** 349 ms
- **Response Body:** *(empty — error response)*
- **Validation:** Expected 200 with pipeline status. Received 503.
- **Error:** `Response status code does not indicate success: 503 (Service Unavailable).`
- **Notes:** ⚠️ **Critical issue — data pipeline is down.** No MLB data is being ingested. Fantasy insights and player stats may be stale.

---

### 3. `GET /health/db` — ⚠️ PARTIAL
- **Status:** `200 OK`
- **Response Time:** 198 ms
- **Response Body:**
  ```json
  {
    "status": "connected",
    "checked_at": "2026-05-11T12:48:22.447236-04:00",
    "table_counts": {
      "games": 4,
      "predictions": 2,
      "data_ingestion_logs": 9171,
      "mlb_players": "error: (psycopg2.errors.UndefinedTable) relation 'mlb_players' does not exist\nLINE 1: SELECT COUNT(*) FROM mlb_players\n                             ^\n[SQL: SELECT COUNT(*) FROM mlb_players]",
      "mlb_matchups": "error: (psycopg2.errors.InFailedSqlTransaction) current transaction is aborted, commands ignored until end of transaction block\n[SQL: SELECT COUNT(*) FROM mlb_matchups]"
    }
  }
  ```
- **Validation:** JSON structure valid, but two table count queries failed:
  - `mlb_players` — **UndefinedTable** — table does not exist in production DB
  - `mlb_matchups` — **InFailedSqlTransaction** — prior SQL error left transaction in aborted state
- **Notes:** ⚠️ **Database schema incomplete or migration failed.** The `mlb_players` table is missing, and a transaction is stuck in failed state. `games` and `predictions` tables have very low row counts (4 and 2), suggesting minimal real data.

---

### 4. `GET /api/fantasy/lineup/current` — ❌ FAIL
- **Status:** `401 Unauthorized`
- **Response Time:** 229 ms
- **Response Body:** *(empty — error response)*
- **Validation:** Expected 200 with lineup data. Received 401.
- **Error:** `Response status code does not indicate success: 401 (Unauthorized).`
- **Notes:** 🔐 `test-key` token rejected. The production deployment may require a different auth mechanism (OAuth/Yahoo session, not bearer token) or the test key is not configured in prod.

---

### 5. `GET /api/fantasy/budget` — ✅ PASS
- **Status:** `200 OK`
- **Response Time:** 1,873 ms (slowest endpoint)
- **Response Body:**
  ```json
  {
    "budget": {
      "acquisitions_used": 0,
      "acquisitions_remaining": 8,
      "acquisition_limit": 8,
      "acquisition_warning": false,
      "il_used": 3,
      "il_total": 3,
      "ip_accumulated": 0.0,
      "ip_minimum": 90.0,
      "ip_pace": "BEHIND",
      "as_of": "2026-05-11T12:48:25.484992-04:00"
    },
    "freshness": {
      "primary_source": "yahoo",
      "fetched_at": "2026-05-11T12:48:24.000273-04:00",
      "computed_at": "2026-05-11T12:48:24.000273-04:00",
      "staleness_threshold_minutes": 60,
      "is_stale": false
    }
  }
  ```
- **Validation:** JSON structure matches expected schema. All fields present with valid types.
- **Notes:** Only fantasy endpoint that works with the `test-key` token. IP pace is "BEHIND" (0.0/90.0 innings pitched minimum). 3 IL slots used out of 3 total. Data is fresh (< 60 min stale).

---

### 6. `GET /api/fantasy/matchup/current` — ❌ FAIL
- **Status:** `404 Not Found`
- **Response Time:** 345 ms
- **Response Body:** *(empty — error response)*
- **Validation:** Expected 200 with current matchup data. Received 404.
- **Error:** `Response status code does not indicate success: 404 (Not Found).`
- **Notes:** No current matchup data available. This could be expected if no matchup is currently active (e.g., all-star break, offseason, or Sunday night game window). However, given the `mlb_matchups` table error in `/health/db`, it may also indicate missing data.

---

### 7. `GET /api/dashboard` — ❌ FAIL
- **Status:** `401 Unauthorized`
- **Response Time:** 276 ms
- **Response Body:** *(empty — error response)*
- **Validation:** Expected 200 with dashboard data. Received 401.
- **Error:** `Response status code does not indicate success: 401 (Unauthorized).`
- **Notes:** 🔐 Same auth issue as `/api/fantasy/lineup/current` and `/api/fantasy/decisions`. `test-key` token not accepted.

---

### 8. `GET /api/fantasy/decisions` — ❌ FAIL
- **Status:** `401 Unauthorized`
- **Response Time:** 394 ms
- **Response Body:** *(empty — error response)*
- **Validation:** Expected 200 with lineup decisions data. Received 401.
- **Error:** `Response status code does not indicate success: 401 (Unauthorized).`
- **Notes:** 🔐 Same auth issue as above. `test-key` token not accepted.

---

## Critical Issues Found

### 🔴 P1 — Data Pipeline Down (`/health/pipeline` → 503)
The data ingestion pipeline is completely unavailable. No new MLB data is being processed.

### 🔴 P1 — Database Schema Incomplete / Migration Failure
- `mlb_players` table **does not exist**
- `mlb_matchups` table query fails due to **stuck failed transaction**
- Only 4 games and 2 predictions in the DB — insufficient for a functional fantasy platform

### 🟡 P2 — Auth Token Rejected for 3 of 4 Fantasy Endpoints
Only `/api/fantasy/budget` accepts the `test-key` token. The other fantasy endpoints (`lineup`, `dashboard`, `decisions`) return 401, suggesting inconsistent auth enforcement or the test token only has partial scope.

### 🟡 P2 — Missing Matchup Data (`/api/fantasy/matchup/current` → 404)
No current matchup found. May be expected depending on schedule, but combined with DB issues raises concern.

### 🟢 Info — Slow Response on Budget Endpoint
`/api/fantasy/budget` took **1,873 ms** — significantly slower than other endpoints (median ~350 ms). Worth monitoring if this endpoint involves heavy computation or external API calls.

---

## Recommendations

1. **Investigate pipeline failure** — Check Railway logs for the ingestion service. The 503 suggests the pipeline container is crashing or not starting.
2. **Fix database migrations** — The `mlb_players` table is missing. Run migrations or check schema drift. Also resolve the stuck transaction on `mlb_matchups`.
3. **Review auth configuration** — The `test-key` token works for `/budget` but not `/lineup`, `/dashboard`, or `/decisions`. Either grant the token broader scope or document which endpoints require different auth.
4. **Verify matchup schedule** — Confirm whether 404 on `/matchup/current` is expected (off-day) or indicates missing data.
5. **Monitor budget endpoint latency** — 1.8s is slow for a simple GET. Investigate if it's doing real-time Yahoo API calls on every request.

---

*Report generated by automated UAT runner.*
