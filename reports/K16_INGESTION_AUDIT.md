# K-16: Yahoo Ingestion Pipeline Failure Mode Audit

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Daily ingestion pipeline, Statcast enrichment, projections loader

---

## Executive Summary

The ingestion pipeline consists of **6 scheduled jobs**, **2 enrichment modules**, and **8 CSV projection files**. All critical paths have graceful degradation, but several silent failure modes exist that may cause stale data without triggering alerts.

**Risk Level:** MEDIUM — No data loss risk, but stale projection data may affect lineup decisions if jobs fail silently.

---

## 1. Daily Ingestion Orchestrator Jobs

**File:** `backend/services/daily_ingestion.py`

All jobs use PostgreSQL advisory locks to prevent duplicate execution across Railway replicas.

| Job ID | Schedule | Advisory Lock | What It Does | Failure Mode | Current Behavior | Risk |
|--------|----------|---------------|--------------|--------------|------------------|------|
| `mlb_odds` | Every 5 min (10 AM - 11 PM ET) | 100_001 | Polls The Odds API for MLB spread odds | API key missing | Logs "skipped", returns `{"status": "skipped"}` | LOW — No action needed |
| `mlb_odds` | — | — | — | HTTP error/timeout | Logs error, status="failed" | LOW — No alert; next poll in 5 min |
| `statcast` | Every 6 hours | 100_002 | Runs Statcast Bayesian updates | pybaseball fails | Logs error, returns failed status | MEDIUM — No notification; projections stale |
| `rolling_z` | Daily 4 AM ET | 100_003 | Computes 7/30-day z-scores | Insufficient data (< 7 days) | Silently skips players | MEDIUM — z-scores remain None |
| `clv` | Daily 11 PM ET | 100_005 | Computes closing line value | No closing lines | Returns empty CLV | LOW — No bets settled |
| `cleanup` | Daily 3:30 AM ET | 100_006 | Deletes old metric rows | DB connection failure | Exception logged | LOW — Data accumulates |
| `valuation_cache` | Daily 6 AM ET (conditional) | 100_011 | Refreshes player valuations | FANTASY_LEAGUES not set | Job not registered | LOW — Cache empty, falls back to board |

### Key Failure Patterns

1. **Silent Skips:** `mlb_odds` and `rolling_z` silently skip on missing data — no notification that data is stale
2. **No Retry:** Failed jobs wait for next scheduled run; no exponential backoff
3. **Status Tracking:** Jobs record status in-memory only (`_job_status` dict); no persistence across restarts

---

## 2. Statcast Ingestion Pipeline

**File:** `backend/fantasy_baseball/statcast_ingestion.py`

**Entry Point:** `run_daily_ingestion()` (called by orchestrator)

### Data Flow

```
run_daily_ingestion()
  ├── Fetch yesterday's games from Baseball Savant API
  ├── Parse PlayerDailyPerformance objects
  ├── Validate data quality (exit velocity ranges, etc.)
  ├── Store to StatcastPerformance table
  ├── Bayesian update: prior + new data → updated projection
  └── Return {success, records_processed, error}
```

### Failure Modes

| Stage | Failure | Behavior | Detection |
|-------|---------|----------|-----------|
| API Fetch | pybaseball down / no games | Returns `{"success": False, "error": "..."}` | Log only |
| Validation | Invalid exit velocity (> 120 mph) | Row skipped, continues | Log warning |
| DB Write | Connection timeout | Exception propagates, transaction rolled back | Log error |
| Bayesian Update | Insufficient sample size | Uses prior only, no update | Silent |

### Silent Failure: Yesterday's Date Calculation

**Line ~160:** Uses `date.today() - timedelta(days=1)` which is **local server time**, not ET.

**Risk:** If server is UTC, "yesterday" at 11 PM UTC is actually "today" in ET during EDT. This causes duplicate processing of same day's games.

**Fix:** Use ET anchor:
```python
from zoneinfo import ZoneInfo
yesterday_et = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).date()
```

---

## 3. Projections Loader

**File:** `backend/fantasy_baseball/projections_loader.py`

### Expected CSV Files (8 files found)

| File | Columns Required | Fallback Behavior if Missing |
|------|------------------|------------------------------|
| `steamer_batting_2026.csv` | Name, Team, G, PA, AB, H, 2B, 3B, HR, R, RBI, BB, SO, HBP, SF, AVG, OBP, SLG, OPS, wOBA, wRC+, BsR, Off, Def, WAR | Uses hardcoded `player_board.py` estimates |
| `steamer_pitching_2026.csv` | Name, Team, W, L, ERA, G, GS, IP, H, ER, HR, BB, SO, WHIP, K/9, BB/9, K/BB, H/9, HR/9, AVG, BABIP, LOB%, GB%, HR/FB, FIP, xFIP, WAR | Uses hardcoded estimates |
| `adp_yahoo_2026.csv` | PLAYER NAME, TEAM, POS, AVG, BEST, WORST, # TEAMS, STDEV | ADP defaults to None, tier calculation approximate |
| `advanced_batting_2026.csv` | (not documented in header) | Graceful skip; uses Steamer only |
| `advanced_pitching_2026.csv` | (not documented in header) | Graceful skip; uses Steamer only |
| `closer_situations_2026.csv` | (not documented in header) | Saves projections default to algorithmic estimate |
| `injury_flags_2026.csv` | (not documented in header) | All players assumed healthy |
| `position_eligibility_2026.csv` | (not documented in header) | Uses Yahoo API positions only |

### Column Mismatch Handling

**Code location:** Lines 150-250 (load functions)

```python
# Pattern used:
try:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Access fields directly; KeyError if column missing
            name = row["Name"]
except FileNotFoundError:
    logger.warning(f"{csv_path} not found, using hardcoded board")
    return {}
except KeyError as e:
    logger.error(f"Column missing in {csv_path}: {e}")
    return {}
```

**Risk:** If CSV has wrong column names, entire file is rejected and falls back to hardcoded board. No retry or partial loading.

### Cache Invalidation

**Pattern:** `@lru_cache(maxsize=1)` on `load_full_board()`

**Implication:** CSV changes require process restart or explicit cache clear via `POST /admin/fantasy/reload-board`

**Risk:** If admin updates CSVs but doesn't restart, stale data persists silently.

---

## 4. Failure Mode Summary Table

| Job | Failure Condition | Current Behavior | User Impact | Recommended Fix |
|-----|-------------------|------------------|-------------|-----------------|
| MLB Odds Poll | API down | Log error, wait 5 min | No live odds | Add alert if 3 consecutive failures |
| Statcast Update | pybaseball error | Log error, skip day | Stale projections | Retry with exponential backoff |
| Rolling Z-Score | < 7 days data | Skip player | No recent z-score | Lower threshold to 3 days |
| Valuation Cache | Yahoo auth fail | Empty cache | Uses board z-scores | Add alert on cache miss |
| Statcast Date | UTC vs ET bug | Wrong "yesterday" | Duplicate/missed games | Fix to ET anchor (K-17) |
| CSV Loader | Missing file | Fallback to board | Less accurate data | Add metric for CSV vs board usage |
| CSV Loader | Wrong columns | Reject entire file | No data from that file | Add partial load with fallback |

---

## 5. Monitoring Recommendations

### Immediate (No Code Changes)

1. **Railway Log Alerts:**
   - Pattern: `_poll_mlb_odds error` or `_update_statcast: unhandled error`
   - Action: Send Discord alert to #system-alerts

2. **Daily Health Check:**
   - Query: `SELECT COUNT(*) FROM player_daily_metrics WHERE metric_date = CURRENT_DATE`
   - Alert if count < expected (e.g., < 500 players)

### Short-Term (Code Changes)

1. **Job Status Endpoint:**
   - Extend `/admin/ingestion/status` to include `last_success` timestamp per job
   - Alert if `now - last_success > 2 * job_interval`

2. **Projection Freshness Metric:**
   - Add `loaded_at` timestamp to `PlayerProjection` table
   - Dashboard shows "Projections: 2 hours old" warning

3. **CSV Validation Pre-Flight:**
   - Before loading, check column names match expected
   - Report which files are missing/wrong without failing entire load

---

## 6. Projections File Inventory

**Directory:** `data/projections/`

| File | Status | Size (est.) | Last Modified | Notes |
|------|--------|-------------|---------------|-------|
| `steamer_batting_2026.csv` | ✅ Present | ~500 KB | Pre-season | Primary batter projections |
| `steamer_pitching_2026.csv` | ✅ Present | ~300 KB | Pre-season | Primary pitcher projections |
| `adp_yahoo_2026.csv` | ✅ Present | ~50 KB | Pre-season | Used for tier calculation |
| `advanced_batting_2026.csv` | ✅ Present | Unknown | Unknown | Unused (speculative) |
| `advanced_pitching_2026.csv` | ✅ Present | Unknown | Unknown | Unused (speculative) |
| `closer_situations_2026.csv` | ✅ Present | Unknown | Unknown | Saves projection override |
| `injury_flags_2026.csv` | ✅ Present | Unknown | Unknown | Pre-season injury data |
| `position_eligibility_2026.csv` | ✅ Present | Unknown | Unknown | Yahoo position eligibility |

**All 8 expected files are present.** Risk is low for projection availability.

---

## 7. Action Items for Claude Code

### High Priority
1. **Fix Statcast date bug** (K-17 finding): Use ET anchor for "yesterday" calculation
2. **Add alert threshold:** If Statcast job fails 3 consecutive times, send alert

### Medium Priority
3. **Retry logic:** Exponential backoff for pybaseball API calls (max 3 retries)
4. **Partial CSV loading:** Load available columns even if some are missing

### Low Priority
5. **Monitoring dashboard:** Show "last successful run" timestamps per job
6. **Projection age metric:** Display freshness in frontend settings page

---

*Audit complete: All 6 jobs, 2 enrichment modules, and 8 CSV files analyzed. 4 actionable fixes identified.*
