# Yahoo Fantasy Baseball Stat Mapping Analysis
## Raw Matchup Stats Response (Week 3)

### Team: Juiced Balls

**Current stats from Yahoo API:**

```json
{
  "stat_id": "60", "value": "48/218"    ← H/AB (Hits/At Bats - display only, non-scoring)
  "stat_id": "7",  "value": "23"        ← R (Runs)
  "stat_id": "8",  "value": "48"        ← H (Hits)
  "stat_id": "12", "value": "7"         ← HR (Home Runs)
  "stat_id": "13", "value": "26"        ← RBI
  "stat_id": "21", "value": "51"        ← TB (Total Bases)
  "stat_id": "23", "value": "79"        ← K (Strikeouts by batters)
  "stat_id": "3",  "value": ".220"      ← AVG (Batting Average)
  "stat_id": "55", "value": ".683"      ← OPS
  "stat_id": "62", "value": "5"         ← NSB (Net Stolen Bases)
  "stat_id": "50", "value": "20.0"      ← IP (Innings Pitched)
  "stat_id": "28", "value": "2"         ← W (Wins)
  "stat_id": "29", "value": "2"         ← L (Losses)
  "stat_id": "38", "value": "1"         ← HR (Home Runs Allowed by pitchers)
  "stat_id": "42", "value": "29"        ← K (Strikeouts by pitchers)
  "stat_id": "26", "value": "2.25"      ← ERA
  "stat_id": "27", "value": "1.05"      ← WHIP
  "stat_id": "57", "value": "13.05"     ← K/9
  "stat_id": "83", "value": "2"         ← NSV (Net Saves)
  "stat_id": "85", "value": "0"         ← QS (Quality Starts)
}
```

---

## Current Mapping vs Correct Mapping

| Stat ID | Value | Current Mapping | **CORRECT Mapping** | Category |
|---------|-------|-----------------|---------------------|----------|
| **60** | 48/218 | ❌ NSB | ✅ **H/AB** (display) | Batting (non-scoring) |
| **7** | 23 | ✅ R | ✅ **R** | Batting |
| **8** | 48 | ✅ H | ✅ **H** | Batting |
| **12** | 7 | ✅ HR_B | ✅ **HR_B** | Batting |
| **13** | 26 | ✅ RBI | ✅ **RBI** | Batting |
| **21** | 51 | ❌ UNMAPPED | ✅ **TB** | Batting |
| **23** | 79 | ❌ W (wrong!) | ✅ **K_B** | Batting |
| **3** | .220 | ❌ UNMAPPED | ✅ **AVG** | Batting |
| **55** | .683 | ❌ UNMAPPED | ✅ **OPS** | Batting |
| **62** | 5 | ❌ UNMAPPED | ✅ **NSB** | Batting |
| **50** | 20.0 | ❌ UNMAPPED | ✅ **IP** | Pitching |
| **28** | 2 | ❌ K_P (wrong!) | ✅ **W** | Pitching |
| **29** | 2 | ❌ QS (wrong!) | ✅ **L** | Pitching |
| **38** | 1 | ❌ UNMAPPED | ✅ **HR_P** | Pitching |
| **42** | 29 | ❌ UNMAPPED | ✅ **K_P** | Pitching |
| **26** | 2.25 | ❌ OPS (wrong!) | ✅ **ERA** | Pitching |
| **27** | 1.05 | ❌ UNMAPPED | ✅ **WHIP** | Pitching |
| **57** | 13.05 | ✅ K_9 | ✅ **K_9** | Pitching |
| **83** | 2 | ✅ NSV | ✅ **NSV** | Pitching |
| **85** | 0 | ❌ UNMAPPED | ✅ **QS** | Pitching |

---

## **Yahoo UI Display Order (Exact Match Required)**

### **Batters Section:**
1. **H/AB** (48/218) — stat_id 60 — *display only, non-scoring*
2. **R** (23) — stat_id 7
3. **H** (48) — stat_id 8
4. **RBI** (26) — stat_id 13
5. **K** (79) — stat_id 23
6. **TB** (51) — stat_id 21
7. **AVG** (.220) — stat_id 3
8. **OPS** (.683) — stat_id 55
9. **NSB** (5) — stat_id 62

### **Pitchers Section:**
1. **IP** (20.0) — stat_id 50
2. **W** (2) — stat_id 28
3. **L** (2) — stat_id 29
4. **HR** (1) — stat_id 38
5. **K** (29) — stat_id 42
6. **ERA** (2.25) — stat_id 26
7. **WHIP** (1.05) — stat_id 27
8. **K/9** (13.05) — stat_id 57
9. **QS** (0) — stat_id 85
10. **NSV** (2) — stat_id 83

---

## Critical Errors in Current Code

1. **stat_id 60** mapped to NSB → should be **H/AB display field**
2. **stat_id 23** mapped to W → should be **K_B** (batter strikeouts)
3. **stat_id 28** mapped to K_P → should be **W** (wins)
4. **stat_id 29** mapped to QS → should be **L** (losses)
5. **stat_id 26** mapped to OPS → should be **ERA**
6. **stat_id 85** missing → should be **QS** (quality starts)
7. **stat_id 62** missing → should be **NSB** (net stolen bases)
8. **stat_id 3, 21, 27, 38, 42, 50, 55** all UNMAPPED

---

## User-Reported Issues Explained

> "NSB showing 10 is wrong - I see it has 10 for Hits as well"

✅ **CONFIRMED**: stat_id 60 contains "48/218" (Hits/At Bats). Current code maps stat_id 60 → NSB. Should be H/AB display field.

> "W showing as 19...this is definitely meant to be strikes"

✅ **CONFIRMED**: stat_id 28 (value "2") currently mapped to K_P but should be W. stat_id 42 (value "29") is the actual K_P but is UNMAPPED.

> "TB, AVG, OPS, NSB - need same order preserved as Yahoo app"

✅ **CONFIRMED**: Need exact order match per Yahoo UI display.

---

## Fix Required

File: `backend/fantasy_baseball/yahoo_client_resilient.py`  
Method: `get_matchup_stats()`  
Lines: ~1078-1092

**Replace entire `yahoo_to_canonical` dict with corrected mapping.**
