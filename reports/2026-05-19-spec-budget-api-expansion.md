# Spec Memo: Budget API Expansion Requirements

**Date:** 2026-05-19  
**Author:** Kimi CLI (Research)  
**Status:** Ready for Claude Code architecture review  
**Priority:** P2 (Polish / Underwhelm — page exists but adds no value beyond Dashboard panel)

---

## 1. Problem Statement

UAT audit (K-NEXT-4, 2026-05-13) rated the Budget page as **P1 Degraded**:

> **"Budget page is extremely sparse — same 3 lines as Dashboard, no added value"**

Current state:
- **Dashboard** embeds `BudgetPanel` (acquisitions bar, IL slots, IP progress)
- **Budget page** (`/war-room/budget`) shows the *exact same* `BudgetPanel` + a "Season Pace" grid (current week, weeks left, days in week, season adds) + a one-line acquisitions callout
- The Budget page is a dedicated route that users navigate to expecting **budget management tools**, but receive only summary metrics

K-1 UAT (2026-05-07) previously noted:
> **"Budget API not integrated into UI — `/api/fantasy/budget` is healthy (576ms) but no frontend page calls it"**

The page now exists, but the underlying API contract is too thin to justify the route.

---

## 2. Current API Contract

### `GET /api/fantasy/budget`

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
    "ip_minimum": 18.0,
    "ip_pace": "BEHIND",
    "as_of": "2026-05-17T12:10:00-04:00",
    "week_label": "Week 8",
    "weeks_remaining": 17,
    "days_in_week_remaining": 7,
    "acquisitions_this_season": 12
  },
  "freshness": { ... }
}
```

### Frontend `BudgetData` Interface
```typescript
export interface BudgetData {
  acquisitions_used: number
  acquisitions_remaining: number
  acquisition_limit: number
  acquisition_warning: boolean
  il_used: number
  il_total: number
  ip_accumulated: number
  ip_minimum: number
  ip_pace: "BEHIND" | "ON_TRACK" | "AHEAD"
  as_of: string
  week_label?: string
  weeks_remaining?: number
  days_in_week_remaining?: number
  acquisitions_this_season?: number
}
```

**Missing entirely:** transaction history, category-level acquisition tracking, IP projection, league context, FAAB data, upcoming schedule impact.

---

## 3. Design Goal

Transform the Budget page from a **summary mirror** into a **budget management cockpit** that answers these manager questions:

1. *"Am I on track to meet the IP minimum?"* → IP projection + schedule-aware forecast
2. *"Where did my acquisitions go?"* → Transaction history with category tags
3. *"How does my pace compare to the league?"* → League percentile rankings
4. *"Should I save adds or spend now?"* → Budget recommendations based on matchup state
5. *"Which categories are driving my add/drop behavior?"* → Acquisition category breakdown

---

## 4. Proposed API Expansion

### 4.1 New Response Shape (Backward Compatible)

All new fields are **optional** (`?`) so existing Dashboard consumption is unaffected.

```typescript
export interface BudgetData {
  // ── Existing fields (unchanged) ──
  acquisitions_used: number
  acquisitions_remaining: number
  acquisition_limit: number
  acquisition_warning: boolean
  il_used: number
  il_total: number
  ip_accumulated: number
  ip_minimum: number
  ip_pace: "BEHIND" | "ON_TRACK" | "AHEAD"
  as_of: string
  week_label?: string
  weeks_remaining?: number
  days_in_week_remaining?: number
  acquisitions_this_season?: number

  // ── NEW: IP Projection & Forecast ──
  ip_projected?: number           // projected IP at week-end based on scheduled SPs
  ip_pace_percentile?: number     // 0-100: how your IP pace ranks in the league
  ip_on_pace_to_hit?: boolean     // true if projected >= minimum
  scheduled_starts_this_week?: number  // count of SP starts remaining

  // ── NEW: Transaction History ──
  recent_transactions?: TransactionRow[]  // last 10 add/drop moves

  // ── NEW: Acquisition Analytics ──
  acquisition_breakdown?: {
    batters_added: number
    pitchers_added: number
    by_category: Record<string, number>  // e.g. { "SV": 3, "ERA": 2, "HR": 1 }
  }

  // ── NEW: League Context ──
  league_median_ip?: number
  league_max_ip?: number
  league_avg_acquisitions_used?: number

  // ── NEW: Recommendations ──
  recommendations?: BudgetRecommendation[]
}

export interface TransactionRow {
  timestamp: string
  type: "add" | "drop" | "trade"
  player_name: string
  player_key: string
  position: string
  direction: "added" | "dropped"
}

export interface BudgetRecommendation {
  type: "ip_urgency" | "save_adds" | "spend_now" | "il_efficiency"
  severity: "info" | "warning" | "critical"
  message: string
  action_hint?: string
}
```

### 4.2 Backend Implementation Plan

#### Phase A: IP Projection (Highest Value / Lowest Effort)

**Data source:** `ProbablePitcherSnapshot` table (already populated daily)

**Algorithm:**
```python
# In backend/routers/fantasy.py:get_constraint_budget()
scheduled_starts = db.query(ProbablePitcherSnapshot).filter(
    ProbablePitcherSnapshot.game_date >= today,
    ProbablePitcherSnapshot.game_date <= week_end,
    ProbablePitcherSnapshot.pitcher_name.in_(my_sp_names)
).count()

avg_ip_per_start = 5.2  # league average; could be personalized from historical data
ip_projected = ip_accumulated + (scheduled_starts * avg_ip_per_start)
ip_on_pace_to_hit = ip_projected >= ip_minimum
```

**Cost:** 1 DB query + 3 lines of math.

#### Phase B: Transaction History

**Data source:** Already fetched via `client.get_transactions(t_type="add")` in the budget endpoint.

**Change:** Instead of discarding the transaction list after counting, enrich and return the last N rows.

```python
recent_transactions = []
for txn in sorted(transactions, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]:
    # Resolve player name from player_key via cached board or Yahoo client
    recent_transactions.append({
        "timestamp": datetime.fromtimestamp(int(txn["timestamp"])).isoformat(),
        "type": "add",
        "player_name": txn.get("player_name", "Unknown"),
        "player_key": txn.get("player_key", ""),
        "position": txn.get("position", ""),
        "direction": "added" if txn.get("type") == "add" else "dropped",
    })
```

**Cost:** Re-use existing `transactions` list; minimal enrichment.

#### Phase C: Acquisition Breakdown

**Data source:** Same `transactions` list + `PlayerProjection` / draft board for position/category tagging.

**Algorithm:**
```python
batters_added = sum(1 for t in week_transactions if _is_batter(t["player_key"]))
pitchers_added = sum(1 for t in week_transactions if _is_pitcher(t["player_key"]))

# Category tagging: infer from player type or use board positions
by_category = defaultdict(int)
for t in week_transactions:
    cat = _infer_target_category(t["player_key"], db)  # e.g. "SV" for RP adds
    if cat:
        by_category[cat] += 1
```

**Cost:** Medium — requires a helper to resolve player type from `PlayerIDMapping` → `PlayerProjection`.

#### Phase D: League Context

**Data source:** Yahoo league scoreboard or cached standings.

**Algorithm:**
```python
# Fetch all team IPs from current week scoreboard
league_ips = []
for team_key in all_team_keys:
    stats = client.get_matchup_stats(week=current_week, my_team_key=team_key)
    league_ips.append(float(stats.get("my_stats", {}).get("IP", 0)))

league_median_ip = median(league_ips)
league_max_ip = max(league_ips)
```

**Cost:** High (N Yahoo API calls for N teams). **Mitigation:** Cache for 1 hour; only compute when `?include_league=true` query param is present.

#### Phase E: Recommendations Engine

**Rules-based (no ML):**
```python
recommendations = []

if ip_pace == "BEHIND" and days_in_week_remaining <= 3:
    recommendations.append({
        "type": "ip_urgency",
        "severity": "critical",
        "message": f"Projected to fall {ip_minimum - ip_projected:.0f} IP short. Stream SPs immediately.",
        "action_hint": "Go to Streaming Station",
    })

if acquisitions_remaining == 0 and days_in_week_remaining > 2:
    recommendations.append({
        "type": "save_adds",
        "severity": "warning",
        "message": "Zero adds remaining. Monitor IL spots for free moves.",
    })

if il_used < il_total and any(p for p in roster if p.get("selected_position") == "BN" and p.get("injury_note")):
    recommendations.append({
        "type": "il_efficiency",
        "severity": "info",
        "message": f"{il_total - il_used} open IL slot(s). Move injured BN players to IL to free roster space.",
    })
```

**Cost:** Low — pure rule evaluation on data already in scope.

---

## 5. Frontend Page Redesign

### New Layout (3-column on desktop, stacked on mobile)

```
┌─────────────────────────────────────────────────────────────┐
│  CONSTRAINT BUDGET                                    [Refresh]
├─────────────────┬──────────────────┬────────────────────────┤
│  BUDGET PANEL   │  IP FORECAST     │  RECOMMENDATIONS       │
│  (existing)     │  (new)           │  (new)                 │
│                 │                  │                        │
│  Acquisitions   │  ┌────────────┐  │  ⚠️ IP URGENCY         │
│  IL Slots       │  │ 0.0 / 18   │  │  Stream SPs now        │
│  IP Progress    │  │  projected │  │                        │
│                 │  │ 12.4 IP    │  │  ℹ️ IL Efficiency      │
│                 │  └────────────┘  │  Move injured BN→IL    │
│                 │  4 starts left   │                        │
├─────────────────┴──────────────────┴────────────────────────┤
│  RECENT TRANSACTIONS                                        │
│  ┌──────────┬──────────┬────────┬──────────┬────────────┐   │
│  │ Time     │ Player   │ Pos    │ Action   │ Direction  │   │
│  ├──────────┼──────────┼────────┼──────────┼────────────┤   │
│  │ Mon 2PM  │ Sánchez  │ SP     │ Add      │ +          │   │
│  │ Mon 10AM │ Crochet  │ SP     │ Drop     │ -          │   │
│  └──────────┴──────────┴────────┴──────────┴────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ACQUISITION ANALYTICS                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Batter/Pitcher│  │ By Category  │  │ vs League Avg    │   │
│  │ Pie Chart    │  │ Bar Chart    │  │ Dot Plot         │   │
│  │ 3 bat / 2 pit│  │ SV:3 ERA:2   │  │ You: 0 IP        │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Inventory

| Component | Source | Effort |
|-----------|--------|--------|
| `BudgetPanel` | Existing (Dashboard) | 0 |
| `IPForecastCard` | New | Small |
| `RecommendationList` | New | Small |
| `TransactionTable` | New | Medium |
| `AcquisitionBreakdown` | New | Medium |
| `LeagueContextChart` | New (recharts) | Medium |

---

## 6. Implementation Phases

| Phase | Scope | Backend | Frontend | Effort |
|-------|-------|---------|----------|--------|
| **1** | IP Forecast + Recommendations | `ip_projected`, `scheduled_starts_this_week`, `ip_on_pace_to_hit`, `recommendations[]` | `IPForecastCard`, `RecommendationList` | Small |
| **2** | Transaction History | `recent_transactions[]` | `TransactionTable` | Small |
| **3** | Acquisition Breakdown | `acquisition_breakdown` | `AcquisitionBreakdown` pie/bar chart | Medium |
| **4** | League Context | `league_median_ip`, `league_avg_acquisitions_used` (cached, opt-in) | `LeagueContextChart` | Medium |

**Recommendation:** Implement Phases 1–2 in a single PR. They require no new data sources (all data is already fetched in the budget endpoint) and deliver immediate user value. Phases 3–4 can be deferred to a follow-up sprint.

---

## 7. API Compatibility

- All new fields are **optional** (`?` / `Optional` / `None` default)
- Existing Dashboard `BudgetPanel` consumption is **unchanged**
- No URL or path changes — same `GET /api/fantasy/budget`
- Frontend feature-gates new sections with `budget.ip_projected != null` checks

---

## 8. Acceptance Criteria

### Phase 1–2 (MVP)
- [ ] `GET /api/fantasy/budget` returns `ip_projected`, `scheduled_starts_this_week`, `ip_on_pace_to_hit`
- [ ] `GET /api/fantasy/budget` returns `recommendations` array with at least "ip_urgency" and "il_efficiency" rules
- [ ] `GET /api/fantasy/budget` returns `recent_transactions` (last 10 add/drop rows)
- [ ] Budget page displays IP Forecast card with projection bar
- [ ] Budget page displays Recommendations panel with severity-colored badges
- [ ] Budget page displays Recent Transactions table
- [ ] UAT re-run: Budget page rated as "useful / not redundant with Dashboard"

### Phase 3–4 (Polish)
- [ ] Acquisition Breakdown chart renders on Budget page
- [ ] League Context comparison renders (optional, cached)
- [ ] All charts use Design System v2 color tokens (`ds-category-*`, `ds-status-*`)

---

## 9. Related Files

| File | Role |
|------|------|
| `backend/routers/fantasy.py:5902-6039` | `get_constraint_budget()` — expand response construction |
| `backend/services/scoreboard_orchestrator.py` | `compute_budget_state()` — may need new fields |
| `backend/schemas.py` | Add `BudgetRecommendation`, `TransactionRow` models |
| `frontend/lib/types.ts:579-605` | `BudgetData`, `BudgetResponse` interfaces |
| `frontend/app/(dashboard)/war-room/budget/page.tsx` | Page layout redesign |
| `frontend/components/dashboard/budget-panel.tsx` | Re-use as-is |
| `reports/2026-05-13-ui-uat-audit.md` | Original UAT finding |
