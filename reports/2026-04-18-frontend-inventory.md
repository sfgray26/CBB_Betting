# K5 Research Memo — Frontend Component Inventory & Archive Plan

**Produced by:** Kimi CLI (Deep Intelligence)  
**Date:** 2026-04-15  
**Scope:** `frontend/` — every source file, dependency, type contract, and backend endpoint mapping  

---

## 1. Full File Inventory with Purpose Classification

### 1.1 Pages (`frontend/app/(dashboard)/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `(dashboard)/page.tsx` | **Shared** | Root redirect (`/` → `/dashboard`). No CBB logic. |
| `(dashboard)/layout.tsx` | **Shared** | Shell: Sidebar + Header + main scroll area. Domain-agnostic. |
| `(dashboard)/dashboard/page.tsx` | **Fantasy-reusable** | Calls `/api/dashboard`. Already renders fantasy payload (`DashboardData`). Skeleton is generic. |
| `(dashboard)/decisions/page.tsx` | **Fantasy-reusable** | Calls `/api/fantasy/decisions` and `/api/fantasy/decisions/status`. Lineup/waiver logic is fantasy-native. |
| `(dashboard)/performance/page.tsx` | **CBB-only** | Betting P&L, ROI, win-rate, bankroll curve. Uses `/api/performance/*`. |
| `(dashboard)/clv/page.tsx` | **CBB-only** | Closing-line value analysis. Uses `/api/performance/clv-analysis`. |
| `(dashboard)/bet-history/page.tsx` | **CBB-only** | Bet log table. Uses `/api/bets`. |
| `(dashboard)/calibration/page.tsx` | **CBB-only** | Model calibration curve. Uses `/api/performance/calibration`. |
| `(dashboard)/alerts/page.tsx` | **CBB-only** | Betting/system alerts. Uses `/api/performance/alerts`. |
| `(dashboard)/today/page.tsx` | **CBB-only** | Daily CBB predictions (BET/CONSIDER/PASS). Uses `/api/predictions/today`. |
| `(dashboard)/live-slate/page.tsx` | **CBB-only** | Tabular view of all games. Uses `/api/predictions/today/all`. |
| `(dashboard)/odds-monitor/page.tsx` | **CBB-only** | Odds polling + portfolio exposure. Uses `/admin/odds-monitor/status`, `/admin/portfolio/status`. |
| `(dashboard)/bracket/page.tsx` | **CBB-only** | NCAA tournament Monte Carlo. Uses `/api/tournament/bracket-projection`. |
| `(dashboard)/bracket/error.tsx` | **CBB-only** | Bracket-specific error boundary. |
| `(dashboard)/bracket/loading.tsx` | **CBB-only** | Bracket skeleton. |
| `(dashboard)/today/error.tsx` | **CBB-only** | Today-page error boundary. |
| `(dashboard)/today/loading.tsx` | **CBB-only** | Today-page skeleton. |
| `(dashboard)/admin/page.tsx` | **CBB-only** | Risk dashboard: portfolio, ratings, scheduler, odds monitor, settlement trigger, bankroll. |
| `(dashboard)/admin/error.tsx` | **CBB-only** | Admin error boundary. |
| `(dashboard)/admin/loading.tsx` | **CBB-only** | Admin skeleton. |
| `login/page.tsx` | **Shared** | API-key login against `/health`. Domain-agnostic. |
| `layout.tsx` | **Shared** | Root layout (fonts, dark mode, metadata). |
| `globals.css` | **Shared** | Tailwind directives + zinc palette + scrollbar styling. |

### 1.2 Components (`frontend/components/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `layout/header.tsx` | **Shared** | Title bar + refresh + logout. `PAGE_TITLES` map is CBB-centric but easily extended. |
| `layout/sidebar.tsx` | **Shared** | Navigation drawer. `navSections` contains CBB routes but structure is generic. |
| `providers.tsx` | **Shared** | TanStack Query `QueryClientProvider`. |
| `shared/tooltip.tsx` | **Shared** | Generic Radix-UI tooltip wrapper. |
| `ui/alert.tsx` | **Shared** | Generic alert primitive (CVA). |
| `ui/badge.tsx` | **Shared** | Generic badge primitive. Includes betting-specific variants (`bet`, `consider`, `pass`) but also generic `win`/`loss`/`pending`. |
| `ui/button.tsx` | **Shared** | Generic button primitive (CVA). |
| `ui/card.tsx` | **Shared** | Card shell + header/title/value/content/description. |
| `ui/data-table.tsx` | **Shared** | Sortable table with generic `Column<T>` interface. Used by both CBB and fantasy pages. |
| `ui/error-boundary.tsx` | **Shared** | Generic class-component error boundary. |
| `ui/input.tsx` | **Shared** | Generic input primitive. |
| `ui/kpi-card.tsx` | **Shared** | KPI card with trend icon. Used by performance, CLV, odds-monitor, bracket, dashboard. |
| `ui/label.tsx` | **Shared** | Generic label primitive (Radix). |
| `ui/progress.tsx` | **Shared** | Generic progress bar (Radix). Unused today but kept. |
| `ui/select.tsx` | **Shared** | Generic select primitive (Radix). Unused today. |
| `ui/separator.tsx` | **Shared** | Generic separator (Radix). Unused today. |
| `ui/skeleton.tsx` | **Shared** | Generic skeleton placeholder. Unused today (pages use inline `animate-pulse` divs). |
| `ui/slider.tsx` | **Shared** | Generic slider (Radix). Unused today. |
| `ui/switch.tsx` | **Shared** | Generic switch (Radix). Unused today. |
| `ui/tabs.tsx` | **Shared** | Generic tabs (Radix). Unused today. |

### 1.3 Library (`frontend/lib/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `lib/api.ts` | **Mixed — keep** | Contains both CBB endpoints (`/api/performance/*`, `/api/bets`, `/api/predictions/*`, `/api/tournament/*`, `/admin/*`) and fantasy endpoints (`/api/dashboard`, `/api/dashboard/streaks`, `/api/dashboard/waiver-targets`, `/api/fantasy/decisions`, `/api/fantasy/decisions/status`, `/api/fantasy/lineup/async-optimize`, `/api/fantasy/jobs/*`). |
| `lib/types.ts` | **Mixed — keep** | Contains both CBB types (`BetLog`, `ClvBetEntry`, `CalibrationBucket`, `Alert`, `PredictionEntry`, `TodaysPredictionsResponse`, `OddsMonitorStatus`, `PortfolioStatusFull`, `BracketProjection`, `SchedulerStatus`, `RatingsStatus`) and fantasy types (`DashboardData`, `DashboardResponse`, `StreakPlayer`, `WaiverTarget`, `LineupGap`, `InjuryFlag`, `MatchupPreviewData`, `ProbablePitcherInfo`, `DecisionResultOut`, `DecisionExplanationOut`, `DecisionWithExplanation`, `DecisionsResponse`, `DecisionPipelineStatus`). |
| `lib/utils.ts` | **Shared** | `cn()` — Tailwind class merge. |
| `lib/constants.ts` | **Shared** | Empty export. Placeholder. |
| `lib/query-client.ts` | **Shared** | TanStack Query client config. |

### 1.4 Config & Tests

| File | Classification |
|------|---------------|
| `package.json` | **Shared** |
| `next.config.ts` | **Shared** |
| `tailwind.config.ts` | **Shared** |
| `middleware.ts` | **Shared** |
| `postcss.config.mjs` | **Shared** |
| `playwright.config.ts` | **Shared** |
| `tests/e2e/smoke.spec.ts` | **Shared** |
| `public/manifest.json` | **Shared** |
| `.env.local` / `.env.local.example` | **Shared** |

---

## 2. Kept Components — Props, API Endpoints, Backend Existence

### 2.1 Fantasy-Reusable Pages

#### `dashboard/page.tsx`
- **Props:** None (page component).
- **API:** `GET /api/dashboard` → `DashboardResponse`
- **Backend exists?** ✅ Yes (`@app.get("/api/dashboard")` in `backend/main.py`).
- **Notes:** Currently renders a minimal placeholder. The full `DashboardData` shape (lineup gaps, streaks, waiver targets, injury flags, matchup preview, probable pitchers) is defined in `types.ts` but the UI only shows a status card and user ID.

#### `decisions/page.tsx`
- **Props:** None (page component).
- **APIs:**
  - `GET /api/fantasy/decisions?decision_type=&as_of_date=&limit=` → `DecisionsResponse`
  - `GET /api/fantasy/decisions/status` → `DecisionPipelineStatus`
- **Backend exists?** ✅ Yes (both routes present in `backend/main.py`).
- **Notes:** Contains hard-coded `LINEUP_SLOT_ORDER` and `LINEUP_SLOT_CAPACITY` constants. These are frontend-side layout rules that may drift from backend contract.

### 2.2 Shared Infrastructure

#### `layout/header.tsx`
- **Props:** `onMenuClick?: () => void`
- **API:** None directly. Uses `queryClient.invalidateQueries()` for global refresh.
- **Backend exists?** N/A
- **Notes:** `PAGE_TITLES` is a static map. Fantasy pages will need entries added here.

#### `layout/sidebar.tsx`
- **Props:** `isOpen?: boolean; onClose?: () => void`
- **API:** `GET /admin/portfolio/status` (for drawdown chip in footer)
- **Backend exists?** ✅ Yes (`/admin/portfolio/status`).
- **Notes:** `navSections` is hard-coded. CBB routes will be archived; fantasy routes need expansion.

#### `ui/data-table.tsx`
- **Props:** `columns: Column<T>[]`, `data: T[]`, `keyExtractor: (row: T) => string | number`, `className?: string`, `emptyMessage?: string`
- **API:** None.
- **Backend exists?** N/A
- **Notes:** Fully generic. Used by `performance`, `clv`, `bet-history`, `calibration`, `live-slate`, `bracket`, `decisions`.

#### `ui/kpi-card.tsx`
- **Props:** `title: string`, `value: string | number`, `unit?: string`, `delta?: number`, `deltaLabel?: string`, `trend?: 'up' | 'down' | 'neutral'`, `loading?: boolean`, `valueClassName?: string`
- **API:** None.
- **Backend exists?** N/A
- **Notes:** Used across CBB and fantasy pages. No domain coupling.

#### `ui/card.tsx` family
- **Props:** Standard `React.HTMLAttributes` + `children`.
- **API:** None.
- **Backend exists?** N/A
- **Notes:** `Card`, `CardHeader`, `CardTitle`, `CardValue`, `CardContent`, `CardDescription`.

#### `ui/badge.tsx`
- **Props:** `variant?: 'bet' | 'consider' | 'pass' | 'win' | 'loss' | 'push' | 'pending' | 'confirmed' | 'caution' | 'volatile' | 'default' | 'secondary'`
- **API:** None.
- **Backend exists?** N/A
- **Notes:** Betting-specific variants (`bet`, `consider`, `pass`) are CBB-legacy. Fantasy can reuse `win`/`loss`/`pending`/`default` or add new variants.

#### `providers.tsx`
- **Props:** `children: React.ReactNode`
- **API:** None.
- **Backend exists?** N/A

#### `login/page.tsx`
- **Props:** None.
- **API:** `GET /health` (key validation)
- **Backend exists?** ✅ Yes.

---

## 3. TypeScript Types vs. Phase 0 Contracts (`backend/contracts.py`)

### 3.1 Types that ALIGN with Phase 0 contracts

| Frontend Type (`lib/types.ts`) | Contract Model (`backend/contracts.py`) | Alignment |
|-------------------------------|----------------------------------------|-----------|
| `FreshnessMetadata` | `FreshnessMetadata` (P0-3) | ✅ Match |
| `CategoryStats` | `CategoryStats` (P0-4) | ✅ Match |
| `MatchupScoreboardRow` (not yet in `types.ts`) | `MatchupScoreboardRow` (P0-5) | ❌ **Missing** |
| `MatchupScoreboardResponse` (not yet in `types.ts`) | `MatchupScoreboardResponse` (P0-5) | ❌ **Missing** |
| `PlayerGameContext` (not yet in `types.ts`) | `PlayerGameContext` (P0-6) | ❌ **Missing** |
| `CanonicalPlayerRow` (not yet in `types.ts`) | `CanonicalPlayerRow` (P0-6) | ❌ **Missing** |
| `ConstraintBudget` (not yet in `types.ts`) | `ConstraintBudget` (P0-2) | ❌ **Missing** |
| `CategoryStatusTag` (not yet in `types.ts`) | `CategoryStatusTag` (P0-1) | ❌ **Missing** |

### 3.2 Types that are STALE or PARTIAL

| Frontend Type | Issue |
|--------------|-------|
| `DashboardData` | Contains fields that mirror contract shapes (`lineup_gaps`, `hot_streaks`, `waiver_targets`, `injury_flags`, `matchup_preview`, `probable_pitchers`) but are **not** typed as canonical contract instances. No `FreshnessMetadata` wrapper. No `ConstraintBudget`. |
| `MatchupPreviewData` | Hand-rolled frontend type. Does not use `MatchupScoreboardResponse` from contracts. Missing `CategoryStatusTag`, `ConstraintBudget`, and per-category rows. |
| `StreakPlayer` / `WaiverTarget` / `InjuryFlag` / `LineupGap` / `ProbablePitcherInfo` | Ad-hoc shapes. Not derived from `CanonicalPlayerRow` or `PlayerValuationReport`. Will diverge as backend evolves. |
| `DecisionResultOut` / `DecisionExplanationOut` / `DecisionWithExplanation` / `DecisionsResponse` | These align with Layer-3F decision outputs, but **do not reference** `PlayerSlot`, `ExecutionDecision`, or `LineupOptimizationRequest` from contracts. The `target_slot` string is unchecked against canonical position codes. |

### 3.3 Types that are CBB-ONLY (archive candidates)

| Type | Used By |
|------|---------|
| `BetLog` | `bet-history/page.tsx`, `lib/api.ts` |
| `ClvBetEntry` | `clv/page.tsx`, `lib/api.ts` |
| `CalibrationBucket` | `calibration/page.tsx`, `lib/api.ts` |
| `Alert` / `LiveAlert` | `alerts/page.tsx`, `lib/api.ts` |
| `GameData` / `PredictionEntry` / `TodaysPredictionsResponse` | `today/page.tsx`, `live-slate/page.tsx`, `lib/api.ts` |
| `OddsMonitorStatus` | `odds-monitor/page.tsx`, `admin/page.tsx`, `lib/api.ts` |
| `PortfolioStatusFull` | `odds-monitor/page.tsx`, `admin/page.tsx`, `lib/api.ts` |
| `BracketProjection` / `UpsetAlert` / `TeamAdvancement` / `AsyncJobStatus` | `bracket/page.tsx`, `lib/api.ts` |
| `SchedulerStatus` / `SchedulerJob` | `admin/page.tsx`, `lib/api.ts` |
| `RatingsStatus` / `RatingSourceStatus` | `admin/page.tsx`, `lib/api.ts` |

---

## 4. Exact Archive List (`frontend/_archive_cbb/`)

When Phase 5 decommissioning starts, move **only** the CBB-only pages and their CBB-specific error/loading shells. Do **not** archive shared UI primitives or fantasy pages.

```
frontend/_archive_cbb/
├── app/
│   ├── (dashboard)/
│   │   ├── performance/
│   │   │   └── page.tsx
│   │   ├── clv/
│   │   │   └── page.tsx
│   │   ├── bet-history/
│   │   │   └── page.tsx
│   │   ├── calibration/
│   │   │   └── page.tsx
│   │   ├── alerts/
│   │   │   └── page.tsx
│   │   ├── today/
│   │   │   ├── page.tsx
│   │   │   ├── error.tsx
│   │   │   └── loading.tsx
│   │   ├── live-slate/
│   │   │   └── page.tsx
│   │   ├── odds-monitor/
│   │   │   └── page.tsx
│   │   ├── bracket/
│   │   │   ├── page.tsx
│   │   │   ├── error.tsx
│   │   │   └── loading.tsx
│   │   └── admin/
│   │       ├── page.tsx
│   │       ├── error.tsx
│   │       └── loading.tsx
│   └── login/
│       └── page.tsx          ← keep in place (shared)
├── lib/
│   └── types.ts              ← keep in place (mixed)
│   └── api.ts                ← keep in place (mixed; prune CBB endpoints in-place)
```

**Cleanup actions at archive time:**
1. Remove CBB entries from `navSections` in `components/layout/sidebar.tsx`.
2. Remove CBB titles from `PAGE_TITLES` in `components/layout/header.tsx`.
3. Delete CBB endpoint functions from `lib/api.ts` (`clvAnalysis`, `performanceSummary`, `performanceTimeline`, `bets`, `calibration`, `alerts`, `acknowledgeAlert`, `portfolioStatus`, `todaysPredictions`, `todaysPredictionsAll`, `oddsMonitorStatus`, `portfolioStatusFull`, `bracketProjection`, `asyncOptimizeLineup`, `getJobStatus`, `schedulerStatus`, `ratingsStatus`, `featureFlags`, `setFeatureFlag`).
4. Delete CBB-only TypeScript interfaces from `lib/types.ts`.

---

## 5. Component Dependency Graph

### 5.1 The Six Canonical Fantasy Pages

Based on `backend/contracts.py` (P0 contracts), the existing `/decisions` page, and the `sidebar.tsx` roadmap grouping, the six canonical fantasy pages are:

1. **Dashboard** (`/dashboard`) — Already exists. Aggregates lineup gaps, streaks, waiver targets, injuries, matchup preview, pitchers.
2. **Daily Decisions** (`/decisions`) — Already exists. Lineup + waiver recommendations with explanations.
3. **Scoreboard** (`/scoreboard` or `/matchup`) — **Not built yet.** Backend contract: `MatchupScoreboardResponse` (P0-5). Backend endpoint: `GET /api/fantasy/matchup`.
4. **Lineup** (`/lineup`) — **Not built yet.** Backend endpoint: `GET /api/fantasy/lineup/{lineup_date}`. Returns `DailyLineupResponse`.
5. **Waiver** (`/waiver`) — **Not built yet.** Backend endpoint: `GET /api/fantasy/waiver`. Returns `WaiverWireResponse`.
6. **Roster / Player Board** (`/roster` or `/player-board`) — **Not built yet.** Backend endpoints: `GET /api/fantasy/roster` (returns `RosterResponse`) and `GET /api/fantasy/draft-board` / `GET /api/fantasy/player/{player_id}`. Phase 0 contract: `CanonicalPlayerRow` (P0-6).

### 5.2 Dependency Graph (pages → components → APIs)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SHARED SHELL                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ layout.tsx      │───→│ Sidebar         │───→│ /admin/portfolio/status │  │
│  │ (dashboard)     │    │ (layout/sidebar)│    │ (drawdown chip)         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│           │                                                                  │
│           └──────────────────────────────────────────────────────────────→  │
│                              Header (layout/header)                          │
│                              ↓ queryClient.invalidateQueries()               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   ┌────▼────┐                ┌─────▼─────┐              ┌──────▼──────┐
   │ Dashboard│                │ Decisions │              │ Scoreboard  │
   │ /dashboard│               │ /decisions│              │ (future)    │
   └────┬────┘                └─────┬─────┘              └──────┬──────┘
        │                           │                           │
   ┌────▼────┐                ┌─────▼─────┐              ┌──────▼──────┐
   │ Card    │                │ Card      │              │ Card        │
   │ KpiCard │                │ Badge     │              │ DataTable   │
   │ DataTable│               │ DataTable │              │ KpiCard     │
   │ (minimal)│               │ (inline)  │              │             │
   └────┬────┘                └─────┬─────┘              └──────┬──────┘
        │                           │                           │
   ┌────▼────┐                ┌─────▼─────┐              ┌──────▼──────┐
   │ GET /api/dashboard       │ GET /api/fantasy/decisions         │ GET /api/fantasy/matchup
   │ GET /api/dashboard/streaks│ GET /api/fantasy/decisions/status │ (MatchupScoreboardResponse)
   │ GET /api/dashboard/waiver-targets│                        │
   └─────────────────────────┘ └─────────────────────────┘ └────────────────┘

   ┌──────────┐                ┌───────────┐              ┌──────────────┐
   │ Lineup   │                │ Waiver    │              │ Roster /     │
   │ (future) │                │ (future)  │              │ Player Board │
   │ /lineup  │                │ /waiver   │              │ (future)     │
   └────┬─────┘                └─────┬─────┘              └──────┬───────┘
        │                            │                            │
   ┌────▼─────┐                ┌─────▼─────┐              ┌───────▼───────┐
   │ Card     │                │ Card      │              │ Card          │
   │ DataTable│                │ DataTable │              │ DataTable     │
   │ Badge    │                │ Badge     │              │ Badge         │
   │ KpiCard  │                │ KpiCard   │              │ Tooltip       │
   └────┬─────┘                └─────┬─────┘              └───────┬───────┘
        │                            │                            │
   ┌────▼────────────────┐     ┌─────▼────────────────┐    ┌──────▼──────────────┐
   │ GET /api/fantasy/lineup/{date}│ GET /api/fantasy/waiver │ GET /api/fantasy/roster
   │ POST /api/fantasy/lineup      │ GET /api/fantasy/waiver/recommendations
   │ POST /api/fantasy/lineup/async-optimize                    │ GET /api/fantasy/player/{id}
   │ GET /api/fantasy/jobs/{job_id}│                         │ GET /api/fantasy/draft-board
   └─────────────────────────┘     └─────────────────────────┘ └───────────────────────┘
```

### 5.3 Key Observations for Build Planning

- **No `hooks/` directory exists today.** All data fetching is inline `useQuery` in page components. A future refactor could extract `useDashboard()`, `useDecisions()`, `useMatchup()` hooks into `frontend/hooks/`.
- **No dedicated fantasy layout variant.** The current `(dashboard)/layout.tsx` is sufficient; fantasy pages live inside the same shell.
- **`DataTable` is the workhorse.** Five of the six canonical pages (all except Dashboard) will likely need it. It is already generic and battle-tested.
- **`KpiCard` is reusable but under-used.** Dashboard currently does not render the rich `DashboardData` fields into KPI cards. That is a pending UI enhancement.
- **Backend endpoints exist for all 6 canonical pages.** The only missing frontend pages are Scoreboard, Lineup, Waiver, and Roster/Player Board.
- **`CanonicalPlayerRow` is not yet reflected in `lib/types.ts`.** Before building Roster/Player Board or Scoreboard pages, `lib/types.ts` must be updated to import/include the P0-5 and P0-6 contract shapes.

---

## Appendix: Backend Endpoint Existence Check

| Endpoint | Used By | Exists in `backend/main.py`? |
|----------|---------|------------------------------|
| `GET /api/dashboard` | `dashboard/page.tsx`, `lib/api.ts` | ✅ |
| `GET /api/dashboard/streaks` | `lib/api.ts` | ✅ |
| `GET /api/dashboard/waiver-targets` | `lib/api.ts` | ✅ |
| `GET /api/fantasy/decisions` | `decisions/page.tsx`, `lib/api.ts` | ✅ |
| `GET /api/fantasy/decisions/status` | `decisions/page.tsx`, `lib/api.ts` | ✅ |
| `GET /api/fantasy/matchup` | (future Scoreboard) | ✅ |
| `GET /api/fantasy/lineup/{date}` | (future Lineup) | ✅ |
| `GET /api/fantasy/waiver` | (future Waiver) | ✅ |
| `GET /api/fantasy/roster` | (future Roster) | ✅ |
| `GET /api/fantasy/player/{player_id}` | (future Player Board) | ✅ |
| `GET /api/fantasy/draft-board` | (future Player Board) | ✅ |
| `POST /api/fantasy/lineup/async-optimize` | `lib/api.ts` | ✅ |
| `GET /api/fantasy/jobs/{job_id}` | `lib/api.ts` | ✅ |
| `GET /admin/portfolio/status` | `sidebar.tsx`, `lib/api.ts` | ✅ |
| `GET /health` | `login/page.tsx` | ✅ |
| `GET /api/performance/summary` | `performance/page.tsx` | ✅ |
| `GET /api/performance/clv-analysis` | `clv/page.tsx` | ✅ |
| `GET /api/performance/calibration` | `calibration/page.tsx` | ✅ |
| `GET /api/performance/timeline` | `performance/page.tsx` | ✅ |
| `GET /api/bets` | `bet-history/page.tsx` | ✅ |
| `GET /api/performance/alerts` | `alerts/page.tsx` | ✅ |
| `POST /admin/alerts/{id}/acknowledge` | `alerts/page.tsx` | ✅ |
| `GET /api/predictions/today` | `today/page.tsx` | ✅ |
| `GET /api/predictions/today/all` | `live-slate/page.tsx` | ✅ |
| `GET /api/tournament/bracket-projection` | `bracket/page.tsx` | ✅ |
| `GET /admin/odds-monitor/status` | `odds-monitor/page.tsx`, `admin/page.tsx` | ✅ |
| `GET /admin/ratings/status` | `admin/page.tsx` | ✅ |
| `GET /admin/scheduler/status` | `admin/page.tsx` | ✅ |
| `GET /admin/bankroll` | `admin/page.tsx` | ✅ |
| `POST /admin/bankroll` | `admin/page.tsx` | ✅ |
| `POST /admin/force-update-outcomes` | `admin/page.tsx` | ✅ |
| `GET /api/feature-flags` | `lib/api.ts` | ✅ |
| `POST /admin/feature-flags/{flag}` | `lib/api.ts` | ✅ |

---

*End of memo.*
