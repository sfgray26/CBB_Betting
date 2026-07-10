# Frontend Reliability & Performance Audit

**Project:** CBB Edge / Fantasy Baseball Frontend  
**Date:** 2026-07-10  
**Scope:** `frontend/` — Next.js 15.5.13 App Router, React 19, TypeScript 5.7.2, Tailwind CSS 3.4.17  
**Methodology:** Read-only static code review. No files were modified.

---

## Executive Summary

The frontend is a modern Next.js App Router application with a consistent Tailwind-based design system and a well-configured TanStack Query client. The biggest reliability risks are **testing-infrastructure drift**, **query-key inconsistency across pages**, and **large, single-file components** that bundle data fetching, mutation logic, and UI. Performance is generally acceptable for an internal dashboard, but synchronous `recharts` imports and the absence of bundle analysis leave room for improvement.

| Rank | Risk | Severity |
|------|------|----------|
| 1 | Jest-style unit test exists with no Jest runner / script | High |
| 2 | Streaming action modal uses hardcoded drop options and position | High |
| 3 | Inline query keys bypass the canonical `FANTASY_QUERY_KEYS` list | Medium |
| 4 | Several pages are >500-line “god components” | Medium |
| 5 | `recharts` is bundled synchronously on analytics pages | Medium |
| 6 | Error/loading boundaries are inconsistent across routes | Medium |
| 7 | Dense analytics tables may be hard to use on mobile | Medium |

---

## 1. Hydration & Rendering Stability

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-HYD-01 | Info | Multiple | — | All `Date`, `localStorage`, and cookie access is guarded inside `'use client'` components or `useEffect`. No obvious server/client mismatch sources. |
| F-HYD-02 | Info | `app/layout.tsx` | 1-30 | Fonts use `preload: false` and `display: 'swap'`, avoiding build-time network fetches and layout-shift risks. |
| F-HYD-03 | Info | `app/(dashboard)/dashboard/page.tsx` | 1-50 | Each dashboard widget is wrapped in its own `ErrorBoundary` + `Suspense` skeleton on the server. |
| F-HYD-04 | Info | `components/freshness/freshness-badge.tsx` | 80-95 | Auto-polling is gated by `enablePolling` and `onRefresh`; component is `'use client'`, so no SSR mismatch. |

**Details**
- `lib/auth.ts` guards `window` / `document` access with `typeof window === 'undefined'` checks (lines 17, 38, 52).
- `middleware.ts` reads the `cbb_api_key` cookie at the edge; no client-side hydration dependency.
- `StreamingPage` initializes `targetDate` with `new Date()` inside a client component state initializer. Because it is `'use client'`, this runs only on the client, so hydration is safe.

**Recommendation:** Maintain current guards. Avoid moving date or storage access into server components.

---

## 2. State Management & Data Fetching

### 2.1 Query-key inconsistency

The shared `FANTASY_QUERY_KEYS` object (`lib/query-client.ts`, lines 31-59) is intended to be the canonical key registry, but many pages inline their own keys. `invalidateAllFantasyCaches()` only invalidates the canonical list, so a global refresh can miss page-specific data.

| Inline key | File | Line | Canonical key? |
|------------|------|------|----------------|
| `['portfolio']` | `components/layout/sidebar.tsx` | 98 | No |
| `['portfolio-full']` | `app/(dashboard)/admin/page.tsx` | 62 | No |
| `['ratings-status']` | `app/(dashboard)/admin/page.tsx` | 150 | No |
| `['scheduler-status']` | `app/(dashboard)/admin/page.tsx` | 222 | No |
| `['odds-monitor-status']` | `app/(dashboard)/admin/page.tsx` | 281 | No |
| `['bankroll-current']` | `app/(dashboard)/admin/page.tsx` | 424 | No |
| `['bracket-projection', nSims]` | `app/(dashboard)/bracket/page.tsx` | 252 | No |
| `['decisions', ...]` | `app/(dashboard)/decisions/page.tsx` | 477, 497, 503 | No |
| `['odds-monitor-status']` | `app/(dashboard)/odds-monitor/page.tsx` | 55 | No |
| `['todays-predictions']` | `app/(dashboard)/today/page.tsx` | 388 | No |
| `['matchup']` | `app/(dashboard)/war-room/page.tsx` | 84 | Yes |
| `['projection-status']` | `app/(dashboard)/war-room/page.tsx` | 91 | Yes |
| `['global-freshness']` | `app/(dashboard)/war-room/page.tsx` | 98 | Yes |
| `['performance-summary']` | `app/(dashboard)/performance/page.tsx` | 41 | No |
| `['performance-timeline']` | `app/(dashboard)/performance/page.tsx` | 48 | No |
| `['roster']` | `app/(dashboard)/war-room/roster/page.tsx` | 880 | Yes |
| `['global-freshness']` | `app/(dashboard)/war-room/roster/page.tsx` | 886 | Yes |
| `['budget']` | `app/(dashboard)/war-room/roster/page.tsx` | 893 | Yes |
| `['scoreboard']` | `app/(dashboard)/war-room/roster/page.tsx` | 899 | Yes |
| `['dashboard-streaks']` | `app/(dashboard)/war-room/roster/page.tsx` | 906 | Yes |
| `['waiver-recommendations']` | `app/(dashboard)/war-room/waiver/page.tsx` | 603 | Yes |
| `['global-freshness']` | `app/(dashboard)/war-room/waiver/page.tsx` | 650 | Yes |
| `['waiver', sort]` | `app/(dashboard)/war-room/waiver/page.tsx` | 657 | No |
| `['alerts']` | `app/(dashboard)/alerts/page.tsx` | 142 | No |
| `['bets', status, days]` | `app/(dashboard)/bet-history/page.tsx` | 46 | No |
| `['canonicalProjections']` | `app/(dashboard)/war-room/roster-lab/page.tsx` | 9 | No |
| `['clv-analysis']` | `app/(dashboard)/clv/page.tsx` | 51 | No |
| `['calibration']` | `app/(dashboard)/calibration/page.tsx` | 38 | No |
| `['streaming-recommendations', targetDate]` | `components/streaming/streaming-recommendations.tsx` | 34 | Yes (base) |
| `["dashboard"]` | `app/(dashboard)/dashboard/_components/dashboard-client.tsx` | 135 | No |
| `["global-freshness"]` | `app/(dashboard)/dashboard/_components/dashboard-client.tsx` | 152 | Yes |
| `["dashboard-waiver-targets"]` | `app/(dashboard)/dashboard/_components/dashboard-client.tsx` | 449 | No |
| `["dashboard-streaks"]` | `app/(dashboard)/dashboard/_components/dashboard-client.tsx` | 538 | Yes |
| `["budget"]` | `app/(dashboard)/dashboard/_components/dashboard-client.tsx` | 610 | Yes |

**Severity:** Medium  
**Evidence:** `lib/query-client.ts` lines 65-76 only invalidates canonical keys.

```ts
export function invalidateAllFantasyCaches(): void {
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.matchup })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.scoreboard })
  // ... only the 9 canonical keys
}
```

**Recommendation:** Migrate every query to `FANTASY_QUERY_KEYS` (or a matching per-domain registry). Add a lint rule or TypeScript constraint that discourages inline string keys.

### 2.2 Broad invalidation in header

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-DATA-02 | Medium | `components/layout/header.tsx` | 54 | `handleRefresh` calls `queryClient.invalidateQueries()` with no filter, invalidating the entire cache at once. |

```tsx
function handleRefresh() {
  setRefreshing(true)
  queryClient.invalidateQueries()  // line 54
  setLastRefresh(Date.now())
  setSecondsAgo(0)
  setTimeout(() => setRefreshing(false), 800)
}
```

**Recommendation:** Scope header refresh to the current route’s relevant keys or reuse `invalidateAllFantasyCaches()` if the intent is fantasy-only refresh.

### 2.3 Redundant refetch after mutation

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-DATA-03 | Low | `app/(dashboard)/war-room/roster/page.tsx` | 976-979 | `moveMutation` calls `invalidateQueries` and then `refetchQueries` for the same key. |

```tsx
void queryClient.invalidateQueries({ queryKey: ['roster'] }).then(() =>
  queryClient.refetchQueries({ queryKey: ['roster'] })
)
```

**Recommendation:** Use `invalidateQueries` alone and let React Query refetch stale queries automatically.

### 2.4 Hardcoded streaming action modal options

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-DATA-04 | High | `components/streaming/action-modal.tsx` | 44-45, 178-180 | Drop-down options are hardcoded placeholder players; the position sent to the API is always `'P'`. |

```tsx
// line 44-45
rosterAction({
  action: 'ADD',
  add_player_id: `bdl.${pitcher.bdl_player_id}`,
  position: 'P',
  drop_player_id: selectedDrop || undefined,
})

// lines 178-180
<option value="example.1">Example Player 1 (0.2 z-score)</option>
<option value="example.2">Example Player 2 (-0.5 z-score)</option>
```

**Recommendation:** Fetch the current roster to populate real drop candidates and derive the correct position from the added player.

### 2.5 Raw `fetch` bypasses API client

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-DATA-05 | Medium | `app/(dashboard)/today/page.tsx` | 101-145 | `logPlacedBet` uses raw `fetch` instead of `apiFetch`, bypassing the 30-second timeout, 401 redirect, and JSON error parsing. |

```ts
const brRes = await fetch(`${BASE_URL}/admin/bankroll`, { ... })
// ...
const res = await fetch(`${BASE_URL}/api/bets/log`, { ... })
```

**Recommendation:** Route these calls through `apiFetch` or extend `endpoints` with bet-logging helpers.

### 2.6 No custom data hooks

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-DATA-06 | Medium | Many | — | Every page inlines `useQuery` / `useMutation` definitions. Roster page alone defines 5+ queries and 4+ mutations. |

**Recommendation:** Extract domain hooks such as `useRoster()`, `useWaiver()`, `useStreaming(targetDate)`, and `useGlobalFreshness()`. This reduces duplication and makes query-key changes safer.

---

## 3. Loading States, Error Boundaries, & User Feedback

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-UX-01 | Info | `app/(dashboard)/dashboard/page.tsx` | 1-50 | Each widget is isolated behind its own `ErrorBoundary` and `Suspense` skeleton. |
| F-UX-02 | Medium | `app/(dashboard)/...` | — | `error.tsx` / `loading.tsx` exist only for `today`, `bracket`, and `admin`. Other routes rely on inline `isLoading` / `isError` checks or component-level `ErrorBoundary`. |
| F-UX-03 | Low | `app/(dashboard)/today/loading.tsx`, `bracket/loading.tsx`, `admin/loading.tsx` | — | Loading skeletons use hardcoded `bg-zinc-800` / dark text colors, inconsistent with the light-theme token system used in `globals.css` and `tailwind.config.ts`. |
| F-UX-04 | Low | `components/error-boundary.tsx` | 1-50 | The “Try again” button only clears local error state; it does not reset React Query or re-fetch, so transient fetch errors may reappear only after a manual page refresh. |

**Details**
- `today/error.tsx`, `bracket/error.tsx`, and `admin/error.tsx` provide App Router error boundaries with `reset`.
- `war-room/roster/page.tsx` is the largest page but has no route-level `loading.tsx` or `error.tsx`; it wraps itself in the reusable `ErrorBoundary` and renders inline spinners.
- `WaiverPage` wraps an inner component in `ErrorBoundary` but provides no skeleton beyond a centered spinner.

**Recommendation:** Standardize on route-level `loading.tsx` / `error.tsx` for all data-heavy routes. Make the reusable `ErrorBoundary` accept an optional `onReset` callback to invalidate related queries.

---

## 4. Mobile Responsiveness

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-MOB-01 | Info | `app/(dashboard)/layout.tsx` | 1-20 | Responsive sidebar drawer with mobile overlay and `md:ml-60` / `min-w-0` containment. |
| F-MOB-02 | Medium | `components/ui/data-table.tsx`, `app/(dashboard)/bracket/page.tsx`, `bet-history/page.tsx`, `live-slate/page.tsx`, `war-room/waiver/page.tsx`, `war-room/roster/page.tsx` | — | Tables rely on generic horizontal scroll (`overflow-x-auto`) but contain many columns that are hard to scan on small screens. |
| F-MOB-03 | Low | `app/(dashboard)/war-room/roster/page.tsx` | — | Many inline panels stack on mobile but the page is extremely dense; touch targets and information hierarchy could be improved. |

**Evidence**
```tsx
// components/ui/data-table.tsx
<div className={cn('overflow-x-auto', className)}>
  <table className="w-full text-sm">...</table>
</div>
```

**Recommendation:** For the densest tables (bracket advancement, live slate, bet history, waiver wire), consider:
- Sticky first column with horizontal scroll.
- Column visibility toggles.
- Card-based mobile layouts that hide secondary stats behind expansion.
- Touch-friendly min-widths on action buttons.

---

## 5. Performance & Bundle Size

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-PERF-01 | Medium | `app/(dashboard)/performance/page.tsx`, `clv/page.tsx`, `calibration/page.tsx` | 7, 13, 13 | `recharts` is imported synchronously on three analytics routes. It is one of the larger charting dependencies and will be included in the main bundle unless code-split. |
| F-PERF-02 | Medium | Multiple | — | Several pages are >500 lines and bundle data fetching, mutations, derived state, and UI with no lazy loading. |
| F-PERF-03 | Low | `components/freshness/freshness-badge.tsx` | 84-95 | Auto-poll interval fires every 30 seconds. Only one badge is currently mounted in the header, but future reuse could duplicate polling. |
| F-PERF-04 | Low | Multiple | — | Very limited use of `React.memo` / `useMemo`. Large parent components will re-render entire sub-trees on local state changes. |
| F-PERF-05 | Info | `lib/query-client.ts` | 12-25 | Default query options are reasonable: `staleTime: 5 min`, `gcTime: 30 min`, `retry: 1`, `refetchOnWindowFocus: true`. |

**Evidence**
```ts
// app/(dashboard)/performance/page.tsx
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
```

**Recommendation:**
- Wrap chart pages in `dynamic(() => import(...), { ssr: false })` to split `recharts` and avoid server rendering issues.
- Add `@next/bundle-analyzer` to confirm bundle impact.
- Consider `React.memo` for expensive table rows (e.g., `PlayerRow`, `BetRow`) and memoized column definitions.

---

## 6. Maintainability & God Components

| ID | Severity | File | Lines | Primary Responsibilities |
|----|----------|------|-------|--------------------------|
| F-MAINT-01 | High | `app/(dashboard)/war-room/roster/page.tsx` | 1,296 | Roster query, budget, scoreboard, streaks, move/optimize/bulk mutations, optimistic updates, plus 10+ inline UI sub-components. |
| F-MAINT-02 | High | `app/(dashboard)/war-room/waiver/page.tsx` | 805 | Waiver queries, recommendations, global freshness, position filters, sort, and inline player/recommendation cards. |
| F-MAINT-03 | Medium | `app/(dashboard)/dashboard/_components/dashboard-client.tsx` | 673 | Header, lineup status, waiver targets, streaks, budget, probable pitchers, two-start pitchers, plus global refresh orchestration. |
| F-MAINT-04 | Medium | `app/(dashboard)/decisions/page.tsx` | 579 | Decision list, filters, status, accuracy, chart, and export. |
| F-MAINT-05 | Medium | `app/(dashboard)/admin/page.tsx` | 506 | Portfolio, ratings, scheduler, odds monitor, bankroll, batch action forms. |
| F-MAINT-06 | Medium | `components/streaming/streaming-recommendations.tsx` | 448 | Streaming recommendations query, tier filter, sort, expandable rows, auto-stream toggle, action modal trigger. |
| F-MAINT-07 | Medium | `app/(dashboard)/today/page.tsx` | 463 | Predictions, KPIs, bet placement, raw fetch logging, and confidence parsing. |
| F-MAINT-08 | Medium | `app/(dashboard)/bracket/page.tsx` | 461 | Bracket projection query, simulation controls, Final Four / upset alerts, advancement table. |
| F-MAINT-09 | Low | `components/yahoo-roster-view.tsx` | 422 | Standalone roster display; large but focused. |
| F-MAINT-10 | Low | `components/war-room/category-battlefield.tsx` | 372 | Category battlefield visualization. |

**Additional concern**
- `lib/types.ts` is 848 lines. While type files can be large, consider splitting into domain-specific type modules (`fantasy.types.ts`, `betting.types.ts`, etc.) as the surface grows.

**Recommendation:**
- Split pages into `page.tsx` (route entry), `*.hooks.ts` (data fetching), `*.components.tsx` (UI), and `*.utils.ts` (derived state).
- Introduce custom React Query hooks as the single source of truth for each domain.

---

## 7. Testing & Quality Infrastructure

| ID | Severity | File | Lines | Finding |
|----|----------|------|-------|---------|
| F-TEST-01 | High | `components/streaming/streaming-recommendations.test.tsx`, `package.json`, `tsconfig.json` | — | A Jest-style test file exists but Jest is not installed and `package.json` has no unit-test script. `tsconfig.json` explicitly excludes `**/*.test.tsx`. |
| F-TEST-02 | Low | `package.json` | — | No lint or format scripts (e.g., `lint`, `format`, `type-check`) are defined in `scripts`. |
| F-TEST-03 | Info | `tsconfig.json` | 7 | `strict: true` is enabled, which helps catch many runtime reliability issues at compile time. |

**Evidence**
```json
// package.json (excerpt)
"scripts": {
  "dev": "next dev",
  "build": "next build",
  "start": "next start",
  "test:e2e": "playwright test"
}
```

```json
// tsconfig.json (excerpt)
"exclude": ["node_modules", "**/*.test.tsx", "**/*.test.ts", "**/*.spec.tsx", "**/*.spec.ts"]
```

**Recommendation:**
- Either add Jest/Vitest + React Testing Library and a `test:unit` script, or delete the dead test file to eliminate confusion.
- Add `lint`, `format`, and `type-check` scripts and run them in CI.
- Consider converting the existing Playwright E2E tests (`tests/e2e/`) into a smoke-test suite that exercises the query-key invalidation paths.

---

## Summary Table

| ID | Area | Severity | File | Lines | Recommendation |
|----|------|----------|------|-------|----------------|
| F-HYD-01 | Hydration | Info | Multiple | — | Keep guarding browser APIs inside client components. |
| F-HYD-02 | Hydration | Info | `app/(dashboard)/dashboard/page.tsx` | 1-50 | Maintain per-widget Suspense/ErrorBoundary pattern. |
| F-HYD-03 | Hydration | Info | `components/freshness/freshness-badge.tsx` | 80-95 | Current client-only polling is safe. |
| F-HYD-04 | Hydration | Info | `app/layout.tsx` | 1-30 | Keep `preload: false` / `display: swap`. |
| F-DATA-01 | Data fetching | Medium | Many | See §2.1 | Centralize all query keys in `FANTASY_QUERY_KEYS` or per-domain registries. |
| F-DATA-02 | Data fetching | Medium | `components/layout/header.tsx` | 54 | Scope header refresh to relevant keys instead of global invalidation. |
| F-DATA-03 | Data fetching | Low | `app/(dashboard)/war-room/roster/page.tsx` | 976-979 | Remove redundant `refetchQueries` after `invalidateQueries`. |
| F-DATA-04 | Data fetching | High | `components/streaming/action-modal.tsx` | 44-45, 178-180 | Wire drop options to the real roster and derive the position dynamically. |
| F-DATA-05 | Data fetching | Medium | `app/(dashboard)/today/page.tsx` | 101-145 | Use `apiFetch` / `endpoints` for bet logging. |
| F-DATA-06 | Data fetching | Medium | Many | — | Extract custom React Query hooks per domain. |
| F-UX-01 | UX/Feedback | Info | `app/(dashboard)/dashboard/page.tsx` | 1-50 | Keep widget-level error/suspense boundaries. |
| F-UX-02 | UX/Feedback | Medium | `app/(dashboard)/...` | — | Add `loading.tsx` / `error.tsx` for remaining data-heavy routes. |
| F-UX-03 | UX/Feedback | Low | `today/loading.tsx`, `bracket/loading.tsx`, `admin/loading.tsx` | — | Align skeleton colors with the light theme token system. |
| F-UX-04 | UX/Feedback | Low | `components/error-boundary.tsx` | 1-50 | Add optional `onReset` callback to invalidate/retry queries. |
| F-MOB-01 | Mobile | Info | `app/(dashboard)/layout.tsx` | 1-20 | Sidebar responsive pattern is solid. |
| F-MOB-02 | Mobile | Medium | Tables across bracket, bet-history, live-slate, waiver, roster | — | Add sticky columns, column toggles, or card mobile layouts. |
| F-MOB-03 | Mobile | Low | `app/(dashboard)/war-room/roster/page.tsx` | — | Reduce density on small screens. |
| F-PERF-01 | Performance | Medium | `performance/page.tsx`, `clv/page.tsx`, `calibration/page.tsx` | 7, 13, 13 | Dynamically import `recharts` pages; add bundle analyzer. |
| F-PERF-02 | Performance | Medium | Multiple large pages | See §6 | Split large pages into hooks/components/utils files. |
| F-PERF-03 | Performance | Low | `components/freshness/freshness-badge.tsx` | 84-95 | Consider a single global freshness poller to avoid duplicate intervals. |
| F-PERF-04 | Performance | Low | Multiple | — | Memoize expensive rows and derived data. |
| F-PERF-05 | Performance | Info | `lib/query-client.ts` | 12-25 | Keep current stale/retry/gc settings. |
| F-MAINT-01 | Maintainability | High | `app/(dashboard)/war-room/roster/page.tsx` | 1,296 | Decompose into domain hooks and sub-components. |
| F-MAINT-02 | Maintainability | High | `app/(dashboard)/war-room/waiver/page.tsx` | 805 | Decompose into domain hooks and sub-components. |
| F-MAINT-03 | Maintainability | Medium | `app/(dashboard)/dashboard/_components/dashboard-client.tsx` | 673 | Split widgets into separate files. |
| F-MAINT-04 | Maintainability | Medium | `app/(dashboard)/decisions/page.tsx` | 579 | Extract hooks and table components. |
| F-MAINT-05 | Maintainability | Medium | `app/(dashboard)/admin/page.tsx` | 506 | Group admin sections into feature folders. |
| F-MAINT-06 | Maintainability | Medium | `components/streaming/streaming-recommendations.tsx` | 448 | Extract filter/sort hooks and row components. |
| F-MAINT-07 | Maintainability | Medium | `app/(dashboard)/today/page.tsx` | 463 | Extract bet-logging and prediction parsing helpers. |
| F-MAINT-08 | Maintainability | Medium | `app/(dashboard)/bracket/page.tsx` | 461 | Separate simulator controls from results UI. |
| F-MAINT-09 | Maintainability | Low | `components/yahoo-roster-view.tsx` | 422 | Acceptable for now; monitor growth. |
| F-MAINT-10 | Maintainability | Low | `components/war-room/category-battlefield.tsx` | 372 | Acceptable for now; monitor growth. |
| F-TEST-01 | Testing | High | `components/streaming/streaming-recommendations.test.tsx`, `package.json`, `tsconfig.json` | — | Add Jest/Vitest or remove the dead test file. |
| F-TEST-02 | Testing | Low | `package.json` | — | Add lint / format / type-check scripts. |
| F-TEST-03 | Testing | Info | `tsconfig.json` | 7 | Maintain `strict: true`. |

---

## Appendix: Largest Files (by line count)

| File | Lines | Notes |
|------|-------|-------|
| `app/(dashboard)/war-room/roster/page.tsx` | 1,296 | Largest route; urgent refactor candidate. |
| `lib/types.ts` | 848 | Central type file; consider domain split. |
| `app/(dashboard)/war-room/waiver/page.tsx` | 805 | Second-largest route. |
| `app/(dashboard)/dashboard/_components/dashboard-client.tsx` | 673 | Dashboard god component. |
| `app/(dashboard)/decisions/page.tsx` | 579 | Betting decisions + charts. |
| `app/(dashboard)/admin/page.tsx` | 506 | Admin dashboard + batch actions. |
| `components/streaming/streaming-recommendations.test.tsx` | 484 | Dead test file. |
| `app/(dashboard)/today/page.tsx` | 463 | Predictions + bet placement. |
| `app/(dashboard)/bracket/page.tsx` | 461 | Bracket simulator. |
| `components/streaming/streaming-recommendations.tsx` | 448 | Streaming UI. |

---

*Report generated by read-only code review. No source files were changed.*
