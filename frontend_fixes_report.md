# Frontend Fixes Report — 2026-05-11

## Summary

Two P0 UI bugs fixed in the Next.js frontend. Both issues were in client-side
React components and did not require any backend changes.

---

## Bug 1: Dashboard Renders Empty

**File:** `frontend/app/(dashboard)/dashboard/page.tsx`

### Root Cause

The component was a stub rendering two generic placeholder `Card` components
("Dashboard Status" / "Data Snapshot") with no actual data. Two secondary issues
compounded this:

1. **`timestamp` extraction was wrong.** The component accessed
   `dashboard.timestamp` (i.e., `response.data.timestamp`), but the backend puts
   `timestamp` at the top-level envelope (`response.timestamp`), not inside
   `data`. The `DashboardData` TypeScript type incorrectly declares `timestamp`
   as a field of `data`. At runtime, `dashboard.timestamp` was always
   `undefined`, falling back to `Date.now()`.

2. **`user_id` is not in `response.data`.** The backend does not include
   `user_id` inside the `data` dict; the component rendered it as `undefined`
   (silently empty).

### Fix

- Extract `timestamp` from `response?.timestamp` (top-level), not from
  `dashboard.timestamp`.
- Removed `user_id` from the rendered output.
- Implemented a full dashboard UI with five data sections:
  - **Lineup status bar** — `lineup_filled_count / lineup_total_count` and
    healthy/injured player counts.
  - **Lineup Gaps** — severity-coded dot (red/amber/grey) per `LineupGap`, with
    suggested add player.
  - **Injury Alerts** — per `InjuryFlag` with status badge, action required,
    and ETA.
  - **Top Waiver Targets** — top-5 from `waiver_targets` with tier badge
    (must_add / strong_add / streamer).
  - **Player Trends** — top-3 hot streaks and top-3 cold streaks with 7-day
    average.
  - **Two-Start Pitchers** — full-width grid when `two_start_pitchers` is
    non-empty.
- Replaced the skeleton and error states with correct zinc-800 palette colours
  matching the site's dark theme.

### Data Contract Confirmed

The `/api/dashboard` backend route returns:
```
{
  "success": true,
  "timestamp": "...",          ← top-level
  "data": {                    ← DashboardData lives here
    "lineup_gaps": [...],
    "waiver_targets": [...],
    "injury_flags": [...],
    "hot_streaks": [...],
    "cold_streaks": [...],
    "two_start_pitchers": [...],
    ...
  }
}
```
The `DashboardResponse` TypeScript type is correct (`response.data` is
`DashboardData`). Only the `timestamp` / `user_id` fields in the `DashboardData`
interface misrepresent where those values actually live. Fixing the type
interface would be a clean-up task, but is not required for the component to
render correctly.

---

## Bug 2: Streaming Page Infinite Loading

**File:** `frontend/app/(dashboard)/war-room/streaming/page.tsx`

### Root Cause

The component used **parallel conditional blocks** in the same render pass:

```tsx
{waiver.isLoading && <LoadingSpinner />}
{waiver.isError && <ErrorBlock />}
{waiver.data && <DataBlock />}
```

In React 19's concurrent rendering, all three condition expressions are
evaluated in every render cycle. If the `data` block causes any interruption
during an in-progress render (concurrent mode pauses and retries renders), React
keeps the **last committed DOM** visible — which was the loading spinner from the
prior render. The component never exits the loading state from the user's
perspective.

The correct pattern — used by the working `war-room/page.tsx` — is **early
returns** that gate the render completely:

```tsx
if (waiver.isLoading) return <LoadingSpinner />
if (waiver.isError) return <ErrorBlock />
if (!waiver.data) return null
return <DataBlock />
```

With early returns, only one branch of code can execute per render cycle,
eliminating any concurrent-rendering state ambiguity.

### Fix

- Converted from parallel conditions to the early-return pattern (consistent
  with the war-room page, which works correctly).
- Added an explicit `if (!waiver.data) return null` guard as a third safety net.
- Replaced the raw `JSON.stringify` data dump with a proper rendered UI:
  - **FAAB balance** in the page header.
  - **Category Deficits** — labelled chips with z-score.
  - **Two-Start Pitchers** — highlighted rows (gold border).
  - **Top Available** — player rows with need score and ownership %.
- Added `WaiverPlayerRow` sub-component for consistent player list rendering.

### API Contract Confirmed

`GET /api/fantasy/waiver` returns `WaiverWireResponse` (backend Pydantic model),
which includes `top_available`, `two_start_pitchers`, `category_deficits`, and
`faab_balance` — matching the `WaiverResponse` frontend type. No shape mismatch
exists; the issue was purely in the React rendering pattern.

---

## Verification

```
npx tsc --noEmit   → 0 errors
npx next lint      → ✔ No ESLint warnings or errors
```

## Out of Scope (Not Fixed in This Pass)

The following P0 / P1 issues were identified in the UAT report but are **not**
addressed here per task scope:

| Issue | File | Notes |
|-------|------|-------|
| CORS block on `POST /api/fantasy/matchup/simulate` | Backend CORS config | Backend change required |
| `GET /api/fantasy/lineup/current` always 422 | Backend route | Backend change required |
| Matchup API latency 3,182ms | Backend / caching | Backend change required |
| Budget API not integrated into UI | New frontend feature | Separate task |
