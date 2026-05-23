# P1 + P3 Frontend Batch Report — 2026-05-20

## Summary

Fixed three frontend issues from audit. All changes build successfully (`npm run build` passes). Light-theme token swaps from Claude's prior work (decisions, today, war-room/page) were absorbed and preserved.

---

## Files Modified

| File | Change |
|------|--------|
| `frontend/app/(dashboard)/war-room/waiver/page.tsx` | BUG 1 — hardened query fns, added ErrorBoundary wrapper, split into `WaiverPageInner` + `WaiverPage` export |
| `frontend/lib/api.ts` | BUG 1 — wrapped success-path `res.json()` in try/catch; added descriptive error for malformed JSON |
| `frontend/components/error-boundary.tsx` | BUG 1 — **new file** — React error boundary with retry button |
| `frontend/components/layout/sidebar.tsx` | BUG 2 — conditional hide of Portfolio chip + Admin section on fantasy routes; context-aware sub-branding |
| `frontend/components/layout/header.tsx` | BUG 3 — added `/war-room/roster`, `/war-room/waiver`, `/war-room/budget` title mappings |
| `frontend/app/(dashboard)/decisions/page.tsx` | Absorbed — `bg-zinc-900`→`bg-bg-surface`, `border-zinc-800`→`border-border-default`, `text-zinc-300`→`text-text-secondary` |
| `frontend/app/(dashboard)/today/page.tsx` | Absorbed — `bg-zinc-900`→`bg-bg-surface`, `border-zinc-800`→`border-border-default`, `text-zinc-300`→`text-text-secondary` |
| `frontend/app/(dashboard)/war-room/page.tsx` | Absorbed — `bg-black`→`bg-bg-base` |

---

## BUG 1 — Waiver Infinite Loading (P1)

### Root Cause

The waiver page had two `useQuery` hooks (main waiver data + recommendations) with no defensive wrapping around the API promise chain. If the backend returned `200 OK` with non-JSON (e.g., HTML error page, partial response), `res.json()` in `apiFetch` would throw an unhandled rejection that React Query *does* surface as `isError`, but the component had no error-boundary guard for render-phase crashes. Additionally, rapid `sort` state toggles could leave stale fetch promises dangling.

### Fixes Applied

1. **`frontend/lib/api.ts`**
   - Wrapped the success-path `res.json()` in `try/catch` so malformed JSON throws a descriptive error (`"200 OK but invalid JSON at ${path}"`) instead of a raw SyntaxError.
   - Added fallback detail message `'Invalid JSON in error response'` when the error-body parse fails.

2. **`frontend/components/error-boundary.tsx`** (new)
   - Class-based React error boundary that catches render-phase crashes.
   - Displays branded error card with message and a **Try again** button that resets state.
   - Can also accept a custom `fallback` prop.

3. **`frontend/app/(dashboard)/war-room/waiver/page.tsx`**
   - Renamed the page component to `WaiverPageInner`.
   - Wrapped the export in `<ErrorBoundary>` so any render crash shows the fallback instead of a blank screen.
   - Wrapped both `queryFn`s in `async/await` with `console.error` logging so failures are visible in devtools.
   - Added `retry: 1` to both `useQuery` configs to prevent aggressive infinite retry loops.

### Build Result

```
✓ Compiled successfully
✓ Generating static pages (24/24)
Route: /war-room/waiver — 7.99 kB
```

---

## BUG 2 — CBB Branding Leak (P3)

### Problem

The sidebar always showed:
- **Portfolio chip** (DD / Exp) — irrelevant for fantasy baseball users
- **"Analytics"** sub-brand — misleading when the user is in Fantasy War Room
- **Risk Dashboard** link under Admin — a CBB-betting-only feature visible on fantasy pages

### Fixes Applied

1. **`frontend/components/layout/sidebar.tsx`**
   - Added `isFantasyRoute(pathname)` helper matching:
     `/war-room/*`, `/roster`, `/waiver`, `/budget`, `/today`, `/decisions`
   - **Portfolio chip** is now conditionally rendered: hidden on fantasy routes.
   - **Admin section** (Risk Dashboard) is filtered out on fantasy routes.
   - **Sub-brand text** under the logo changes from `"Analytics"` to `"Fantasy Baseball"` when on fantasy routes.

### Build Result

```
✓ Compiled successfully
✓ Generating static pages (24/24)
```

---

## BUG 3 — Page Titles (P3)

### Problem

`frontend/components/layout/header.tsx` uses a `PAGE_TITLES` record for the header bar. Several War Room sub-pages were missing, causing the budget page (and others) to fall back to `'Dashboard'`.

### Fixes Applied

Added three missing mappings to `PAGE_TITLES`:

```ts
'/war-room/roster':  'My Roster',
'/war-room/waiver':  'Waiver Wire',
'/war-room/budget':  'Budget',
```

### Build Result

```
✓ Compiled successfully
✓ Generating static pages (24/24)
```

---

## Light-Theme Token Absorption

Claude's uncommitted light-theme swaps were already in the working tree. This batch did **not** revert them; they were preserved and are included in the diff:

- `decisions/page.tsx`: `bg-zinc-900` → `bg-bg-surface`, `border-zinc-800` → `border-border-default`, `text-zinc-300` → `text-text-secondary`
- `today/page.tsx`: same token families replaced
- `war-room/page.tsx`: `bg-black` → `bg-bg-base`

---

## Verification Checklist

- [x] `npm run build` passes after BUG 1
- [x] `npm run build` passes after BUG 2
- [x] `npm run build` passes after BUG 3
- [x] No new TypeScript errors
- [x] ErrorBoundary created and imported correctly
- [x] All `export default` declarations are singular
- [x] Fantasy route detection covers all specified paths
- [x] Page title mapping covers all war-room sub-pages
