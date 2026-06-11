# WAVE 4 FEATURE: Weekly Matchup Preview Frontend — 2026-05-20

## Summary

Created a new `/war-room/preview` page that displays next-week matchup projections, category-by-category analysis, streaming recommendations, and schedule advantage. Build passes cleanly (`npx tsc --noEmit` zero errors, `npm run build` 25/25 static pages).

---

## Files Created / Modified

| File | Change |
|------|--------|
| `frontend/app/(dashboard)/war-room/preview/page.tsx` | **New** — Weekly Preview page with opponent card, category projections, needs streaming, schedule advantage |
| `frontend/lib/types.ts` | Added `MatchupPreviewResponse`, `ScheduleAdvantage`, `WeakCategory` types |
| `frontend/lib/api.ts` | Added `getMatchupPreview()` endpoint + imported new type |
| `frontend/components/layout/sidebar.tsx` | Added "Preview" nav link under Fantasy section with `Eye` icon |
| `frontend/components/layout/header.tsx` | Added `/war-room/preview` → `"Weekly Preview"` title mapping |

---

## Page Structure

```
Weekly Preview (/war-room/preview)
├── Header: "Weekly Preview" + week number
├── Opponent Card
│   ├── Opponent name + logo (or fallback initial)
│   └── Overall projected win %
├── Category Projections Table
│   ├── Column headers: CAT | ME | [bar] | OPP | WIN% | STATUS
│   ├── BATTING section (green/red/yellow tags)
│   └── PITCHING section (green/red/yellow tags)
├── Needs Streaming (conditional)
│   └── Weak categories with links to /war-room/waiver?category=X
└── Schedule Advantage
    ├── My games vs opponent games
    └── +/- game advantage callout
```

---

## Type Definition

```ts
export interface MatchupPreviewResponse {
  week_number: number
  opponent_name: string
  opponent_logo?: string | null
  overall_win_prob: number
  category_projections: CategoryProjection[]
  weak_categories: WeakCategory[]
  schedule_advantage: ScheduleAdvantage
  message?: string | null
}
```

---

## Visual Design

- **Opponent card**: `bg-bg-surface` with border, opponent avatar + name on left, win % on right
- **Category rows**: Reuse the split-bar pattern from `CategoryBattlefield`
  - My side bar uses category color when projected win > 50%
  - Status tags: `PROJECTED WIN` (green), `BUBBLE` (amber), `PROJECTED LOSS` (red)
- **Needs Streaming**: Amber bubble section with category dots and direct links to waiver
- **Schedule Advantage**: Side-by-side game count cards with +/- indicator

---

## Error Handling

### 404 Graceful Fallback

If the backend endpoint `/api/fantasy/matchup-preview` returns 404 (not yet deployed), the page shows:

```
┌─ Weekly Preview ─────────────────────────┐
│     👁️                                   │
│  Weekly Preview                          │
│  Next-week matchup projections are       │
│  coming soon. Check back after the       │
│  backend endpoint is deployed.           │
│  Endpoint: /api/fantasy/matchup-preview  │
└──────────────────────────────────────────┘
```

### Generic Error Fallback

For non-404 errors, a standard error card with **Retry** button is shown.

---

## Build Verification

```
✓ Compiled successfully
✓ Generating static pages (25/25)
Route: /war-room/preview — 4.77 kB
```

### Type Safety

- `npx tsc --noEmit` — zero errors
- ESLint clean (only pre-existing `<img>` warning from another component)

---

## Navigation Integration

### Sidebar (Fantasy section)

```
Fantasy
  War Room      ⚔️
  My Roster     👥
  Waiver Wire   🔍
  Streaming     🌊
  Budget        💵
  Preview       👁️   ← NEW
```

### Header Title

Navigating to `/war-room/preview` displays **"Weekly Preview"** in the top header bar.

---

## Backend Contract

The frontend expects Claude's backend endpoint to return:

```
GET /api/fantasy/matchup-preview
→ MatchupPreviewResponse
```

All fields are typed defensively (`| null` on optional data) so partial responses won't crash the UI.
