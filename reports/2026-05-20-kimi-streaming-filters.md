# WAVE 2 — P4 LOW: Streaming Station Filters — 2026-05-20

## Summary

Added two frontend filters to the Streaming Station (`/war-room/streaming`) to reduce noise from rostered players and low-value recommendations. Both `npx tsc --noEmit` and `npm run build` pass cleanly.

---

## File Modified

| File | Change |
|------|--------|
| `frontend/app/(dashboard)/war-room/streaming/page.tsx` | Added `hideOwned` + `minNeedScore` state, filter UI, and filter logic for both player lists |

---

## Filter Details

### 1. Hide Owned Players — checkbox, default **ON**

- **Proxy for "owned / on roster"**: `player.percent_owned == null && player.owned_pct == null`
- Players with no ownership data are considered already rostered and are excluded when the toggle is ON.
- Uses a native checkbox styled with design-system tokens (`text-accent-primary`, `border-border-default`).

### 2. Minimum Need Score — range slider, default **0.0**

- Slider range: **-5.0 to 10.0** with **0.1** step.
- Players with `need_score <= minNeedScore` are excluded.
- Current value displayed in a monospaced gold label next to the slider.

---

## UI Placement

A new filter bar sits directly below the page header (`STREAMING STATION` + FAAB balance):

```
┌─ Filters ──────────────────────────────────────┐
│ [✓] Hide Owned Players    Min Need Score [━━━●─] 0.0 │
└──────────────────────────────────────────────────┘
```

- Background: `bg-bg-surface`
- Border: `border-border-subtle`
- Responsive layout: stacked on mobile, row on desktop (`sm:flex-row`)

---

## Filter Logic

```ts
function passesFilters(p: WaiverAvailablePlayer): boolean {
  if (hideOwned && (p.percent_owned == null && p.owned_pct == null)) return false
  if (p.need_score != null && p.need_score <= minNeedScore) return false
  return true
}
```

Both `two_start_pitchers` and `top_available` arrays are filtered through `passesFilters` before rendering.

---

## Count Indicators

When filters hide some players, the section header shows a "hidden" count:

```
Two-Start Pitchers (3) · 2 hidden
Top Available (8) · 4 hidden
```

This makes it obvious that results are being suppressed rather than absent.

---

## Empty State

When all players are filtered out, the message updates from:

> "No waiver targets found for the current period."

To:

> "No waiver targets match the current filters."

This signals to the user that data exists but is being hidden by their filter choices.

---

## Verification

- [x] `npx tsc --noEmit` — zero type errors
- [x] `npm run build` — compiles successfully, 24/24 static pages generated
- [x] `/war-room/streaming` route size: **4.52 kB** (up from 4.06 kB — expected increase for new UI + state)
- [x] No new runtime dependencies
- [x] Native HTML inputs used for maximum theme compatibility
