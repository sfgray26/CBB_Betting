# CLAUDE CODE PICKUP — CLV Page Validation Results

**Date:** March 19, 2026  
**Component:** `frontend/app/(dashboard)/clv/page.tsx`  
**Validator:** Kimi CLI  
**Full Report:** `VALIDATION_REPORT_CLV.md`

---

## TL;DR — ACTION REQUIRED

❌ **FAIL** — 1 blocking issue must be fixed before merge.

**Critical Fix:** Line 119 — Add null check before `clv_prob * 100`

---

## The Problem (Line 119)

```tsx
// CURRENT (CRASHES if clv_prob is null):
accessor: (r) => (
  <span className={`... ${r.clv_prob >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
    {signed(r.clv_prob * 100)}%
  </span>
)

// FIXED:
accessor: (r) => {
  if (r.clv_prob == null) return <span className="text-zinc-500">—</span>
  return (
    <span className={`... ${r.clv_prob >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
      {signed(r.clv_prob * 100)}%
    </span>
  )
}
```

---

## Also Fix (Non-Blocking)

1. **Update `signed()` helper** to handle null/undefined:
```tsx
const signed = (v: number | undefined | null, decimals = 2) => {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}`
}
```

2. **Remove non-null assertions** (lines 159, 173):
```tsx
// Change:
value={hasData ? `${signed(data!.mean_clv * 100)}` : '--'}

// To:
value={hasData && data?.mean_clv != null ? `${signed(data.mean_clv * 100)}` : '--'}
```

---

## What's Correct ✅

| Check | Status |
|-------|--------|
| Decimal → Percentage | ✅ All CLV values ×100 |
| Loading states | ✅ Skeleton loaders present |
| Empty state | ✅ User message shown |
| Object.entries() | ✅ `?? {}` guard used |
| Empty arrays | ✅ `?? []` fallback used |

---

## API Spec Compliance

Per `reports/api_ground_truth.md`:
- `clv_prob` in array items is **nullable** — currently not checked (Line 119)
- `clv_points` is nullable — correctly checked ✅
- `mean_clv` is nullable — correctly handled via `hasData` guard ✅

---

## Commands to Run After Fix

```bash
cd frontend
npm run build        # Must pass
npx tsc --noEmit     # Must pass
```

---

## Full Details

See `VALIDATION_REPORT_CLV.md` for:
- Complete line-by-line breakdown
- All issues with severity levels
- Correct patterns to preserve
- Re-validation checklist

---

**Status:** Awaiting fix on Line 119. Re-assign to Kimi CLI for re-validation after fix.
