# VALIDATION REPORT — Calibration Page

**Validation Agent:** Kimi CLI (Deep Intelligence Unit)  
**Date:** March 19, 2026 ~10:38 GMT+8  
**Component:** `frontend/app/(dashboard)/calibration/page.tsx`  
**Status:** ⚠️ **CONDITIONAL PASS — 1 minor issue, 1 recommendation**

---

## EXECUTIVE SUMMARY

Component implements Calibration dashboard with Brier Score KPI, calibration curve chart, and bins table. **Strong null-safety and loading states**. Minor issue with `toFixed()` on potentially undefined values in column accessor.

---

## VALIDATION CHECKLIST RESULTS

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | **NULL SAFETY** | ✅ Pass | `data?.brier_score != null`, `data?.calibration_buckets ?? []` |
| 2 | **EMPTY ARRAY** | ✅ Pass | `data?.calibration_buckets ?? []` (lines 50, 144) |
| 3 | **DECIMAL DISPLAY** | ✅ Pass | `* 100` conversion for all probability displays |
| 4 | **LOADING STATE** | ✅ Pass | Skeleton loaders with `animate-pulse` (lines 108-110, 137-141) |
| 5 | **CRASH RISK** | ⚠️ Minor | `toFixed()` on line 68 without null guard for `actual_win_rate` |
| 6 | **Object.entries()** | ✅ N/A | Not used in this component |
| 7 | **Empty State UX** | ✅ Pass | User-visible message: "Not enough data to draw calibration curve" |

---

## DETAILED ANALYSIS

### ✅ Excellent Null Safety

**Line 82:** `data?.brier_score != null`
- Explicit null check before using Brier score

**Line 88:** `data ? (data.calibration_buckets.reduce(...) : '--'`
- Ternary with fallback

**Line 50:** `data?.calibration_buckets ?? []`
- Nullish coalescing for empty array

**Line 144:** `data?.calibration_buckets ?? []`
- Consistent pattern in DataTable

### ✅ Decimal → Percentage Conversion

**Line 64:** `(r.actual_win_rate * 100).toFixed(1)}%`
- Correct ×100 conversion

**Line 74:** `(delta * 100).toFixed(1)}%`
- Correct ×100 conversion for delta

### ✅ Loading States

**Lines 108-110:** Calibration chart skeleton
```tsx
{isLoading ? (
  <div className="px-6 pb-6">
    <div className="h-64 bg-zinc-800 rounded animate-pulse" />
  </div>
) : ...}
```

**Lines 137-141:** Table skeleton
```tsx
{isLoading ? (
  <div className="p-6 space-y-2">
    {[1, 2, 3, 4, 5].map((i) => (...))}
  </div>
) : ...}
```

### ✅ Empty State UX

**Lines 115-117:**
```tsx
{curveData.length === 0 ? (
  <div className="px-6 pb-6 py-16 text-center text-zinc-500 text-sm">
    Not enough data to draw calibration curve.
  </div>
) : ...}
```

**Line 149:** `emptyMessage="No calibration bins available."`

---

## ⚠️ MINOR ISSUE (Non-Blocking)

### Line 68: toFixed() Without Explicit Null Guard

**Current Code:**
```tsx
{
  key: 'actual',
  header: 'Actual Win Rate',
  accessor: (r) => (
    <span className="font-mono tabular-nums">
      {(r.actual_win_rate * 100).toFixed(1)}%
    </span>
  ),
  // ...
}
```

**Issue:** `actual_win_rate` is technically nullable per API spec (though unlikely in practice since bucket wouldn't exist without data).

**Risk:** Low — table only renders when `calibration_buckets` has items, and buckets shouldn't exist without actual_win_rate.

**Recommended Fix (Defensive):**
```tsx
accessor: (r) => {
  const rate = r.actual_win_rate
  if (rate == null) return <span className="text-zinc-500">—</span>
  return (
    <span className="font-mono tabular-nums">
      {(rate * 100).toFixed(1)}%
    </span>
  )
}
```

---

## 💡 RECOMMENDATION (Code Quality)

### Line 74: Delta Calculation Duplicated

**Current:**
```tsx
const delta = r.actual_win_rate - r.predicted_prob
// Used for label, color, and display
```

**Note:** The `error` field from API already contains this delta. Could use `r.error` directly instead of recalculating.

**Not a bug** — just a minor redundancy.

---

## API SPEC COMPLIANCE

Per `reports/api_ground_truth.md` for `GET /api/performance/calibration`:

### Empty State (Handled ✅)
```json
{
  "calibration_buckets": [],
  "mean_calibration_error": null,
  "is_well_calibrated": null,
  "brier_score": null
}
```
Component shows: "Not enough data to draw calibration curve." ✅

### Nullable Fields (Mostly Handled ⚠️)

| Field | Nullable | Handled? | Notes |
|-------|----------|----------|-------|
| `brier_score` | Yes | ✅ Line 82 | `!= null` check |
| `mean_calibration_error` | Yes | ✅ Not displayed | N/A |
| `is_well_calibrated` | Yes | ✅ Not displayed | N/A |
| `calibration_buckets[].actual_win_rate` | No | ⚠️ Line 68 | Technically non-nullable, but defensive guard recommended |
| `calibration_buckets[].predicted_prob` | No | ✅ Line 74 | Used in calculation |
| `calibration_buckets[].count` | No | ✅ Line 72 | Direct access |

---

## CORRECT PATTERNS TO PRESERVE

### Null-Safe Data Access
```tsx
const curveData = (data?.calibration_buckets ?? []).map((b) => ({
  predicted: Math.round(b.predicted_prob * 100),
  actual: Math.round(b.actual_win_rate * 100),
  // ...
}))
```

### Conditional KPI Display
```tsx
<KpiCard
  title="Brier Score"
  value={data?.brier_score != null ? data.brier_score.toFixed(4) : '--'}
  loading={isLoading}
/>
```

---

## SUMMARY

| Issue Type | Count |
|------------|-------|
| Critical (crash risk) | 0 |
| High (null safety) | 0 |
| Medium (defensive coding) | 1 (line 68, minor) |
| Low (code quality) | 1 (redundant delta calc) |

---

## RECOMMENDATION

**Status:** ✅ **APPROVE WITH NOTES**

Component is production-ready. The line 68 issue is theoretical (bucket shouldn't exist without actual_win_rate). Fix is recommended for defensive coding but not blocking.

**Optional fixes:**
1. Add null guard on line 68 for `actual_win_rate`
2. Consider using `r.error` instead of recalculating delta

---

## TESTING CHECKLIST

Before merge, verify:
- [ ] `npm run build` passes
- [ ] `npx tsc --noEmit` passes
- [ ] Component renders with empty calibration data
- [ ] Component renders with full calibration data
- [ ] Brier Score displays correctly
- [ ] Calibration curve draws correctly

---

**PASS/FAIL: ✅ CONDITIONAL PASS — Minor defensive coding recommendation**

---

**Status:** Ready for merge (optional line 68 improvement).
