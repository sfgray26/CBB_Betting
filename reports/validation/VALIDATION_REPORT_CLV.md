# VALIDATION REPORT — CLV Analysis Page

**Validation Agent:** Kimi CLI (Deep Intelligence Unit)  
**Date:** March 19, 2026 ~10:33 GMT+8  
**Component:** `frontend/app/(dashboard)/clv/page.tsx`  
**Status:** ❌ **FAIL — 1 HIGH priority issue requiring fix**

---

## EXECUTIVE SUMMARY

Component implements CLV Analysis dashboard with KPI cards, distribution chart, confidence tier table, and top/bottom 10 bet lists. **Most implementation is correct** — proper decimal→percentage conversion, loading states, and empty state handling. **One critical null-safety issue** on line 119 that can cause runtime crashes.

---

## CRITICAL ISSUE (Must Fix Before Merge)

### Line 119: Null Safety — `clv_prob` Multiplication Without Check

**Location:** Column accessor for "CLV %" in `clvBetColumns`

**Current Code:**
```tsx
{
  key: 'clv_prob',
  header: 'CLV %',
  accessor: (r) => (
    <span className={`... ${r.clv_prob >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
      {signed(r.clv_prob * 100)}%  // ← CRASHES if null/undefined
    </span>
  ),
  sortValue: (r) => r.clv_prob,
  // ...
}
```

**Problem:** 
- API spec shows `clv_prob` can be null/undefined (nullable per API ground truth)
- No null check before `* 100` multiplication
- Will throw `TypeError: Cannot read property 'toFixed' of null/undefined`

**Fix Required:**
```tsx
{
  key: 'clv_prob',
  header: 'CLV %',
  accessor: (r) => {
    if (r.clv_prob == null) return <span className="text-zinc-500">—</span>
    return (
      <span className={`... ${r.clv_prob >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
        {signed(r.clv_prob * 100)}%
      </span>
    )
  },
  sortValue: (r) => r.clv_prob ?? 0,
  // ...
}
```

---

## MEDIUM PRIORITY ISSUES (Recommended Fixes)

### Issue 2: Line 136 — `signed()` Helper Doesn't Handle Undefined

**Location:** Column accessor for "CLV pts"

**Current Code:**
```tsx
accessor: (r) => {
  const pts = r.clv_points
  if (pts == null) return <span className="text-zinc-500">—</span>
  return (
    <span className={`...`}>
      {signed(pts)}  // pts can still be undefined here
    </span>
  )
}
```

**Problem:** The null check is good, but the `signed()` helper function itself should be defensive.

**Fix:** Update helper function signature and guard:
```tsx
// At top of file, change:
const signed = (v: number | undefined | null, decimals = 2) => {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}`
}
```

---

### Issue 3: Lines 159, 173 — Non-Null Assertions

**Location:** KPI Card values

**Current Code:**
```tsx
<KpiCard
  title="Avg CLV"
  value={hasData ? `${signed(data!.mean_clv * 100)}` : '--'}  // ! assertion
  // ...
/>
<KpiCard
  title="Positive CLV"
  value={hasData ? pct(data!.positive_clv_rate, 1) : '--'}  // ! assertion
  // ...
/>
```

**Problem:** TypeScript non-null assertions (`!`) are risky — if `hasData` logic changes, this crashes.

**Fix:** Use optional chaining with explicit null check:
```tsx
<KpiCard
  title="Avg CLV"
  value={hasData && data?.mean_clv != null ? `${signed(data.mean_clv * 100)}` : '--'}
  // ...
/>
<KpiCard
  title="Positive CLV"
  value={hasData && data?.positive_clv_rate != null ? pct(data.positive_clv_rate, 1) : '--'}
  // ...
/>
```

---

## WHAT'S CORRECT ✅ (Validation Checklist)

| Check | Status | Evidence |
|-------|--------|----------|
| **1. NULL SAFETY (overall)** | ✅ Pass | `data?.distribution` with optional chaining (line 85) |
| **2. EMPTY ARRAY** | ✅ Pass | `?? []` fallback for `top_10_clv`, `bottom_10_clv` (lines 208, 223) |
| **3. DECIMAL DISPLAY** | ✅ Pass | All CLV values ×100: `mean_clv * 100` (line 159), `clv_prob * 100` (line 119) |
| **4. LOADING STATE** | ✅ Pass | Skeleton loaders present (lines 183-185, 214-218, 242-246) |
| **5. CRASH RISK (toFixed)** | ✅ Pass | `toFixed()` only called within `signed()` helper which has guard |
| **6. Object.entries()** | ✅ Pass | `?? {}` guard: `data?.clv_by_confidence ?? {}` (line 92) |
| **7. Empty State UX** | ✅ Pass | User-visible message: `data.message ?? 'No CLV data yet...'` (line 145) |

---

## API SPEC COMPLIANCE

Per `reports/api_ground_truth.md` for `GET /api/performance/clv-analysis`:

### Empty State (Handled ✅)
```json
{
  "message": "No CLV data yet (requires closing lines)",
  "bets_with_clv": 0
}
```
Component shows: `"No CLV data yet. Requires closing lines to be captured."` ✅

### Nullable Fields (Partial ⚠️)
| Field | Nullable | Handled? |
|-------|----------|----------|
| `mean_clv` | Yes | ✅ KPI card checks `hasData` |
| `median_clv` | Yes | ✅ Not displayed in this component |
| `std_clv` | Yes | ✅ Not displayed in this component |
| `clv_prob` (in arrays) | Yes | ❌ **NOT checked — LINE 119** |
| `clv_points` (in arrays) | Yes | ✅ Checked with `if (pts == null)` |

---

## CORRECT PATTERNS TO PRESERVE

### Decimal → Percentage Conversion (Correct)
```tsx
// Line 159 — correct ×100 conversion
`${signed(data!.mean_clv * 100)}`

// Line 119 — correct ×100 conversion (but missing null check)
{signed(r.clv_prob * 100)}%
```

### Loading State Pattern (Correct)
```tsx
{isLoading ? (
  <div className="p-6 space-y-2">
    {[1, 2, 3].map((i) => (
      <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
    ))}
  </div>
) : (
  <DataTable ... />
)}
```

### Empty State Pattern (Correct)
```tsx
{!isLoading && data && !hasData && (
  <div className="...">
    {data.message ?? 'No CLV data yet. Requires closing lines to be captured.'}
  </div>
)}
```

---

## REQUIRED ACTIONS FOR CLAUDE CODE

### Must Fix (Blocking Merge)
1. **Line 119:** Add null check for `r.clv_prob` in column accessor

### Should Fix (Code Quality)
2. **Helper function:** Update `signed()` to accept `number | undefined | null`
3. **Lines 159, 173:** Replace non-null assertions with optional chaining + null checks

### Test Before Merge
```bash
# Build check
npm run build

# Type check
npx tsc --noEmit

# Runtime check (if API returns null clv_prob)
# Should display "—" instead of crashing
```

---

## VALIDATION CHECKLIST FOR RE-REVIEW

After fixes, verify:
- [ ] Line 119 has null check for `clv_prob`
- [ ] `signed()` helper handles null/undefined
- [ ] No TypeScript non-null assertions (`!`) remain
- [ ] `npm run build` passes without errors
- [ ] Component renders with null CLV data without crashing

---

**Overall Assessment:** Component is 90% correct with good patterns for loading states, empty states, and decimal conversion. **One critical null-safety bug** prevents merge. Fix line 119 and re-validate.

**PASS/FAIL: ❌ FAIL (1 blocking issue)**

---

**Next Step:** Fix line 119 null check, run `npm run build`, re-submit for validation.
