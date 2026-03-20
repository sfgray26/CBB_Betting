# VALIDATION REPORT — Alerts Page

**Validation Agent:** Kimi CLI (Deep Intelligence Unit)  
**Date:** March 19, 2026 ~10:40 GMT+8  
**Component:** `frontend/app/(dashboard)/alerts/page.tsx`  
**Status:** ⚠️ **CONDITIONAL PASS — 1 medium issue requiring fix**

---

## EXECUTIVE SUMMARY

Component implements Alerts dashboard with live alerts, active/acknowledged alert lists, and acknowledgment functionality. **Good structure overall**, but **one medium-priority null-safety issue** on line 69 where `parseISO` is called without null check, potentially crashing on malformed API data.

---

## VALIDATION CHECKLIST RESULTS

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | **NULL SAFETY** | ⚠️ Issue | Line 69: `parseISO(alert.created_at)` without null check |
| 2 | **EMPTY ARRAY** | ✅ Pass | `data?.alerts ?? []` (lines 101, 102, 103) |
| 3 | **DECIMAL DISPLAY** | ✅ N/A | No percentages in this component |
| 4 | **LOADING STATE** | ✅ Pass | Skeleton loaders with `animate-pulse` (lines 133-137) |
| 5 | **CRASH RISK** | ⚠️ Issue | `parseISO()` can throw on null/invalid date |
| 6 | **Object.entries()** | ✅ N/A | Not used |
| 7 | **Empty State UX** | ✅ Pass | "No active alerts. All systems nominal." message |

---

## ISSUES FOUND

### 🔴 Line 69: parseISO Without Null Guard

**Current Code:**
```tsx
function AlertCard({ alert, onAck }: { alert: Alert; onAck: (id: number) => void }) {
  // ...
  return (
    <div>
      {/* ... */}
      <span className="text-xs text-zinc-500 whitespace-nowrap">
        {formatDistanceToNow(parseISO(alert.created_at), { addSuffix: true })}
      </span>
      {/* ... */}
    </div>
  )
}
```

**Problem:**
- `alert.created_at` could be null/undefined per API spec (though unlikely)
- `parseISO(null)` throws: `RangeError: Invalid time value`
- Component crashes, breaks entire alert list

**Fix Required:**
```tsx
<span className="text-xs text-zinc-500 whitespace-nowrap">
  {alert.created_at 
    ? formatDistanceToNow(parseISO(alert.created_at), { addSuffix: true })
    : 'Unknown time'
  }
</span>
```

---

### 🟡 Line 77: Similar Issue with acknowledged_at

**Current Code:**
```tsx
{alert.acknowledged && alert.acknowledged_at && (
  <p className="text-xs text-zinc-600 mt-1">
    Acknowledged{' '}
    {formatDistanceToNow(parseISO(alert.acknowledged_at), { addSuffix: true })}
  </p>
)}
```

**Problem:** Same pattern — `parseISO` without null guard.

**Note:** Already has `&& alert.acknowledged_at` check, so actually safe. But defensive coding recommended.

---

## WHAT'S CORRECT ✅

### Empty Array Handling
```tsx
const active = (data?.alerts ?? []).filter((a) => !a.acknowledged)
const acknowledged = (data?.alerts ?? []).filter((a) => a.acknowledged)
const liveAlerts = data?.live_alerts ?? []
```
- Consistent nullish coalescing ✅

### Loading State
```tsx
{isLoading ? (
  <div className="space-y-3">
    {[1, 2, 3].map((i) => (
      <div key={i} className="h-20 bg-zinc-800 rounded-lg animate-pulse" />
    ))}
  </div>
) : ...}
```
- 3 skeleton cards with pulse animation ✅

### Empty State UX
```tsx
{active.length === 0 ? (
  <Card>
    <div className="flex items-center gap-3 text-emerald-400">
      <CheckCircle2 className="h-5 w-5" />
      <span className="text-sm">No active alerts. All systems nominal.</span>
    </div>
  </Card>
) : ...}
```
- User-friendly success message ✅

### Error State
```tsx
if (isError) {
  return <ErrorCard message="Failed to load alerts. Check your connection." />
}
```
- Proper error boundary ✅

---

## API SPEC COMPLIANCE

Per typical Alert API patterns:

### Nullable Fields

| Field | Nullable | Handled? | Line |
|-------|----------|----------|------|
| `alert.created_at` | Possibly | ❌ No | 69 |
| `alert.acknowledged_at` | Yes | ✅ Yes | 77 (has guard) |
| `alert.message` | No | ✅ Direct access | 64 |
| `alert.severity` | No | ✅ Switch statement | 29 |
| `alert.alert_type` | No | ✅ Direct access | 63 |

---

## CORRECT PATTERNS TO PRESERVE

### Severity Configuration
```tsx
function severityConfig(severity: string) {
  switch (severity) {
    case 'CRITICAL':
      return { Icon: AlertOctagon, color: 'text-rose-500', ... }
    case 'WARNING':
      return { Icon: AlertTriangle, color: 'text-amber-400', ... }
    default:
      return { Icon: Info, color: 'text-sky-400', ... }
  }
}
```
- Clean mapping with default fallback ✅

### Acknowledgment Handler
```tsx
async function handleAck(id: number) {
  setAckingId(id)
  try {
    await endpoints.acknowledgeAlert(id)
    queryClient.invalidateQueries({ queryKey: ['alerts'] })
  } catch (e) {
    console.error('Failed to acknowledge alert', e)
  } finally {
    setAckingId(null)
  }
}
```
- Proper loading state, error handling, cache invalidation ✅

---

## SUMMARY

| Issue Type | Count |
|------------|-------|
| Critical (crash risk) | 0 |
| High (null safety) | 1 (line 69, parseISO) |
| Medium | 0 |
| Low | 0 |

---

## REQUIRED ACTIONS

### Must Fix (Before Merge)
1. **Line 69:** Add null guard for `alert.created_at` before `parseISO`

### Code Change
```tsx
// From:
{formatDistanceToNow(parseISO(alert.created_at), { addSuffix: true })}

// To:
{alert.created_at 
  ? formatDistanceToNow(parseISO(alert.created_at), { addSuffix: true })
  : 'Unknown time'
}
```

---

## TESTING CHECKLIST

After fix, verify:
- [ ] `npm run build` passes
- [ ] `npx tsc --noEmit` passes
- [ ] Component renders with no alerts (empty state)
- [ ] Component renders with alerts
- [ ] Alert acknowledgment works
- [ ] Auto-refresh every 30s works
- [ ] No crash if alert has null `created_at`

---

**PASS/FAIL: ⚠️ CONDITIONAL PASS — 1 blocking fix required**

---

**Next Step:** Fix line 69 null guard, re-test, then merge.
