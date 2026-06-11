# War-Room Audit + Design System v3 Consistency Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all dark-theme color remnants in war-room pages so the UI is fully consistent with Design System v3 (light theme semantic tokens).

**Architecture:** Swap hardcoded dark Tailwind classes (`*-900/30 *-400`) for either DS v3 semantic tokens or light-theme equivalents (`*-50 *-700`). No new tokens needed — position badge colors stay categorical, semantic status colors use DS v3 tokens.

**Tech Stack:** Next.js 14, Tailwind CSS v3, Design System v3 (defined in `frontend/tailwind.config.ts`)

---

## Audit Summary (No Code Changes)

### Commit 74c2752 — Cherry-pick status: ✅ ALREADY IN CODEBASE
- Content: AGENTS.md, HANDOFF.md, HERMES.md, docs/metadata only. No code logic.
- Both `main` and `stable/cbb-prod` are at `add7395`, which includes this commit.
- **Action: None.**

### HERMES P1 Bugs — Status: ✅ ALL FIXED
| Bug | Location | Fixed In |
|-----|----------|---------|
| 1 · Wrong solver in roster optimize | `fantasy.py:3447` | `aa32233` |
| 2 · Inverted implied runs sign | `daily_lineup_optimizer.py:420` | `51b4bb9` |
| 3 · Silent empty roster on count field | `yahoo_client_resilient.py:693` | `8e331c4` |
| 4 · Disabled handedness signal | `matchup_engine.py:227` | `51b4bb9` |
| 5 · Unsafe ilike fallback | `projection_assembly_service.py:503` | `51b4bb9` |
- **Action: None.**

### War-Room Design System v3 Compliance — ❌ 3 pages non-compliant
See Tasks 1–3 below.

---

## Design System v3 Token Reference

From `frontend/tailwind.config.ts`:

| Token | Hex | Use for |
|-------|-----|---------|
| `text-status-safe` | `#16a34a` | winning, ahead, positive |
| `text-status-bubble` | `#d97706` | caution, on-pace, neutral |
| `text-status-behind` | `#ea580c` | behind, warning |
| `text-status-lost` | `#dc2626` | losing, error, danger |
| `text-accent-primary` | `#2563eb` | primary action |
| `text-signal-consider` | `#0891b2` | watch, cold, neutral signal |
| `bg-accent-primary` | `#2563eb` | progress bars, primary fill |

---

## Files to Modify

- Modify: `frontend/app/(dashboard)/war-room/roster/page.tsx` (lines 59–61, 187–190, 228)
- Modify: `frontend/app/(dashboard)/war-room/waiver/page.tsx` (lines 77, 84–90)
- Modify: `frontend/app/(dashboard)/war-room/streaming/page.tsx` (line 36)

---

## Task 1: Fix `roster/page.tsx` — SLOT_COLORS + IP pace + progress bar

**Files:**
- Modify: `frontend/app/(dashboard)/war-room/roster/page.tsx`

### Issue 1a: SLOT_COLORS — dark badges on light background (lines 59–61)
Current (`bg-blue-900/30 text-blue-400` renders as low-contrast light text on semi-transparent dark background — bad on white):
```tsx
const SLOT_COLORS: Record<string, string> = {
  BN: 'bg-bg-elevated text-text-secondary',
  IL: 'bg-status-lost/10 text-status-lost',
  IL60: 'bg-status-lost/10 text-status-lost',
  SP: 'bg-blue-900/30 text-blue-400',
  RP: 'bg-purple-900/30 text-purple-400',
  P: 'bg-purple-900/30 text-purple-400',
}
```

### Issue 1b: IP pace color (lines 186–190)
Current (dark-only colors):
```tsx
const paceColor = budget.ip_pace === 'BEHIND'
  ? 'text-rose-400'
  : budget.ip_pace === 'AHEAD'
    ? 'text-emerald-400'
    : 'text-amber-400'
```

### Issue 1c: IP pace progress bar fill (line 228)
Current:
```tsx
className="h-full rounded-full bg-blue-500"
```

- [ ] **Step 1: Fix SLOT_COLORS**

Replace lines 55–62 in `frontend/app/(dashboard)/war-room/roster/page.tsx`:
```tsx
const SLOT_COLORS: Record<string, string> = {
  BN: 'bg-bg-elevated text-text-secondary',
  IL: 'bg-status-lost/10 text-status-lost',
  IL60: 'bg-status-lost/10 text-status-lost',
  SP: 'bg-blue-50 text-blue-700',
  RP: 'bg-purple-50 text-purple-700',
  P: 'bg-purple-50 text-purple-700',
}
```

- [ ] **Step 2: Fix IP pace color**

Replace lines 186–190 in `frontend/app/(dashboard)/war-room/roster/page.tsx`:
```tsx
const paceColor = budget.ip_pace === 'BEHIND'
  ? 'text-status-lost'
  : budget.ip_pace === 'AHEAD'
    ? 'text-status-safe'
    : 'text-status-bubble'
```

- [ ] **Step 3: Fix progress bar fill**

Replace line 228 in `frontend/app/(dashboard)/war-room/roster/page.tsx`:
```tsx
className="h-full rounded-full bg-accent-primary"
```

- [ ] **Step 4: Verify no other dark-theme remnants in roster/page.tsx**

Run:
```powershell
Select-String -Path "frontend/app/(dashboard)/war-room/roster/page.tsx" -Pattern "blue-[3-9]00|purple-[3-9]00|rose-[34]|emerald-[34]|amber-[34]"
```
Expected: no matches.

- [ ] **Step 5: Commit**

```powershell
git add "frontend/app/(dashboard)/war-room/roster/page.tsx"
git commit -m "fix(ui): DS v3 — roster page slot badge + IP pace colors"
```

---

## Task 2: Fix `waiver/page.tsx` — positionBadgeClass + COLD badge

**Files:**
- Modify: `frontend/app/(dashboard)/war-room/waiver/page.tsx`

### Issue 2a: COLD badge (line 77)
Current (sky-400 not in DS v3):
```tsx
<span className="flex items-center gap-0.5 text-[10px] text-sky-400 font-semibold">
  <Snowflake className="h-3 w-3" /> COLD
</span>
```

### Issue 2b: positionBadgeClass (lines 84–90)
Current (all dark theme `*-900/30 *-400`):
```tsx
function positionBadgeClass(pos: string): string {
  if (pos === 'SP') return 'bg-blue-900/30 text-blue-400'
  if (pos === 'RP' || pos === 'P') return 'bg-purple-900/30 text-purple-400'
  if (pos === 'OF' || pos === 'LF' || pos === 'CF' || pos === 'RF') return 'bg-emerald-900/30 text-emerald-400'
  if (pos === 'C') return 'bg-amber-900/30 text-amber-400'
  if (pos === '1B' || pos === '3B') return 'bg-orange-900/30 text-orange-400'
  if (pos === '2B' || pos === 'SS' || pos === 'MI') return 'bg-sky-900/30 text-sky-400'
  return 'bg-bg-elevated text-text-secondary'
}
```

- [ ] **Step 1: Fix COLD badge**

Replace lines 76–80 in `frontend/app/(dashboard)/war-room/waiver/page.tsx`:
```tsx
return (
  <span className="flex items-center gap-0.5 text-[10px] text-signal-consider font-semibold">
    <Snowflake className="h-3 w-3" /> COLD
  </span>
)
```

- [ ] **Step 2: Fix positionBadgeClass**

Replace lines 83–91 in `frontend/app/(dashboard)/war-room/waiver/page.tsx`:
```tsx
function positionBadgeClass(pos: string): string {
  if (pos === 'SP') return 'bg-blue-50 text-blue-700'
  if (pos === 'RP' || pos === 'P') return 'bg-purple-50 text-purple-700'
  if (pos === 'OF' || pos === 'LF' || pos === 'CF' || pos === 'RF') return 'bg-emerald-50 text-emerald-700'
  if (pos === 'C') return 'bg-amber-50 text-amber-700'
  if (pos === '1B' || pos === '3B') return 'bg-orange-50 text-orange-700'
  if (pos === '2B' || pos === 'SS' || pos === 'MI') return 'bg-sky-50 text-sky-700'
  return 'bg-bg-elevated text-text-secondary'
}
```

- [ ] **Step 3: Verify no other dark-theme remnants in waiver/page.tsx**

Run:
```powershell
Select-String -Path "frontend/app/(dashboard)/war-room/waiver/page.tsx" -Pattern "blue-[3-9]00|purple-[3-9]00|sky-[34]00|emerald-[3-9]00|amber-[3-9]00|orange-[3-9]00"
```
Expected: no matches.

- [ ] **Step 4: Commit**

```powershell
git add "frontend/app/(dashboard)/war-room/waiver/page.tsx"
git commit -m "fix(ui): DS v3 — waiver page position badges + COLD badge color"
```

---

## Task 3: Fix `streaming/page.tsx` — error state

**Files:**
- Modify: `frontend/app/(dashboard)/war-room/streaming/page.tsx`

### Issue: Error state uses `text-rose-400` (line 36)
Current:
```tsx
<div className="flex items-center gap-2 text-rose-400">
```

- [ ] **Step 1: Fix error state color**

Replace line 36 in `frontend/app/(dashboard)/war-room/streaming/page.tsx`:
```tsx
<div className="flex items-center gap-2 text-status-lost">
```

- [ ] **Step 2: Verify no other dark-theme remnants in streaming/page.tsx**

Run:
```powershell
Select-String -Path "frontend/app/(dashboard)/war-room/streaming/page.tsx" -Pattern "rose-[34]|sky-[34]00|blue-[3-9]00|purple-[3-9]00"
```
Expected: no matches.

- [ ] **Step 3: Commit**

```powershell
git add "frontend/app/(dashboard)/war-room/streaming/page.tsx"
git commit -m "fix(ui): DS v3 — streaming page error state color"
```

---

## Task 4: Verify full suite still passes + merge to stable

- [ ] **Step 1: Run backend tests**

```powershell
venv/Scripts/python -m pytest tests/ -q --tb=short --ignore=tests/test_roster_waiver_enrichment_contract.py
```
Expected: all tests pass (ignore the known-slow `test_waiver_populates_percent_owned_from_ownership_subresource`).

- [ ] **Step 2: Verify frontend TypeScript compiles**

```powershell
cd frontend && npx tsc --noEmit 2>&1 | head -30
```
Expected: no errors.

- [ ] **Step 3: Merge to stable/cbb-prod**

```powershell
git checkout stable/cbb-prod
git merge main --no-edit
git checkout main
```

---

## Verification Checklist

| Check | Expected |
|-------|----------|
| `roster/page.tsx` — no `*-900/30` or `*-400` | ✅ no matches |
| `waiver/page.tsx` — no `*-900/30` or `*-400` | ✅ no matches |
| `streaming/page.tsx` — no `rose-4` | ✅ no matches |
| SP badge renders on white background | Light blue pill — readable |
| RP badge renders on white background | Light purple pill — readable |
| IP pace BEHIND shows DS v3 red | `text-status-lost` = `#dc2626` |
| IP pace AHEAD shows DS v3 green | `text-status-safe` = `#16a34a` |
| COLD badge color | `text-signal-consider` = cyan `#0891b2` |
| Error state on streaming | `text-status-lost` = `#dc2626` |
