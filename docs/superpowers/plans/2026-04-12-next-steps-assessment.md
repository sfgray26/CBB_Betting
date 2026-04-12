# Next Steps Assessment & Fantasy Baseball Frontend Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assess project state, identify the highest-impact next step, and provide a plan to build the Fantasy Baseball frontend — the single biggest gap in the platform.

**Architecture:** Next.js pages consuming existing `/api/fantasy/*` endpoints, applying Revolut design tokens from K-44. Sidebar restructured from CBB-only to dual-purpose with Fantasy Baseball as the primary section.

**Tech Stack:** Next.js 14, React Query, Tailwind CSS, Lucide icons, existing `apiFetch` helper

---

## Strategic Assessment

### Current State (April 12, 2026)

| Area | Status | Impact |
|------|--------|--------|
| **CBB Season** | CLOSED permanently | The entire current frontend serves a dead product |
| **MLB Fantasy (Priority 1)** | Backend: 40+ files, 25+ API endpoints. Frontend: **ZERO pages** | Critical gap — users have no UI |
| **MLB Betting (Priority 2)** | Stub-level (`mlb_analysis.py`) | Not ready for frontend yet |
| **Tasks 4/8/9** | Code complete, pending Gemini deploy | Blocked on ops, not Claude |
| **Task 5 (Statcast backfill)** | Gemini assignment | Blocked on ops |
| **Task 6 (Ingestion logging)** | Medium priority | Not urgent |
| **K-44 (UI/UX Revolut redesign)** | Plan written, ready to implement | Should be applied to new pages, not old CBB ones |
| **K-39..K-43 (Infra hardening)** | Kimi research pending | Independent track |

### The Biggest Gap

The frontend is **100% CBB betting** — sidebar says "CBB EDGE Analytics", every page is betting-focused (Dashboard, Performance, CLV, Bet History, Calibration, Alerts, Today's Bets, Live Slate, Odds Monitor, Bracket, Risk Dashboard). **There is not a single page for fantasy baseball.**

Meanwhile, the backend has a complete fantasy baseball API:

| Endpoint Group | Endpoints | Purpose |
|----------------|-----------|---------|
| **Roster** | `GET /api/fantasy/roster` | Current Yahoo roster |
| **Lineup** | `GET /api/fantasy/lineup/{date}`, `POST /api/fantasy/lineup`, `GET /api/fantasy/lineup/elite-optimize/{date}` | Daily lineup optimization |
| **Matchup** | `GET /api/fantasy/matchup`, `POST /api/fantasy/matchup/simulate` | H2H matchup analysis + Monte Carlo |
| **Waiver** | `GET /api/fantasy/waiver`, `GET /api/fantasy/waiver/recommendations`, `POST /api/fantasy/waiver/add` | Waiver wire intelligence |
| **Briefing** | `GET /api/fantasy/briefing/{date}` | Daily morning briefing |
| **Valuations** | `GET /api/fantasy/players/valuations` | Player value rankings |
| **Dashboard Stream** | `GET /api/fantasy/dashboard/stream` | Real-time fantasy dashboard |
| **Draft** | `GET /api/fantasy/draft-board`, `POST /api/fantasy/draft-session`, etc. | Draft assistant (6 endpoints) |
| **Scarcity** | `POST /api/fantasy/lineup/analyze-scarcity` | Positional scarcity analysis |

### Recommended Priority Order

1. **Fantasy Baseball Frontend** — This is what users actually need right now. The backend is ready. Build 5-6 core pages consuming existing API endpoints. Apply K-44 Revolut design while building (don't redesign CBB pages that serve a dead product).

2. **Sidebar Restructure** — Rebrand from "CBB EDGE" to something dual-purpose. Make Fantasy Baseball the primary nav section. Move CBB pages to an "Archive" or "CBB (Closed)" section.

3. **Gemini Deploy (Task 5)** — Parallel track, no Claude dependency.

4. **Data Ingestion Logging (Task 6)** — After frontend is in place.

5. **Simulation Staleness Investigation** — Quick sidebar task.

---

## Plan: Fantasy Baseball Frontend (Phase 1 — Core Pages)

This plan covers the minimum viable fantasy frontend: 5 pages that surface the most critical daily-use features. Each page consumes existing backend endpoints — no backend changes needed.

### File Structure

```
frontend/
  app/(dashboard)/
    fantasy/                        # NEW - Fantasy Baseball section
      page.tsx                      # Fantasy home / daily briefing
      roster/page.tsx               # Current roster view
      lineup/page.tsx               # Lineup optimizer
      matchup/page.tsx              # H2H matchup analysis
      waiver/page.tsx               # Waiver wire recommendations
  components/
    fantasy/                        # NEW - Fantasy-specific components
      player-card.tsx               # Reusable player display card
      stat-bar.tsx                  # Horizontal stat comparison bar
      position-badge.tsx            # Position indicator pill
      category-tracker.tsx          # H2H category win/loss tracker
  lib/
    fantasy-api.ts                  # NEW - Fantasy API endpoint functions
    fantasy-types.ts                # NEW - Fantasy TypeScript types
  components/layout/
    sidebar.tsx                     # MODIFY - Add Fantasy section, restructure nav
```

---

### Task 1: Fantasy API Client & Types

**Files:**
- Create: `frontend/lib/fantasy-api.ts`
- Create: `frontend/lib/fantasy-types.ts`
- Modify: `frontend/lib/api.ts` (add reexport)

- [ ] **Step 1: Explore backend response shapes**

Read the backend endpoint handlers to determine response schemas:

```bash
# Check roster response shape
grep -A 30 'class RosterResponse' backend/main.py
# Check lineup response shape
grep -A 30 'class DailyLineupResponse' backend/main.py
# Check matchup response shape
grep -A 30 'class MatchupResponse' backend/main.py
# Check waiver response shape
grep -A 30 'class WaiverWireResponse' backend/main.py
```

- [ ] **Step 2: Create `frontend/lib/fantasy-types.ts`**

Define TypeScript interfaces matching every backend response model used by the fantasy endpoints. Include at minimum:

- `FantasyPlayer` (name, team, positions, stats, yahoo_key)
- `RosterEntry` (player + roster slot + eligibility)
- `RosterResponse` (entries[], league_info)
- `LineupSlot` (player, position, projected_points, is_starter)
- `DailyLineupResponse` (lineup[], bench[], optimization_score)
- `MatchupCategory` (category, my_value, opp_value, projected_winner)
- `MatchupResponse` (categories[], win_prob, my_team, opp_team)
- `WaiverCandidate` (player, add_value, drop_candidate, priority_score)
- `WaiverWireResponse` (candidates[], last_updated)
- `DailyBriefing` (headline, lineup_notes, waiver_alerts, injury_updates, matchup_summary)

> **Important:** Read each backend response model before typing. Do not guess field names.

- [ ] **Step 3: Create `frontend/lib/fantasy-api.ts`**

```typescript
import { apiFetch } from '@/lib/api'
import type {
  RosterResponse,
  DailyLineupResponse,
  MatchupResponse,
  WaiverWireResponse,
  DailyBriefing,
  PlayerValuation,
} from './fantasy-types'

// Format: YYYY-MM-DD
function today(): string {
  return new Date().toISOString().slice(0, 10)
}

export const fantasyEndpoints = {
  getRoster: () => apiFetch<RosterResponse>('/api/fantasy/roster'),

  getLineup: (date?: string) =>
    apiFetch<DailyLineupResponse>(`/api/fantasy/lineup/${date ?? today()}`),

  optimizeLineup: (date?: string) =>
    apiFetch<DailyLineupResponse>(`/api/fantasy/lineup/elite-optimize/${date ?? today()}`),

  saveLineup: (lineup: unknown) =>
    apiFetch<{ success: boolean }>('/api/fantasy/lineup', {
      method: 'POST',
      body: JSON.stringify(lineup),
    }),

  getMatchup: () => apiFetch<MatchupResponse>('/api/fantasy/matchup'),

  simulateMatchup: () =>
    apiFetch<unknown>('/api/fantasy/matchup/simulate', { method: 'POST' }),

  getWaiverWire: () => apiFetch<WaiverWireResponse>('/api/fantasy/waiver'),

  getWaiverRecommendations: () =>
    apiFetch<unknown>('/api/fantasy/waiver/recommendations'),

  getBriefing: (date?: string) =>
    apiFetch<DailyBriefing>(`/api/fantasy/briefing/${date ?? today()}`),

  getValuations: () =>
    apiFetch<PlayerValuation[]>('/api/fantasy/players/valuations'),

  getDashboardStream: () =>
    apiFetch<unknown>('/api/fantasy/dashboard/stream'),
}
```

- [ ] **Step 4: Verify types compile**

```bash
cd frontend && npx tsc --noEmit lib/fantasy-types.ts lib/fantasy-api.ts
```

Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/fantasy-types.ts frontend/lib/fantasy-api.ts
git commit -m "feat: add Fantasy Baseball API client and TypeScript types"
```

---

### Task 2: Sidebar Restructure

**Files:**
- Modify: `frontend/components/layout/sidebar.tsx`

- [ ] **Step 1: Write the updated sidebar nav structure**

Replace the current `navSections` array with a restructured version:

```typescript
import {
  LayoutDashboard, Users, Calendar, Swords, Search,
  BarChart2, TrendingUp, ClipboardList, Target, Bell,
  Zap, Activity, Radio, ShieldAlert,
} from 'lucide-react'

const navSections = [
  {
    label: 'Fantasy Baseball',
    items: [
      { href: '/fantasy', label: 'Home', icon: LayoutDashboard },
      { href: '/fantasy/roster', label: 'My Roster', icon: Users },
      { href: '/fantasy/lineup', label: 'Lineup Optimizer', icon: Calendar },
      { href: '/fantasy/matchup', label: 'Matchup', icon: Swords },
      { href: '/fantasy/waiver', label: 'Waiver Wire', icon: Search },
    ],
    soon: false,
    hidden: false,
  },
  {
    label: 'CBB Archive',
    items: [
      { href: '/dashboard', label: 'Dashboard', icon: BarChart2 },
      { href: '/today', label: "Today's Bets", icon: Zap },
      { href: '/performance', label: 'Performance', icon: TrendingUp },
      { href: '/bet-history', label: 'Bet History', icon: ClipboardList },
    ],
    soon: false,
    hidden: false,
  },
  {
    label: 'Admin',
    items: [
      { href: '/admin', label: 'Admin', icon: ShieldAlert },
    ],
    soon: false,
    hidden: false,
  },
]
```

- [ ] **Step 2: Update the logo/branding area**

Change the sidebar header from:
```tsx
<div className="font-bold text-lg text-amber-400 tracking-tight">
  CBB EDGE
</div>
<div className="text-xs text-zinc-500 mt-0.5">
  Analytics
</div>
```

To:
```tsx
<div className="font-bold text-lg text-amber-400 tracking-tight">
  EDGE
</div>
<div className="text-xs text-zinc-500 mt-0.5">
  Fantasy + Analytics
</div>
```

- [ ] **Step 3: Remove the portfolio chip from the bottom panel**

The portfolio chip shows CBB betting drawdown/exposure — irrelevant for fantasy. Replace with a simpler status indicator or remove entirely.

- [ ] **Step 4: Update the root redirect**

Change `frontend/app/(dashboard)/page.tsx` to redirect to `/fantasy` instead of `/dashboard`:

```typescript
import { redirect } from 'next/navigation'

export default function DashboardRoot() {
  redirect('/fantasy')
}
```

- [ ] **Step 5: Verify sidebar renders**

```bash
cd frontend && npm run dev
```

Navigate to `http://localhost:3000` in browser. Verify:
- Sidebar shows "Fantasy Baseball" as the primary section
- CBB pages are grouped under "CBB Archive"
- Logo says "EDGE" not "CBB EDGE"
- Root `/` redirects to `/fantasy`

- [ ] **Step 6: Commit**

```bash
git add frontend/components/layout/sidebar.tsx frontend/app/\(dashboard\)/page.tsx
git commit -m "feat: restructure sidebar — Fantasy Baseball primary, CBB archived"
```

---

### Task 3: Fantasy Home Page (Daily Briefing)

**Files:**
- Create: `frontend/app/(dashboard)/fantasy/page.tsx`

- [ ] **Step 1: Read the briefing endpoint response shape**

```bash
grep -A 50 'def.*briefing_date' backend/main.py | head -60
```

- [ ] **Step 2: Create the Fantasy Home page**

This page is the daily command center. It calls `GET /api/fantasy/briefing/{date}` and displays:

1. **Headline section** — date, matchup opponent, win probability
2. **Lineup alerts** — players on IL, bench bats with good matchups, pitchers with bad weather
3. **Waiver wire highlights** — top 3 pickups ranked by add value
4. **Matchup preview** — category-by-category projected wins/losses
5. **Injury updates** — roster players with status changes

Use React Query with `staleTime: 5 * 60_000` and `refetchInterval: 10 * 60_000`.

Layout: Single-column stack of cards. Each card has a header with icon + title, and content area.

> Read the actual backend response before building this. The briefing endpoint may return a different shape than expected. Adapt the UI to what the endpoint actually returns.

- [ ] **Step 3: Verify the page renders**

```bash
cd frontend && npm run dev
```

Navigate to `/fantasy`. Verify it loads without errors (it may show empty state if backend isn't running — that's fine, but it should not crash).

- [ ] **Step 4: Commit**

```bash
git add frontend/app/\(dashboard\)/fantasy/page.tsx
git commit -m "feat: add Fantasy home page with daily briefing"
```

---

### Task 4: Roster Page

**Files:**
- Create: `frontend/app/(dashboard)/fantasy/roster/page.tsx`
- Create: `frontend/components/fantasy/position-badge.tsx`

- [ ] **Step 1: Read the roster endpoint response shape**

```bash
grep -A 50 'class RosterResponse' backend/main.py
grep -A 80 '/api/fantasy/roster' backend/main.py | head -100
```

- [ ] **Step 2: Create the position badge component**

Small pill component showing position (C, 1B, 2B, SS, 3B, OF, SP, RP, BN, IL) with color coding:
- Catchers/Infield: blue-ish
- Outfield: green
- Pitchers: orange
- Bench/IL: gray

```tsx
// frontend/components/fantasy/position-badge.tsx
const positionColors: Record<string, string> = {
  C: 'bg-blue-500/20 text-blue-400',
  '1B': 'bg-blue-500/20 text-blue-400',
  '2B': 'bg-blue-500/20 text-blue-400',
  SS: 'bg-blue-500/20 text-blue-400',
  '3B': 'bg-blue-500/20 text-blue-400',
  OF: 'bg-emerald-500/20 text-emerald-400',
  SP: 'bg-orange-500/20 text-orange-400',
  RP: 'bg-orange-500/20 text-orange-400',
  BN: 'bg-zinc-500/20 text-zinc-400',
  IL: 'bg-rose-500/20 text-rose-400',
  'IL+': 'bg-rose-500/20 text-rose-400',
  Util: 'bg-purple-500/20 text-purple-400',
}

export function PositionBadge({ position }: { position: string }) {
  const colors = positionColors[position] ?? 'bg-zinc-500/20 text-zinc-400'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${colors}`}>
      {position}
    </span>
  )
}
```

- [ ] **Step 3: Create the Roster page**

Displays the current Yahoo roster in a table:
- Columns: Position | Player | Team | Key Stats (AVG/HR/RBI for batters, ERA/WHIP/K for pitchers)
- Group by: Starters (by position) → Bench → IL
- Include an "auto" refresh every 5 minutes via React Query
- Show "Last synced" timestamp if available in the response

> Read the actual backend response before building. Adapt columns and grouping to match what the API returns.

- [ ] **Step 4: Verify the page renders**

Navigate to `/fantasy/roster`. Verify layout, no console errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/\(dashboard\)/fantasy/roster/page.tsx frontend/components/fantasy/position-badge.tsx
git commit -m "feat: add Fantasy roster page with position badges"
```

---

### Task 5: Lineup Optimizer Page

**Files:**
- Create: `frontend/app/(dashboard)/fantasy/lineup/page.tsx`

- [ ] **Step 1: Read the lineup endpoint response shapes**

```bash
grep -A 50 'class DailyLineupResponse' backend/main.py
grep -A 80 '/api/fantasy/lineup/' backend/main.py | head -100
grep -A 30 'elite-optimize' backend/main.py | head -40
```

- [ ] **Step 2: Create the Lineup Optimizer page**

Core features:
1. **Date picker** — defaults to today, can select future dates
2. **Current lineup display** — table showing who's starting at each position, with projected points
3. **"Optimize" button** — calls `GET /api/fantasy/lineup/elite-optimize/{date}` and shows the optimized result
4. **Diff view** — highlight changes between current and optimized lineup (swaps in green, drops in red)
5. **"Set Lineup" button** — calls `POST /api/fantasy/lineup` with the optimized lineup
6. **Optimization score** — show the total projected score for current vs. optimized

State management:
- `currentLineup` from `GET /api/fantasy/lineup/{date}`
- `optimizedLineup` from the optimize button (stored in local state)
- `isOptimizing` loading state
- `isSaving` loading state

- [ ] **Step 3: Handle the async optimization flow**

The backend has `POST /api/fantasy/lineup/async-optimize` which returns a job ID, and `GET /api/fantasy/jobs/{job_id}` to poll. If the sync `elite-optimize` endpoint is slow (>5s), switch to the async flow with a polling interval.

- [ ] **Step 4: Verify the page renders**

Navigate to `/fantasy/lineup`. Verify date picker works, optimize button shows loading state, layout is correct.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/\(dashboard\)/fantasy/lineup/page.tsx
git commit -m "feat: add Fantasy lineup optimizer page with diff view"
```

---

### Task 6: Matchup Analysis Page

**Files:**
- Create: `frontend/app/(dashboard)/fantasy/matchup/page.tsx`
- Create: `frontend/components/fantasy/category-tracker.tsx`

- [ ] **Step 1: Read the matchup endpoint response shapes**

```bash
grep -A 50 'class MatchupResponse' backend/main.py
grep -A 80 '/api/fantasy/matchup' backend/main.py | head -100
```

- [ ] **Step 2: Create the category tracker component**

Horizontal bar showing each H2H category (R, HR, RBI, SB, AVG, W, SV, K, ERA, WHIP) with:
- Your value on the left
- Opponent value on the right
- Color: green if winning, red if losing, gray if tied
- A total "Categories Won" counter at the top

```tsx
// frontend/components/fantasy/category-tracker.tsx
interface CategoryRowProps {
  category: string
  myValue: number
  oppValue: number
  lowerIsBetter?: boolean  // ERA, WHIP
}
```

- [ ] **Step 3: Create the Matchup page**

1. **Header** — "Week X vs. [Opponent Name]" with overall win probability
2. **Category grid** — using CategoryTracker component for each stat
3. **"Simulate" button** — calls `POST /api/fantasy/matchup/simulate` for Monte Carlo projection
4. **Simulation results** — win%, tie%, loss% displayed as a horizontal stacked bar
5. **Key differentials** — highlight the 2-3 categories that are closest (toss-ups)

- [ ] **Step 4: Verify the page renders**

Navigate to `/fantasy/matchup`. Verify category bars render, simulate button works.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/\(dashboard\)/fantasy/matchup/page.tsx frontend/components/fantasy/category-tracker.tsx
git commit -m "feat: add Fantasy matchup analysis page with category tracker"
```

---

### Task 7: Waiver Wire Page

**Files:**
- Create: `frontend/app/(dashboard)/fantasy/waiver/page.tsx`

- [ ] **Step 1: Read the waiver endpoint response shapes**

```bash
grep -A 50 'class WaiverWireResponse' backend/main.py
grep -A 80 '/api/fantasy/waiver' backend/main.py | head -100
```

- [ ] **Step 2: Create the Waiver Wire page**

1. **Top recommendations** — ranked list of waiver pickups with add-value scores
2. **For each candidate**: player name, team, positions, key stats, ownership %, recommended drop
3. **"Add" button** per candidate — calls `POST /api/fantasy/waiver/add`
4. **Filter controls** — position filter (All, C, 1B, 2B, SS, 3B, OF, SP, RP), sort by (value, recent performance)
5. **Recommendation engine results** — separate section from `GET /api/fantasy/waiver/recommendations`

- [ ] **Step 3: Verify the page renders**

Navigate to `/fantasy/waiver`. Verify list renders, filters work, add button shows confirmation.

- [ ] **Step 4: Commit**

```bash
git add frontend/app/\(dashboard\)/fantasy/waiver/page.tsx
git commit -m "feat: add Fantasy waiver wire page with recommendations"
```

---

## Post-Plan: What Comes After

Once the 7 tasks above are complete, the next priorities are:

| Priority | Item | Est. Effort |
|----------|------|-------------|
| 1 | Apply K-44 Revolut design tokens to new fantasy pages | 1-2 days |
| 2 | Fantasy player detail page (`/fantasy/player/{id}`) | 0.5 day |
| 3 | Data ingestion logging (Task 6) | 0.5 day |
| 4 | Investigate simulation staleness | 1 hour |
| 5 | Draft assistant pages (off-season priority) | 1-2 days |

---

## Scope Check

This plan covers a single subsystem (Fantasy Baseball Frontend) and produces working, testable software on its own. The K-44 Revolut design system is intentionally deferred to a follow-up pass — getting functional pages live is more important than pixel-perfect design when users currently have zero UI.
