# CBB Edge — Frontend Migration Workstream

> **Workstream owner:** Claude Code (Master Architect)
> **Started:** March 18, 2026
> **Goal:** Replace Streamlit dashboard with Next.js 15 production UI. Streamlit stays live on :8501 until Phase 5. New frontend runs on :3000.

---

## Agent Roles in This Workstream

| Agent | Role | What they do NOT do |
|-------|------|----------------------|
| **Claude Code** | Architecture, all TypeScript/React code, debugging, design system | Does not delegate type decisions to other agents |
| **Kimi CLI** | Reads backend source → produces API Ground Truth specs | Does NOT write production code |
| **OpenClaw** | Post-write validation checklist on each component | Does NOT make code changes |

---

## Source of Truth Docs

| Doc | Purpose |
|-----|---------|
| `reports/api_ground_truth.md` | Exact JSON shapes for every API endpoint (produced by Kimi) |
| `frontend/lib/types.ts` | TypeScript interfaces — derived from ground truth, never guessed |
| `tasks/todo.md` | Phase checklist with per-task status |

**Rule:** `lib/types.ts` is only updated from `api_ground_truth.md`. Never from browser error messages or assumptions.

---

## Phase Status

### Phase 0 — Foundation ✅
- [x] Next.js 15 scaffold (`frontend/`)
- [x] Design system: zinc palette, amber/sky/emerald/rose signals, JetBrains Mono for numbers
- [x] TanStack Query v5 provider
- [x] API client (`lib/api.ts`) — cookie-based auth, Railway URL via `NEXT_PUBLIC_API_URL`
- [x] Auth middleware — redirects unauthenticated users to `/login`
- [x] Login page — validates key against `/health`, stores in cookie
- [x] Dashboard shell: sidebar + header layout
- [x] Sidebar: nav items, live portfolio status footer, "Streamlit ↗" link
- [x] CORS fixed on backend — `allow_origins=["*"]` with env var override

### Phase 1 — Core Analytics Pages ✅ COMPLETE
- [x] Performance page (`/performance`)
- [x] CLV Analysis page (`/clv`) — fixed against ground truth spec
- [x] Bet History page (`/bet-history`) — fixed against ground truth spec
- [x] Calibration page (`/calibration`) — fixed against ground truth spec
- [x] Alerts page (`/alerts`) — fixed against ground truth spec

**Source of truth:** `reports/api_ground_truth.md` (produced by Kimi CLI, 2026-03-12)

### Phase 2 — Live Slate & Odds Monitor ✅ DONE
- [x] Today page (`/today`)
- [x] Live Slate page (`/live-slate`)
- [x] Odds Monitor page (`/odds-monitor`)

### Phase 3 — Tournament Experience ✅ DONE
- [x] Bracket page (`/bracket`) with Monte Carlo simulator (10k sims)
- [x] Tournament game tracking
- [x] First Four monitoring

### Phase 4 — Mobile & PWA ✅ DONE
- [x] Viewport meta tag configuration
- [x] Mobile sidebar drawer
- [x] Touch targets (44px minimum)
- [x] Responsive grid patterns
- [x] PWA manifest.json

### Phase 5 — Polish & Decommission ✅ DONE
- [x] Error boundaries (`error.tsx`)
- [x] Loading states (`loading.tsx`)
- [x] Fantasy Baseball integration (`/fantasy/*`)
- [x] Admin Risk Dashboard (`/admin`)

---

## Known Type Mismatches Fixed

These were discovered at runtime — document here so Kimi's spec can prevent recurrence.

| Endpoint | Assumed field | Actual field | Notes |
|----------|--------------|--------------|-------|
| `/api/performance/summary` | `summary.roi` | `summary.overall.roi` | Nested under `overall` |
| `/api/performance/summary` | `summary.win_rate` | `summary.overall.win_rate` | Same nesting |
| `/api/performance/summary` | `summary.avg_clv` | `summary.overall.mean_clv` | Different name |
| `/api/performance/summary` | `summary.total_profit_units` | `summary.overall.total_profit_dollars` | Dollars not units |
| `/api/performance/summary` | `summary.by_type` | `summary.by_bet_type` | Different key name |
| `/api/performance/summary` | `by_type[x].wins` | `by_type[x].win_rate` | Rate not count |
| `/api/performance/summary` | `summary.rolling_7d` | `summary.rolling_windows.last_10` | Different structure entirely |
| `/api/performance/summary` | `roi` as percentage | `roi` as decimal (0.043) | Must × 100 for display |
| All rate fields | percentage | decimal | `win_rate`, `roi`, `mean_clv` all need × 100 |
| Empty state | fields present | `{ message: "...", total_bets: 0 }` | No `overall` key when no data |

---

## OpenClaw Validation Checklist

Run after every component Claude Code writes. Report file + line number. Do not suggest style changes.

```
Review this React/TypeScript component. Check ONLY these issues:

1. NULL SAFETY — any .field access on potentially undefined/null without ?. guard
2. EMPTY ARRAY — any .map() without a ?? [] fallback on the source
3. DECIMAL DISPLAY — any API field named roi/win_rate/prob/clv displayed without ×100 conversion
4. LOADING STATE — every async section has a loading skeleton or spinner
5. CRASH RISK — toFixed/toString/toLocaleString called on a value that could be undefined
6. Object.entries() called without ?? {} guard on the argument
7. Empty state — if data is an empty array or null, is there a user-visible message?

Output: PASS or list of issues with line numbers.
```

---

## Kimi Delegation Prompt — API Ground Truth

Copy-paste to Kimi CLI session:

```
MISSION: Read the CBB Edge backend Python source and produce a definitive API shape document.
Save output to: reports/api_ground_truth.md

Files to read completely:
- backend/services/performance.py
- backend/services/alerts.py
- backend/services/bet_tracker.py
- backend/services/portfolio.py
- backend/services/odds.py
- backend/main.py (find every @app.get/@app.post endpoint and its return statement)

For EACH of these endpoints, document the EXACT JSON response shape:
  GET /api/performance/summary
  GET /api/performance/clv-analysis
  GET /api/performance/calibration
  GET /api/performance/timeline?days=30
  GET /api/bets?status=all&days=60
  GET /api/performance/alerts
  GET /admin/portfolio/status
  GET /api/predictions/today
  GET /api/predictions/today/all

For each endpoint provide ALL of:
  1. Complete JSON shape — every key, nested exactly as returned
  2. Type of each field: str | int | float | bool | null | list | dict
  3. Which fields can be null or absent
  4. The empty/no-data response (many return {"message": "...", count: 0} when DB is empty)
  5. Decimal vs percentage — flag any field that looks like a % but is stored as a decimal
  6. Whether lists can be empty []

Format as TypeScript interfaces + example JSON side by side.
Be exhaustive. Every missing field costs a debug cycle. Do not summarize.
```

---

## Design System Reference

```
background:      zinc-950  (#09090b)
surface:         zinc-900  (#18181b)
surface-raised:  zinc-800  (#27272a)
border:          zinc-700/40

signal-bet:      amber-400 (#fbbf24)  — BET verdict
signal-consider: sky-400   (#38bdf8)  — CONSIDER verdict
signal-win:      emerald-400 (#34d399)
signal-loss:     rose-500  (#f43f5e)

Numbers/odds:    font-mono tabular-nums  (JetBrains Mono)
Body:            Inter
```

**Display conversion rules:**
- All `roi`, `win_rate`, `prob`, `clv` fields from API are decimals → multiply × 100 before displaying as %
- Always show sign on P&L and edge: `+4.2%` not `4.2%`
- Negative values: `text-rose-400`, Positive: `text-emerald-400`, Zero: `text-zinc-400`

---

## Environment

```bash
# Start frontend (port 3000)
cd frontend && npm run dev

# Backend (Railway — no local DB needed)
# Set frontend/.env.local:
NEXT_PUBLIC_API_URL=https://cbbbetting-production.up.railway.app

# Run local backend if needed (requires local PostgreSQL)
uvicorn backend.main:app --reload
```

---

## Architectural Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Mar 18 | Frontend in `frontend/` subdirectory (monorepo) | Keeps everything in one repo, Railway deploys backend only |
| Mar 18 | Client-side data fetching (TanStack Query) over RSC | Simpler auth flow, no server-side cookie forwarding complexity |
| Mar 18 | API key in cookie, direct browser→Railway calls | Matches existing auth model; no Next.js API proxy layer needed |
| Mar 18 | `allow_origins=["*"]` on backend CORS | API key auth makes wildcard safe; `allow_credentials=False` required with wildcard |
| Mar 18 | Kimi produces spec before Claude writes types | Prevents the runtime type mismatch bug class entirely |
