# Lessons Learned — CBB Edge Analyzer
*Updated: 2026-03-19*

---

## Test Infrastructure

### pytest-asyncio strict mode
- **Lesson**: When `asyncio_mode = strict` is set in `pytest.ini`, ALL async test functions must have `@pytest.mark.asyncio`. Do NOT rely on auto-discovery.
- **Context**: CI failure on `test_line_monitor.py` — async tests silently passed/failed without the decorator.

### GitHub Actions PostgreSQL service container
- **Lesson**: The PostgreSQL service container only creates the default `postgres` database. Named databases (e.g., `test_cbb_edge`) must be created explicitly with `psql -c "CREATE DATABASE ..."` before running `init_db.py`.
- **Context**: CI job "Set up test database" failed because `test_cbb_edge` didn't exist.

### Class method mocks vs instance method mocks
- **Lesson**: `OddsAPIClient.quota_is_low()` is a CLASS METHOD. When mocking it, mock on the class (`mock_cls.quota_is_low.return_value = False`), not on the instance (`mock_instance.quota_is_low`). A `MagicMock()` without a return value is truthy, which caused the quota guard to always fire.
- **Context**: `test_check_line_movements_*` tests always hit the early `return` because `quota_is_low()` returned a truthy MagicMock.

---

## Build System

### Backend image must not depend on frontend assets
- **Lesson**: If `.dockerignore` excludes `frontend/`, backend runtime code cannot import JSON/config files from `frontend/*` at module import time. Keep backend runtime assets under `backend/` and optionally probe frontend paths only as non-required fallbacks.
- **Context**: Railway boot crash after introducing shared fantasy stat contract path in backend loader (`FileNotFoundError: /app/frontend/lib/fantasy-stat-contract.json`).

### Next.js Google Fonts in network-isolated builders
- **Lesson**: `next/font/google` fetches fonts at build time. In Railway's network-isolated build environment, this causes a build failure. Always pass `preload: false` as an option:
  ```typescript
  const inter = Inter({ subsets: ['latin'], preload: false })
  ```
- **Context**: Railway CI check failing with network error during font fetch.

### TypeScript `&&` vs `??` operator precedence (TS5076)
- **Lesson**: TypeScript 5.x errors on `a && b ?? c` because it's ambiguous. Always parenthesize: `(a ? b : undefined) ?? c`.
- **Context**: `bracket/page.tsx` had `champion && data?.advancement_probs[champion]?.champion_pct ?? 0`.

### Next.js 15 Viewport export
- **Lesson**: In Next.js 15, `viewport` and `themeColor` fields are no longer valid inside `Metadata`. They must be exported separately as `export const viewport: Viewport = { ... }`. Importing `Viewport` type from `next` is required.
- **Context**: TypeScript CI error on `frontend/app/layout.tsx`.

---

## Team Mapping

*(Add entries here when fuzzy matching failures occur)*

---

## Shin Vig Removal

*(Add entries here when Shin method produces unexpected results)*

---

## UAT Fixes — Verify Before Fixing

### Always read the code before acting on UAT bug reports
- **Lesson**: UAT issues may already be fixed in the codebase. Read the actual file before writing a fix. In EMAC-086, issue #10a (min="20" on settlement input) was already `min={1}` in the code — no change needed. Issue #8 (z-scores) was already fully wired in both backend and frontend. Only issues #10b (drawdown 0%) and #3 endpoint (mark-as-placed) required actual changes.
- **Context**: Three out of five sub-issues in UAT Phase 4 were already resolved. Acting without reading would have introduced unnecessary diffs.

### PortfolioManager singleton drawdown always returns 0 without DB load
- **Lesson**: `PortfolioManager` is a singleton that initializes `current_bankroll = starting_bankroll` (drawdown = 0%). It only computes real drawdown after `load_from_db()` is called. Any endpoint that surfaces `drawdown_pct` must call `pm.load_from_db(db)` first. Without this, the Risk Dashboard drawdown gauge always shows 0%.
- **Context**: EMAC-086 fix 1b — `get_portfolio_status` endpoint now accepts `db: Session` dependency and calls `pm.load_from_db(db)` before returning state.

---

## Fantasy Baseball

### Existing backend module
- `backend/fantasy_baseball/` has a rich set of modules: `player_board.py`, `draft_engine.py`, `projections_loader.py`, `daily_lineup_optimizer.py`, `keeper_engine.py`, `yahoo_client.py`, `qwen_advisor.py`.
- Yahoo API credentials are already in `.env.example`.
- `player_board.get_board()` returns a list of player dicts sorted by rank; each has `id`, `name`, `team`, `positions`, `type`, `tier`, `rank`, `adp`, `proj`, `z_score`, `cat_scores`.
- When adding fantasy API routes, wire directly to these modules — do NOT reimplement ranking logic.

### Yahoo fantasy contract must stay league-specific
- **Lesson**: For Yahoo league 72586, roster slots and scoring categories must come from one backend-owned SSOT contract. Do not rebuild stat maps inline in `main.py` or rely on generic `is_only_display_stat` heuristics alone. Parse league settings against the canonical 18-category contract and fall back only to the league-specific stat-id map, never a generic mixed stat map.
- **Context**: Fantasy data-layer hard reset on Apr 3, 2026 fixed zero active scoring categories, matchup column drift, waiver stat leakage, and async lineup payload validation failures.

### Lineup actuation must use Yahoo player keys and full slot expansion
- **Lesson**: The resilient lineup path must operate on Yahoo player keys, not numeric player IDs or names, and must expand the active roster into all 16 concrete league slots including duplicate `SP`, `RP`, and `P` slots. Generic `OF` must never be submitted to Yahoo; outfield eligibility is fine, but actuation positions must be `LF`, `CF`, or `RF`.
- **Context**: Apr 3, 2026 hard reset fixed `Player does not play that position`, `game_ids don't match for player key`, and `get_roster(date=...)` crashes.

### Shared frontend surfaces must be classified, not bulk-deleted
- **Lesson**: In the fantasy UI removal pass, `frontend/lib/api.ts` and `frontend/lib/types.ts` looked fantasy-heavy but were mixed surfaces. `UserPreferences` remained live in `frontend/app/(dashboard)/settings/page.tsx`, dashboard types remained live in `frontend/app/(dashboard)/dashboard/page.tsx`, and async job polling methods had to remain exported even after the fantasy routes were deleted. Always classify each symbol by consumer before deleting a shared file block.
- **Context**: Apr 3, 2026 contract-preserving frontend excision removed `frontend/app/(dashboard)/fantasy/**` while preserving dashboard/settings contracts and async polling helpers.
