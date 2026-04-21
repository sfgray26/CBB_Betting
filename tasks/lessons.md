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

### Exact Yahoo keys still need identity validation
- **Lesson**: A matching `player_id_mapping.yahoo_key` is not sufficient to trust roster membership for lineup decisions. Stale or incorrect Yahoo-to-BDL rows can still point at the wrong BDL player. Validate the mapped player's normalized name against the live Yahoo roster name before admitting that BDL ID into lineup optimization.
- **Context**: Decisions page showed non-roster players (for example, Shane Smith) because roster filtering trusted a bad `yahoo_key -> bdl_id` mapping row. The fix rejects mismatched names during `daily_ingestion` roster resolution.

### Per-date decision snapshots must replace, not accumulate
- **Lesson**: `decision_results` is effectively a daily snapshot, not an append-only event stream. Rerunning decision optimization for the same `as_of_date` must delete stale lineup/waiver rows before inserting fresh ones, otherwise impossible duplicate slots and removed players persist in the UI.
- **Context**: The lineup page showed two `Util` rows and two `P` rows after reruns because rows were upserted by `(as_of_date, decision_type, bdl_player_id)` without clearing players that dropped out of the latest optimized lineup.

### Existing backend module
- `backend/fantasy_baseball/` has a rich set of modules: `player_board.py`, `draft_engine.py`, `projections_loader.py`, `daily_lineup_optimizer.py`, `keeper_engine.py`, `yahoo_client.py`, `qwen_advisor.py`.
- Yahoo API credentials are already in `.env.example`.
- `player_board.get_board()` returns a list of player dicts sorted by rank; each has `id`, `name`, `team`, `positions`, `type`, `tier`, `rank`, `adp`, `proj`, `z_score`, `cat_scores`.
- When adding fantasy API routes, wire directly to these modules — do NOT reimplement ranking logic.

### Waiver drop logic must blend short-term output with hold value
- **Lesson**: Waiver recommendations cannot compare a free agent only against a roster player's current z-score or recent signal. Drop logic needs a long-term hold floor from projection metadata such as tier, ADP, ownership, and locked-upside risk profiles, or it will recommend shallow recency-biased cuts.
- **Context**: April 20 fantasy UAT surfaced risky suggestions like dropping Eury Perez for a short-term streamer and concern about elite players like Juan Soto being treated as disposable after a short slump.

### Roster optimize identity should key off yahoo_key first
- **Lesson**: `/api/fantasy/roster/optimize` cannot rely on the numeric Yahoo tail alone when resolving roster players to `PlayerIDMapping`. The stable linkage is `yahoo_key` first, with name-validated fallback paths; otherwise valid roster players miss `player_scores` and collapse to neutral fallback scoring.
- **Context**: April 20 optimizer triage showed live responses where every player returned `lineup_score: 50.0` because full roster keys were not matching the canonical identity table reliably.

### Gemini swimlane violations must be treated as unreviewed patches
- **Lesson**: If Gemini reports Python code edits, do not treat them as ready-to-deploy fixes. Gemini is restricted from runtime code changes in this repo; any such edits need architect review plus targeted tests before they can enter a deployment bundle.
- **Context**: April 20 post-deploy triage included Gemini edits to `row_projector.py`, `category_math.py`, and `scoreboard_orchestrator.py`. The patch compiled but immediately failed `tests/test_scoreboard_orchestrator.py` with `ValueError: Denominator must be positive for OPS, got 0.0`.

### ROW projection helpers must preserve default games_remaining behavior
- **Lesson**: Do not coerce an omitted `games_remaining` input to `{}` before calling `compute_row_projection()`. The projector treats missing games as zero, so converting `None` to an empty dict silently zeroes every player projection and collapses ratio denominators like `OPS` to 0.
- **Context**: April 20 scoreboard triage showed `GET /api/fantasy/scoreboard` failing with `ValueError: Denominator must be positive for OPS, got 0.0` because `_project_row_from_player_scores()` replaced the projector's default one-game fallback with an empty dict.

### Read endpoints need schema-tolerant fallbacks for evolving analytics tables
- **Lesson**: Endpoints that expose derived analytics tables should not hard-crash when production lags the newest ORM model by a nullable column or two. For read-only paths like `player_scores`, add a schema-inspected fallback query so old tables degrade to 200/404 instead of 500.
- **Context**: April 20 UAT showed `GET /api/fantasy/players/1/scores` returning `ProgrammingError` in production even though the logical outcome should have been 200 or 404.
