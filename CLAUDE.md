# CLAUDE.md — Project Orientation

> Read this once per session. Complements `.claude/rules/workflow.md` (wake-up routine) and
> `HANDOFF.md` (current mission state). Do not duplicate — reference those files for detail.

---

## What This Project Is

**Dual-purpose platform: Fantasy Baseball (priority 1) + MLB Sports Betting (priority 2).**
Previously a CBB betting system (V9.2). CBB season is over. MLB season is active (opened March 2026).
Deployed on Railway. Python/FastAPI backend. Next.js frontend. PostgreSQL database.

---

## Current Season State (as of April 5, 2026)

| Fact | Status |
|------|--------|
| CBB season | CLOSED — permanently archived |
| CBB betting model | FROZEN permanently — season over, no recalibration planned |
| MLB fantasy app | LIVE on Railway — active season, data layer under validation |
| MLB betting model | In development — `mlb_analysis.py` stub-level, BDL now active |
| BDL NCAAB subscription | CANCELLED — do not call `/ncaab/v1/` endpoints (will 401) |
| BDL GOAT MLB | ACTIVE — purchased. Expand `balldontlie.py` with `/mlb/v1/` endpoints |
| OddsAPI Basic | ACTIVE — 20k calls/month. CBB archival closing lines only. MLB odds via BDL. |

**Always read `HANDOFF.md` for the exact current task queue before touching any code.**

---

## Tech Stack

```
Backend:   Python 3.11 / FastAPI / SQLAlchemy / Alembic / APScheduler
Database:  PostgreSQL (Railway managed)
Frontend:  Next.js (canonical UI — Streamlit retired)
Deploy:    Railway (not local — use `railway run` for prod commands)
Test:      pytest (Windows: venv/Scripts/python -m pytest tests/ -q --tb=short)
Compile:   venv/Scripts/python -m py_compile <file>
Lint:      flake8
Solver:    OR-Tools (installed via requirements.txt)
Statcast:  pybaseball>=2.2.5 (wraps Baseball Savant)
```

---

## Key File Map

| What you need | Where it is |
|---------------|-------------|
| Current mission + task queue | `HANDOFF.md` |
| Agent roles + swimlanes | `ORCHESTRATION.md`, `AGENTS.md` |
| Risk posture + Kelly math | `.claude/skills/cbb-identity/SKILL.md` |
| Agent team protocol | `.claude/skills/emac-protocol/SKILL.md` |
| Workflow rules | `.claude/rules/workflow.md` |
| Architecture gaps analysis | `SYSTEM_ARCHITECTURE_ANALYSIS.md` |
| MLB data provider decisions | `HANDOFF.md` §1 + `research/baseball-api-analysis/EXECUTIVE_SUMMARY.md` |
| Fantasy baseball module | `backend/fantasy_baseball/` (35+ files) |
| CBB betting model | `backend/betting_model.py` (FROZEN) |
| MLB analysis | `backend/services/mlb_analysis.py` |
| Odds client (CBB) | `backend/services/odds.py` |
| BDL client (needs MLB expansion) | `backend/services/balldontlie.py` |
| Daily ingestion scheduler | `backend/services/daily_ingestion.py` |
| Main FastAPI app | `backend/main.py` (239KB — read specific sections only) |
| Yahoo OAuth client | `backend/fantasy_baseball/yahoo_client_resilient.py` |

---

## Hard Stops — Do Not Cross Without Explicit Instruction

| Rule | Reason |
|------|--------|
| Do NOT modify Kelly math in `betting_model.py` | CBB season closed — model archived, not recalibrating |
| Do NOT call BDL `/ncaab/v1/` endpoints | Subscription cancelled — will 401 |
| Do NOT route MLB odds through OddsAPI | 20k/month budget — MLB odds via BDL only |
| Do NOT touch `dashboard/` (Streamlit) | Retired — Next.js is canonical |
| Do NOT write test files outside `tests/` | Architecture decision locked |
| Do NOT use `THE_ODDS_API_KEY` for new MLB features | OddsAPI Basic reserved for CBB archival closing lines only |
| Do NOT use `datetime.utcnow()` for game times | Use `datetime.now(ZoneInfo("America/New_York"))` |

---

## Agent Team (Current)

The emac-protocol skill lists Gemini+OpenClaw — **Kimi CLI has replaced OpenClaw** as the deep research agent.

| Agent | Role | Do NOT assign |
|-------|------|---------------|
| **Claude Code** (you) | Master Architect — algorithms, schema, core logic | DevOps, deployment, Railway ops |
| **Gemini CLI** | Ops/DevOps — Railway deploy, py_compile verify, smoke tests | Backend schema changes, Yahoo API |
| **Kimi CLI** | Deep research, spec memos, API audits, MLB analysis research | Production code without Claude delegation bundle |

---

## Data Providers — Current State

```
LIVE NOW:
  Yahoo Fantasy API     — OAuth 2.0, league/roster/matchup ops (yahoo_client_resilient.py)
  BallDontLie GOAT MLB  — ACTIVE. Primary source for MLB schedule, scores, injuries, odds.
                          Expand balldontlie.py with /mlb/v1/ endpoints.
                          Migrate mlb_analysis._fetch_mlb_odds() to BDL.
                          Migrate daily_ingestion._poll_mlb_odds() to BDL.
  MLB Stats API         — statsapi library, schedule/scores (mlb_analysis._fetch_schedule)
  pybaseball            — Statcast/FanGraphs, xERA/wRC+ (pybaseball_loader.py)
  OddsAPI Basic         — 20k calls/month. CBB archival closing lines ONLY (odds.py).
                          Do NOT use for any new MLB feature — route all MLB odds through BDL.
  OpenWeatherMap        — Park weather (park_weather.py)

SAVANT PITCH QUALITY:
  savant_pitch_quality  — In-house 100-centered Baseball Savant pitcher score for
                          waiver/breakout detection. Implemented behind disabled
                          feature flags; not a FanGraphs Stuff+/Location+ clone.
                          See backend/fantasy_baseball/savant_pitch_quality.py and
                          HANDOFF.md before activating waiver or projection behavior.

SAVANT PARK FACTORS:
  park_factors           — Baseball Savant Statcast park factors are canonical for
                          projection context. Snapshot lives at
                          data/park_factors/savant_park_factors_2025_3yr.json.
                          Savant 100-index values are normalized to 1.00 factors
                          by backend/fantasy_baseball/savant_park_factors.py.
                          Roll out with migration_savant_park_factors.py then
                          seed_savant_park_factors.py before relying on DB lookup.

AGENT RESEARCH MCP:
  BallDontLie MCP       — Official hosted MCP at https://mcp.balldontlie.io/mcp.
                          Available to agents as @balldontlie for ad-hoc endpoint discovery,
                          payload inspection, odds/injury/stat checks, and cross-sport research.
                          Do not replace backend/services/balldontlie.py with MCP for production
                          ingestion; keep direct REST + Pydantic contracts for runtime jobs.

CANCELLED:
  BallDontLie NCAAB     — Subscription ended with CBB season
  OddsAPI Champion      — Downgraded to Basic plan

DO NOT REPLACE:
  pybaseball/Statcast   — BDL does not expose xwOBA/barrel%/exit velocity. Keep forever.
  savant_pitch_quality  — Production runtime must use project-owned scripts/services;
                          MCP is for research and validation only.
```

---

## Stale References to Be Aware Of

These exist in the codebase and are known-outdated — do not fix without a dedicated task:

- `.claude/agents/cbb-architect.md` — agent named `cbb-architect`, references "CBB Edge Analyzer" and "V9 Predictive Confidence Engine". Rename to `mlb-architect` in a future housekeeping session.
- `.claude/skills/cbb-identity/SKILL.md` — mission statement says "NCAA D1 basketball bets". Now dual-purpose (fantasy + MLB betting). Update when CBB archive task runs post-Apr 7.
- `backend/services/mlb_analysis.py` — `GUARDIAN FREEZE` comment says do not import `betting_model`. CBB season is closed but the architectural boundary (ADR-004) stays — MLB analysis must never import the CBB betting model.

---

## Test Pattern (Windows dev environment)

```bash
# Full suite
venv/Scripts/python -m pytest tests/ -q --tb=short

# Targeted (fantasy)
venv/Scripts/python -m pytest tests/test_mlb_analysis.py tests/test_lineup_optimizer.py -q --tb=short

# Syntax check before commit
venv/Scripts/python -m py_compile backend/services/balldontlie.py
venv/Scripts/python -m py_compile backend/services/mlb_analysis.py
venv/Scripts/python -m py_compile backend/main.py

# Railway prod commands
railway run python -m py_compile <file>
railway run python -c "from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient; c = YahooFantasyClient(); print(c.get_my_team_key())"
```

---

## Advisory Lock IDs (daily_ingestion.py — do not reuse taken IDs)

```
100_001 mlb_odds        | 100_002 statcast          | 100_003 rolling_z         | 100_004 cbb_ratings
100_005 clv             | 100_006 cleanup           | 100_007 waiver_scan        | 100_008 mlb_brief
100_009 openclaw_perf   | 100_010 openclaw_sweep    | 100_011 valuation_cache    | 100_012 fangraphs_ros
100_013 yahoo_adp_injury| 100_014 ensemble_update   | 100_015 projection_freshness| 100_016 mlb_game_log
100_017 mlb_box_stats   | 100_018 rolling_windows   | 100_019 player_scores      | 100_020 player_momentum
100_021 ros_simulation  | 100_022 decision_optimization | 100_023 backtesting    | 100_024 explainability
100_025 snapshot        | 100_026 statsapi_supplement | 100_027 position_eligibility | 100_028 probable_pitchers
100_029 player_id_mapping | 100_030 vorp            | 100_031 projection_cat_scores | 100_032 savant_ingestion
100_033 bdl_injuries    | 100_034 yahoo_id_sync     | 100_035 cat_scores_backfill | 100_036 ros_projection_refresh
100_037 opportunity_update | 100_038 market_signals_update | 100_039 matchup_context_update | 100_040 canonical_projection_refresh
100_041 bridge_mapping_to_identities
Next available: 100_042
```
