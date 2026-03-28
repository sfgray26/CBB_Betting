# ORCHESTRATION.md — Swarm Swimlane Rules & Task Routing

> Maintained by: Claude Code (Master Architect). Authority: absolute.
> Last consolidated: March 28, 2026
> This document defines WHO does WHAT. Violations require a post-mortem in HANDOFF.md.

---

## System Overview

This monorepo contains two production systems that share infrastructure but have distinct domains:

| System | Domain | Model Version | Season Status |
|--------|--------|---------------|---------------|
| **CBB Betting Analyzer** | NCAA D1 basketball betting, Kelly sizing, integrity validation | V9.1 | Tournament active; recalibration blocked until Apr 7 |
| **Fantasy Baseball Platform** | Yahoo Fantasy lineup optimization, waiver intelligence, dashboard | Season Live | Opening Day March 26, 2026 — Day 3 |

All agents operate across both systems within their swimlanes.

---

## Task Routing Matrix

| Task Type | Owner | Notes |
|-----------|-------|-------|
| **ARCHITECTURE / BACKEND** | | |
| Risk math, Kelly sizing, circuit breakers | Claude | Never delegated |
| CBB model changes (spread, SD, CI) | Claude | Never delegated |
| New FastAPI route | Claude | Always grep existing routes before creating |
| Pydantic schema change | Claude | Validate against all callers before merging |
| SQLAlchemy model change | Claude | Check migration impact before applying |
| Yahoo client changes | Claude | `yahoo_client_resilient.py` is the single file — no forks |
| Lineup optimizer logic | Claude | ILP math, scoring formula changes |
| Waiver edge detection | Claude | Category deficit scoring, move ranking |
| Dashboard service methods | Claude | All panel stub wiring |
| **FRONTEND / UI** | | |
| Next.js page routing | Claude | Architecture decisions |
| React component build (delegated) | Kimi | Only when Claude issues explicit delegation bundle |
| CSS / Tailwind fixes | Kimi | Delegated; Claude reviews before merge |
| API response shape changes (frontend impact) | Claude | Always check Next.js consumers |
| Streamlit (any) | NOBODY | Retired. Never touch `dashboard/` again. |
| **DEVOPS / INFRASTRUCTURE** | | |
| Railway deployment / redeploy | Gemini | `railway up` or dashboard trigger |
| Env var changes | Gemini | Railway dashboard only |
| Log tailing | Gemini | `railway logs --follow` |
| DB migration (write) | Claude | Script authored by Claude |
| DB migration (run) | Gemini | `railway run python scripts/migrate_vN.py` |
| CI/CD pipeline | Claude | Gemini must not touch |
| **RESEARCH / ANALYSIS** | | |
| Multi-doc research synthesis (3+ long docs) | Kimi | 1M context window |
| Full season performance attribution | Kimi | >500 records — whole-corpus only |
| Tournament intelligence packages | Kimi | Full bracket + all team profiles in one shot |
| Codebase-wide anti-pattern audit | Kimi | Reads all files simultaneously |
| Single-doc API research | Gemini | No code output |
| **RUNTIME INTELLIGENCE** | | |
| CBB integrity check (all BET games, nightly) | OpenClaw | qwen2.5:3b — fast, cheap |
| CBB integrity second opinion (Elite 8+, ≥1.5u) | Kimi | After OpenClaw first pass |
| Fantasy morning brief generation | OpenClaw | `openclaw_briefs_improved.py` |
| Waiver move digest (batch) | OpenClaw | `discord_notifier.send_batch_digest` |
| Monitoring, health checks | Gemini | DevOps domain |

---

## Swimlane Violation Examples (with historical root causes)

| Violation | What Happened | Consequence |
|-----------|---------------|-------------|
| Gemini edits backend code | EMAC-075: duplicate FastAPI routes, invalid dict keys | Gemini permanently restricted from code writes |
| Kimi writes production code without delegation | — | Reverted; Kimi re-proposes via HANDOFF.md memo |
| Claude uses `datetime.utcnow()` for MLB games | West Coast games (9pm+ EDT) dropped as "no game" | Fixed Mar 28, 2026. UTC is banned for game_date. |
| `yahoo_client.py` forked from `yahoo_client_resilient.py` | Split-brain client state, duplicate auth logic | Merged Mar 28, 2026. Single file only. |

---

## Hive Protocol

### Session Startup (All Agents)
1. Read `HANDOFF.md`
2. Read `ORCHESTRATION.md` (this file)
3. Read `IDENTITY.md`
4. Read `HEARTBEAT.md`
5. Read `memory/YYYY-MM-DD.md` (today + yesterday)

### Handoff Quality Standard
Every HANDOFF.md must pass the cold-start test:
> "If I gave this handoff document to the target agent with zero prior context, could they execute the task completely and correctly?"

If not — add the missing detail. Handoffs are operational briefings, not task lists.

### Change Justification Rule
Every production file modification must be traceable to:
- A HANDOFF.md delegation bundle, OR
- An explicit in-session directive from the human operator

Silent edits that appear in git history without a corresponding HANDOFF entry are violations.

---

## No-Fail Rules

1. **Gemini does not write code.** Not `.py`, not `.ts`, not config files with runtime impact.
2. **Kimi proposes, Claude approves.** Kimi output → `reports/` or HANDOFF.md → Claude implements.
3. **No UTC for baseball.** `datetime.now(ZoneInfo("America/New_York"))` everywhere game dates are computed.
4. **No split clients.** `yahoo_client_resilient.py` is the one Yahoo client. No new yahoo_* files.
5. **Tier your integrity.** OpenClaw runs on every CBB BET game. Kimi escalation for Elite 8+ or VOLATILE.
6. **Test before marking done.** No task is complete without `py_compile` passing and relevant tests green.
7. **Policy lives in IDENTITY.md.** Risk parameters must be documented there before they appear in code.
8. **Streamlit is dead.** Never reference, import, or edit anything in `dashboard/`.
