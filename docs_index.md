# docs_index — Minified System Reference

> **Read this first.** Use `scripts/doc_retriever.py` to fetch full documents on demand.
> **Last updated:** April 15, 2026

---

## What This Repo Is

Two production systems sharing a FastAPI + PostgreSQL + Railway backend:
- **CBB Edge Betting Analyzer (V9.1)** — NCAA D1 basketball positive-EV bet finder
- **Fantasy Baseball Platform (2026 Season)** — Yahoo H2H lineup optimization, waiver intelligence

## Production Environment

| Resource | Value | Notes |
|----------|-------|-------|
| **Production URL** | `https://observant-benevolence-production.up.railway.app/` | Live FastAPI deployment |
| **API Auth** | `API_KEY_USER1` header | Required for all protected endpoints |
| **UI Inspection** | Chrome DevTools MCP available | Use for frontend debugging, DOM inspection, network analysis |
| **Database** | Railway PostgreSQL | Connection via `DATABASE_URL` env var |

**Claude:** DevTools MCP is configured in `~/.claude.json` (project-level). Use it to inspect the live UI without guessing markup structure.
**Kimi:** DevTools MCP can be added via `kimi mcp add --transport stdio chrome-devtools -- npx chrome-devtools-mcp@latest`

---

## Agent Roles & Swimlanes

| Agent | Can Do | Cannot Do |
|-------|--------|-----------|
| **Claude Code** | Architecture, backend code, FastAPI routes, Pydantic schemas, SQLAlchemy models, tests | Infrastructure ops, frontend CSS/React without delegating |
| **Gemini CLI** | Railway deploys, env vars, log tailing, running pre-approved DB migrations | Write `.py`, `.ts`, `.tsx`, `.js`, or CI/CD pipelines |
| **Kimi CLI** | Research memos (`reports/`), codebase-wide audits, doc maintenance | Production backend code without explicit delegation |
| **OpenClaw** | DDGS integrity checks, Discord alerts, morning briefs, waiver digests | Architecture changes |

**Routing:**
- Backend/algorithm work → Claude
- Railway ops → Gemini
- Multi-doc research (>3 docs) or whole-corpus audit → Kimi
- CBB integrity checks → OpenClaw first; escalate to Kimi for Elite 8+ or ≥1.5u or VOLATILE

---

## Critical Rules (No Exceptions)

1. **No ghost changes.** Every modification justified in `HANDOFF.md`.
2. **Kimi proposes, Claude approves.** Kimi writes to `reports/` or `HANDOFF.md`; Claude implements code.
3. **Gemini does not write code.** Not even "trivial" one-liners.
4. **No `datetime.utcnow()` for MLB.** Always `datetime.now(ZoneInfo("America/New_York"))`.
5. **No bool-to-string leakage** in Pydantic schemas (`status: False` is forbidden).
6. **Single Yahoo client.** `backend/fantasy_baseball/yahoo_client_resilient.py` only. No forks.
7. **Streamlit is dead.** Never touch `dashboard/`.
8. **Test before marking done.** `py_compile` pass + relevant `pytest` subset green.

---

## Code Quality Gates

Before any `backend/` file is marked complete:
1. `venv/Scripts/python -m py_compile <file>` passes
2. Relevant `pytest tests/` subset passes
3. No `datetime.utcnow()` — use `America/New_York`
4. No `status: False` or other bool-as-string leakage in schemas

---

## Risk Posture (CBB Betting)

- **Kelly scaling:** Base Kelly × SNR Scalar × Integrity Scalar = Final Kelly
- **SNR floor:** 0.5 (`SNR_KELLY_FLOOR`)
- **Integrity scalars:** CONFIRMED=1.0×, CAUTION=0.75×, VOLATILE=0.50×, ABORT/RED FLAG=**0.0× HARD GATE**
- **Circuit breakers:**
  - Integrity Abort Gate: `"ABORT"` or `"RED FLAG"` → `kelly_frac = 0`
  - Portfolio drawdown >15% → all new bets paused
  - `|model_margin - market_margin| > 2.5 × effective_base_sd` → hard PASS

---

## Current Focus (April 15)

1. Redeploy production so live code matches the repo
2. Verify `data_ingestion_logs` persists durable rows in production
3. Verify `probable_pitchers` populates usable rows in production
4. Remove false-green health reporting from operational interpretation
5. Complete canonical environment snapshots as the final major Layer 2 gap

**Operating rule:** Only Layer 2 work is active. Derived stats, decision engines, simulation, Yahoo automation, and frontend work are all frozen until Layer 2 is proven complete in production.

**Live pipeline truth:** critical Layer 2 tables are not yet certified live in production. Do not infer readiness from legacy or stale health summaries.

---

## Document Map

| Need | File | Retriever Command |
|------|------|-------------------|
| Current operational state + task queue | `HANDOFF.md` | `python scripts/doc_retriever.py HANDOFF.md` |
| Full agent role definitions | `AGENTS.md` | `python scripts/doc_retriever.py AGENTS.md` |
| Swimlane routing matrix | `ORCHESTRATION.md` | `python scripts/doc_retriever.py ORCHESTRATION.md` |
| Risk posture + circuit breakers | `IDENTITY.md` | `python scripts/doc_retriever.py IDENTITY.md` |
| Operational loops / heartbeats | `HEARTBEAT.md` | `python scripts/doc_retriever.py HEARTBEAT.md` |
| Historical HANDOFF context | `HANDOFF_ARCHIVE.md` | `python scripts/doc_retriever.py HANDOFF_ARCHIVE.md` |
| Quick command reference | `QUICKREF.md` | `python scripts/doc_retriever.py QUICKREF.md` |
| Prompt index | `CLAUDE_PROMPTS_INDEX.md` | `python scripts/doc_retriever.py CLAUDE_PROMPTS_INDEX.md` |
| Recent audit (Apr 15) | `reports/2026-04-15-comprehensive-application-audit.md` | `python scripts/doc_retriever.py reports/2026-04-15-comprehensive-application-audit.md` |

---

## Retrieval Tool

```bash
# Read a full document on demand
python scripts/doc_retriever.py <relative-path>

# Example
python scripts/doc_retriever.py IDENTITY.md
python scripts/doc_retriever.py reports/2026-04-15-comprehensive-application-audit.md
```

**Session startup rule:** Read this file (`docs_index.md`) first. Then read `HANDOFF.md`. For deep dives into specific domains, use the retriever instead of loading all docs at once.
