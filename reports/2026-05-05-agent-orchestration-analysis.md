# Agent Orchestration Analysis: Ruflo & Alternatives for CBB Edge Swarm

**Date:** 2026-05-05
**Author:** Kimi CLI (Research & Analysis)
**Scope:** Evaluate tools to automate handoff/delegation between Claude Code, Kimi CLI, Gemini CLI, and Codex — eliminating manual `HANDOFF.md` maintenance.

---

## Executive Summary

| Tool | Fit for Your Setup | Verdict |
|------|-------------------|---------|
| **Ruflo** | ⚠️ Partial | Claude-only orchestration; doesn't natively reach Kimi/Gemini CLI processes |
| **LangGraph** | ❌ Poor | Single-runtime framework; assumes all agents in one Python process |
| **CrewAI** | ❌ Poor | Same-runtime assumption; no cross-CLI coordination |
| **OpenAI Agents SDK** | ❌ Poor | Locked to OpenAI models; single-runtime |
| **Letta (MemGPT)** | ⚠️ Narrow | Shared memory layer possible, but requires adopting Letta runtime |
| **Custom (PostgreSQL + FastAPI + MCP)** | ✅ Best fit | Builds on your existing infrastructure; no vendor lock-in |

**Bottom line:** Your architecture — four *separate CLI tools* (Claude Code, Kimi CLI, Gemini CLI, Codex) communicating via filesystem — is fundamentally different from what every major multi-agent framework assumes. They all expect agents to be Python objects or API calls within a single runtime. You need **cross-process orchestration**, not intra-process orchestration.

---

## 1. Ruflo Deep Dive

### What It Actually Is

Ruflo (formerly Claude Flow) is a **middleware layer that sits between you and Claude Code**. It installs hooks, skills, and an MCP server into Claude Code's environment to enable:

- Multi-agent swarms (queen + worker hierarchy)
- Persistent memory (AgentDB + HNSW vector search)
- Self-learning (SONA pattern matching)
- Task routing across LLM providers (Claude, GPT, Gemini, Ollama)
- Federation (Ruflo-to-Ruflo communication across machines)

### Critical Architecture Point

```
User → Claude Code CLI
           ↓
      Ruflo Middleware (MCP Server, Hooks, Router)
           ↓
      Swarm of "agents" — ALL RUNNING INSIDE CLAUDE CODE
           ↓
      LLM Provider Switching (Claude/GPT/Gemini/Ollama)
```

Ruflo's "agents" are **not separate CLI processes**. They are prompt templates + tool definitions that Claude Code invokes sequentially or in parallel. When Ruflo routes to "Gemini," it makes an API call to Gemini — it does **not** launch the Gemini CLI tool you have installed.

### Relevance to Your Swarm

| Your Need | Ruflo Can? | Notes |
|-----------|-----------|-------|
| Orchestrate Claude Code tasks | ✅ Yes | Core purpose |
| Route coding tasks to "Kimi" | ⚠️ Partial | API call to Moonshot API, not Kimi CLI |
| Respect EMAC-075 (Gemini code ban) | ❌ No | Ruflo has no concept of your EMAC-075 policy |
| Run Kimi's 1M-context codebase audits | ❌ No | Kimi CLI's 1M token window is a client feature; API has different limits |
| Execute Gemini's Railway ops | ❌ No | Ruflo would call Gemini API, not `railway` CLI via Gemini CLI |
| Maintain AGENTS.md authority | ❌ No | Ruflo defines its own agent roles; no mechanism to enforce your AGENTS.md hierarchy |
| Eliminate HANDOFF.md maintenance | ⚠️ Partial | Replaces it with AgentDB memory, but your Kimi/Gemini agents wouldn't see it |

### Red Flags from Community

Per GitHub Discussion #1666 and issues #126, #430, #640:

1. **MCP tool namespace mismatch** — `mcp__claude-flow__*` vs `mcp__ruv-swarm__*` causes 100% failure rate for swarm coordination features
2. **Self-reported success is broken** — "Agents self-report 'success' when 89% actually fail. No enforcement mechanism between claim and acceptance"
3. **Concurrency blocked by sync SQLite** — `better-sqlite3` blocks the Node.js event loop; no real parallel execution
4. **Overwhelming complexity** — "314 native MCP tools, 100+ agents" but beginners report "paradox of choice and confusion"

**Verdict:** Ruflo is over-engineered for your needs and doesn't solve your cross-CLI coordination problem. The federation feature sounds relevant but is designed Ruflo-to-Ruflo, not Ruflo-to-Kimi CLI.

---

## 2. LangGraph, CrewAI, OpenAI SDK — Why They Don't Fit

These are **single-runtime orchestration frameworks**. They assume:

```python
# LangGraph pattern — all agents in one Python process
from langgraph import StateGraph

graph = StateGraph()
graph.add_node("claude_agent", call_claude_api)
graph.add_node("kimi_agent", call_kimi_api)  # Just another API call
graph.add_node("gemini_agent", call_gemini_api)
```

Your actual architecture:

```
┌─────────────────┐     ┌─────────────────┐
│  Claude Code    │     │   Kimi CLI      │
│  (Windows CLI)  │     │  (Windows CLI)  │
│                 │     │                 │
│  Owns backend/  │◄────┤  Owns reports/  │
│  main.py, tests/│ HAND│  audits, deep   │
│                 │OFF  │  research       │
└─────────────────┘.md  └─────────────────┘
         ▲                       ▲
         │                       │
         │   ┌─────────────────┐ │
         └───┤   Gemini CLI    ├─┘
             │  (Windows CLI)  │
             │                 │
             │  DevOps only,   │
             │  no code writes │
             └─────────────────┘
```

These CLI tools are **opaque processes** that can't be invoked as Python functions. LangGraph can't spawn Kimi CLI, wait for it to finish a 1M-token audit, and consume its markdown report.

| Framework | Orchestration Model | Your Fit |
|-----------|---------------------|----------|
| **LangGraph** | Directed graph, checkpointed state | ❌ Assumes single Python runtime |
| **CrewAI** | Role-based crews, sequential/parallel tasks | ❌ Same assumption |
| **OpenAI Agents SDK** | Explicit handoffs, guardrails | ❌ OpenAI-only, single runtime |
| **Google ADK** | Hierarchical agent tree, A2A protocol | ⚠️ A2A is interesting, but ADK is Google Cloud-native |
| **Smolagents** | Code-generating agents, ~1K LOC | ❌ Single-runtime |

---

## 3. Letta (MemGPT) — Shared Memory Layer

Letta provides **persistent memory across sessions** via a three-tier architecture (core/recall/archival). It's model-agnostic and has a REST API.

### How It Could Help

You could theoretically:
1. Run a Letta server (self-hosted, free)
2. Have each CLI tool (Claude, Kimi, Gemini) call Letta's REST API to read/write shared memory
3. Replace `HANDOFF.md` with Letta's archival memory as the canonical task registry

### Why It's Not a Clean Fit

1. **Letta is a full runtime** — your agents don't run "inside" Letta; they would be external clients polling a memory service
2. **Complexity** — you're adding a new server (Letta) just to replace a markdown file
3. **No task queue semantics** — Letta handles memory, not job delegation, priority, or completion tracking
4. **Lock-in** — adopting Letta's memory model means all agents must learn its API

**Verdict:** Overkill. You need a task board, not an agent operating system.

---

## 4. Recommended Approach: Build on Your Existing Stack

You already have the infrastructure. Use it.

### Architecture: PostgreSQL Task Queue + FastAPI MCP

```
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL (Railway)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ agent_tasks  │  │ agent_memory │  │ handoff_state   │   │
│  │ (task queue) │  │ (context)    │  │ (sync registry) │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ REST / MCP
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────┴────┐           ┌────┴────┐          ┌────┴────┐
   │ Claude  │           │  Kimi   │          │ Gemini  │
   │  Code   │◄─────────►│  CLI    │◄────────►│  CLI    │
   │         │  polling  │         │  polling │         │
   └─────────┘           └─────────┘          └─────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   FastAPI MCP     │
                    │  /mcp/task_claim  │
                    │  /mcp/task_submit │
                    │  /mcp/state_read  │
                    └───────────────────┘
```

### Minimal Implementation

**Step 1: Add a task table** (5 minutes)

```sql
-- Run this migration
CREATE TABLE agent_tasks (
    id SERIAL PRIMARY KEY,
    task_id TEXT UNIQUE NOT NULL,  -- e.g., "K-NEXT-3"
    title TEXT NOT NULL,
    description TEXT,
    assigned_to TEXT CHECK (assigned_to IN ('claude', 'kimi', 'gemini', 'openclaw')),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'blocked', 'done')),
    priority INT DEFAULT 0,  -- P0 = 0, P1 = 1, etc.
    input_payload JSONB,
    output_payload JSONB,
    blocked_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_agent_tasks_assigned ON agent_tasks(assigned_to, status);
CREATE INDEX idx_agent_tasks_priority ON agent_tasks(priority, created_at);
```

**Step 2: Expose via your existing MCP server** (already mounted at `/mcp`)

Add these to `backend/main.py` as FastAPI routes (they'll auto-expose as MCP tools):

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import get_db

orchestration_router = APIRouter(prefix="/orchestration", tags=["orchestration"])

class TaskClaim(BaseModel):
    agent_name: str  # "claude", "kimi", "gemini", "openclaw"
    task_id: str | None = None  # Claim specific task, or next available

class TaskSubmit(BaseModel):
    task_id: str
    status: str  # "done", "blocked", "in_progress"
    output_payload: dict | None = None
    blocked_reason: str | None = None

@orchestration_router.post("/claim")
def claim_task(claim: TaskClaim, db: Session = Depends(get_db)):
    """Agent claims its next highest-priority pending task."""
    query = db.query(AgentTask).filter(
        AgentTask.assigned_to == claim.agent_name,
        AgentTask.status == "pending"
    ).order_by(AgentTask.priority, AgentTask.created_at)
    
    if claim.task_id:
        query = query.filter(AgentTask.task_id == claim.task_id)
    
    task = query.first()
    if not task:
        return {"task": None}
    
    task.status = "in_progress"
    task.claimed_at = datetime.now(ZoneInfo("America/New_York"))
    db.commit()
    return {"task": task.to_dict()}

@orchestration_router.post("/submit")
def submit_task(submit: TaskSubmit, db: Session = Depends(get_db)):
    """Agent submits completed work or reports blockage."""
    task = db.query(AgentTask).filter(AgentTask.task_id == submit.task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")
    
    task.status = submit.status
    task.output_payload = submit.output_payload
    task.blocked_reason = submit.blocked_reason
    task.updated_at = datetime.now(ZoneInfo("America/New_York"))
    
    if submit.status == "done":
        task.completed_at = task.updated_at
    
    db.commit()
    return {"status": "ok", "task": task.to_dict()}

@orchestration_router.get("/queue/{agent_name}")
def get_queue(agent_name: str, db: Session = Depends(get_db)):
    """View pending tasks for any agent."""
    tasks = db.query(AgentTask).filter(
        AgentTask.assigned_to == agent_name,
        AgentTask.status.in_(["pending", "in_progress"])
    ).order_by(AgentTask.priority).all()
    return {"tasks": [t.to_dict() for t in tasks]}
```

**Step 3: Each agent gets a lightweight poll script**

```python
# scripts/agent_poll.py — run by each CLI tool
import sys, requests, os

AGENT_NAME = sys.argv[1]  # "claude", "kimi", "gemini"
API_BASE = os.environ["CBB_API_URL"]  # your Railway deployment

def poll():
    resp = requests.post(f"{API_BASE}/orchestration/claim", 
                        json={"agent_name": AGENT_NAME})
    task = resp.json()["task"]
    if not task:
        print(f"No pending tasks for {AGENT_NAME}")
        return
    
    print(f"CLAIMED: {task['task_id']} — {task['title']}")
    # Agent does work here...
    # Then submits:
    requests.post(f"{API_BASE}/orchestration/submit",
                 json={"task_id": task["task_id"], "status": "done",
                       "output_payload": {"report_path": "reports/..."}})

if __name__ == "__main__":
    poll()
```

### What This Replaces

| Current Pain | Solution |
|-------------|----------|
| Manual `HANDOFF.md` edits | Task queue is single source of truth |
| Agents don't know what's in progress | `/orchestration/queue/{agent}` shows live state |
| Kimi starts cold without context | `input_payload` carries task briefing |
| Claude doesn't see Kimi's findings | `output_payload` carries report paths and summaries |
| No priority enforcement | `priority` column + `ORDER BY` |
| No completion tracking | `claimed_at`, `completed_at`, `status` |

---

## 5. Alternative: A2A Protocol (Google ADK)

Google's **Agent-to-Agent (A2A) protocol** (April 2025) is designed for cross-framework agent communication. An agent built with any framework can expose an "Agent Card" (JSON metadata) and receive tasks via standardized REST endpoints.

### Relevance

- **Pros:** Vendor-neutral, designed for exactly your problem (different agents on different systems)
- **Cons:** Very new (April 2025), immature ecosystem, Google Cloud-centric tooling
- **Verdict:** Monitor but don't adopt yet. The protocol is promising but you'd be writing the integration layer yourself anyway.

---

## 6. Recommendation Matrix

| Approach | Effort | Value | Risk | Timeline |
|----------|--------|-------|------|----------|
| **PostgreSQL task queue** | Low (1 day) | High | Low | Immediate |
| **Ruflo** | Medium (2-3 days setup) | Low | High (broken features) | Skip |
| **Letta memory layer** | Medium (2 days) | Medium | Medium | Later |
| **A2A protocol** | High (1-2 weeks) | High (future) | High (immature) | Q3 2026 |
| **Collapse to single runtime** | Very High (months) | Very High | Very High | Not recommended |

---

## 7. Suggested Immediate Action

1. **Create the `agent_tasks` table** — 5-minute migration
2. **Add the 3 orchestration routes** to `backend/main.py` — 30 minutes
3. **Write `scripts/agent_poll.py`** — 15 minutes
4. **Test manually** — have Claude create a task for Kimi, watch Kimi claim it
5. **Replace `HANDOFF.md` P0/P1 sections** with DB-driven generation — write a script that renders `agent_tasks` into markdown for human readability

This gives you:
- Machine-readable task queue (API + MCP accessible)
- Human-readable HANDOFF.md (auto-generated from DB)
- No new infrastructure (uses existing PostgreSQL + FastAPI)
- No new dependencies (no npm, no Letta server, no Ruflo hooks)
- EMAC-075 compliance preserved (Gemini still can't write code; task assignments respect AGENTS.md roles)

---

## 8. Open Questions for Claude

1. Should the orchestration layer enforce AGENTS.md boundaries? (e.g., reject a task assigned to Gemini that modifies `.py` files)
2. Should OpenClaw (`scout.py`) auto-create integrity-check tasks in the queue?
3. Should DailyIngestionOrchestrator publish its job status to the task queue so agents know when data is fresh?
4. Do we want bidirectional sync — Kimi CLI polling the API, or also push notifications (webhooks, Discord)?

---

*Report generated by Kimi CLI. Proposes; Claude approves and implements.*
