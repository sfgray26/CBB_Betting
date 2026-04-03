# HANDOFF.md — Repository Review: CBB_Betting AI Workforce Implementation

**From:** Kimi CLI (Deep Intelligence Unit)  
**To:** Claude Code (Master Architect)  
**Date:** 2026-03-25  
**Task:** Review CBB_Betting repo against BentoBoiNFT's AI workforce patterns

---

## Summary

Reviewed the CBB_Betting repository against the 5-pillar AI workforce framework from [BentoBoiNFT's article](https://x.com/bentoboinft/status/2036827922565042415). The codebase already implements ~70% of the patterns well. Documenting gaps and opportunities for tightening the system.

---

## What's Already Implemented (EMAC Compliant)

### ✅ 1. Multi-Agent Specialization
The EMAC structure maps well to the article's recommendation:
- **Claude Code** → Alex (coordinator/architect)
- **Kimi CLI** → Deep research specialist
- **Gemini CLI** → Ops/research (restricted post-EMAC-075)
- **OpenClaw (qwen2.5:3b)** → Real-time integrity/execution

### ✅ 2. Memory Architecture
Full stack operational:
- `MEMORY.md` — Long-term storage
- `memory/YYYY-MM-DD.md` — Daily logs
- `SOUL.md` — Agent personality
- `AGENTS.md` — Role registry
- `HANDOFF.md` — Operational briefings

### ✅ 3. Automated Workflows (Cron Jobs)
Extensive scheduler in `main.py`:
- 3 AM nightly analysis
- Opening line attacks (10:30 PM, 12:30 AM)
- Odds monitoring every 5 min
- Health checks at 5 AM
- Morning briefings at 7 AM
- Weekly recalibration Sundays

### ✅ 4. Self-Improvement Loop
- Correction tracking via `tasks/lessons.md`
- Dynamic weight calibration in `recalibration.py`
- Pattern graduation logic (3+ occurrences → permanent rule)

### ✅ 5. Tiered Integrity (O-9)
- OpenClaw runs first pass on all BET candidates
- Kimi escalation for Elite Eight+, ≥1.5u, VOLATILE verdicts
- Coordinator routing in `.openclaw/coordinator.py`

---

## Implementation Gaps & Opportunities

### 🔧 1. Dedicated Discord Channels per Agent
**Current:** Notifications go to generic channels.  
**Opportunity:** Channel separation for signal clarity:
```
#claude-architect — Model changes, design decisions
#kimi-intel — Research reports, tournament briefs
#openclaw-integrity — Real-time sanity check results  
#system-health — Cron failures, circuit breakers
```

### 🔧 2. Skill Encapsulation
**Current:** Complex logic in `analysis.py` (~800 lines), `main.py` (~1500 lines).  
**Opportunity:** Extract to skills:
```
skills/
├── nightly-analysis/      # Pass 1/Pass 2 workflow
├── integrity-sweep/       # DDGS + sanity check pattern
├── kelly-sizing/          # Simultaneous Kelly + global scaling
├── parlay-engine/         # Cross-game parlay construction
├── morning-brief/         # Structured daily briefing
└── performance-review/    # Weekly attribution reports
```

Each skill gets `SKILL.md` with input/output contracts, examples, error handling.

### 🔧 3. Structured Agent Output
**Current:** Kimi outputs to `reports/` and `HANDOFF.md` but format varies.  
**Opportunity:** Standardized YAML frontmatter for machine parsing:
```yaml
---
agent: kimi-cli
task_type: tournament_intelligence
confidence: high
key_findings:
  - finding: "Model undervalues high-seed underdogs"
    evidence: "+3.2% ROI on 10+ seeds with AdjEM > +15"
    recommendation: "Add SEED_UPSET_SCALAR to betting_model.py"
action_items:
  - file: backend/betting_model.py
    change: "Add seed-upset scalar"
    priority: high
---
```

### 🔧 4. Data-Driven Escalation Rules
**Current:** Routing logic embedded in `coordinator.py`.  
**Opportunity:** Externalize to `.openclaw/escalation_rules.yaml`:
```yaml
rules:
  - name: "elite_eight_high_stakes"
    condition: "tournament_round >= 4 AND recommended_units >= 1.5"
    escalate_to: "kimi-cli"
    sla_seconds: 300
    
  - name: "volatile_verdict"
    condition: "integrity_verdict == 'VOLATILE'"
    escalate_to: "kimi-cli"
    context: "Include DDGS results"
    
  - name: "routine_bet"
    condition: "recommended_units < 1.0"
    engine: "local"
    model: "qwen2.5:3b"
```

### 🔧 5. Correction Database
**Current:** Corrections logged but not systematically tracked.  
**Opportunity:** New model `AgentCorrection`:
```python
class AgentCorrection(Base):
    id: int
    timestamp: datetime
    agent: str  # "claude", "kimi", "openclaw"
    correction_type: str  # "math_error", "logic_error", "style"
    original_output: str
    corrected_output: str
    pattern_count: int
    promoted_to_rule: bool
    rule_location: str
```
Weekly cron auto-promotes 3+ occurrence patterns to permanent rules.

### 🔧 6. Agent Health Dashboard
**Current:** Health logged but not visualized.  
**Opportunity:** New endpoint `/admin/agents/health`:
```json
{
  "agents": {
    "claude-code": {"last_contribution": "...", "files_modified_24h": 12},
    "kimi-cli": {"reports_generated": 3, "avg_response_time_sec": 45},
    "openclaw": {"integrity_checks_24h": 47, "error_rate": 0.02}
  },
  "coordination": {
    "handoffs_pending": 0,
    "last_handoff": "..."
  }
}
```

---

## Quick Wins (Prioritized)

| Priority | Task | Effort | Owner |
|----------|------|--------|-------|
| 1 | Create `skills/morning-brief/` skill | 2 hrs | Claude |
| 2 | Add `AgentCorrection` model + weekly cron | 4 hrs | Claude |
| 3 | Extract `integrity-sweep/` skill | 3 hrs | Claude |
| 4 | Create escalation rules YAML | 2 hrs | Kimi/Claude |
| 5 | Add Discord channel routing per agent | 2 hrs | Claude |

---

## Architectural Recommendation

`main.py` is ~1500 lines, `analysis.py` ~800 lines. Consider the "manager as router" pattern:

```
main.py → FastAPI setup + scheduler only
  ├── routers/analysis.py → orchestration endpoints
  ├── routers/agents.py → agent coordination  
  ├── skills/nightly-analysis/ → actual workflow
  └── skills/integrity-sweep/ → sub-workflow
```

This aligns with the article's insight: *"Your main agent becomes a manager, not a worker."*

---

## Questions for Claude

1. **Skill encapsulation:** Should I (Kimi) draft the `SKILL.md` templates for morning-brief and performance-review, or do you prefer to architect the skill structure first?

2. **Escalation rules:** Should the YAML live in `.openclaw/` or `config/`? And should I write the parser/loader or do you want to integrate it into the existing coordinator?

3. **Correction database:** This touches models.py and requires a migration. Safe for me to propose the schema, or do you want to handle all DB changes?

4. **Discord channels:** Do you want me to research the current Discord notifier structure and propose channel routing, or is this better handled by you since it touches the notification system?

---

## Files Reviewed

- `AGENTS.md` — Agent role registry (comprehensive)
- `SOUL.md` — Personality definitions
- `HEARTBEAT.md` — Operational loops (well-structured)
- `ORCHESTRATION.md` — Task routing matrix
- `backend/main.py` — FastAPI app + scheduler (~1500 lines)
- `backend/services/analysis.py` — Nightly analysis orchestration (~800 lines)

---

## Status

**Kimi's Assessment:** The foundation is solid. The gaps are in **encapsulation** (skills), **visibility** (dashboards, structured output), and **self-improvement automation** (correction DB). None are urgent, but implementing the Quick Wins would make the system significantly more maintainable.

**Next Step:** Awaiting Claude's direction on which items to implement and how to divide work between agents.

---

*Documented by Kimi CLI*  
*Review source: https://x.com/bentoboinft/status/2036827922565042415*
