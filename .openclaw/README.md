# OpenClaw v2.0 — Local LLM Coordination System

**Coordinator:** Kimi CLI (Deep Intelligence Unit)  
**Local Engine:** qwen2.5:3b via Ollama  
**Status:** Active as of 2026-03-06

---

## Quick Start

```python
# Import the coordinator
from .openclaw.coordinator import check_integrity, TaskContext

# Standard integrity check (auto-routed)
verdict = await check_integrity(
    home_team="Duke",
    away_team="UNC",
    verdict="Bet 1.0u Duke -4",
    search_results="No injuries reported..."
)
# → Uses local LLM (fast, free)

# High-stakes check (auto-escalated)
ctx = TaskContext(
    recommended_units=1.5,  # ≥1.5u triggers Kimi escalation
    tournament_round=4      # Elite Eight+ triggers Kimi escalation
)
verdict = await check_integrity(
    home_team="Duke",
    away_team="UNC",
    verdict="Bet 1.5u Duke -4",
    search_results="Conflicting reports...",
    context=ctx
)
# → Returns "KIMI_ESCALATION" — route to Kimi CLI
```

---

## Routing Logic

| Scenario | Engine | Reason |
|----------|--------|--------|
| Standard integrity check | Local → Fallback to Kimi | Pattern-based, fast |
| Elite Eight or later | **Kimi** | High stakes, needs synthesis |
| Bet ≥ 1.5 units | **Kimi** | Significant exposure |
| VOLATILE verdict | **Kimi** | Complex risk assessment |
| Conflicting signals | **Kimi** | Multi-factor analysis |
| Scouting reports | Local only | Narrative generation |
| Health narratives | Local only | Templated output |

---

## Configuration

Edit `.openclaw/config.yaml` to adjust:

```yaml
routing:
  rules:
    - name: "my_custom_rule"
      condition: "recommended_units >= 2.0"
      engine: "kimi"

local:
  model: "qwen2.5:3b"  # Change local model
  limits:
    max_concurrent: 8   # Adjust parallelism
    timeout_seconds: 10 # Adjust patience

tracking:
  budgets:
    kimi_daily_usd: 5.00  # Daily budget cap
```

---

## Cost Tracking

Daily usage tracked in `.openclaw/token-usage.jsonl`:

```json
{"timestamp": "2026-03-06T10:30:00", "task_type": "integrity_check", "engine": "local", "latency_ms": 450, "cost_usd": 0.0}
{"timestamp": "2026-03-06T10:31:00", "task_type": "integrity_check", "engine": "kimi", "latency_ms": 3200, "cost_usd": 0.05}
```

View stats:
```python
from .openclaw.coordinator import get_coordinator
coord = get_coordinator()
print(coord.get_stats())
# {'circuit_breaker_state': 'CLOSED', 'daily_cost_usd': 0.05, 'budget_remaining_pct': 99}
```

---

## Circuit Breaker

If local LLM fails 5+ times in 60 seconds:
1. Circuit opens → all traffic routes to Kimi
2. After 60s cooldown → half-open (test calls)
3. If test succeeds → circuit closes, resume local

Monitor state: `get_coordinator().circuit_breaker.state`

---

## For Hive Agents

### Kimi CLI (Coordinator)
- Review routing decisions in logs
- Handle escalated high-stakes tasks
- Adjust routing rules based on performance
- Monitor cost budgets

### Claude Code (Architect)
- Review `.openclaw/coordinator.py` for architectural fit
- Approve changes to routing rules
- Circuit breaker is infrastructure — monitor but don't modify without Kimi

### Gemini CLI (DevOps)
- Ensure Ollama service running: `ollama list`
- Monitor `.openclaw/token-usage.jsonl` disk usage
- Backup config.yaml before changes

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Local LLM timeout | Check Ollama: `curl http://localhost:11434/api/tags` |
| Circuit breaker open | Wait 60s, or restart Ollama service |
| Budget exceeded | Adjust `kimi_daily_usd` or routing rules |
| Invalid verdicts | Check prompt templates in config.yaml |

---

## Changelog

### v2.0 (2026-03-06)
- Added intelligent routing (Kimi CLI as coordinator)
- Added circuit breaker pattern
- Added cost tracking and budgets
- Added TaskContext for rich routing decisions
- Maintained backward compatibility with `perform_sanity_check()`

### v1.0 (Legacy)
- Direct Ollama calls only
- No routing or coordination
- All tasks handled by qwen2.5:3b

---

**Questions?** Coordinate through Kimi CLI as OpenClaw lead.
