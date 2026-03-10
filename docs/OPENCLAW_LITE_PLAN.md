# OpenClaw Lite — Simplified Replacement Plan

> **Status:** Proposed  
> **Replaces:** OpenClaw v2.0 (`.openclaw/coordinator.py`)  
> **Motivation:** Reduce complexity, remove Ollama dependency

---

## Current Problems

| Issue | Impact |
|-------|--------|
| Ollama not running | Local LLM calls fail entirely |
| Complex coordinator | 400+ lines, hard to debug |
| Circuit breaker logic | Over-engineered for CBB use case |
| Config YAML | Another file to maintain |
| Discord integration | Requires tokens, rarely used |

---

## Proposed Solution: OpenClaw Lite

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Betting Model │────>│  OpenClaw Lite   │────>│   Heuristic     │
│   (analyze_game)│     │  (simplified)    │     │   (80% cases)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               └──────────────────>┐
                                                   │
                         ┌──────────────────┐     │
                         │   sessions_spawn │<────┘
                         │   (qwen model)   │
                         │   (if needed)    │
                         └──────────────────┘
```

### Key Differences

| Feature | OpenClaw v2.0 | OpenClaw Lite |
|---------|---------------|---------------|
| **Local LLM** | Ollama (qwen2.5:3b) | Heuristics + spawn |
| **Lines of code** | ~450 | ~150 |
| **Config files** | YAML + Python | Python only |
| **Circuit breaker** | Yes | No (simpler retry) |
| **Cost tracking** | Complex JSONL | Simple stats dict |
| **Discord** | Configurable | Removed |
| **Fallback** | Kimi escalation | Heuristic fallback |

---

## Routing Logic (Simplified)

```python
if recommended_units < 0.5:
    return heuristic_check()  # Fast, no LLM

elif recommended_units >= 1.5 or is_elite_eight:
    return deep_analysis()    # You're already Kimi, just analyze

else:
    return heuristic_check()  # Medium stakes, rules are fine
    # Could add: await spawn_qwen_if_available()
```

---

## Heuristic Rules (80% Coverage)

The lite version uses keyword matching for fast decisions:

```python
HIGH_RISK = ["injury", "suspension", "out", "doubtful", ...]
CONFLICT = ["conflicting", "disagree", "rumor", ...]

if high_risk_count >= 3:
    return "CAUTION"

if conflict_count >= 2:
    return "VOLATILE"

if "star" in text and "out" in text:
    return "CAUTION"

return "CONFIRMED"
```

**Why this works:** Most integrity checks are looking for obvious red flags. Keyword matching catches these without LLM latency.

---

## Implementation Plan

### Phase 1: Create Lite Version (Done)
- [x] `backend/services/openclaw_lite.py` — 150 lines
- [x] Heuristic-based integrity checks
- [x] Backward-compatible wrapper

### Phase 2: Testing
- [ ] Unit tests for heuristic rules
- [ ] Compare output vs. v2.0 on historical data
- [ ] Measure latency improvement

### Phase 3: Migration
- [ ] Update `scout.py` to use `openclaw_lite`
- [ ] Deprecate `.openclaw/coordinator.py`
- [ ] Remove Ollama dependency from docs
- [ ] Update HANDOFF.md

### Phase 4: Cleanup
- [ ] Delete `.openclaw/` directory (or archive)
- [ ] Update requirements.txt
- [ ] Deploy to Railway

---

## Expected Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Cold start** | Ollama service required | Instant | ∞ |
| **Latency (heuristic)** | 500ms (Ollama) | 5ms | 100× |
| **Code complexity** | 450 lines | 150 lines | 3× simpler |
| **Dependencies** | Ollama + requests | None | Zero local deps |
| **Maintenance** | YAML + circuit breaker | Python only | Easier |

---

## Trade-offs

### What We Lose
- Local LLM for edge cases
- Circuit breaker resilience
- Cost tracking (simplified instead)
- Discord notifications

### What We Keep
- Kimi for high-stakes analysis
- Routing logic (simplified)
- Backward compatibility
- Same verdict outputs

---

## Migration Decision

**Option A: Full Replacement**
- Delete v2.0, use Lite exclusively
- Risk: Lose Ollama for complex edge cases

**Option B: Parallel Operation**
- Keep both, toggle via env var
- Risk: Code duplication

**Option C: Gradual Migration**
- Start with heuristic-only
- Add spawn integration later
- Risk: None (iterative)

**Recommendation:** Option C — start simple, add complexity only if needed.

---

## Code Comparison

### v2.0 (Current)
```python
# 450 lines, requires Ollama
coordinator = OpenClawCoordinator()
result = await coordinator.route_task(
    task_type=TaskType.INTEGRITY_CHECK,
    context=TaskContext(...),
    prompt=prompt
)
```

### Lite (Proposed)
```python
# 150 lines, no dependencies
checker = OpenClawLite()
result = await checker.check_integrity(
    search_text=results,
    home_team="Duke",
    away_team="UNC",
    recommended_units=1.0
)
```

---

## Testing Checklist

Before migrating:
- [ ] Run 100 historical integrity checks through both systems
- [ ] Compare verdicts (should match >90% for obvious cases)
- [ ] Measure latency (heuristic should be 10× faster)
- [ ] Test fallback behavior (simulate spawn failure)

---

## Decision Required

**Should we proceed with OpenClaw Lite?**

Pros:
- Simpler codebase
- No Ollama dependency
- Faster for typical cases
- Easier to maintain

Cons:
- Less sophisticated for edge cases
- Requires testing to validate heuristics
- Some v2.0 features lost (Discord, detailed tracking)

**Next step:** Approve and run comparison tests, or keep v2.0 and fix Ollama setup.

---

*Document created by Kimi CLI for CBB Edge Analyzer*
