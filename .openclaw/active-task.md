# OpenClaw Active Task — O-7: Coordinator Validation Test

**Date:** 2026-03-06  
**Coordinator:** Kimi CLI  
**Status:** READY (no prerequisites)

---

## Mission

Validate the OpenClaw v2.0 coordinator system is functioning correctly by running a series of test tasks through each routing path.

---

## Test Cases

### Test 1: Low-Stakes Local Routing
**Task:** Generate scouting report  
**Expected Route:** Local (qwen2.5:3b)  
**Context:** Standard game, no special flags  
**Success Criteria:** Response received within 10s, no escalation

### Test 2: High-Stakes Escalation
**Task:** Integrity check with high units  
**Expected Route:** Kimi escalation  
**Context:** recommended_units=1.5, tournament_round=4  
**Success Criteria:** Returns "KIMI_ESCALATION"

### Test 3: Circuit Breaker Monitoring
**Task:** Check circuit breaker state  
**Expected:** CLOSED (normal operation)

---

## Execution Steps

1. **Validate Ollama connection**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Run Test 1** — Low-stakes scouting report
   - Use example data: Duke vs UNC
   - Verify local response
   - Log latency

3. **Run Test 2** — High-stakes escalation test
   - Create TaskContext with recommended_units=1.5
   - Verify escalation signal returned

4. **Report results**
   - Document latency for local calls
   - Confirm routing logic works
   - Report any failures

---

## Output

Save results to: `.openclaw/test-results-2026-03-06.json`

Format:
```json
{
  "test_date": "2026-03-06T11:30:00",
  "tests": [
    {"name": "low_stakes_local", "passed": true, "latency_ms": 450},
    {"name": "high_stakes_escalation", "passed": true, "route": "kimi"},
    {"name": "circuit_breaker", "state": "CLOSED"}
  ],
  "overall_status": "PASS"
}
```

---

## Coordination Notes

- O-6 is BLOCKED on G-10 (Railway DB) — do not attempt
- This test validates the v2.0 coordinator before O-6 runs
- Kimi CLI standing by to handle escalated tasks
- No token costs for local tests

---

**Execute when ready.** Report completion to Kimi CLI.
