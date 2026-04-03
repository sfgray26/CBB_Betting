---
agent: kimi-cli
task_type: skills_assessment
confidence: high
key_findings:
  - finding: "everything-claude-code repo has 125+ skills, 28 agents, and production patterns"
    evidence: "108k stars, Anthropic hackathon winner, 10+ months of battle-tested configs"
    recommendation: "Adopt select patterns, not the whole system — focus on error recovery, memory, and verification"
  
  - finding: "Their skill architecture uses YAML frontmatter + SKILL.md format"
    evidence: "skills/<name>/SKILL.md with standardized headers (name, description, tools)"
    recommendation: "Align our skill structure with theirs for compatibility"
  
  - finding: "Continuous learning v2 uses 'instincts' — confidence-scored patterns"
    evidence: "/instinct-status, /instinct-export, /evolve commands for pattern graduation"
    recommendation: "Replace tasks/lessons.md with instinct-based learning post-April 7"
  
  - finding: "Circuit breaker and stale cache patterns are well-documented"
    evidence: "api-design, backend-patterns, and deployment-patterns skills"
    recommendation: "Implement in waiver_recovery skill immediately"
  
  - finding: "Token optimization settings can reduce costs 60%+"
    evidence: "model: sonnet, MAX_THINKING_TOKENS: 10000, CLAUDE_AUTOCOMPACT_PCT_OVERRIDE: 50"
    recommendation: "Apply to our OpenClaw config for cost control"

action_items:
  - file: skills/waiver_recovery/SKILL.md
    change: "Created resilient Yahoo API client with circuit breaker, fallback, and cache patterns"
    priority: high
  
  - file: .openclaw/settings.json
    change: "Add token optimization settings (MAX_THINKING_TOKENS, model routing)"
    priority: medium
  
  - file: skills/continuous-learning-v2/
    change: "Adopt instinct-based learning after GUARDIAN FREEZE lifts (April 7)"
    priority: low
    blocked_until: "2026-04-07"
  
  - file: backend/fantasy_baseball/yahoo_client.py
    change: "Implement ResilientYahooClient with circuit breaker and fallback strategies"
    priority: high
    blocked_until: "user approves or post-April 7"

---

# Skills Adoption Analysis
## Source: everything-claude-code by affaan-m

---

## What's Relevant vs. What's Noise

### ✅ HIGH VALUE — Adopt Now

| Pattern | Status | Why It Helps |
|---------|--------|--------------|
| **Circuit Breaker** | Created `skills/waiver_recovery/` | Stops cascading Yahoo API failures |
| **Stale Cache Fallback** | Included in waiver_recovery | Serves last-known-good waiver data |
| **Token Optimization** | Recommend settings below | Cut AI costs 60%+ without quality loss |
| **SKILL.md Format** | Already aligned | YAML frontmatter standardization |

### ⏳ DEFER TO APRIL 7 (Post GUARDIAN FREEZE)

| Pattern | Blocked Reason | Future Value |
|---------|---------------|--------------|
| **Continuous Learning v2** | Requires structural changes | Auto-extract patterns into instincts |
| **Multi-agent Orchestration** | Complex, needs testing | Parallel analysis, faster workflows |
| **Verification Loops** | Needs eval harness setup | Quality gates before deployment |
| **PM2 Commands** | Not needed for current infra | If we scale to multi-service |

### ❌ SKIP — Not Relevant

| Pattern | Why Skip |
|---------|----------|
| TypeScript/React skills | Wrong stack |
| E2E testing (Playwright) | No frontend to test |
| Docker patterns | Already containerized |
| Database migration skills | Using SQLAlchemy, not Prisma/Drizzle |

---

## Immediate Implementation: Waiver Recovery

The skill I created addresses your exact log errors:

```
Yahoo API error 400: Invalid subresource percent_owned requested
Skipped Marcus Semien (pos=2B): game_id mismatch
```

**What it adds:**
1. **Circuit Breaker** — Opens after 3 failures, prevents hammering dead API
2. **Metadata Fallback** — When `percent_owned` fails, fetch metadata + ADP estimate
3. **Position Normalizer** — Validates lineup assignments before hitting Yahoo
4. **Stale Cache** — Serves 24h-old data if everything else fails

**Integration point:**
```python
# Replace in backend/main.py or backend/fantasy_baseball/yahoo_client.py
from yahoo_client_resilient import ResilientYahooClient

# Old: yahoo = YahooClient()
# New: yahoo = ResilientYahooClient()  # Drop-in replacement
```

---

## Token Optimization Settings

From ECC's research — apply to OpenClaw for cost savings:

```json
// ~/.openclaw/settings.json or equivalent
{
  "model": "kimi-coding/k2p5",
  "env": {
    "MAX_THINKING_TOKENS": "10000",
    "AUTOCOMPACT_PCT_OVERRIDE": "50"
  }
}
```

| Setting | Impact |
|---------|--------|
| `MAX_THINKING_TOKENS=10000` | ~70% reduction in hidden reasoning cost |
| `AUTOCOMPACT_PCT_OVERRIDE=50` | Compact at 50% context instead of 95%, better quality |

---

## Post-April 7 Roadmap

When GUARDIAN FREEZE lifts, consider adopting:

### 1. Continuous Learning v2
Replace `tasks/lessons.md` with instinct-based system:

```
/instinct-status     # View learned patterns with confidence scores
/instinct-export     # Share patterns with other agents
/evolve              # Cluster instincts into formal skills
```

**Benefit:** Automatic pattern extraction from betting decisions.

### 2. Verification Loops
Add checkpoint/verify pattern for model calibration:

```
/checkpoint "Model retraining completed"
/verify calibration --against last_30_days
```

**Benefit:** Catch model drift before it costs money.

### 3. Multi-Agent Orchestration
For tournament season parallel analysis:

```
/multi-plan "Analyze Sweet 16 bracket"
# Spawns: odds_agent, injury_agent, historical_agent
# Combines outputs into unified report
```

**Benefit:** Faster tournament analysis with specialized subagents.

---

## What Makes ECC Good (vs. Generic Advice)

1. **Battle-tested** — 10+ months of daily production use
2. **Measurable** — Token costs, pass@k metrics, concrete settings
3. **Modular** — Take what you need, ignore the rest
4. **Cross-platform** — Works across Claude Code, Cursor, OpenCode, Codex

The repo is 90% web dev (TypeScript/React/Next.js) — but the 10% that's 
generic (error handling, memory, verification) is gold.

---

## Recommendation

**Now:** Implement `waiver_recovery` skill to fix Yahoo API issues
**April 7:** Evaluate continuous learning + verification loops
**Later:** Multi-agent orchestration if workload justifies it

---

*Analysis based on: https://github.com/affaan-m/everything-claude-code*
*Created: 2026-03-26*
