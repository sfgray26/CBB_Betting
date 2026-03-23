# 📚 REFERENCE — Agent Team Best Practices (C Compiler Patterns)

> **Status:** Reference Document (Mar 2026)  
> **Purpose:** Research notes from Anthropic's C Compiler project  
> **Note:** Proposed patterns (`.agent_tasks/` structure, validation scripts) not yet fully implemented  
> **Current Task System:** See `.agent_tasks/README.md` for working implementation

---

# AGENT TEAM BEST PRACTICES — Applied to CBB Edge

**Source:** Anthropic's "Building a C Compiler with 16 Agents" by Nicholas Carlini  
**Adapted for:** CBB Edge Betting Platform  
**Date:** March 19, 2026  
**For:** Claude Code (Master Architect)

---

## 🎯 EXECUTIVE SUMMARY

Anthropic's C compiler project (100k lines, 2,000 Claude sessions, $20k) succeeded through **agent teams** — multiple Claude instances working in parallel with minimal human intervention. This document extracts applicable patterns for the CBB Edge project.

**Key Insight:** The harness around the agent matters more than the agent itself. Tests, feedback loops, and parallelization strategies determine success.

---

## 🔑 KEY LESSONS & APPLICATIONS

### 1. AGENT TEAM PARALLELIZATION

**From C Compiler:**
- 16 Claude agents worked in parallel
- Each agent cloned repo to `/workspace`, pushed to `/upstream`
- Simple file-based locking: `current_tasks/parse_if_statement.txt`

**Apply to CBB Edge:**

```bash
# Task locking mechanism for multiple agents
mkdir -p .agent_tasks/

# Agent claims task by creating file
touch .agent_tasks/fix_discord_notifications.md

# If another agent tries same task, git conflict forces different choice
```

**Implementation:**
- **Frontend agents:** One per page (Today, Live Slate, Odds Monitor, etc.)
- **Backend agents:** One per service (Discord, Analysis, Bracket)
- **Test agents:** Specialized for validation (like Kimi CLI role)
- **Docs agents:** Maintain HANDOFF.md, README, API specs

**Files to create:**
```
.agent_tasks/
├── frontend_today_page.md
├── frontend_live_slate.md
├── backend_discord_fix.md
├── test_validation.md
└── docs_update.md
```

---

### 2. HIGH-QUALITY TESTS AS GUIDANCE

**From C Compiler:**
> "Claude will work autonomously to solve whatever problem I give it. So it's important that the task verifier is nearly perfect, otherwise Claude will solve the wrong problem."

**Apply to CBB Edge:**

| Current | Improved |
|---------|----------|
| Manual testing | Automated validation scripts |
| Runtime errors | Pre-commit type checking |
| Silent failures | Explicit test assertions |

**Implement:**
```bash
# Pre-commit validation (like compiler test suite)
./scripts/validate.sh
  ├── npm run build        # TypeScript check
  ├── npx tsc --noEmit     # Type check
  ├── pytest backend/      # API tests
  └── ./scripts/validate_frontend.sh  # Kimi CLI-style checks
```

**Frontend Validation (Kimi CLI Pattern):**
```bash
# scripts/validate_frontend.sh
#!/bin/bash
# 7-point checklist automation

for page in frontend/app/(dashboard)/*/page.tsx; do
  echo "Validating $page..."
  
  # 1. Check null safety
  grep -n "\.field" "$page" | grep -v "?\.|??" && echo "WARNING: Possible null safety issue"
  
  # 2. Check empty array fallbacks
  grep -n "\.map(" "$page" | grep -v "?? \[\]" && echo "WARNING: Missing empty array fallback"
  
  # 3. Check percentage conversion
  grep -n "toFixed" "$page" | grep -v "\* 100" && echo "INFO: Check decimal vs percentage"
  
  # ... etc
done
```

---

### 3. CONTEXT WINDOW MANAGEMENT

**From C Compiler:**
> "The test harness should not print thousands of useless bytes. At most, it should print a few lines of output and log all important information to a file so Claude can find it when needed."

**Apply to CBB Edge:**

| Anti-Pattern | Best Practice |
|--------------|---------------|
| `console.log` everywhere | Structured logging with levels |
| Full stack traces in output | Error codes + log file references |
| Raw API responses in context | Pre-computed summaries |
| Verbose test output | `--fast` mode with sampling |

**Implement:**
```typescript
// lib/logging.ts
export function log(level: 'ERROR' | 'WARN' | 'INFO', message: string, details?: object) {
  const timestamp = new Date().toISOString()
  const logEntry = { timestamp, level, message, details }
  
  // Write to file for later analysis
  fs.appendFileSync('logs/agent.log', JSON.stringify(logEntry) + '\n')
  
  // Console: only high-level summary
  if (level === 'ERROR') {
    console.error(`[ERROR] ${message} (see logs/agent.log)`)
  }
}
```

**Fast Mode for Development:**
```bash
# Run 10% sample of tests for quick feedback
npm run test:fast    # Random 10% of test suite
npm run test:full    # Full test suite (CI only)
```

---

### 4. ORIENTATION DOCUMENTATION

**From C Compiler:**
> "Each agent is dropped into a fresh container with no context... I included instructions to maintain extensive READMEs and progress files that should be updated frequently with the current status."

**Apply to CBB Edge:**

**Current State:**
- ✅ `HANDOFF.md` exists
- ✅ `CLAUDE_HANDOFF_VALIDATION.md` exists
- ✅ Validation reports created

**Improvements:**
```
AGENT_CONTEXT.md          # What this agent should do
CURRENT_STATUS.md         # What's working vs broken  
NEXT_TASKS.md            # Prioritized task list
DECISIONS.md             # Why certain choices were made
```

**AGENT_CONTEXT.md Template:**
```markdown
# Agent Context: [Task Name]

## Goal
[One sentence objective]

## Current State
- Working: [list]
- Broken: [list]
- Unknown: [list]

## Constraints
- DO NOT modify: [files]
- TEST before committing: [commands]
- ASK before: [actions]

## Next Steps
1. [First task]
2. [Second task]
3. [etc]

## References
- API spec: [link]
- Similar implementation: [file path]
```

---

### 5. AGENT SPECIALIZATION

**From C Compiler:**
> "I tasked one agent with coalescing any duplicate code it found. I put another in charge of improving the performance... and a third I made responsible for outputting efficient compiled code."

**Apply to CBB Edge — Define Roles:**

| Role | Responsibility | Example Task |
|------|----------------|--------------|
| **Feature Agent** | Build new features | "Add live odds widget" |
| **Refactor Agent** | Code quality | "Remove duplicate API calls" |
| **Test Agent** | Validation | "Write tests for bracket sim" |
| **Docs Agent** | Documentation | "Update HANDOFF.md with V9.2" |
| **Debug Agent** | Bug fixes | "Fix Discord notification gap" |
| **Perf Agent** | Optimization | "Reduce DB query time" |

**Implementation:**
```bash
# scripts/start_agent.sh
ROLE=$1  # feature|refactor|test|docs|debug|perf

claude -p "$(cat AGENT_PROMPT.md)" \
       --env ROLE=$ROLE \
       --env CONTEXT=$(cat CURRENT_STATUS.md)
```

---

### 6. ORACLE-BASED TESTING

**From C Compiler:**
> "The fix was to use GCC as an online known-good compiler oracle to compare against... randomly compiled most of the kernel using GCC, and only the remaining files with Claude's C Compiler."

**Apply to CBB Edge:**

**Concept:** Compare our model against known-good baselines

| Our System | Oracle | Comparison |
|------------|--------|------------|
| V9.1 Model | KenPom + BartTorvik consensus | If our pick differs from consensus by >5%, flag for review |
| Bracket Sim | Historical upset rates | If 5v12 upset rate is <30% or >50%, investigate |
| CLV Tracking | Closing lines | If CLV < -5%, model is mispricing |

**Implement:**
```typescript
// backend/services/oracle_validation.ts

export function validateAgainstOracle(
  ourPrediction: Prediction,
  consensusLine: number
): ValidationResult {
  const diff = Math.abs(ourPrediction.spread - consensusLine)
  
  if (diff > 5) {
    return {
      status: 'WARNING',
      message: `Prediction differs ${diff} points from consensus`,
      confidence: 'low'
    }
  }
  
  return { status: 'OK', confidence: 'high' }
}
```

---

### 7. CONTINUOUS INTEGRATION PIPELINE

**From C Compiler:**
> "I built a continuous integration pipeline and implemented stricter enforcement that allowed Claude to better test its work so that new commits can't break existing code."

**Apply to CBB Edge:**

```yaml
# .github/workflows/agent-ci.yml
name: Agent CI

on: [push]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Type Check
        run: npx tsc --noEmit
        
      - name: Build
        run: npm run build
        
      - name: Backend Tests
        run: pytest tests/ -q
        
      - name: Frontend Validation
        run: ./scripts/validate_frontend.sh
        
      - name: Check Documentation
        run: ./scripts/check_docs.sh
        
      - name: Agent Report
        if: failure()
        run: |
          echo "## Agent CI Failed" >> $GITHUB_STEP_SUMMARY
          echo "Review logs and fix before continuing" >> $GITHUB_STEP_SUMMARY
```

---

### 8. TASK GRANULARITY FOR PARALLELISM

**From C Compiler:**
> "When there are many distinct failing tests, parallelization is trivial: each agent picks a different failing test to work on."

**Apply to CBB Edge:**

**Good Task Size:** Can be completed in 1-3 Claude sessions

| Too Big ❌ | Just Right ✅ |
|------------|---------------|
| "Rewrite entire frontend" | "Build Today's Bets page" |
| "Fix all Discord issues" | "Fix morning briefing job" |
| "Implement V9.2" | "Recalibrate sd_mult parameter" |

**Task Template:**
```markdown
## Task: [Name]
**Estimated:** 1-2 sessions
**Dependencies:** [list]
**Files to modify:** [list]
**Test command:** [command]

### Success Criteria
- [ ] Feature works as expected
- [ ] TypeScript compiles
- [ ] Tests pass
- [ ] Documentation updated
```

---

## 🛠️ IMPLEMENTATION ROADMAP

### Phase 1: Infrastructure (This Week)
- [ ] Create `.agent_tasks/` directory with template files
- [ ] Implement `scripts/validate_frontend.sh`
- [ ] Add `AGENT_CONTEXT.md` template
- [ ] Set up structured logging

### Phase 2: Agent Roles (Next Week)
- [ ] Define 6 agent roles in `AGENTS.md`
- [ ] Create role-specific prompts
- [ ] Implement task locking mechanism
- [ ] Set up parallel agent harness

### Phase 3: Oracle Testing (Post-Tournament)
- [ ] Implement consensus comparison
- [ ] Add historical upset rate validation
- [ ] Create CLV oracle checks
- [ ] Build automated regression detection

### Phase 4: Full Autonomy (V9.2+)
- [ ] CI pipeline blocks bad commits
- [ ] Agents self-assign from task pool
- [ ] Automatic documentation updates
- [ ] Performance optimization agents

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Mitigation |
|------|------------|
| Agents overwrite each other | Git-based locking + merge conflict resolution |
| Context window overflow | Structured logging + file-based storage |
| Wrong problem solved | High-quality tests + oracle validation |
| Quality degradation | Mandatory validation before commit |
| Time blindness | Progress indicators + fast mode sampling |

---

## 📊 SUCCESS METRICS

| Metric | Current | Target |
|--------|---------|--------|
| Validation time per page | 2 hours | 30 minutes (automated) |
| Issues caught pre-merge | 60% | 95% |
| Agent sessions per feature | 5-10 | 2-3 |
| Parallel agent efficiency | N/A | 4x speedup |

---

## 🎯 RECOMMENDED NEXT STEPS

1. **Immediate:** Implement `scripts/validate_frontend.sh` based on Kimi CLI's 7-point checklist
2. **This Week:** Create `.agent_tasks/` with current tournament prep tasks
3. **Next Week:** Run 2 agents in parallel on V9.2 recalibration (one per parameter)
4. **Post-Tournament:** Full agent team harness for Fantasy Baseball

---

## 📁 FILES TO CREATE

```
CBB_Betting/
├── .agent_tasks/              # Task pool for agents
│   ├── README.md
│   └── [task_files].md
├── scripts/
│   ├── validate_frontend.sh   # Automated 7-point check
│   ├── start_agent.sh         # Launch agent with role
│   └── check_oracle.sh        # Compare vs known-good
├── AGENT_CONTEXT.md           # Template for context
├── CURRENT_STATUS.md          # Live project status
└── AGENTS.md                  # Role definitions
```

---

## 💡 KEY TAKEAWAY

> "The harness around the agent matters more than the agent itself."

For CBB Edge, this means:
1. **Tests are the spec** — If it passes validation, it's correct
2. **Context is king** — Every agent needs CURRENT_STATUS.md
3. **Parallelize wisely** — Small, independent tasks maximize throughput
4. **Validate everything** — No commit without automated checks

---

**Source:** [Anthropic: Building a C Compiler](https://www.anthropic.com/engineering/building-c-compiler)  
**Adapted by:** Kimi CLI  
**For:** Claude Code (Master Architect)  
**Date:** March 19, 2026
