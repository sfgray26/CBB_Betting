# OpenClaw 24/7 Agent Orchestration Workflow

**Current Friction:** Manual handoffs between Claude/Kimi/Gemini via separate CLIs  
**Solution:** Use OpenClaw as the central orchestrator with ACP (Agent Client Protocol)

---

## Current State (The Problem)

```
You → Check HANDOFF.md
  → Open Claude Code CLI → "Do your tasks"
  → Wait
  → Check HANDOFF.md
  → Open Kimi CLI → "Do your audits"  
  → Wait
  → Check HANDOFF.md
  → Open Gemini CLI → "Do your ops"
  → Wait
  → Repeat...
```

**Issues:**
- Context switching overhead
- Agents can't see each other's work in real-time
- You become the bottleneck
- Risk of missing updates

---

## Target State (The Solution)

```
OpenClaw Gateway (24/7)
  ├─→ ACP Session: Claude Code ─┐
  ├─→ ACP Session: Kimi CLI ────┼──→ Shared Context
  ├─→ ACP Session: Gemini CLI ──┘      (HANDOFF.md via MCP)
  └─→ Sub-Agent: Task Router
       
You → OpenClaw Discord/Chat → "Start daily workflow"
  → OpenClaw spawns all 3 agents
  → Agents collaborate via shared memory
  → OpenClaw reports status
  → You intervene only when needed
```

---

## Implementation Options

### Option A: OpenClaw + ACPX (Recommended - Minimal Change)

**What:** Use `acpx` (ACP client) to spawn agents via OpenClaw, but keep your current HANDOFF.md workflow

**Setup:**
```bash
# OpenClaw is already running
# Just configure ACP agents in OpenClaw

# Add to ~/.openclaw/config.json:
{
  "agents": {
    "claude": {
      "runtime": "acp",
      "command": "claude-agent-acp"
    },
    "kimi": {
      "runtime": "acp", 
      "command": "kimi acp"
    },
    "gemini": {
      "runtime": "acp",
      "command": "gemini --acp"
    }
  }
}

# Restart OpenClaw gateway
openclaw gateway restart
```

**Usage:**
```
You in Discord/Telegram:
  "@openclaw Start daily development workflow"

OpenClaw:
  1. Spawns Claude (architecture tasks)
  2. Spawns Kimi (audit tasks)
  3. Spawns Gemini (ops tasks)
  4. Reports: "3 agents active. Claude working on MLB model, 
               Kimi auditing OpenClaw, Gemini setting Railway vars"
  5. Each agent updates HANDOFF.md as they work
  6. OpenClaw notifies you when tasks complete
```

**Pros:**
- Keeps your existing HANDOFF.md workflow
- Agents work in parallel
- You get notifications, not constant checking
- One interface (Discord/Chat) instead of 3 CLIs

**Cons:**
- Need to configure ACP in OpenClaw
- Initial setup time

---

### Option B: OpenClaw Sub-Agents with Task Queue

**What:** OpenClaw manages a task queue, assigns to appropriate agent

**Setup:**
```json
// ~/.openclaw/skills/dev-workflow/skill.json
{
  "name": "dev-workflow",
  "triggers": ["start sprint", "daily workflow"],
  "agents": ["claude", "kimi", "gemini"],
  "workflow": [
    {"agent": "gemini", "task": "check railway status", "priority": "critical"},
    {"agent": "claude", "task": "review handoff.md architecture tasks", "priority": "high"},
    {"agent": "kimi", "task": "audit recent code changes", "priority": "medium"}
  ]
}
```

**Usage:**
```
You: "@openclaw start daily workflow"

OpenClaw:
  - Assigned to Gemini: Check Railway (ETA 2 min)
  - Assigned to Claude: Review architecture (ETA 15 min)
  - Assigned to Kimi: Audit code (ETA 10 min)
  
[2 min later]
OpenClaw: "✅ Railway: All systems operational"

[10 min later]
OpenClaw: "✅ Kimi: Audit complete - 2 minor issues found, 
          see HANDOFF.md §15.3"

[15 min later]
OpenClaw: "✅ Claude: Architecture review complete, 
          starting MLB model implementation"
```

**Pros:**
- True automation - you just trigger it
- Parallel execution
- Time estimates
- Status tracking

**Cons:**
- More complex to set up
- Need to define workflows

---

### Option C: Keep Current + Add OpenClaw Notifications

**What:** Minimal change - just add Discord/Telegram notifications for HANDOFF.md updates

**Setup:**
```bash
# Create a simple watcher script
# ~/.openclaw/handoff-watcher.js

const fs = require('fs');
const { exec } = require('child_process');

fs.watchFile('HANDOFF.md', () => {
  // Parse what changed
  const changes = parseHandoffChanges();
  
  // Notify you
  exec(`openclaw notify "HANDOFF.md updated: ${changes.summary}"`);
});
```

**Usage:**
- You still manually switch CLIs
- But you get pinged when HANDOFF.md updates
- No guesswork about when to check

**Pros:**
- Minimal setup
- Keeps current workflow
- Just adds notifications

**Cons:**
- Still manual context switching
- You remain the bottleneck

---

## Recommended: Hybrid Approach

### Phase 1: Notifications (This Week)
Set up OpenClaw to notify you when HANDOFF.md changes

### Phase 2: Parallel Spawn (Next Week) 
Use OpenClaw to spawn all 3 agents at once with their tasks

### Phase 3: Full Automation (Later)
Define workflows that run without your trigger

---

## Specific Configuration for Your Setup

Since OpenClaw is already running 24/7:

### Step 1: Verify ACP Support
```bash
# Check if OpenClaw can see agents
openclaw agents list

# Should show: claude, kimi, gemini (if configured)
# If not, need to add ACP adapters
```

### Step 2: Add MCP for HANDOFF.md
```bash
# Install file watcher MCP
openclaw mcp add filesystem

# Configure to watch HANDOFF.md
openclaw config set mcp.filesystem.watch ["HANDOFF.md"]
```

### Step 3: Create Trigger Phrase
```
In OpenClaw chat (Discord/Telegram):

"Start work session"
→ OpenClaw reads HANDOFF.md
→ Spawns Claude: "Work on tasks in §X"
→ Spawns Kimi: "Audit items in §Y"  
→ Spawns Gemini: "Complete ops in §Z"
→ Reports: "3 agents active, monitoring..."
```

---

## What You Need to Decide

1. **Interface:** Discord, Telegram, or something else?
2. **Trigger:** "Start work" or scheduled (9 AM daily)?
3. **Intervention:** Auto-run or wait for your approval?
4. **Reporting:** Real-time updates or summary when done?

---

## Next Steps

**Option A (Easiest):**
Tell me your preferred chat platform, I'll give you exact config.

**Option B (Best):**
Give Claude the prompt to set up ACP orchestration in your existing OpenClaw.

**Option C (Safest):**
Start with notifications only, add orchestration later.

Which feels right for your workflow?
