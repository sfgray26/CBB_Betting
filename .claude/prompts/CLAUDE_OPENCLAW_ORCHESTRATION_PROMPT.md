> **Note:** This is a copy. The canonical version is in the repository root: $(Split-Path System.Collections.Hashtable.Path -Leaf)`n
---

# OpenClaw Orchestration Prompt# Prompt for Claude Code: OpenClaw 24/7 Agent Orchestration

> **Related:** `AGENTS.md` (agent roles) | `ORCHESTRATION.md` (task routing)

Copy-paste this into Claude Code to automate your agent handoffs:

---

## PROMPT START

Configure OpenClaw to orchestrate Claude Code, Kimi CLI, and Gemini CLI so they work in parallel instead of requiring manual handoffs.

### Current Setup (Keep This)
- OpenClaw running 24/7 ✓
- Claude Code on fixed plan ✓
- Kimi CLI on fixed plan ✓
- Gemini CLI for ops ✓
- HANDOFF.md as coordination point ✓

### Goal
Replace this manual workflow:
```
You → Claude CLI → "Do tasks" → Wait → Kimi CLI → "Do audit" → Wait...
```

With this automated workflow:
```
You → OpenClaw → "Start work session" → All 3 agents spawn in parallel
```

### Implementation

#### Step 1: Verify OpenClaw ACP Support
```bash
# Check current OpenClaw version and ACP status
openclaw --version
openclaw agents list
openclaw config get agents

# If agents not configured, check if ACP is available
which acpx
acpx --version
```

#### Step 2: Configure ACP Agents in OpenClaw

Add to `~/.openclaw/config.json`:

```json
{
  "agents": {
    "claude": {
      "name": "Claude Code",
      "runtime": "acp",
      "command": "claude-agent-acp",
      "description": "Architecture and implementation"
    },
    "kimi": {
      "name": "Kimi CLI",
      "runtime": "acp", 
      "command": "kimi acp",
      "description": "Audit and analysis"
    },
    "gemini": {
      "name": "Gemini CLI",
      "runtime": "acp",
      "command": "gemini --acp",
      "description": "DevOps and Railway operations"
    }
  },
  "workflows": {
    "daily-sprint": {
      "trigger": ["start work", "daily sprint"],
      "steps": [
        {
          "agent": "gemini",
          "prompt": "Check Railway status and complete any pending ops from HANDOFF.md §16.4. Update HANDOFF.md when done.",
          "priority": "critical"
        },
        {
          "agent": "claude",
          "prompt": "Read HANDOFF.md and complete your assigned architecture/development tasks. Update HANDOFF.md §X when done.",
          "priority": "high"
        },
        {
          "agent": "kimi",
          "prompt": "Read HANDOFF.md and complete pending audits. Update HANDOFF.md §Y with findings.",
          "priority": "medium"
        }
      ],
      "notify_on_complete": true
    }
  }
}
```

#### Step 3: Add HANDOFF.md Watcher

Create `~/.openclaw/handoff-watcher.js`:

```javascript
// Watches HANDOFF.md and notifies when agents update it
const fs = require('fs');
const path = require('path');

const HANDOFF_PATH = path.join(process.cwd(), 'HANDOFF.md');
let lastContent = '';

function checkHandoff() {
  try {
    const content = fs.readFileSync(HANDOFF_PATH, 'utf8');
    
    if (content !== lastContent && lastContent !== '') {
      // Parse what changed
      const lines = content.split('\n');
      const lastLines = lastContent.split('\n');
      
      // Find section that was updated
      const updatedSection = findUpdatedSection(lines, lastLines);
      
      // Notify via OpenClaw
      console.log(`HANDOFF.md updated: ${updatedSection}`);
      // This will be picked up by OpenClaw notification system
    }
    
    lastContent = content;
  } catch (err) {
    console.error('Error reading HANDOFF.md:', err);
  }
}

function findUpdatedSection(current, previous) {
  // Simple heuristic: find first line that changed
  for (let i = 0; i < Math.min(current.length, previous.length); i++) {
    if (current[i] !== previous[i]) {
      // Look for section header above
      for (let j = i; j >= 0; j--) {
        if (current[j].startsWith('## §')) {
          return current[j];
        }
      }
      return 'Unknown section';
    }
  }
  return 'End of document';
}

// Watch for changes
fs.watchFile(HANDOFF_PATH, { interval: 5000 }, checkHandoff);
console.log('Watching HANDOFF.md for changes...');
```

#### Step 4: Create OpenClaw Skill

Create `.gemini/skills/openclaw-orchestrator/SKILL.md`:

```markdown
# OpenClaw Agent Orchestrator

## Purpose
Coordinates Claude, Kimi, and Gemini agents to work in parallel on CBB Edge tasks.

## Usage

Trigger workflows:
- "Start work session" - Spawns all 3 agents with their tasks
- "Status check" - Reports what each agent is working on
- "Claude status" - Check only Claude's progress
- "Kimi audit" - Trigger Kimi audit specifically

## Handoff Protocol

All agents use HANDOFF.md as the coordination point:
1. Read HANDOFF.md for assigned tasks
2. Complete work
3. Update HANDOFF.md with results
4. OpenClaw notifies user of completion

## Commands

/start-work - Begin daily development workflow
/status - Check all agent statuses  
/notify [on|off] - Enable/disable HANDOFF.md change notifications
```

#### Step 5: Test the Setup

```bash
# Restart OpenClaw with new config
openclaw gateway restart

# Verify agents are registered
openclaw agents list
# Should show: claude, kimi, gemini

# Test spawning agents
openclaw agent spawn claude "Test connection"
openclaw agent spawn kimi "Test connection"
openclaw agent spawn gemini "Test connection"

# Trigger workflow
openclaw workflow run daily-sprint
```

### Deliverables

- [ ] OpenClaw config updated with ACP agents
- [ ] HANDOFF.md watcher script created
- [ ] Skill documentation added
- [ ] Test workflow executed successfully
- [ ] User can trigger "Start work" and all 3 agents spawn
- [ ] Notifications sent when HANDOFF.md updated

### Success Criteria

**Before:**
- User manually opens Claude CLI, types prompt, waits
- User manually opens Kimi CLI, types prompt, waits
- User manually opens Gemini CLI, types prompt, waits

**After:**
- User types "Start work" in OpenClaw chat
- All 3 agents spawn automatically with context from HANDOFF.md
- Agents work in parallel
- User gets notified as each completes

## PROMPT END

---

## Quick Start (Minimal Version)

If the full setup is too complex, start with just notifications:

```bash
# Add to your shell profile (.bashrc/.zshrc)
alias watch-handoff='watch -n 5 "grep -A2 \"Next operator\" HANDOFF.md"'

# Or use fswatch (install first)
fswatch HANDOFF.md | xargs -I {} sh -c 'echo "HANDOFF.md changed at $(date)"'
```

This gives you instant notification when HANDOFF.md changes, so you know when to check the next agent.

