> **Note:** This is a copy. The canonical version is in the repository root: $(Split-Path System.Collections.Hashtable.Path -Leaf)`n
---

# Local LLM Prompt (OpenClaw)# Prompt for Claude Code: Local LLM Integration Implementation

Copy-paste the following into Claude Code:

---

## PROMPT START

Implement local LLM integration and cost optimization for CBB Edge using the architecture specified in `reports/MULTI_AGENT_ORCHESTRATION_ANALYSIS.md`.

### Phase 1: LLM Router (Immediate - 2 hours)

Install and configure **NadirClaw** for intelligent request routing:

```bash
# Install
npm install -g nadirclaw

# Configure for CBB Edge
export NADIRCLAW_SIMPLE_MODEL=gemini-2.5-flash
export NADIRCLAW_COMPLEX_MODEL=claude-sonnet-4-5
export NADIRCLAW_FREE_MODEL=ollama/llama3.2:8b  # placeholder until ollama setup

# Create config file
mkdir -p ~/.nadirclaw
cat > ~/.nadirclaw/config.json << 'EOF'
{
  "simple_model": "gemini-2.5-flash",
  "complex_model": "claude-sonnet-4-5",
  "routing_rules": {
    "prompt_length_threshold": 1000,
    "complexity_detection": true,
    "keywords": {
      "simple": ["fix typo", "add comment", "explain", "rename"],
      "complex": ["architect", "design", "refactor", "audit", "security"]
    }
  },
  "fallback_chain": ["gemini-2.5-flash", "claude-sonnet-4-5"]
}
EOF

# Test routing
curl http://localhost:8856/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Fix this typo"}]}'
```

Deliverables:
- [ ] NadirClaw installed and running
- [ ] Router configured with tiered models
- [ ] Claude Code configured to use router
- [ ] Test: Verify routing works for simple vs complex prompts
- [ ] Document: Update HANDOFF.md §17 with setup instructions

### Phase 2: Ollama Local LLM (Post-Apr 7 - 1-2 days)

Install Ollama and configure for CBB Edge development:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull coding-optimized models
ollama pull llama3.2:8b
ollama pull qwen2.5-coder:14b

# Verify installation
ollama list
ollama run llama3.2:8b "Explain this Python function"
```

Deliverables:
- [ ] Ollama installed and running
- [ ] At least 2 models downloaded
- [ ] Performance benchmark (tokens/sec)
- [ ] Router updated to include local tier
- [ ] Documentation: Model selection guide

### Phase 3: ACPX Orchestration (Optional - Evaluate need)

If multi-agent coordination becomes necessary:

```bash
# Install ACPX
npm install -g acpx

# Configure for CBB Edge agents
acpx config init

# Test agent spawning
acpx claude "analyze backend architecture"
acpx kimi "audit betting model"
```

### Constraints

- Do NOT implement full OpenClaw yet (defer until post-Apr 7)
- Keep existing workflows intact
- Document everything in HANDOFF.md
- Measure cost savings after 1 week

### References

- Report: `reports/MULTI_AGENT_ORCHESTRATION_ANALYSIS.md`
- NadirClaw: https://github.com/doramirdor/NadirClaw
- Ollama: https://ollama.com
- ACPX: https://github.com/openclaw/acpx

## PROMPT END

---

## Usage

1. Read the full analysis in `reports/MULTI_AGENT_ORCHESTRATION_ANALYSIS.md`
2. Copy-paste the prompt above into Claude Code
3. Claude will implement Phase 1 (router) immediately
4. Phases 2-3 can be deferred until after Apr 7 deadline

## Expected Results

**Phase 1 Only:**
- 30-40% cost reduction immediately
- No workflow changes
- 2-hour setup

**Phases 1-2:**
- 50-70% cost reduction
- Local LLM for simple tasks
- 1-2 day setup

**Full Implementation:**
- Structured multi-agent orchestration
- Persistent sessions
- Parallel agent execution

