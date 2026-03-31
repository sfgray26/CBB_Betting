# Gemini CLI MCP Server Analysis for CBB Edge

**Date:** March 31, 2026  
**Analyst:** Claude Code (Master Architect)  
**Subject:** Assessment of Gemini CLI MCP integration for operational efficiency  
**Reference:** AGENTS.md swimlane restrictions, ORCHESTRATION.md routing rules

---

## Executive Summary

**Verdict: Limited benefit for current workflow; potential value if scoped carefully to DevOps-only operations.**

The Gemini CLI MCP server would allow Gemini CLI to discover and execute custom tools. However, given **EMAC-075 restrictions** (Gemini CLI is hard-restricted from code writes), the MCP integration must be carefully scoped to avoid policy violations.

| Scenario | Benefit | Risk | Recommendation |
|----------|---------|------|----------------|
| Railway DevOps automation | Medium | Low | ✅ Consider |
| Code generation/IDE tools | High | **Critical** | ❌ Policy violation |
| Report/data access | Low | Low | ⚠️ Nice-to-have |
| Database operations | Medium | Medium | ⚠️ Use existing scripts |

---

## 1. What MCP Would Enable

### 1.1 Custom Tool Discovery
Gemini CLI could discover and execute project-specific tools:

```json
// .gemini/settings.json
{
  "mcpServers": {
    "cbb-edge-ops": {
      "command": "python",
      "args": ["scripts/mcp_server.py"],
      "env": {
        "RAILWAY_TOKEN": "$RAILWAY_TOKEN",
        "DATABASE_URL": "$DATABASE_URL"
      },
      "trust": false  // Require confirmation for each tool call
    }
  }
}
```

### 1.2 Potential Custom Tools

| Tool | Purpose | Policy Check |
|------|---------|--------------|
| `railway_logs` | Tail production logs | ✅ Permitted per AGENTS.md |
| `railway_vars` | Check env var values | ✅ Permitted |
| `railway_deploy` | Trigger redeploy | ✅ Permitted |
| `run_migration` | Execute pre-approved DB script | ✅ Permitted with constraints |
| `health_check` | Run py_compile + smoke test | ✅ Permitted |
| `generate_code` | Write Python/TypeScript | ❌ **EMAC-075 violation** |
| `edit_file` | Modify source files | ❌ **EMAC-075 violation** |

---

## 2. Current vs. Proposed Workflow

### Current Workflow (AGENTS.md Compliant)

```
User: "Check Railway logs"
Gemini CLI: railway logs --follow
           ↓
        (reads logs)
           ↓
User: "Deploy the fix"
Gemini CLI: Claude, the code is ready
           ↓
Claude Code: Implements changes
           ↓
Gemini CLI: railway up (trigger deploy)
```

### Proposed MCP Workflow

```
User: "Check system health"
Gemini CLI: Discovers mcp_railway_health_check tool
           ↓
           Executes: python scripts/mcp_server.py health_check
           ↓
           Returns: {status: "healthy", logs: "...", vars: "..."}
           ↓
User: "Deploy the fix"
Gemini CLI: Discovers mcp_railway_deploy tool
           ↓
           (same escalation to Claude for code)
```

**Key Difference:** MCP provides a structured interface, but the fundamental workflow remains the same.

---

## 3. Specific Use Cases Analysis

### 3.1 Use Case: Railway Health Dashboard

**Current:**
```bash
railway logs --follow
railway variables | grep -i error
railway status
```

**With MCP:**
```python
# scripts/mcp_server.py
@mcp.tool()
def railway_health_snapshot() -> dict:
    """Get comprehensive Railway health status."""
    return {
        "logs_last_hour": get_recent_logs(hours=1),
        "error_patterns": analyze_errors(),
        "env_vars_status": check_required_vars(),
        "deploy_status": get_deploy_status()
    }
```

**Benefit:** Single command gets comprehensive status.  
**Effort:** 2-3 days to build and test MCP server.  
**Value:** Medium — saves typing, not a bottleneck currently.

---

### 3.2 Use Case: Pre-Deploy Validation

**Current (per AGENTS.md §7):**
```bash
# Gemini CLI runs manually:
railway run python -m py_compile backend/services/balldontlie.py
railway run python -m py_compile backend/services/mlb_analysis.py
railway variables | grep -i balldontlie
railway up
```

**With MCP:**
```python
@mcp.tool()
def validate_before_deploy(files: list[str]) -> dict:
    """Run pre-deploy validation checks."""
    return {
        "py_compile": [check_syntax(f) for f in files],
        "env_vars": verify_env_vars(),
        "tests": run_smoke_tests(),
        "can_deploy": all_checks_pass()
    }
```

**Benefit:** Structured, repeatable validation.  
**Risk:** Must ensure tool doesn't write code (read-only validation only).  
**Value:** Medium — codifies existing AGENTS.md playbook.

---

### 3.3 Use Case: Database Query Access

**Current:**
- Claude Code writes migration scripts
- Gemini CLI runs: `railway run python scripts/migration.py`

**With MCP:**
```python
@mcp.tool()
def query_db(query: str, read_only: bool = True) -> dict:
    """Execute read-only database queries."""
    if not read_only:
        raise PermissionError("Write queries require Claude Code approval")
    return execute_query(query)
```

**Risk Assessment:**  
⚠️ **HIGH RISK** — Tool could be misused for unauthorized DB modifications.  
Mitigation: `read_only=True` default + tool confirmation prompts.

---

## 4. Policy Compliance Analysis

### 4.1 AGENTS.md Constraints

> **GEMINI CLI — DevOps Lead**  
> **Restriction level:** HARD — no Python or TypeScript code writes.  
> **Root cause of restriction (EMAC-075, Mar 20, 2026):** Duplicate FastAPI route creation, invalid dict key references, testing against production without deploying.

### 4.2 MCP Tool Design Principles

To remain compliant, MCP tools must:

| Principle | Implementation | Violation Example |
|-----------|---------------|-------------------|
| Read-only by default | Database queries with `read_only=True` | Tool that writes migration files |
| No code generation | Return data, never source code | Tool that generates Python routes |
| No file modification | Read reports, don't edit them | Tool that patches .env files |
| Confirmation prompts | `trust: false` in settings | `trust: true` bypassing safety |
| Audit logging | Log all tool executions | Silent tool execution |

---

## 5. Implementation Cost-Benefit

### 5.1 Implementation Requirements

| Component | Effort | Description |
|-----------|--------|-------------|
| MCP server scaffold | 4-6 hours | Python MCP SDK setup |
| Railway tool wrappers | 4-8 hours | Wrap railway CLI commands |
| Security hardening | 4-8 hours | Read-only enforcement, audit logs |
| Testing | 4-6 hours | Validate against AGENTS.md constraints |
| Documentation | 2-4 hours | Update AGENTS.md with MCP scope |
| **Total** | **18-32 hours** | ~3-4 days of focused work |

### 5.2 Ongoing Maintenance

| Task | Frequency | Effort |
|------|-----------|--------|
| MCP server dependency updates | Monthly | 30 min |
| Tool validation after railway CLI updates | As needed | 1-2 hours |
| Security audit | Quarterly | 2-4 hours |

### 5.3 Value Assessment

**Quantifiable Time Savings:**
- Pre-deploy validation: 5 min/task × 20 deploys/month = 100 min/month
- Health checks: 2 min/task × 30 checks/month = 60 min/month
- Log analysis: 3 min/task × 10 incidents/month = 30 min/month
- **Total:** ~3 hours/month saved

**ROI Calculation:**
- Implementation: 25 hours (avg)
- Monthly savings: 3 hours
- Break-even: ~8 months

---

## 6. Risks and Mitigations

### 6.1 Policy Violation Risk (Critical)

**Risk:** MCP tools could bypass EMAC-075 restrictions by providing indirect code modification capabilities.

**Examples of indirect violations:**
```python
# DANGEROUS - violates EMAC-075
@mcp.tool()
def update_env_var(key: str, value: str):
    """Update Railway env var."""
    # This is fine per AGENTS.md...
    railway.vars.set(key, value)
    
    # ...but what if value contains Python code?
    # "DEPLOY_SCRIPT=import os; os.system('rm -rf /')"
```

**Mitigation:**
1. All tools reviewed by Claude Code before deployment
2. Input validation schemas strict
3. `trust: false` — require confirmation for every call
4. Audit log of all tool executions

### 6.2 Security Risk (Medium)

**Risk:** MCP server exposes internal systems to Gemini CLI context.

**Mitigation:**
1. No sensitive env vars in MCP server scope
2. Use explicit env var passing (not full environment)
3. Network isolation (localhost only)
4. Rate limiting on tool calls

### 6.3 Complexity Risk (Medium)

**Risk:** Additional moving part that can break.

**Mitigation:**
1. Graceful degradation (fallback to manual CLI commands)
2. Health check endpoint for MCP server itself
3. Version pinning for MCP SDK

---

## 7. Alternative: Claude Code MCP Integration

**Observation:** Claude Code (not Gemini CLI) could also benefit from MCP.

### Potential Claude Code MCP Tools

| Tool | Use Case | Value |
|------|----------|-------|
| `search_codebase` | Semantic code search | High — faster than grep |
| `run_tests` | Execute pytest subset | Medium — already in workflow |
| `validate_schema` | Check Pydantic schemas | Medium — prevent type mismatches |
| `generate_migration` | Scaffold DB migrations | Medium — boilerplate reduction |

**Policy Note:** This would NOT violate AGENTS.md because Claude Code has code-writing authority.

---

## 8. Recommendation

### Short Term (Next 2 Weeks): No Action

**Rationale:**
1. Current AGENTS.md workflow is functional
2. EMAC-075 restriction requires careful scoping
3. MLB season just started — minimize infrastructure changes
4. ROI break-even is 8 months

### Medium Term (Post-MLB Season): Consider Limited MCP

**Scope:** DevOps-only tools, read-only operations
**Tools:**
- `railway_health_check` — Comprehensive status
- `validate_deploy_ready` — Pre-deployment checks
- `query_reports` — Read report files from `reports/`

**Explicitly Excluded:**
- Code generation tools
- File modification tools
- Database write operations
- Any tool that could indirectly modify source code

### Implementation Plan (If Approved)

```bash
# Week 1: Scaffold and Railway tools
- Create scripts/mcp_server.py
- Implement railway_health_check tool
- Add validation and confirmation prompts

# Week 2: Testing and documentation
- Test against AGENTS.md constraints
- Update AGENTS.md with MCP scope
- Add audit logging

# Week 3: Deploy and validate
- Configure .gemini/settings.json
- Validate in staging
- Document runbook
```

---

## 9. Conclusion

The Gemini CLI MCP server offers **moderate operational efficiency gains** but introduces **policy complexity** given EMAC-075 restrictions.

**My recommendation:** 
1. **Defer** MCP setup until after MLB season (post-October 2026)
2. **Scope strictly** to DevOps read-only operations
3. **Audit thoroughly** to ensure no EMAC-075 violations
4. **Consider Claude Code MCP** instead — higher value, no policy conflicts

The current manual workflow, while less elegant, is **explicit, auditable, and compliant** with your established agent governance model.

---

*Report prepared by Claude Code (Master Architect)*  
*Reference: AGENTS.md §2 (Gemini CLI restrictions), ORCHESTRATION.md routing rules*
