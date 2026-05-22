#!/bin/bash
# Major Improvement Sprint - Multi-Agent Execution
# Executes 8 improvement tasks across Codex, Gemini, and Claude (Hermes)

set -e

cd /mnt/c/Users/sfgra/repos/Fixed/cbb-edge

echo "=========================================="
echo "  MAJOR IMPROVEMENT SPRINT"
echo "  8 Tasks | 3 Agents | Parallel Execution"
echo "=========================================="
echo ""

# Function to run codex tasks
run_codex_tasks() {
    echo "🤖 CODEX: Starting tasks..."
    
    # Task 1: Test Coverage
    echo "  → Task 1: Creating comprehensive test suite..."
    git checkout -b agent/codex/test-coverage-20260516 || true
    codex -p "Create comprehensive test suite for core services. Current coverage is critically low (only 3 test files for 210 Python files). 

Create tests for:
1. backend/services/matchup_engine.py - test pitcher stat fetching, matchup scoring
2. backend/services/player_mapper.py - test player card mapping, Yahoo integration
3. backend/services/row_projector.py - test weekly projections, ROS calculations

Requirements:
- Use pytest with fixtures
- Mock all external API calls (Yahoo, BDL, Statcast)
- Test edge cases and error handling
- Target 80%+ line coverage
- Follow patterns in backend/test_p1_bugs_fixes.py
- Create backend/tests/conftest.py with shared fixtures

Files to create:
- backend/tests/test_matchup_engine.py
- backend/tests/test_player_mapper.py
- backend/tests/test_row_projector.py
- backend/tests/conftest.py

Run tests with: python -m pytest backend/tests/ -v --cov=backend/services" \
    --permission-mode bypassPermissions
    
    # Task 2: Remove Prints
    echo "  → Task 2: Replacing print statements with logging..."
    git checkout -b agent/codex/remove-prints-20260516 || true
    codex -p "Replace all print statements with proper logging across the codebase.

Files to modify:
- backend/models.py - line 622: print to logger.info
- backend/services/team_conference_lookup.py - lines 211-229: prints to logger
- backend/services/sentinel.py - line 213: print to logger
- backend/services/possession_sim.py - line 23: print to logger
- backend/services/openclaw_telemetry.py - lines 508, 526-544: prints to logger

Requirements:
- Add 'import logging' and 'logger = logging.getLogger(__name__)' at module level
- Replace print() with appropriate logger level (info/debug/warning)
- Keep __main__ block prints (they're for CLI usage)
- Add context to log messages (e.g., 'Team Conference Lookup: {team}')
- Do NOT change test files or debugging scripts

Run: python -c 'import backend.models; import backend.services.team_conference_lookup' to verify no errors" \
    --permission-mode bypassPermissions
    
    echo "✅ CODEX: Tasks complete"
}

# Function to run gemini tasks  
run_gemini_tasks() {
    echo "🌟 GEMINI: Starting tasks..."
    
    # Task 5: Documentation
    echo "  → Task 5: Documenting all TODOs..."
    git checkout -b agent/gemini/todo-documentation-20260516 || true
    gemini -p "Document all TODOs found in the codebase (30+ found in search).

Search these files for TODO/FIXME/XXX comments:
- backend/routers/fantasy.py (lines 2074, 2253, 2254, 2270, 3277, 3280, etc.)
- backend/services/player_mapper.py (lines 112, 203, 233, 236)
- backend/services/row_projector.py (lines 251, 255, 570, 601)
- backend/services/scoreboard_orchestrator.py (lines 394, 397)

Create docs/TODO.md with:
1. Table of all TODOs: Location | Description | Priority | Est. Effort
2. Organize by category: API, Data Pipeline, Features, Performance
3. Add dependencies between TODOs
4. Mark which are blocked by others

Also update HERMES.md:
- Add 'Technical Debt' section
- Reference TODO.md
- List the 5 highest priority TODOs

Be thorough - check every TODO comment in the codebase." \
    --permission-mode bypassPermissions
    
    # Task 8: Security Audit
    echo "  → Task 8: Security audit and hardening..."
    git checkout -b agent/gemini/security-audit-20260516 || true
    gemini -p "Perform comprehensive security audit of the codebase.

Files to review:
- backend/routers/*.py - All API endpoints
- backend/main.py - App configuration
- backend/auth.py - Authentication (if exists)
- backend/middleware/*.py - Middleware

Check for:
1. SQL Injection risks - Look for raw SQL, f-strings in queries
2. Input validation - Are all inputs validated/sanitized?
3. Authentication - Are protected endpoints properly guarded?
4. CORS configuration - Is it too permissive?
5. Secrets in logs - Are API keys/tokens being logged?
6. Path traversal - File operations with user input
7. XSS vulnerabilities - User input rendered in responses

Create docs/SECURITY_AUDIT.md with:
1. Executive Summary (Critical/High/Medium/Low counts)
2. Detailed Findings (location, severity, recommendation)
3. Input validation checklist
4. Recommended security improvements

Add security middleware if missing:
- Rate limiting
- Input sanitization
- Security headers" \
    --permission-mode bypassPermissions
    
    echo "✅ GEMINI: Tasks complete"
}

# Function for Claude tasks (run by Hermes)
run_claude_tasks() {
    echo "🧠 CLAUDE (Hermes): Starting complex tasks..."
    echo "  These will be executed in subsequent prompts"
    echo "✅ CLAUDE: Task plan created"
}

# Main execution
echo "📋 Execution Plan:"
echo "  Codex:  2 tasks (test coverage, remove prints)"
echo "  Gemini: 2 tasks (documentation, security audit)"
echo "  Claude: 4 tasks (quality score, stale data, game context, performance)"
echo ""

# Run in parallel
echo "🚀 Launching parallel execution..."
echo ""

# Start Codex and Gemini in background
run_codex_tasks &
CODEX_PID=$!

run_gemini_tasks &
GEMINI_PID=$!

# Show Claude tasks that need manual execution
run_claude_tasks

echo ""
echo "⏳ Waiting for Codex and Gemini to complete..."
wait $CODEX_PID
codex_status=$?
wait $GEMINI_PID
gemini_status=$?

echo ""
echo "=========================================="
echo "  SPRINT STATUS"
echo "=========================================="
if [ $codex_status -eq 0 ]; then
    echo "✅ Codex: Complete"
else
    echo "❌ Codex: Failed (exit $codex_status)"
fi

if [ $gemini_status -eq 0 ]; then
    echo "✅ Gemini: Complete"
else
    echo "❌ Gemini: Failed (exit $gemini_status)"
fi

echo "⏳ Claude: Execute manually via prompts"
echo ""
echo "Next: Run the 4 Claude tasks with separate prompts"
echo "=========================================="
