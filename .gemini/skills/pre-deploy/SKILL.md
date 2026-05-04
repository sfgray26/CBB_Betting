---
name: pre-deploy
description: Run pre-deployment validation checks before pushing to Railway. Use when Claude Code says code is ready to deploy or when asked to validate before deploying.
---

# Pre-Deploy Validation

## When to Use

- Claude Code says "code is ready, deploy it"
- User asks "is it safe to deploy"
- Before any `railway up` command

## Workflow (ALWAYS run in order)

### Step 1: Syntax check all modified Python files
```bash
# Find modified Python files and py_compile them
for f in $(git diff --name-only HEAD | grep '\.py$'); do
  echo "Checking: $f"
  python -m py_compile "$f" || { echo "SYNTAX ERROR in $f"; exit 1; }
done
```

### Step 2: Run targeted tests
```bash
# Run fantasy baseball tests (the critical path)
python -m pytest tests/test_waiver_integration.py tests/test_mlb_analysis.py -q --tb=short
```

### Step 3: Check env vars are set
```bash
bash .gemini/skills/env-check/scripts/check-vars.sh --critical-only
```

### Step 4: Check system health BEFORE deploy
```bash
bash .gemini/skills/health-check/scripts/check-health.sh
```

### Step 5: Deploy (only if all above pass)
```bash
railway up
```

## Rules

- **NEVER** run `railway up` if Step 1–4 fail
- If tests fail → STOP, tell Claude Code the test output
- If env vars missing → STOP, use env-check skill to set them
- If health check fails → STOP, investigate with railway-logs skill
- After deploy → wait 60s, then run post-deploy skill
