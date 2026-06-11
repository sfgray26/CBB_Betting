#!/usr/bin/env bash
# setup_test_env.sh — Ensure the Python venv matches requirements.txt for local/WSL testing.
#
# Usage:
#   bash scripts/setup_test_env.sh           # normal setup
#   bash scripts/setup_test_env.sh --force   # delete and recreate venv from scratch
#
# Requires: python3.11+ on PATH

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# ── 1. Optionally nuke venv ───────────────────────────────────────────────────
if [[ $FORCE -eq 1 && -d "venv" ]]; then
    echo "[setup_test_env] --force: removing existing venv..."
    rm -rf venv
fi

# ── 2. Create venv if needed ──────────────────────────────────────────────────
if [[ ! -f "venv/bin/python" ]]; then
    echo "[setup_test_env] Creating virtual environment..."
    python3 -m venv venv
fi

PYTHON="venv/bin/python"
PIP="$PYTHON -m pip"

# ── 3. Upgrade pip ───────────────────────────────────────────────────────────
echo "[setup_test_env] Upgrading pip..."
$PIP install --upgrade pip --quiet

# ── 4. Install requirements.txt ──────────────────────────────────────────────
echo "[setup_test_env] Installing requirements.txt..."
$PIP install -r requirements.txt --quiet
echo "[setup_test_env] requirements.txt installed OK"

# ── 5. Smoke-test critical packages ─────────────────────────────────────────
CRITICAL=(requests apscheduler pytest sqlalchemy pydantic fastapi)
MISSING=()
for pkg in "${CRITICAL[@]}"; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        echo "  OK:      $pkg"
    else
        echo "  MISSING: $pkg"
        MISSING+=("$pkg")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: Missing packages after install: ${MISSING[*]}"
    echo "Check requirements.txt and Python version compatibility."
    exit 1
fi

# ── 6. Quick pytest smoke test ───────────────────────────────────────────────
echo "[setup_test_env] Running smoke tests..."
$PYTHON -m pytest tests/test_injury_overlay.py tests/test_fantasy_budget.py -q --tb=short

echo ""
echo "[setup_test_env] Environment is ready. Run tests with:"
echo "  venv/bin/python -m pytest tests/ -q --tb=short"
