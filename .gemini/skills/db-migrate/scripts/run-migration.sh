#!/bin/bash
# run-migration.sh — Railway database migration runner
# Usage: run-migration.sh <migration_name> [--dry-run]
#        run-migration.sh --verify-only
# Requires: railway CLI authenticated

set -euo pipefail

DRY_RUN=false
VERIFY_ONLY=false
MIGRATION=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)       DRY_RUN=true ;;
        --verify-only)   VERIFY_ONLY=true ;;
        migrate_*)       MIGRATION="$arg" ;;
        *)               echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

if [ "$VERIFY_ONLY" = true ]; then
    echo "=== POST-MIGRATION VERIFICATION ==="
    echo "Checking player_daily_metrics table..."
    railway run python -c "
from backend.models import engine
from sqlalchemy import inspect, text
insp = inspect(engine)
tables = insp.get_table_names()
print('player_daily_metrics:', 'OK' if 'player_daily_metrics' in tables else 'MISSING')
print('projection_snapshots:', 'OK' if 'projection_snapshots' in tables else 'MISSING')
with engine.connect() as c:
    col_names = [col['name'] for col in insp.get_columns('predictions')]
    print('pricing_engine column:', 'OK' if 'pricing_engine' in col_names else 'MISSING')
" 2>&1
    exit 0
fi

if [ -z "$MIGRATION" ]; then
    echo "ERROR: No migration name provided" >&2
    echo "Usage: run-migration.sh <migrate_name> [--dry-run]" >&2
    exit 1
fi

SCRIPT="scripts/${MIGRATION}.py"

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN: $SCRIPT ==="
    railway run python "$SCRIPT" --dry-run
    echo ""
    echo "Review SQL above. If correct, run without --dry-run."
else
    echo "=== APPLYING MIGRATION: $SCRIPT ==="
    railway run python "$SCRIPT"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Migration SUCCEEDED."
    else
        echo "Migration FAILED (exit $EXIT_CODE). Escalate to Claude Code." >&2
        exit $EXIT_CODE
    fi
fi
