#!/bin/bash
# filter-logs.sh — Railway log filter wrapper
# Usage: filter-logs.sh [--errors] [--grep=<keyword>] [--lines=<n>]
# Requires: railway CLI authenticated

set -euo pipefail

ERRORS=false
GREP_PATTERN=""
LINES=200

for arg in "$@"; do
    case "$arg" in
        --errors)        ERRORS=true ;;
        --grep=*)        GREP_PATTERN="${arg#--grep=}" ;;
        --lines=*)       LINES="${arg#--lines=}" ;;
        *)               echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# Fetch log snapshot
LOGS=$(railway logs 2>&1 | tail -"$LINES")

if [ "$ERRORS" = true ]; then
    echo "=== ERROR/WARNING LINES ==="
    echo "$LOGS" | grep -iE "(error|critical|warning|exception|traceback|restart|crash)" || echo "(none found)"
elif [ -n "$GREP_PATTERN" ]; then
    echo "=== LINES MATCHING: $GREP_PATTERN ==="
    echo "$LOGS" | grep -i "$GREP_PATTERN" || echo "(none found)"
else
    echo "$LOGS"
fi
