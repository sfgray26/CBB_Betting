#!/bin/bash
# check-health.sh — CBB Edge production health checker
# Usage: check-health.sh [--component=<railway|api|scheduler|all>]
# Requires: railway CLI authenticated, RAILWAY_API_URL env var (optional)

set -euo pipefail

COMPONENT="all"
for arg in "$@"; do
    case "$arg" in
        --component=*) COMPONENT="${arg#--component=}" ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# Try to get the API base URL from railway or fall back to env var
API_URL="${RAILWAY_API_URL:-}"
if [ -z "$API_URL" ]; then
    # Attempt to derive from railway domain (best-effort)
    DOMAIN=$(railway domain 2>/dev/null | grep -oP 'https?://[^\s]+' | head -1 || true)
    API_URL="${DOMAIN:-}"
fi

ERRORS=0

check_railway() {
    echo "--- Railway Service Status ---"
    railway status 2>&1 || { echo "  WARN: railway status failed (check auth)"; ((ERRORS++)); }
}

check_api() {
    echo "--- API Connectivity ---"
    if [ -z "$API_URL" ]; then
        echo "  SKIP: RAILWAY_API_URL not set (run: export RAILWAY_API_URL=https://your-app.railway.app)"
        return
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health" --max-time 10 || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "  OK   GET /health -> 200"
    else
        echo "  FAIL GET /health -> $HTTP_CODE" >&2
        ((ERRORS++))
    fi
}

check_scheduler() {
    echo "--- Scheduler Status ---"
    if [ -z "$API_URL" ]; then
        echo "  SKIP: RAILWAY_API_URL not set"
        return
    fi
    API_KEY="${API_KEY_USER1:-}"
    if [ -z "$API_KEY" ]; then
        echo "  SKIP: API_KEY_USER1 not set in local env"
        return
    fi
    RESP=$(curl -s -o - -w "\nHTTP:%{http_code}" \
        -H "X-API-Key: $API_KEY" \
        "${API_URL}/admin/scheduler/status" \
        --max-time 10 || echo "FAIL\nHTTP:000")
    HTTP_CODE=$(echo "$RESP" | grep "HTTP:" | cut -d: -f2)
    if [ "$HTTP_CODE" = "200" ]; then
        JOB_COUNT=$(echo "$RESP" | grep -c '"id"' || echo "0")
        echo "  OK   Scheduler active — $JOB_COUNT job(s) registered"
    else
        echo "  FAIL Scheduler responded HTTP $HTTP_CODE" >&2
        ((ERRORS++))
    fi
}

case "$COMPONENT" in
    railway)    check_railway ;;
    api)        check_api ;;
    scheduler)  check_scheduler ;;
    all)
        check_railway
        echo ""
        check_api
        echo ""
        check_scheduler
        ;;
    *)
        echo "Unknown component: $COMPONENT" >&2
        echo "Valid: railway, api, scheduler, all" >&2
        exit 1
        ;;
esac

echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo "HEALTH: DEGRADED ($ERRORS issue(s) detected)"
    exit 1
else
    echo "HEALTH: OK"
    exit 0
fi
