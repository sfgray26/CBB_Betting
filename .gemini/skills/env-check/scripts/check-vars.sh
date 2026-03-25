#!/bin/bash
# check-vars.sh — Railway environment variable checker
# Usage: check-vars.sh [--critical-only]
# Requires: railway CLI authenticated

set -euo pipefail

CRITICAL_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --critical-only) CRITICAL_ONLY=true ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# Fetch all Railway env vars (suppress credentials from output)
ALL_VARS=$(railway variables 2>&1)

check_var() {
    local name="$1"
    local expected="$2"  # empty means "just check it exists"
    local val
    val=$(echo "$ALL_VARS" | grep "^${name}" | head -1 | cut -d'=' -f2- | tr -d ' ')

    if [ -z "$val" ]; then
        echo "  MISSING  $name"
        return 1
    elif [ -n "$expected" ] && [ "$val" != "$expected" ]; then
        echo "  WRONG    $name = '$val' (expected: '$expected')"
        return 1
    else
        # Mask secrets in output
        case "$name" in
            *TOKEN*|*SECRET*|*KEY*|DATABASE_URL)
                echo "  OK       $name = [SET]"
                ;;
            *)
                echo "  OK       $name = '$val'"
                ;;
        esac
        return 0
    fi
}

ERRORS=0

echo "=== CRITICAL: App startup variables ==="
check_var "DATABASE_URL"             "" || ((ERRORS++))
check_var "THE_ODDS_API_KEY"         "" || ((ERRORS++))
check_var "API_KEY_USER1"            "" || ((ERRORS++))

echo ""
echo "=== CRITICAL: Feature flags (must be correct NOW) ==="
check_var "INTEGRITY_SWEEP_ENABLED"  "false" || ((ERRORS++))
check_var "ENABLE_MLB_ANALYSIS"      "true"  || ((ERRORS++))
check_var "ENABLE_INGESTION_ORCHESTRATOR" "true" || ((ERRORS++))

if [ "$CRITICAL_ONLY" = false ]; then
    echo ""
    echo "=== OPTIONAL: Yahoo Fantasy Baseball ==="
    check_var "YAHOO_CLIENT_ID"      "" || ((ERRORS++))
    check_var "YAHOO_CLIENT_SECRET"  "" || ((ERRORS++))
    check_var "YAHOO_REFRESH_TOKEN"  "" || ((ERRORS++))

    echo ""
    echo "=== OPTIONAL: Discord routing ==="
    check_var "DISCORD_BOT_TOKEN"    "" || true  # not critical if missing
    check_var "DISCORD_CHANNEL_FANTASY_WAIVERS" "" || true
    check_var "DISCORD_CHANNEL_OPENCLAW_BRIEFS" "" || true

    echo ""
    echo "=== OPTIONAL: Other ==="
    check_var "KENPOM_API_KEY"       "" || true
fi

echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo "RESULT: $ERRORS critical variable(s) MISSING or INCORRECT. Fix with: railway variables set NAME=value"
    exit 1
else
    echo "RESULT: All critical variables OK."
    exit 0
fi
