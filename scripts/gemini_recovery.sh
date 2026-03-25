#!/bin/bash
# Gemini Session Recovery Script
# Run this when Gemini CLI loses context or authentication
# 
# Usage: bash scripts/gemini_recovery.sh

set -e

echo "=========================================="
echo "   Gemini CLI Session Recovery"
echo "   EMAC-082 DevOps Lead Profile"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check HANDOFF.md for current status
echo -e "${GREEN}[1/5] Reading HANDOFF.md for pending operations...${NC}"
echo "---"
grep -A3 "Next operator (Gemini" HANDOFF.md 2>/dev/null || echo "HANDOFF.md not found in current directory"
echo "---"
echo ""

# 2. Check Railway authentication
echo -e "${GREEN}[2/5] Checking Railway authentication...${NC}"
if command -v railway &> /dev/null; then
    if railway whoami &> /dev/null; then
        echo -e "${GREEN}✓ Railway authenticated${NC}"
        railway whoami 2>/dev/null | head -1
    else
        echo -e "${RED}✗ Railway NOT authenticated${NC}"
        echo "  Run: railway login"
        echo "  Or set RAILWAY_TOKEN environment variable"
    fi
else
    echo -e "${RED}✗ Railway CLI not installed${NC}"
    echo "  Install: npm install -g @railway/cli"
fi
echo ""

# 3. Show pending env var operations
echo -e "${GREEN}[3/5] Pending Railway env var operations (from HANDOFF.md §16.4)...${NC}"
echo "---"
grep -A20 "Current Pending Operations" HANDOFF.md 2>/dev/null | grep "PENDING" || echo "Check HANDOFF.md section 16.4 for pending ops"
echo "---"
echo ""

# 4. Quick env var check
echo -e "${GREEN}[4/5] Current Railway env vars (critical ones)...${NC}"
if railway whoami &> /dev/null; then
    railway variables 2>/dev/null | grep -E "(INTEGRITY_SWEEP|ENABLE_MLB|ENABLE_INGESTION)" || echo "Could not fetch variables (may need project selection)"
else
    echo "Skip: Railway not authenticated"
fi
echo ""

# 5. Summary
echo -e "${GREEN}[5/5] Recovery Summary${NC}"
echo "=========================================="
echo "Next steps for Gemini CLI:"
echo ""
echo "1. If Railway not authenticated:"
echo "   railway login"
echo ""
echo "2. Set pending env vars:"
echo "   railway variables set INTEGRITY_SWEEP_ENABLED=false"
echo "   railway variables set ENABLE_MLB_ANALYSIS=true"
echo "   railway variables set ENABLE_INGESTION_ORCHESTRATOR=true"
echo ""
echo "3. Verify settings:"
echo "   railway variables"
echo ""
echo "4. Watch logs:"
echo "   railway logs --follow"
echo ""
echo -e "${YELLOW}Remember:${NC} Gemini CLI is DevOps Lead only — NO code changes!"
echo -e "${YELLOW}Escalate to:${NC} Claude Code (code) | Kimi CLI (audit)"
echo "=========================================="
