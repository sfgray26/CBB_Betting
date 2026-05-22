#!/bin/bash
# CBB Edge - UAT Automation Runner
# This script runs the full UAT suite and can be scheduled via cron

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_DIR/reports/uat"
DATE=$(date +%Y%m%d_%H%M%S)
BASE_URL="${UAT_BASE_URL:-https://your-app.railway.app}"
API_KEY="${UAT_API_KEY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
        exit 1
    fi
    
    if ! python3 -c "import playwright" 2>/dev/null; then
        log "Installing Playwright..."
        pip install playwright
        playwright install chromium
    fi
    
    log "Dependencies OK"
}

# Setup environment
setup() {
    log "Setting up UAT environment..."
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR"
    
    # Check API key
    if [ -z "$API_KEY" ]; then
        if [ -f "$PROJECT_DIR/.env" ]; then
            API_KEY=$(grep "API_KEY" "$PROJECT_DIR/.env" | cut -d '=' -f2 | tr -d '"')
        fi
    fi
    
    if [ -z "$API_KEY" ]; then
        error "API key not found. Set UAT_API_KEY environment variable or add to .env file"
        exit 1
    fi
    
    log "Setup complete"
}

# Run UAT
run_uat() {
    log "Starting UAT execution..."
    log "Base URL: $BASE_URL"
    log "Report directory: $REPORTS_DIR"
    
    OUTPUT_FILE="$REPORTS_DIR/uat_report_${DATE}.md"
    
    cd "$PROJECT_DIR"
    
    if python3 "$SCRIPT_DIR/uat_automation.py" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        --output "$OUTPUT_FILE" \
        --json; then
        log "UAT completed successfully"
        log "Report saved to: $OUTPUT_FILE"
        return 0
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 2 ]; then
            warn "UAT completed with warnings (exit code 2)"
            log "Report saved to: $OUTPUT_FILE"
            return 0
        else
            error "UAT failed (exit code $EXIT_CODE)"
            log "Report saved to: $OUTPUT_FILE"
            return 1
        fi
    fi
}

# Generate summary report
generate_summary() {
    log "Generating summary report..."
    
    SUMMARY_FILE="$REPORTS_DIR/latest_summary.md"
    
    cat > "$SUMMARY_FILE" << EOF
# CBB Edge UAT - Latest Summary
**Generated:** $(date)
**Latest Report:** [uat_report_${DATE}.md](./uat_report_${DATE}.md)

## Quick Links
- [Latest Full Report](uat_report_${DATE}.md)
- [Latest JSON Data](uat_report_${DATE}.json)

## Historical Reports
EOF
    
    # List all reports
    ls -1t "$REPORTS_DIR"/uat_report_*.md | head -20 | while read -r report; do
        basename=$(basename "$report")
        echo "- [$basename](./$basename)" >> "$SUMMARY_FILE"
    done
    
    log "Summary updated: $SUMMARY_FILE"
}

# Send notification (uses existing Discord service from Python script)
send_notification() {
    # Notifications are now handled directly by uat_automation.py
    # This function is kept for compatibility but does nothing
    log "Discord notifications will be sent by the Python script"
}

# Main execution
main() {
    log "========================================"
    log "CBB Edge UAT Automation"
    log "========================================"
    
    check_dependencies
    setup
    
    if run_uat; then
        generate_summary
        send_notification
        log "UAT workflow completed successfully"
    else
        generate_summary
        send_notification
        error "UAT workflow completed with failures"
        exit 1
    fi
}

# Run main
main "$@"
