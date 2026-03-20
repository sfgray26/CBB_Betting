#!/usr/bin/env bash
# =============================================================================
# validate_frontend.sh — CBB Edge Frontend Safety Checker
#
# Implements automated pre-merge validation based on the OpenClaw checklist.
# Methodology: Anthropic "Building a C Compiler with 16 Agents" — harness-first.
#
# SIGNAL DESIGN: This script targets nullable API fields defined in lib/types.ts.
# It intentionally excludes typed helper functions, literal arrays, and patterns
# where context makes clear a null check exists. TypeScript is the authoritative
# type checker — this script catches runtime anti-patterns TypeScript may allow.
#
# Usage:
#   ./scripts/validate_frontend.sh                    # all dashboard pages
#   ./scripts/validate_frontend.sh path/to/page.tsx   # single file
#
# Exit: 0 = pass/warn only, 1 = blocking issues found
# =============================================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

TMPLIST=$(mktemp)
COUNTER=$(mktemp)
WARN_TOTAL=0
PAGE_PASS=0
PAGE_FAIL=0

trap 'rm -f "$TMPLIST" "$COUNTER"' EXIT

if [ $# -gt 0 ]; then
  for f in "$@"; do echo "$f"; done > "$TMPLIST"
else
  find "frontend/app/(dashboard)" -name "page.tsx" 2>/dev/null | sort > "$TMPLIST"
fi

FILE_COUNT=$(wc -l < "$TMPLIST" | tr -d ' ')
[ "$FILE_COUNT" -eq 0 ] && { echo "No .tsx files found. Run from repo root."; exit 1; }

echo ""
echo -e "${BOLD}CBB Edge Frontend Validator${NC} — safety checklist"
echo -e "Checking $FILE_COUNT file(s)...\n"

# Increment the per-file blocking issue counter
add_issue() {
  local current
  current=$(cat "$COUNTER" 2>/dev/null || echo 0)
  echo $((current + 1)) > "$COUNTER"
  echo -e "  ${RED}[BLOCK]${NC} $1"
}

add_warn() {
  ((WARN_TOTAL++)) || true
  echo -e "  ${YELLOW}[WARN]${NC}  $1"
}

# =============================================================================
# CHECK 1: Nullable API fields called directly without null guard
# Only targets fields typed "number | null" in lib/types.ts
# =============================================================================
check_nullable_fields() {
  local file="$1"
  local TMPG
  TMPG=$(mktemp)

  # These are the exact nullable number fields from lib/types.ts + api ground truth
  local NULLABLE='\.(projected_margin|edge_conservative|recommended_units|model_prob|profit_loss_units|profit_loss_dollars|clv_points|clv_prob|mean_clv|brier_score|log_loss|std_clv|outcome|halt_reason|quota_remaining)\.'

  grep -nE "$NULLABLE(toFixed|toLocaleString|toString)\(" "$file" 2>/dev/null \
    | grep -vE "//|interface |type |import |!= null|=== null|!== null" \
    > "$TMPG" || true

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    lno=$(echo "$line" | cut -d: -f1)
    content=$(echo "$line" | cut -d: -f2-)
    add_issue ":$lno — nullable field accessed without ?. guard: $content"
  done < "$TMPG"

  rm -f "$TMPG"
}

# =============================================================================
# CHECK 2: Object.entries() on API data without ?? {}
# =============================================================================
check_object_entries() {
  local file="$1"
  local TMPG
  TMPG=$(mktemp)

  grep -nE "Object\.entries\([a-zA-Z]" "$file" 2>/dev/null \
    | grep -vE "//|interface |type " \
    > "$TMPG" || true

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    lno=$(echo "$line" | cut -d: -f1)
    content=$(echo "$line" | cut -d: -f2-)
    # Check for ?? {} on same line
    echo "$content" | grep -qE '\?\?\ \{\}|\?\?\{' && continue
    add_warn ":$lno — Object.entries() without ?? {} guard: $content"
  done < "$TMPG"

  rm -f "$TMPG"
}

# =============================================================================
# CHECK 3: API decimal fields (0.043) displayed as % without × 100
# =============================================================================
check_decimal_display() {
  local file="$1"
  local TMPG
  TMPG=$(mktemp)

  # Look for known decimal fields used in display context without * 100
  local FIELDS='\.(roi|win_rate|mean_clv|edge_conservative|clv_prob|model_prob|positive_clv_rate)'

  grep -nE "$FIELDS" "$file" 2>/dev/null \
    | grep -vE "interface |type |import |export |//|Record<|\?: |null$|\bnull\b" \
    | grep -E 'toFixed|%' \
    | grep -vE '\* 100|pct\(|signed\(|displayPercent\(|× 100' \
    > "$TMPG" || true

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    lno=$(echo "$line" | cut -d: -f1)
    content=$(echo "$line" | cut -d: -f2-)
    add_warn ":$lno — decimal API field may need × 100 for % display: $content"
  done < "$TMPG"

  rm -f "$TMPG"
}

# =============================================================================
# CHECK 4: Loading + error states presence
# =============================================================================
check_loading_error() {
  local file="$1"
  local qcount
  qcount=$(grep -cE "useQuery\(" "$file" 2>/dev/null || echo 0)
  qcount=$(echo "$qcount" | tr -d '[:space:]')

  if [ "$qcount" -gt 0 ]; then
    local scount ecount
    scount=$(grep -cE "animate-pulse|isLoading\b|Loading" "$file" 2>/dev/null || echo 0)
    ecount=$(grep -cE "isError\b|border-rose|ErrorCard|error state" "$file" 2>/dev/null || echo 0)
    scount=$(echo "$scount" | tr -d '[:space:]')
    ecount=$(echo "$ecount" | tr -d '[:space:]')

    [ "$scount" -eq 0 ] && add_issue "— $qcount useQuery() calls but no loading skeleton"
    [ "$ecount" -eq 0 ] && add_issue "— $qcount useQuery() calls but no error state"
  fi
}

# =============================================================================
# CHECK 5: Empty state visibility
# =============================================================================
check_empty_state() {
  local file="$1"
  local count
  count=$(grep -cE "emptyMessage|No [a-zA-Z]+ (yet|data|picks|found)|text-center.*zinc" \
    "$file" 2>/dev/null || echo 0)
  count=$(echo "$count" | tr -d '[:space:]')
  [ "$count" -eq 0 ] && add_warn "— no visible empty state found (emptyMessage / 'No ... yet')"
}

# =============================================================================
# Main loop
# =============================================================================

while IFS= read -r file; do
  [ -z "$file" ] && continue
  echo -e "${BOLD}━━━ $file${NC}"

  echo 0 > "$COUNTER"

  check_nullable_fields  "$file"
  check_object_entries   "$file"
  check_decimal_display  "$file"
  check_loading_error    "$file"
  check_empty_state      "$file"

  file_issues=$(cat "$COUNTER" | tr -d '[:space:]')

  if [ "$file_issues" -eq 0 ]; then
    echo -e "  ${GREEN}✓ PASS${NC}"
    ((PAGE_PASS++)) || true
  else
    echo -e "  ${RED}✗ FAIL${NC} ($file_issues blocking issue(s))"
    ((PAGE_FAIL++)) || true
  fi
  echo ""
done < "$TMPLIST"

# =============================================================================
# Summary
# =============================================================================

echo -e "${BOLD}━━━ SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
printf "  %-20s %d\n" "Files checked:" "$((PAGE_PASS + PAGE_FAIL))"
printf "  %-20s %d\n" "PASS:" "$PAGE_PASS"
printf "  %-20s %d\n" "FAIL:" "$PAGE_FAIL"
printf "  %-20s %d\n" "Warnings:" "$WARN_TOTAL"
echo ""

if [ "$PAGE_FAIL" -gt 0 ]; then
  echo -e "${RED}✗ Validation failed — fix blocking issues before merging.${NC}"
  exit 1
else
  [ "$WARN_TOTAL" -gt 0 ] && \
    echo -e "${GREEN}✓ All pages passed.${NC} ${YELLOW}($WARN_TOTAL warning(s) — review manually)${NC}" || \
    echo -e "${GREEN}✓ All pages passed validation.${NC}"
  exit 0
fi
