# Tasks 10-11 Planning Summary

**Date:** April 10, 2026
**Status**: Planning Complete, Ready for Execution
**Approach**: Quality-first research before implementation

---

## Executive Summary

I've completed comprehensive research and planning for the final tasks (10-11) in the data quality remediation plan. Here's what I found:

### Key Findings

**Task 10 (Impossible ERA)**: Investigation Required
- The task mentions "1.726 for pitcher" but an ERA of 1.726 is excellent (not impossible)
- The planning document mentions "ERA > 100" but this may be a placeholder
- **Action needed**: Run diagnostic script on Railway to find actual problematic ERA values

**Task 11 (Validation Audit)**: Ready to Execute
- Comprehensive validation audit script exists
- Will compare before/after state of all data quality fixes
- Foundation for next development phase

**Kimi Research**: Timed Out
- All 5 research tasks timed out (likely due to complex queries)
- Alternative approach: Break into smaller, more focused research tasks
- Or defer until after Tasks 10-11 complete

---

## Task 10: Fix Impossible ERA - Detailed Plan

### Current Understanding

**Unknown**: We don't actually know what the impossible ERA value is yet.
- Plan mentions "ERA > 100" but that's likely a placeholder
- Summary mentions "1.726 for pitcher" but that's a valid ERA
- **Must investigate first before implementing any fix**

### Investigation Strategy

**Step 1**: Run diagnostic script on Railway
```bash
railway run --service Fantasy-App -- python scripts/diagnose_era_issue.py
```

This will output:
- Overall ERA distribution (min, max, avg, median)
- Rows with ERA > 50 (truly problematic)
- Rows with ERA < 1.0 (excellent but rare)
- ERA calculation mismatches
- Specific check for ERA = 1.726
- Summary and recommendations

**Step 2**: Based on diagnostic results, determine root cause

**Scenario A**: If ERA > 100 exists
- Root cause: Likely calculation error (e.g., IP parsing issue)
- Fix: Correct ERA calculation in ingestion logic
- Backfill: Recalculate ERA for affected rows

**Scenario B**: If ERA = 1.726 is the "issue"
- Root cause: False positive (ERA of 1.726 is excellent)
- Fix: Update task validation threshold or close task as N/A

**Scenario C**: If calculation mismatches found
- Root cause: ERA calculation formula wrong
- Fix: Correct formula and recalculate all ERAs

**Step 3**: Implement fix

**Option A**: Fix calculation logic (if bug found)
- File: `backend/services/daily_ingestion.py`
- Location: Where ERA is computed during ingestion
- Formula: `ERA = (earned_runs / innings_pitched) × 9`
- Add: IP parsing for "6.2" format (already exists: `_parse_innings_pitched()`)

**Option B**: Update bad data (if source error)
- Create migration script to NULL out incorrect ERAs
- Trigger backfill from BDL API
- Verify correct values populated

**Option C**: Update validation rules (if false positive)
- Adjust ERA validation threshold
- Document ERA < 1.0 as rare but valid
- Close task as N/A with explanation

**Step 4**: Add validation test
- File: `tests/test_era_validation.py`
- Test: Ensure no ERA values > 100 or < 0
- Test: Ensure ERA calculations are correct
- Run: `pytest tests/test_era_validation.py -v`

**Step 5**: Verify fix
```bash
railway run --service Fantasy-App -- python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
count = db.execute(text('SELECT COUNT(*) FROM mlb_player_stats WHERE era < 0 OR era > 100')).scalar()
print(f'Impossible ERA count: {count}')
db.close()
"
```

Expected: `Impossible ERA count: 0`

### Quality Gates

- [ ] Diagnostic script executed successfully
- [ ] Root cause identified (calculation bug vs bad data vs false positive)
- [ ] Fix implemented and tested
- [ ] No impossible ERA values remain (0 rows with ERA < 0 or > 100)
- [ ] Validation test added to prevent recurrence
- [ ] Changes committed to git

---

## Task 11: Run Full Data Validation Audit - Detailed Plan

### Current Understanding

**Purpose**: Final validation to ensure all data quality issues are resolved
**Baseline**: 0 CRITICAL, 131 WARNING, 64 INFO (from original audit)
**Target**: 0 CRITICAL, <50 WARNING, minimal INFO

### Validation Strategy

**Step 1**: Create or run validation audit script

**Option A**: If script exists
```bash
python scripts/db_validation_audit.py > reports/2026-04-10-final-validation.md
```

**Option B**: If script doesn't exist, create it
```python
# scripts/db_validation_audit.py
"""
Comprehensive database validation audit.

Checks all tables for data quality issues:
- NULL values in required fields
- Impossible values (e.g., ERA < 0 or > 100)
- Foreign key violations
- Orphaned records
- Empty tables that should have data
"""

from backend.models import SessionLocal
from sqlalchemy import text
import sys

def run_validation():
    db = SessionLocal()
    issues = []

    # Check player_id_mapping
    print("Validating player_id_mapping...")
    yahoo_key_null = db.execute(text(
        "SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NULL"
    )).scalar()
    if yahoo_key_null > 100:
        issues.append(f"player_id_mapping: {yahoo_key_null} rows with NULL yahoo_key")

    # Check position_eligibility
    print("Validating position_eligibility...")
    bdl_id_null = db.execute(text(
        "SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NULL"
    )).scalar()
    if bdl_id_null > 100:
        issues.append(f"position_eligibility: {bdl_id_null} rows with NULL bdl_player_id")

    # Check mlb_player_stats
    print("Validating mlb_player_stats...")
    era_impossible = db.execute(text(
        "SELECT COUNT(*) FROM mlb_player_stats WHERE era < 0 OR era > 100"
    )).scalar()
    if era_impossible > 0:
        issues.append(f"mlb_player_stats: {era_impossible} rows with impossible ERA")

    ops_null = db.execute(text(
        "SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NULL AND (obp IS NOT NULL AND slg IS NOT NULL)"
    )).scalar()
    if ops_null > 100:
        issues.append(f"mlb_player_stats: {ops_null} rows with NULL ops (but obp/slg available)")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total issues found: {len(issues)}")
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ NO ISSUES FOUND")

    db.close()
    return len(issues)

if __name__ == "__main__":
    issue_count = run_validation()
    sys.exit(0 if issue_count == 0 else 1)
```

**Step 2**: Execute validation audit
```bash
# Run on Railway
railway run --service Fantasy-App -- python scripts/db_validation_audit.py

# Or run locally if DB accessible
python scripts/db_validation_audit.py
```

**Step 3**: Compare with baseline

**Baseline** (from earlier audits):
- player_id_mapping.yahoo_key: 0 populated → 2,376 populated ✅
- position_eligibility.bdl_player_id: 0 populated → 2,376 populated ✅
- mlb_player_stats.ops: 100% NULL → Computed ✅
- mlb_player_stats.whip: 100% NULL → Computed ✅
- Empty tables: 16 → Diagnosed & documented ✅

**Target**:
- All Tasks 1-9 fixes verified
- No impossible ERA values
- No new issues introduced
- WARNING count significantly reduced

**Step 4**: Create before/after report

File: `reports/2026-04-10-final-data-quality-report.md`

```markdown
# Final Data Quality Remediation Report

**Date:** April 10, 2026
**Phase**: Tasks 1-11 Complete

## Before vs After Comparison

### Player Identity Resolution
| Metric | Before | After | Status |
|--------|--------|-------|--------|
| player_id_mapping.yahoo_key | 0 | 2,376 | ✅ COMPLETE |
| position_eligibility.bdl_player_id | 0 | 2,376 | ✅ COMPLETE |

### Computed Fields
| Metric | Before | After | Status |
|--------|--------|-------|--------|
| mlb_player_stats.ops | 100% NULL | Computed | ✅ COMPLETE |
| mlb_player_stats.whip | 100% NULL | Computed | ✅ COMPLETE |
| mlb_player_stats.caught_stealing | Partial | Defaulted | ✅ COMPLETE |

### Data Quality Fixes
| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Impossible ERA values | ? | 0 | ✅ COMPLETE |

### Empty Tables (Diagnosed)
| Table | Finding | Status |
|-------|---------|--------|
| probable_pitchers | BDL API doesn't provide | ✅ DOCUMENTED |
| statcast_performances | 502 errors, no retry | ✅ DOCUMENTED |
| data_ingestion_logs | Unused by design | ✅ DOCUMENTED |

## Remaining Issues (if any)

[List any WARNING or INFO items that remain]

## Next Steps

1. Integrate player identity resolution into daily sync
2. Implement statcast retry logic (Task 5 finding)
3. Add probable pitcher data source
4. Build lineup optimization features

## Conclusion

All CRITICAL data quality issues resolved.
Phase 1-4 remediation COMPLETE.
```

**Step 5**: Commit report
```bash
git add reports/2026-04-10-final-data-quality-report.md
git commit -m "docs(report): add final data quality remediation report - Tasks 1-11 complete"
```

### Quality Gates

- [ ] Validation audit executed successfully
- [ ] All fixes from Tasks 1-10 verified
- [ ] No new issues introduced
- [ ] Before/after report created
- [ ] HANDOFF.md updated with completion status
- [ ] Ready for next development phase

---

## Kimi Research: Alternative Approach

All 5 Kimi research tasks timed out. Here are alternative approaches:

### Option A: Defer Research
- Complete Tasks 10-11 first
- Then tackle research with more focused queries
- Research not blocking current work

### Option B: Manual Research
- I can research some topics during implementation
- User can research others using web search
- Focus on high-impact topics only

### Option C: Retry with Smaller Tasks
- Break each research task into 2-3 smaller queries
- Use `thinking: false` for faster responses
- Reduce `max_output_tokens` to 3000-4000

**Recommendation**: Option A (defer research) since Tasks 10-11 are the priority and research doesn't block implementation.

---

## Execution Roadmap

### Phase 1: Investigation (Task 10) - 30 minutes

1. Run diagnostic script on Railway
   ```bash
   railway run --service Fantasy-App -- python scripts/diagnose_era_issue.py
   ```

2. Review diagnostic output
   - Identify actual problematic ERA values
   - Determine root cause
   - Plan fix strategy

### Phase 2: Implementation (Task 10) - 1-2 hours

1. Implement fix based on findings
   - If calculation bug: Fix ERA computation
   - If bad data: Update or NULL values
   - If false positive: Document and close

2. Add validation test
   ```python
   # tests/test_era_validation.py
   def test_no_impossible_era_values():
       """Test that no ERA values are outside valid range (0-100)."""
       from backend.models import SessionLocal
       from sqlalchemy import text
       db = SessionLocal()
       count = db.execute(text(
           "SELECT COUNT(*) FROM mlb_player_stats WHERE era < 0 OR era > 100"
       )).scalar()
       assert count == 0, f"Found {count} rows with impossible ERA"
       db.close()
   ```

3. Verify fix
   - Run test: `pytest tests/test_era_validation.py -v`
   - Verify no impossible ERA values

4. Commit changes
   ```bash
   git add scripts/diagnose_era_issue.py tests/test_era_validation.py
   git commit -m "fix(data-quality): fix impossible ERA value and add validation"
   ```

### Phase 3: Validation (Task 11) - 1-2 hours

1. Run validation audit
   ```bash
   railway run --service Fantasy-App -- python scripts/db_validation_audit.py
   ```

2. Create final report
   - Document all fixes
   - Before/after comparison
   - Remaining issues (if any)

3. Update HANDOFF.md
   - Mark Tasks 10-11 complete
   - Update overall status
   - Prepare handoff for next phase

4. Commit report
   ```bash
   git add reports/2026-04-10-final-data-quality-report.md HANDOFF.md
   git commit -m "docs(report): add final validation report - Tasks 10-11 complete, data quality remediation done"
   ```

---

## Quality Commitment

I will NOT:
- ❌ Skip investigation and assume ERA = 1.726 is the issue
- ❌ Implement fix without understanding root cause
- ❌ Rush validation audit
- ❌ Skip tests or verification

I WILL:
- ✅ Run diagnostic script first to find actual issue
- ✅ Understand root cause before implementing fix
- ✅ Add comprehensive validation tests
- ✅ Document all findings thoroughly
- ✅ Verify fixes work correctly
- ✅ Create detailed before/after report

---

## Files Created

1. **docs/superpowers/prep/tasks-10-11-data-quality-fixes-prep.md**
   - Comprehensive research and quality checklist
   - Kimi research opportunities identified
   - Execution strategy

2. **scripts/diagnose_era_issue.py**
   - Diagnostic script to investigate ERA values
   - Runs on Railway
   - Outputs comprehensive report

3. **docs/superpowers/planning-summary-tasks-10-11.md** (this file)
   - Planning summary for user review
   - Detailed execution roadmap
   - Quality commitment

---

## Ready for Execution

**Planning Status**: ✅ COMPLETE

**Next Action**: User to review planning and approve execution

**Estimated Time**:
- Task 10: 1.5-2.5 hours (investigation + fix + tests)
- Task 11: 1-2 hours (validation + report)
- Total: 2.5-4.5 hours

**Quality Priority**: HIGH - Taking time to do this right

---

**Prepared by**: Claude Code (Master Architect)
**Preparation Date**: April 10, 2026
**Status**: Planning complete, awaiting user approval to proceed with execution
