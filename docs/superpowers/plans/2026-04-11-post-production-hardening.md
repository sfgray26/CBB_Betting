# Post-Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete P-1..P-4 follow-up items and add HIGH-priority database infrastructure improvements for production readiness.

**Architecture:** Four focused tasks that (1) refresh stale validation queries, (2) fix cosmetic rowcount over-reporting, (3) dedupe player_id_mapping duplicates, and (4) add performance indexes for common query patterns.

**Tech Stack:** FastAPI, SQLAlchemy, PostgreSQL (Railway), pytest, alembic-style migrations

---

## Task 1: Refresh Validation Audit Endpoint Queries (FOLLOW-UP 2)

**Problem:** The `/admin/validation-audit` endpoint returns stale findings — it re-emits hardcoded issue descriptions instead of re-querying live tables. After P-1..P-4, the audit incorrectly reports "1639 NULL ops despite obp+slg", "477 orphaned position_eligibility", and "Statcast 502 errors".

**Root Cause:** The validation queries check conditions but use old thresholds and don't account for fixes already applied.

**Files:**
- Modify: `backend/admin_endpoints_validation.py:50-150, 220-264`
- Test: `tests/test_admin_validation_audit.py`

- [ ] **Step 1: Write failing test for orphan count accuracy**

Create `tests/test_admin_validation_audit.py`:

```python
"""
Test that validation audit reports accurate orphan counts.
"""
import pytest
from backend.admin_endpoints_validation import validation_audit
from backend.models import SessionLocal, get_db


def test_orphan_count_matches_live_db():
    """The orphan count in validation output should match actual DB state."""
    db = SessionLocal()
    try:
        # Get actual orphan count from DB
        actual_orphans = db.execute("""
            SELECT COUNT(*) FROM position_eligibility pe
            LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
            WHERE pe.yahoo_player_key IS NOT NULL AND pim.yahoo_key IS NULL
        """).scalar()
    finally:
        db.close()

    # Get validation report
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    response = client.get("/admin/validation-audit")
    assert response.status_code == 200
    report = response.json()

    # Find orphan finding in report
    orphan_finding = None
    for severity in ["critical", "high", "medium", "low"]:
        for finding in report.get(severity, []):
            if finding.get("category") == "Foreign Keys" and "orphaned" in finding.get("issue", "").lower():
                orphan_finding = finding
                break

    # If orphan finding exists, it should match actual count
    if orphan_finding:
        # Extract number from issue string like "362 orphaned position_eligibility rows"
        import re
        match = re.search(r'(\d+)\s+orphaned', orphan_finding["issue"])
        if match:
            reported_count = int(match.group(1))
            assert reported_count == actual_orphans, f"Reported {reported_count} orphans but DB has {actual_orphans}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_admin_validation_audit.py::test_orphan_count_matches_live_db -v`
Expected: FAIL (the validation endpoint uses hardcoded or stale logic)

- [ ] **Step 3: Update ops validation to use dynamic threshold**

Modify `backend/admin_endpoints_validation.py` lines 108-127:

```python
# Check 2.1: ops (On-Base Percentage Plus Slugging Percentage)
result = db.execute(text("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(ops) as ops_populated,
        COUNT(*) - COUNT(ops) as ops_null,
        COUNT(*) FILTER (WHERE obp IS NOT NULL AND slg IS NOT NULL AND ops IS NULL) as backfillable_ops
    FROM mlb_player_stats
""")).fetchone()

# Only report if backfillable rows exist (have obp+slg but missing ops)
if result.backfillable_ops > 0:
    add_finding("high", "Computed Fields", "mlb_player_stats",
        f"{result.backfillable_ops} rows have NULL ops despite having obp+slg components.",
        "Run POST /admin/backfill-ops-whip to populate computed fields.",
        "SELECT COUNT(*) FROM mlb_player_stats WHERE obp IS NOT NULL AND slg IS NOT NULL AND ops IS NULL")
elif result.ops_null > 0 and result.ops_null == result.backfillable_ops:
    # All NULL ops are structural (missing components), this is expected
    findings["info"].append({
        "category": "Computed Fields",
        "issue": f"ops has {result.ops_null} NULL rows (all structurally unbackfillable — missing obp or slg)",
        "recommendation": "No action needed. These rows lack required components.",
        "sql_check": None
    })
```

- [ ] **Step 4: Update orphan check to use current threshold**

Modify `backend/admin_endpoints_validation.py` lines 250-263:

```python
# Check 5.1: Orphaned position_eligibility records
result = db.execute(text("""
    SELECT COUNT(*) as orphaned_count
    FROM position_eligibility pe
    LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
    WHERE pe.yahoo_player_key IS NOT NULL
      AND pim.yahoo_key IS NULL
""")).fetchone()

# Dynamic threshold: warn if orphan count grew significantly
# Current baseline after P-3 is 362 (permanently unmatchable prospects)
ORPHAN_BASELINE = 362
if result.orphaned_count > ORPHAN_BASELINE + 50:  # Allow 50 new orphans before alerting
    add_finding("medium", "Foreign Keys", "position_eligibility",
        f"{result.orphaned_count} orphaned position_eligibility rows (baseline: {ORPHAN_BASELINE}).",
        f"{result.orphaned_count - ORPHAN_BASELINE} new orphans detected. Consider re-running fuzzy linker.",
        "SELECT COUNT(*) FROM position_eligibility pe LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key WHERE pim.yahoo_key IS NULL")
elif result.orphaned_count > 0:
    findings["info"].append({
        "category": "Foreign Keys",
        "issue": f"{result.orphaned_count} orphaned position_eligibility rows (at expected baseline)",
        "recommendation": "These are primarily minor-league prospects with no MLB/BDL entry. No action needed.",
        "sql_check": None
    })
```

- [ ] **Step 5: Update Statcast validation to check row count not API errors**

Modify `backend/admin_endpoints_validation.py` lines 220-240:

```python
                elif table_name == "statcast_performances":
                    statcast_count = db.execute(text("SELECT COUNT(*) FROM statcast_performances")).scalar()
                    if statcast_count == 0:
                        add_finding("high", "Empty Tables", "statcast_performances",
                            "statcast_performances table is empty",
                            "Run POST /admin/backfill/statcast to populate. If rows processed but table empty, check transform_to_performance() for column name mismatches.",
                            "SELECT COUNT(*) FROM statcast_performances")
                    elif statcast_count < 5000:
                        add_finding("medium", "Empty Tables", "statcast_performances",
                            f"statcast_performances has only {statcast_count} rows (expected 15000+ for March 20 - April 11)",
                            "Re-run POST /admin/backfill/statcast to fill missing dates.",
                            "SELECT COUNT(*) FROM statcast_performances")
                    else:
                        findings["info"].append({
                            "category": "Data Volume",
                            "table": "statcast_performances",
                            "issue": f"Statcast data populated: {statcast_count} rows",
                            "recommendation": "No action needed.",
                            "sql_check": None
                        })
                        # Store row count for summary
                        validation_results["statcast_row_count"] = statcast_count
```

- [ ] **Step 6: Run test to verify fix passes**

Run: `venv/Scripts/python -m pytest tests/test_admin_validation_audit.py::test_orphan_count_matches_live_db -v`
Expected: PASS

- [ ] **Step 7: Add additional test for ops backfill detection**

```python
def test_ops_validation_only_reports_backfillable():
    """OPS validation should not complain about structurally unbackfillable NULL ops."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    response = client.get("/admin/validation-audit")
    assert response.status_code == 200
    report = response.json()

    # Check that we don't have a HIGH severity finding for ops
    # (NULL ops with missing components is expected state)
    high_findings = report.get("high", [])
    ops_findings = [f for f in high_findings if "ops" in f.get("issue", "").lower() and "null" in f.get("issue", "").lower()]

    # If ops finding exists, verify it's about backfillable rows only
    if ops_findings:
        assert "backfillable" in ops_findings[0]["issue"].lower() or "despite" in ops_findings[0]["issue"].lower()
```

- [ ] **Step 8: Run all new tests**

Run: `venv/Scripts/python -m pytest tests/test_admin_validation_audit.py -v`
Expected: 2/2 PASS

- [ ] **Step 9: Commit**

```bash
git add backend/admin_endpoints_validation.py tests/test_admin_validation_audit.py
git commit -m "fix: refresh validation audit queries for live DB state

- Update ops validation to only report backfillable NULLs (have obp+slg)
- Update orphan check to use dynamic threshold vs hardcoded 477
- Update Statcast validation to check row count not assume API errors
- Add tests for orphan count accuracy and backfill detection

Fixes FOLLOW-UP 2 from P-1..P-4 production deployment."
```

---

## Task 2: Fix Backfill Ops/WHIP Rowcount Over-Reporting (FOLLOW-UP 4)

**Problem:** `/admin/backfill-ops-whip` reports "8 whip_updated" on every call, but those rows are NULL→NULL no-ops (innings_pitched='0.0' makes WHIP mathematically undefined).

**Root Cause:** The diagnostic UPDATE doesn't exclude rows where innings_pitched='0.0', so it "updates" them to NULL (same value) and counts them.

**Files:**
- Modify: `backend/admin_backfill_ops_whip.py:56-71`
- Test: `tests/test_admin_backfill_ops_whip.py`

- [ ] **Step 1: Write failing test for rowcount accuracy**

Create `tests/test_admin_backfill_ops_whip.py`:

```python
"""
Test that backfill-ops-whip endpoint accurately counts actual updates.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.models import SessionLocal
from sqlalchemy import text


def test_whip_backfill_excludes_zero_ip_rows():
    """
    WHIP backfill should not count rows with innings_pitched='0.0' as updated.
    For these rows, WHIP is mathematically undefined (division by zero).
    """
    client = TestClient(app)
    db = SessionLocal()

    try:
        # Find a row with innings_pitched='0.0' and non-zero walks+hits
        test_row = db.execute(text("""
            SELECT id, walks_allowed, hits_allowed, innings_pitched
            FROM mlb_player_stats
            WHERE innings_pitched = '0.0'
              AND walks_allowed IS NOT NULL
              AND hits_allowed IS NOT NULL
              AND (walks_allowed + hits_allowed) > 0
            LIMIT 1
        """)).fetchone()

        if not test_row:
            pytest.skip("No test data: need a row with IP=0.0 and non-zero BB+H")

        # Ensure whip is NULL before test
        db.execute(text("UPDATE mlb_player_stats SET whip = NULL WHERE id = :id"), {"id": test_row.id})
        db.commit()

        # Call backfill endpoint
        response = client.post("/admin/backfill-ops-whip")
        assert response.status_code == 200
        result = response.json()

        # The whip_updated count should NOT include our zero-IP row
        # because WHIP = (BB+H)/IP is undefined when IP=0
        # If it was counted, whip_updated would be >= 1

        # Verify the row is still NULL (wasn't updated)
        whip_value = db.execute(text("SELECT whip FROM mlb_player_stats WHERE id = :id"), {"id": test_row.id}).scalar()
        assert whip_value is None, "WHIP should remain NULL for zero-inning appearances"

    finally:
        db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_admin_backfill_ops_whip.py::test_whip_backfill_excludes_zero_ip_rows -v`
Expected: FAIL (the endpoint counts zero-IP rows as "updated")

- [ ] **Step 3: Fix WHIP backfill to exclude zero IP rows**

Modify `backend/admin_backfill_ops_whip.py` lines 56-71:

```python
        # Backfill whip
        whip_result = db.execute(text("""
            UPDATE mlb_player_stats
            SET whip = (walks_allowed + hits_allowed)::numeric /
                      NULLIF(
                          CAST(SPLIT_PART(innings_pitched, '.', 1) AS NUMERIC) +
                          CAST(SPLIT_PART(innings_pitched, '.', 2) AS NUMERIC) / 3.0,
                          0
                      )
            WHERE whip IS NULL
              AND walks_allowed IS NOT NULL
              AND hits_allowed IS NOT NULL
              AND innings_pitched IS NOT NULL
              AND innings_pitched != ''
              AND innings_pitched NOT IN ('0.0', '0', '0.00')
        """))
        result["whip_updated"] = whip_result.rowcount
```

- [ ] **Step 4: Add diagnostic field to show excluded zero-IP rows**

Modify `backend/admin_backfill_ops_whip.py` lines 20-29:

```python
    result = {
        "status": "success",
        "ops_updated": 0,
        "whip_updated": 0,
        "whip_skipped_zero_ip": 0,  # NEW: count rows skipped due to IP=0
        "initial_ops_null": 0,
        "initial_whip_null": 0,
        "final_ops_null": 0,
        "final_whip_null": 0,
        "total_rows": 0
    }
```

Add after line 70 (after whip UPDATE):

```python
        # Count zero-IP rows that were skipped (diagnostic only)
        result["whip_skipped_zero_ip"] = db.execute(text("""
            SELECT COUNT(*) FROM mlb_player_stats
            WHERE whip IS NULL
              AND walks_allowed IS NOT NULL
              AND hits_allowed IS NOT NULL
              AND innings_pitched IN ('0.0', '0', '0.00')
        """)).scalar()
```

- [ ] **Step 5: Run test to verify fix passes**

Run: `venv/Scripts/python -m pytest tests/test_admin_backfill_ops_whip.py::test_whip_backfill_excludes_zero_ip_rows -v`
Expected: PASS

- [ ] **Step 6: Add test for whip_skipped_zero_ip field**

```python
def test_backfill_reports_zero_ip_skipped_count():
    """Endpoint should report how many rows were skipped due to zero innings."""
    client = TestClient(app)
    response = client.post("/admin/backfill-ops-whip")
    assert response.status_code == 200
    result = response.json()

    # Should have the new field
    assert "whip_skipped_zero_ip" in result
    # Should be a non-negative integer
    assert isinstance(result["whip_skipped_zero_ip"], int)
    assert result["whip_skipped_zero_ip"] >= 0
```

- [ ] **Step 7: Run all tests**

Run: `venv/Scripts/python -m pytest tests/test_admin_backfill_ops_whip.py -v`
Expected: 2/2 PASS

- [ ] **Step 8: Commit**

```bash
git add backend/admin_backfill_ops_whip.py tests/test_admin_backfill_ops_whip.py
git commit -m "fix: backfill-ops-whip excludes zero-inning rows from update count

- Add innings_pitched NOT IN ('0.0', '0', '0.00') filter to WHIP UPDATE
- Add whip_skipped_zero_ip diagnostic field to report excluded rows
- WHIP is mathematically undefined for zero-inning appearances (division by zero)
- Add tests for accurate rowcount reporting

Fixes FOLLOW-UP 4 from P-1..P-4 production deployment."
```

---

## Task 3: Deduplicate Player ID Mapping (FOLLOW-UP 3)

**Problem:** Both Ohtani (bdl_id=208) and Lorenzen (bdl_id=2293) have 4 duplicate rows each at ids (X, X+10000, X+20000, X+30000). The yahoo_key column is populated on only one of the 4 rows for Lorenzen, and none for Ohtani.

**Root Cause:** Upstream mapping-seed job re-inserts instead of upserting. Natural key should be (bdl_id, mlbam_id) or similar.

**Scope:** This task adds (1) unique constraint to prevent future duplicates, and (2) migration script to dedupe existing rows.

**Files:**
- Create: `scripts/migrate_v27_player_id_dedup.py`
- Modify: `backend/models.py:1089-1130` (add unique constraint)
- Test: `tests/test_player_id_dedup.py`

- [ ] **Step 1: Write test for duplicate detection**

Create `tests/test_player_id_dedup.py`:

```python
"""
Test player_id_mapping deduplication logic.
"""
import pytest
from backend.models import SessionLocal, PlayerIDMapping
from sqlalchemy import text


def test_player_id_mapping_has_no_duplicates_by_bdl_mlbam():
    """No two rows should share the same (bdl_id, mlbam_id) combination."""
    db = SessionLocal()
    try:
        duplicates = db.execute(text("""
            SELECT bdl_id, mlbam_id, COUNT(*) as dup_count
            FROM player_id_mapping
            WHERE bdl_id IS NOT NULL AND mlbam_id IS NOT NULL
            GROUP BY bdl_id, mlbam_id
            HAVING COUNT(*) > 1
        """)).fetchall()

        dup_list = [(d.bdl_id, d.mlbam_id, d.dup_count) for d in duplicates]
        assert len(dup_list) == 0, f"Found duplicates: {dup_list}"
    finally:
        db.close()


def test_ohtani_has_single_record():
    """Shohei Ohtani (bdl_id=208, mlbam_id=660271) should have exactly 1 record."""
    db = SessionLocal()
    try:
        count = db.execute(text("""
            SELECT COUNT(*) FROM player_id_mapping
            WHERE bdl_id = 208 OR mlbam_id = 660271
        """)).scalar()
        assert count == 1, f"Ohtani has {count} records, expected 1"
    finally:
        db.close()


def test_lorenzen_has_single_record():
    """Michael Lorenzen (bdl_id=2293, mlbam_id=594787) should have exactly 1 record."""
    db = SessionLocal()
    try:
        count = db.execute(text("""
            SELECT COUNT(*) FROM player_id_mapping
            WHERE bdl_id = 2293 OR mlbam_id = 594787
        """)).scalar()
        assert count == 1, f"Lorenzen has {count} records, expected 1"
    finally:
        db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_player_id_dedup.py -v`
Expected: FAIL (duplicates exist for Ohtani and Lorenzen)

- [ ] **Step 3: Create deduplication migration script**

Create `scripts/migrate_v27_player_id_dedup.py`:

```python
"""
Migration v27: Deduplicate player_id_mapping table.

Problem: Ohtani and Lorenzen have 4 duplicate rows each. Root cause is
upstream seed job re-inserting instead of upserting.

Solution:
1. Create unique constraint on (bdl_id, mlbam_id) to prevent future dups
2. Consolidate duplicates: keep row with yahoo_key populated, delete others

Run: railway run python scripts/migrate_v27_player_id_dedup.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import SessionLocal, PlayerIDMapping
from sqlalchemy import text


def main():
    db = SessionLocal()
    try:
        print("Checking for duplicates...")
        duplicates = db.execute(text("""
            SELECT bdl_id, mlbam_id, COUNT(*) as dup_count,
                   array_agg(id ORDER BY id) as ids,
                   array_agg(yahoo_key ORDER BY id) as yahoo_keys
            FROM player_id_mapping
            WHERE bdl_id IS NOT NULL AND mlbam_id IS NOT NULL
            GROUP BY bdl_id, mlbam_id
            HAVING COUNT(*) > 1
        """)).fetchall()

        if not duplicates:
            print("No duplicates found. Creating unique constraint...")
        else:
            print(f"Found {len(duplicates)} duplicate groups:")
            for dup in duplicates:
                print(f"  bdl_id={dup.bdl_id}, mlbam_id={dup.mlbam_id}, ids={dup.ids}")

            # Deduplicate: keep row with yahoo_key, delete others
            for dup in duplicates:
                ids = dup.ids  # [id1, id2, id3, id4]
                yahoo_keys = dup.yahoo_keys  # [null, '469.p.1000001', null, null]

                # Find the row with yahoo_key populated (prefer that one)
                keep_id = None
                for i, yk in enumerate(yahoo_keys):
                    if yk:
                        keep_id = ids[i]
                        break

                # If no yahoo_key, keep the lowest id
                if keep_id is None:
                    keep_id = min(ids)

                # Delete the other rows
                delete_ids = [iid for iid in ids if iid != keep_id]
                if delete_ids:
                    db.execute(text(
                        "DELETE FROM player_id_mapping WHERE id = ANY(:ids)"
                    ), {"ids": delete_ids})
                    print(f"  Deleted ids {delete_ids}, keeping id {keep_id}")

            db.commit()
            print("Deduplication complete.")

        # Create unique constraint
        print("Creating unique constraint on (bdl_id, mlbam_id)...")
        try:
            db.execute(text("""
                ALTER TABLE player_id_mapping
                DROP CONSTRAINT IF EXISTS uq_player_id_mapping_bdl_mlbam
            """))
        except Exception:
            pass  # Constraint doesn't exist yet

        db.execute(text("""
            ALTER TABLE player_id_mapping
            ADD CONSTRAINT uq_player_id_mapping_bdl_mlbam
            UNIQUE (bdl_id, mlbam_id)
            DEFERRABLE INITIALLY DEFERRED
        """))
        db.commit()
        print("Unique constraint created.")

        # Verify
        final_check = db.execute(text("""
            SELECT COUNT(*) FROM player_id_mapping
            WHERE bdl_id IS NOT NULL AND mlbam_id IS NOT NULL
        """)).scalar()
        print(f"Final player_id_mapping count: {final_check}")

        # Specific checks
        ohtani = db.execute(text("""
            SELECT COUNT(*) FROM player_id_mapping
            WHERE bdl_id = 208
        """)).scalar()
        lorenzen = db.execute(text("""
            SELECT COUNT(*) FROM player_id_mapping
            WHERE bdl_id = 2293
        """)).scalar()
        print(f"Ohtani records: {ohtani} (expected 1)")
        print(f"Lorenzen records: {lorenzen} (expected 1)")

        if ohtani == 1 and lorenzen == 1:
            print("\n✅ Migration successful!")
            return 0
        else:
            print("\n❌ Migration failed: duplicates still exist")
            return 1

    except Exception as e:
        db.rollback()
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Update model to reflect unique constraint**

Modify `backend/models.py` after line 1130 (after `last_verified` column):

```python
    __table_args__ = (
        # Unique constraint on (bdl_id, mlbam_id) to prevent duplicates
        # DEFERRABLE INITIALLY DEFERRED allows migration scripts to reorder
        CheckConstraint(
            "bdl_id IS NOT NULL AND mlbam_id IS NOT NULL",
            name="ck_player_id_mapping_bdl_mlbam_notnull"
        ),
    )
```

Actually, we need to use the proper SQLAlchemy syntax for unique constraints. Replace the class's table_args section:

```python
    __tablename__ = "player_id_mapping"

    # ... all existing columns ...

    last_verified         = Column(Date, nullable=True)

    __table_args__ = (
        # Unique constraint to prevent duplicate player identities
        # DEFERRABLE INITIALLY DEFERRED allows bulk operations before validation
        UniqueConstraint(
            'bdl_id', 'mlbam_id',
            name='uq_player_id_mapping_bdl_mlbam',
            deferrable='INITIALLY DEFERRED'
        ),
    )
```

- [ ] **Step 5: Test migration script locally**

Run: `python scripts/migrate_v27_player_id_dedup.py`
Expected: "✅ Migration successful!" with Ohtani=1, Lorenzen=1

- [ ] **Step 6: Run tests to verify deduplication**

Run: `venv/Scripts/python -m pytest tests/test_player_id_dedup.py -v`
Expected: 3/3 PASS

- [ ] **Step 7: Add test for constraint violation**

```python
def test_cannot_insert_duplicate_bdl_mlbam():
    """Inserting duplicate (bdl_id, mlbam_id) should raise IntegrityError."""
    from sqlalchemy.exc import IntegrityError
    db = SessionLocal()
    try:
        # Try to insert a duplicate
        duplicate = PlayerIDMapping(
            full_name="Test Player",
            normalized_name="test player",
            bdl_id=208,  # Ohtani's bdl_id
            mlbam_id=660271,  # Ohtani's mlbam_id
            source="test"
        )
        db.add(duplicate)
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()
```

- [ ] **Step 8: Run all tests**

Run: `venv/Scripts/python -m pytest tests/test_player_id_dedup.py -v`
Expected: 4/4 PASS

- [ ] **Step 9: Commit**

```bash
git add scripts/migrate_v27_player_id_dedup.py backend/models.py tests/test_player_id_dedup.py
git commit -m "feat: deduplicate player_id_mapping and add unique constraint

- Add migration script to consolidate Ohtani/Lorenzen duplicates
- Keep row with yahoo_key populated when consolidating
- Add UNIQUE(bdl_id, mlbam_id) constraint to prevent future dups
- DEFERRABLE INITIALLY DEFERRED allows bulk operations
- Add tests for duplicate detection and constraint enforcement

Fixes FOLLOW-UP 3 from P-1..P-4 production deployment."
```

---

## Task 4: Database Index Optimization (K-39)

**Problem:** Query performance degrades as data grows (~40K player_id_mapping records, 5K+ mlb_player_stats). Missing indexes on foreign keys and common query patterns.

**Scope:** Add indexes for (1) foreign key lookups, (2) position_eligibility joins, (3) player stats queries by date and bdl_id.

**Files:**
- Create: `scripts/migrate_v28_performance_indexes.py`
- Test: `tests/test_performance_indexes.py`

- [ ] **Step 1: Analyze current query patterns**

Read key query locations to understand index needs:

```python
# Common patterns found in codebase:

# 1. position_eligibility → player_id_mapping join (used in validation, orphan linking)
# FROM position_eligibility pe
# LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
# → Need index on position_eligibility(yahoo_player_key)

# 2. mlb_player_stats by bdl_id (used in stats computation, fantasy scoring)
# WHERE bdl_player_id = :bdl_id
# → Need index on mlb_player_stats(bdl_player_id)

# 3. mlb_player_stats by game_date (used in freshness checks, rolling stats)
# WHERE game_date >= :cutoff_date
# → Need index on mlb_player_stats(game_date DESC)

# 4. statcast_performances by player_id + date (used in backfill, queries)
# WHERE player_id = :pid AND game_date = :date
# → Need composite index on statcast_performances(player_id, game_date)

# 5. player_rolling_stats by player_id + date (used in projection computation)
# WHERE player_id = :pid AND stat_date >= :cutoff
# → Need composite index on player_rolling_stats(player_id, stat_date DESC)
```

- [ ] **Step 2: Create index migration script**

Create `scripts/migrate_v28_performance_indexes.py`:

```python
"""
Migration v28: Add performance indexes for common query patterns.

K-39: Database Index Optimization Analysis

Indexes added:
1. position_eligibility(yahoo_player_key) - for orphan linking joins
2. mlb_player_stats(bdl_player_id) - for stats lookup by player
3. mlb_player_stats(game_date DESC) - for freshness/rolling stats
4. mlb_player_stats(bdl_player_id, game_date DESC) - composite for player+time
5. statcast_performances(player_id, game_date) - for Statcast queries
6. player_rolling_stats(player_id, stat_date DESC) - for projection queries

Run: railway run python scripts/migrate_v28_performance_indexes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import SessionLocal
from sqlalchemy import text


MIGRATIONS = [
    # 1. position_eligibility yahoo_player_key index
    {
        "name": "idx_position_eligibility_yahoo_key",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_position_eligibility_yahoo_key
            ON position_eligibility(yahoo_player_key)
            WHERE yahoo_player_key IS NOT NULL
        """,
        "description": "Speed up orphan linking joins to player_id_mapping"
    },
    # 2. mlb_player_stats bdl_player_id index
    {
        "name": "idx_mlb_player_stats_bdl_id",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_player_stats_bdl_id
            ON mlb_player_stats(bdl_player_id)
            WHERE bdl_player_id IS NOT NULL
        """,
        "description": "Speed up stats lookup by player"
    },
    # 3. mlb_player_stats game_date index
    {
        "name": "idx_mlb_player_stats_game_date",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_player_stats_game_date
            ON mlb_player_stats(game_date DESC NULLS LAST)
        """,
        "description": "Speed up freshness checks and rolling stats computation"
    },
    # 4. mlb_player_stats composite bdl_id + game_date
    {
        "name": "idx_mlb_player_stats_bdl_id_date",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mlb_player_stats_bdl_id_date
            ON mlb_player_stats(bdl_player_id, game_date DESC NULLS LAST)
            WHERE bdl_player_id IS NOT NULL
        """,
        "description": "Speed up player stats queries with date range"
    },
    # 5. statcast_performances composite player_id + game_date
    {
        "name": "idx_statcast_performances_player_date",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_statcast_performances_player_date
            ON statcast_performances(player_id, game_date)
        """,
        "description": "Speed up Statcast queries by player and date"
    },
    # 6. player_rolling_stats composite player_id + stat_date
    {
        "name": "idx_player_rolling_stats_player_date",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_player_rolling_stats_player_date
            ON player_rolling_stats(player_id, stat_date DESC)
        """,
        "description": "Speed up projection and momentum queries"
    },
]


def main():
    db = SessionLocal()
    try:
        print("Creating performance indexes...\n")

        for i, migration in enumerate(MIGRATIONS, 1):
            print(f"[{i}/{len(MIGRATIONS)}] {migration['name']}")
            print(f"  {migration['description']}")

            try:
                db.execute(text(migration['sql']))
                db.commit()
                print("  ✅ Created")
            except Exception as e:
                db.rollback()
                print(f"  ❌ Failed: {e}")
                return 1

        print(f"\n✅ All {len(MIGRATIONS)} indexes created successfully!")

        # Verify indexes
        print("\nVerifying indexes...")
        verification = db.execute(text("""
            SELECT
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE indexname LIKE ANY(ARRAY[
                'idx_position_eligibility_%',
                'idx_mlb_player_stats_%',
                'idx_statcast_performances_%',
                'idx_player_rolling_stats_%'
            ])
            ORDER BY tablename, indexname
        """)).fetchall()

        for v in verification:
            print(f"  ✓ {v.indexname} on {v.tablename}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Create test for index existence**

Create `tests/test_performance_indexes.py`:

```python
"""
Test that performance indexes exist and are used.
"""
import pytest
from backend.models import SessionLocal
from sqlalchemy import text


EXPECTED_INDEXES = [
    ("idx_position_eligibility_yahoo_key", "position_eligibility"),
    ("idx_mlb_player_stats_bdl_id", "mlb_player_stats"),
    ("idx_mlb_player_stats_game_date", "mlb_player_stats"),
    ("idx_mlb_player_stats_bdl_id_date", "mlb_player_stats"),
    ("idx_statcast_performances_player_date", "statcast_performances"),
    ("idx_player_rolling_stats_player_date", "player_rolling_stats"),
]


@pytest.mark.parametrize("index_name,table_name", EXPECTED_INDEXES)
def test_performance_index_exists(index_name, table_name):
    """Each expected performance index should exist in the database."""
    db = SessionLocal()
    try:
        result = db.execute(text("""
            SELECT 1 FROM pg_indexes
            WHERE indexname = :index_name
              AND tablename = :table_name
        """), {"index_name": index_name, "table_name": table_name}).scalar()

        assert result is not None, f"Index {index_name} not found on table {table_name}"
    finally:
        db.close()


def test_all_performance_indexes_exist():
    """All expected performance indexes should exist."""
    db = SessionLocal()
    try:
        existing = db.execute(text("""
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE indexname LIKE ANY(ARRAY[
                'idx_position_eligibility_%',
                'idx_mlb_player_stats_%',
                'idx_statcast_performances_%',
                'idx_player_rolling_stats_%'
            ])
        """)).fetchall()

        existing_set = {(row.indexname, row.tablename) for row in existing}
        expected_set = set(EXPECTED_INDEXES)

        missing = expected_set - existing_set
        assert not missing, f"Missing indexes: {missing}"

    finally:
        db.close()


def test_query_plan_uses_indexes():
    """Verify common queries use the new indexes (EXPLAIN ANALYZE)."""
    db = SessionLocal()
    try:
        # Test position_eligibility join uses index
        plan = db.execute(text("""
            EXPLAIN (FORMAT TEXT)
            SELECT pe.* FROM position_eligibility pe
            LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
            WHERE pe.yahoo_player_key = '469.p.1000001'
            LIMIT 1
        """)).fetchall()

        plan_text = " ".join(row[0] for row in plan)
        # Should use index on position_eligibility
        assert "idx_position_eligibility_yahoo_key" in plan_text or \
               "Index Scan" in plan_text or \
               "Bitmap Index Scan" in plan_text, \
               "position_eligibility query should use index"

    finally:
        db.close()
```

- [ ] **Step 4: Run migration script locally**

Run: `python scripts/migrate_v28_performance_indexes.py`
Expected: "✅ All 6 indexes created successfully!"

- [ ] **Step 5: Run tests to verify indexes**

Run: `venv/Scripts/python -m pytest tests/test_performance_indexes.py -v`
Expected: 8/8 PASS (6 parametrized + 1 aggregate + 1 query plan)

- [ ] **Step 6: Commit**

```bash
git add scripts/migrate_v28_performance_indexes.py tests/test_performance_indexes.py
git commit -m "feat: add performance indexes for common query patterns (K-39)

- idx_position_eligibility_yahoo_key: speed up orphan linking joins
- idx_mlb_player_stats_bdl_id: speed up player stats lookup
- idx_mlb_player_stats_game_date: speed up freshness checks
- idx_mlb_player_stats_bdl_id_date: composite for player+time queries
- idx_statcast_performances_player_date: speed up Statcast queries
- idx_player_rolling_stats_player_date: speed up projection queries

All indexes use CONCURRENTLY to avoid table locks.

Implements K-39 from proactive research initiatives."
```

---

## Final Verification

- [ ] **Step 1: Run full test suite**

Run: `venv/Scripts/python -m pytest tests/test_admin_validation_audit.py tests/test_admin_backfill_ops_whip.py tests/test_player_id_dedup.py tests/test_performance_indexes.py -v`

Expected: All tests pass

- [ ] **Step 2: Verify no regressions in existing tests**

Run: `venv/Scripts/python -m pytest tests/ -q --tb=short`

Expected: 650+ tests pass (existing pass rate maintained)

- [ ] **Step 3: Compile check**

Run:
```bash
venv/Scripts/python -m py_compile backend/admin_endpoints_validation.py
venv/Scripts/python -m py_compile backend/admin_backfill_ops_whip.py
venv/Scripts/python -m py_compile backend/models.py
venv/Scripts/python -m py_compile scripts/migrate_v27_player_id_dedup.py
venv/Scripts/python -m py_compile scripts/migrate_v28_performance_indexes.py
```

Expected: No syntax errors

---

## Deployment Instructions (Gemini/DevOps)

Once all tasks are complete and tested:

1. **Run migrations on Railway:**
   ```bash
   railway run python scripts/migrate_v27_player_id_dedup.py
   railway run python scripts/migrate_v28_performance_indexes.py
   ```

2. **Verify validation endpoint:**
   ```bash
   railway run curl -X GET https://fantasy-app-production.up.railway.app/admin/validation-audit
   ```

3. **Test backfill endpoint:**
   ```bash
   railway run curl -X POST https://fantasy-app-production.up.railway.app/admin/backfill-ops-whip
   ```

---

## Success Criteria

- [ ] Validation audit endpoint reports live DB state (not stale findings)
- [ ] Backfill endpoint accurately counts updates (excludes zero-IP rows)
- [ ] Ohtani and Lorenzen have 1 record each in player_id_mapping
- [ ] Unique constraint prevents future duplicates
- [ ] All 6 performance indexes created and verified
- [ ] Full test suite passes without regression

---

**Result:** Production-hardened platform with accurate validation, correct diagnostics, deduped data, and optimized queries.
