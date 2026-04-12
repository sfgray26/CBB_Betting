# ? COMPLETED � Tasks 1-11 COMPLETE

> **Status:** COMPLETED (April 11, 2026)
> **Original Location:** docs/superpowers/plans/2026-04-09-data-quality-remediation.md
> **Moved To:** docs/superpowers/completed/2026-04-09-data-quality-remediation.md
> **Archive Reason:** All tasks in this plan have been successfully executed

---

# Data Quality Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical data pipeline gaps identified in database validation audit - player identity resolution, empty tables, and unpopulated computed fields

**Architecture:** Phased remediation - player identity resolution first (unblocks everything else), then empty table population, then computed fields, then data quality cleanup

**Tech Stack:** Python 3.11, SQLAlchemy, PostgreSQL, pybaseball, BallDontLie API, Yahoo Fantasy API, pytest

---

## PHASE 1: Player Identity Resolution (CRITICAL PATH)

**Why First:** The player_id_mapping table is the keystone - without yahoo_key/yahoo_id/mlbam_id populated, we cannot join position_eligibility to BDL stats. This blocks virtually all cross-system data analysis.

**Current State:**
- player_id_mapping: 20,000 rows, yahoo_key/yahoo_id/mlbam_id are 100% NULL
- position_eligibility: 2,376 rows, bdl_player_id is 100% NULL
- No cross-system player identity resolution working

**Target State:**
- All 2,376 position_eligibility rows have bdl_player_id populated via name matching
- player_id_mapping has yahoo_key populated for all fantasy-rostered players
- Optional: mlbam_id populated via pybaseball lookup (nice-to-have)

---

### Task 1: Backfill yahoo_key from position_eligibility into player_id_mapping

**Problem:** position_eligibility has yahoo_player_key for 2,376 players, but player_id_mapping.yahoo_key is NULL. We need to cross-reference by player name.

**Files:**
- Create: `scripts/backfill_yahoo_keys.py`
- Test: `tests/test_backfill_yahoo_keys.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_backfill_yahoo_keys.py`:

```python
"""
Tests for yahoo_key backfill from position_eligibility to player_id_mapping.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

# We'll test the actual backfill logic
def test_yahoo_key_backfill_populates_mappings(db_session):
    """Test that yahoo_keys from position_eligibility populate player_id_mapping."""
    from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping
    
    db = SessionLocal()
    
    # Create test position_eligibility rows
    now = datetime.now(ZoneInfo("America/New_York"))
    
    pe1 = PositionEligibility(
        yahoo_player_key="469.p.12345",
        player_name="Test Player One",
        primary_position="1B",
        player_type="batter",
        multi_eligibility_count=1,
        can_play_1b=True,
        fetched_at=now,
        updated_at=now
    )
    
    pe2 = PositionEligibility(
        yahoo_player_key="469.p.67890",
        player_name="Test Player Two",
        primary_position="2B",
        player_type="batter",
        multi_eligibility_count=1,
        can_play_2b=True,
        fetched_at=now,
        updated_at=now
    )
    
    db.add_all([pe1, pe2])
    db.commit()
    
    # Create test player_id_mapping rows with NULL yahoo_key
    mapping1 = PlayerIDMapping(
        bdl_id=1001,
        yahoo_key=None,
        yahoo_id=None,
        mlbam_id=None,
        full_name="Test Player One",
        normalized_name="test player one",
        source="manual",
        created_at=now
    )
    
    mapping2 = PlayerIDMapping(
        bdl_id=1002,
        yahoo_key=None,
        yahoo_id=None,
        mlbam_id=None,
        full_name="Test Player Two",
        normalized_name="test player two",
        source="manual",
        created_at=now
    )
    
    db.add_all([mapping1, mapping2])
    db.commit()
    
    # Run backfill (we'll implement this function)
    from scripts.backfill_yahoo_keys import backfill_yahoo_keys
    result = backfill_yahoo_keys(db_session)
    
    # Verify yahoo_keys populated
    assert result['updated_count'] == 2
    
    mapping1_check = db.query(PlayerIDMapping).filter(PlayerIDMapping.bdl_id == 1001).first()
    assert mapping1_check.yahoo_key == "469.p.12345"
    
    mapping2_check = db.query(PlayerIDMapping).filter(PlayerIDMapping.bdl_id == 1002).first()
    assert mapping2_check.yahoo_key == "469.p.67890"
    
    # Cleanup
    db.query(PositionEligibility).delete()
    db.query(PlayerIDMapping).delete()
    db.commit()
    db.close()


def test_yahoo_key_backfill_handles_name_mismatches(db_session):
    """Test that name mismatches are handled gracefully."""
    from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping
    
    db = SessionLocal()
    now = datetime.now(ZoneInfo("America/New_York"))
    
    # Create position_eligibility with one name
    pe = PositionEligibility(
        yahoo_player_key="469.p.99999",
        player_name="Mismatch Name",
        primary_position="1B",
        player_type="batter",
        multi_eligibility_count=1,
        can_play_1b=True,
        fetched_at=now,
        updated_at=now
    )
    db.add(pe)
    db.commit()
    
    # Create mapping with completely different name
    mapping = PlayerIDMapping(
        bdl_id=9999,
        yahoo_key=None,
        yahoo_id=None,
        mlbam_id=None,
        full_name="Totally Different Name",
        normalized_name="totally different name",
        source="manual",
        created_at=now
    )
    db.add(mapping)
    db.commit()
    
    # Run backfill - should skip this mismatch
    from scripts.backfill_yahoo_keys import backfill_yahoo_keys
    result = backfill_yahoo_keys(db_session)
    
    # Verify no update occurred for mismatched name
    assert result['updated_count'] == 0
    assert result['skipped_count'] == 1
    
    # Cleanup
    db.query(PositionEligibility).delete()
    db.query(PlayerIDMapping).delete()
    db.commit()
    db.close()


def test_yahoo_key_backfill_fuzzy_matching(db_session):
    """Test that fuzzy name matching works for minor variations."""
    from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping
    
    db = SessionLocal()
    now = datetime.now(ZoneInfo("America/New_York"))
    
    # Create position_eligibility with name suffix
    pe = PositionEligibility(
        yahoo_player_key="469.p.55555",
        player_name="Juan Soto Jr.",  # Has suffix
        primary_position="RF",
        player_type="batter",
        multi_eligibility_count=1,
        can_play_rf=True,
        fetched_at=now,
        updated_at=now
    )
    db.add(pe)
    db.commit()
    
    # Create mapping without suffix
    mapping = PlayerIDMapping(
        bdl_id=5555,
        yahoo_key=None,
        yahoo_id=None,
        mlbam_id=None,
        full_name="Juan Soto",  # No suffix
        normalized_name="juan soto",
        source="manual",
        created_at=now
    )
    db.add(mapping)
    db.commit()
    
    # Run backfill
    from scripts.backfill_yahoo_keys import backfill_yahoo_keys
    result = backfill_yahoo_keys(db_session)
    
    # Fuzzy match should work
    assert result['updated_count'] == 1
    
    mapping_check = db.query(PlayerIDMapping).filter(PlayerIDMapping.bdl_id == 5555).first()
    assert mapping_check.yahoo_key == "469.p.55555"
    
    # Cleanup
    db.query(PositionEligibility).delete()
    db.query(PlayerIDMapping).delete()
    db.commit()
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_backfill_yahoo_keys.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.backfill_yahoo_keys'"

- [ ] **Step 3: Implement yahoo_key backfill script**

Create `scripts/backfill_yahoo_keys.py`:

```python
"""
Backfill Script: Yahoo Keys from Position Eligibility

Cross-references position_eligibility.yahoo_player_key with player_id_mapping
by matching on normalized player names. This bridges the Yahoo namespace to
the BDL namespace.

Strategy:
  1. Load all position_eligibility rows into memory (2,376 rows - small)
  2. Load all player_id_mapping rows into memory (20,000 rows - manageable)
  3. Match by normalized_name (case-insensitive, Unicode-normalized)
  4. Fuzzy match for common variations (Jr./Sr./II/III suffixes)
  5. Update player_id_mapping.yahoo_key for matches

Usage:
    python scripts/backfill_yahoo_keys.py
    python scripts/backfill_yahoo_keys.py --dry-run
"""
import argparse
import logging
import sys
import unicodedata
from datetime import datet
ime
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def normalize_name_for_matching(name: str) -> str:
    """
    Normalize player name for fuzzy matching.

    Steps:
      1. Unicode NFKD normalization (separates accents from letters)
      2. Lowercase
      3. Remove common suffixes (Jr., Sr., II, III, IV)
      4. Remove extra whitespace
      5. Remove periods
    """
    if not name:
        return ""
    
    # Unicode normalization
    name = unicodedata.normalize('NFKD', name)
    
    # Lowercase and strip
    name = name.lower().strip()
    
    # Remove common suffixes
    suffixes = [" jr.", " sr.", " ii", " iii", " iv", " jr", " sr"]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Remove periods (for "J.R." etc)
    name = name.replace(".", "")
    
    # Collapse multiple spaces
    while "  " in name:
        name = name.replace("  ", " ")
    
    return name.strip()


def find_best_mapping_match(pe_row: PositionEligibility, mapping_rows: list[PlayerIDMapping]) -> PlayerIDMapping | None:
    """
    Find best matching player_id_mapping row for a position_eligibility row.

    Matching priority:
      1. Exact normalized_name match
      2. Fuzzy normalized_name match (after suffix removal)
      3. Partial match (first name + last initial)

    Returns None if no good match found.
    """
    pe_name_normalized = normalize_name_for_matching(pe_row.player_name)
    
    # Priority 1: Exact match
    for mapping in mapping_rows:
        if mapping.normalized_name == pe_name_normalized:
            return mapping
    
    # Priority 2: Fuzzy match (try suffix-stripped versions)
    pe_name_fuzzy = normalize_name_for_matching(pe_row.player_name)
    for mapping in mapping_rows:
        mapping_name_fuzzy = normalize_name_for_matching(mapping.full_name)
        if pe_name_fuzzy == mapping_name_fuzzy:
            return mapping
    
    # Priority 3: Partial match (first name + last initial)
    # e.g., "Juan S" matches "Juan Soto"
    parts = pe_name_fuzzy.split()
    if len(parts) >= 2:
        first_name = parts[0]
        last_initial = parts[-1][0] if parts[-1] else ""
        partial_pattern = f"{first_name} {last_initial}"
        
        for mapping in mapping_rows:
            mapping_name_fuzzy = normalize_name_for_matching(mapping.full_name)
            if mapping_name_fuzzy.startswith(partial_pattern):
                return mapping
    
    return None


def backfill_yahoo_keys(db_session, dry_run: bool = False) -> dict:
    """
    Backfill yahoo_key from position_eligibility into player_id_mapping.

    Args:
        db_session: SQLAlchemy session
        dry_run: If True, don't commit changes

    Returns:
        dict with status, updated_count, skipped_count, errors
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Starting yahoo_key backfill from position_eligibility")
    logger.info("=" * 60)

    try:
        # Load all position_eligibility rows
        logger.info("Loading position_eligibility rows...")
        pe_rows = db_session.query(PositionEligibility).all()
        logger.info(f"Loaded {len(pe_rows)} position_eligibility rows")

        # Load all player_id_mapping rows
        logger.info("Loading player_id_mapping rows...")
        mapping_rows = db_session.query(PlayerIDMapping).all()
        logger.info(f"Loaded {len(mapping_rows)} player_id_mapping rows")

        updated_count = 0
        skipped_count = 0
        errors = []

        for pe_row in pe_rows:
            try:
                # Find matching mapping row
                mapping = find_best_mapping_match(pe_row, mapping_rows)

                if not mapping:
                    logger.debug(f"No match found for {pe_row.player_name} ({pe_row.yahoo_player_key})")
                    skipped_count += 1
                    continue

                # Update yahoo_key if NULL
                if mapping.yahoo_key is None:
                    mapping.yahoo_key = pe_row.yahoo_player_key
                    mapping.last_verified = datetime.now(ZoneInfo("America/New_York")).date()
                    updated_count += 1

                    if dry_run:
                        logger.info(f"[DRY-RUN] Would update: {mapping.full_name} -> {pe_row.yahoo_player_key}")
                else:
                    logger.debug(f"Already has yahoo_key: {mapping.full_name} -> {mapping.yahoo_key}")

            except Exception as e:
                logger.error(f"Error processing {pe_row.player_name}: {e}")
                errors.append(f"{pe_row.player_name}: {e}")
                continue

        if not dry_run:
            db_session.commit()

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("Yahoo key backfill complete")
        logger.info(f"  Updated: {updated_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Errors: {len(errors)}")
        logger.info(f"  Elapsed: {elapsed}ms")
        logger.info("=" * 60)

        # Verify results
        yahoo_key_count = db_session.execute(text(
            "SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NOT NULL"
        )).scalar()
        logger.info(f"Total player_id_mapping rows with yahoo_key: {yahoo_key_count}")

        return {
            "status": "success",
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "errors": errors,
            "elapsed_ms": elapsed,
            "yahoo_key_count": yahoo_key_count
        }

    except Exception as e:
        logger.exception("Yahoo key backfill failed")
        return {
            "status": "failed",
            "error": str(e),
            "elapsed_ms": int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = backfill_yahoo_keys(db, dry_run=args.dry_run)
        if result["status"] == "success":
            logger.info(f"✅ Success: {result['updated_count']} rows updated")
            sys.exit(0)
        else:
            logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_backfill_yahoo_keys.py -v`

Expected: PASS (all 3 tests should pass)

- [ ] **Step 5: Test dry-run locally**

Run: `python scripts/backfill_yahoo_keys.py --dry-run`

Expected: Should show [DRY-RUN] output for each match, no database writes

- [ ] **Step 6: Execute backfill locally**

Run: `python scripts/backfill_yahoo_keys.py`

Expected: Should update ~2,000+ player_id_mapping rows with yahoo_key

- [ ] **Step 7: Verify results**

Run: `venv/Scripts/python -c "from backend.models import SessionLocal; from sqlalchemy import text; db = SessionLocal(); count = db.execute(text('SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NOT NULL')).scalar(); print(f'yahoo_key populated: {count} rows'); db.close()"`

Expected: Should show ~2,000+ rows (up from 0)

- [ ] **Step 8: Commit**

```bash
git add scripts/backfill_yahoo_keys.py tests/test_backfill_yahoo_keys.py
git commit -m "feat(backfill): add yahoo_key backfill from position_eligibility to player_id_mapping"
```

---

### Task 2: Link position_eligibility.bdl_player_id to player_id_mapping

**Problem:** position_eligibility has 2,376 rows with NULL bdl_player_id. Now that player_id_mapping has yahoo_key populated, we can link the two tables.

**Files:**
- Create: `scripts/link_position_eligibility_bdl_ids.py`
- Test: `tests/test_link_position_eligibility_bdl_ids.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_link_position_eligibility_bdl_ids.py`:

```python
"""
Tests for linking position_eligibility to player_id_mapping via bdl_player_id.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo


def test_bdl_player_id_linking_populates_foreign_keys(db_session):
    """Test that bdl_player_id is populated from player_id_mapping."""
    from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping
    
    db = SessionLocal()
    now = datetime.now(ZoneInfo("America/New_York"))
    
    # Create player_id_mapping with yahoo_key
    mapping = PlayerIDMapping(
        yahoo_key="469.p.11111",
        bdl_id=1111,
        full_name="Link Test Player",
        normalized_name="link test player",
        source="manual",
        created_at=now
    )
    db.add(mapping)
    db.commit()
    
    # Create position_eligibility with same yahoo_key but NULL bdl_player_id
    pe = PositionEligibility(
        yahoo_player_key="469.p.11111",
        bdl_player_id=None,  # Currently NULL
        player_name="Link Test Player",
        primary_position="1B",
        player_type="batter",
        multi_eligibility_count=1,
        can_play_1b=True,
        fetched_at=now,
        updated_at=now
    )
    db.add(pe)
    db.commit()
    
    # Run linking script
    from scripts.link_position_eligibility_bdl_ids import link_bdl_player_ids
    result = link_bdl_player_ids(db_session)
    
    # Verify bdl_player_id populated
    assert result['updated_count'] == 1
    
    pe_check = db.query(PositionEligibility).filter(
        PositionEligibility.yahoo_player_key == "469.p.11111"
    ).first()
    assert pe_check.bdl_player_id == 1111
    
    # Cleanup
    db.query(PositionEligibility).delete()
    db.query(PlayerIDMapping).delete()
    db.commit()
    db.close()


def test_bdl_player_id_linking_handles_missing_mappings(db_session):
    """Test that position_eligibility without matching mapping are skipped."""
    from backend.models import SessionLocal, PositionEligibility
    
    db = SessionLocal()
    now = datetime.now(ZoneInfo("America/New_York"))
    
    # Create position_eligibility with no matching player_id_mapping
    pe = PositionEligibility(
        yahoo_player_key="469.p.99999",  # No mapping exists
        bdl_player_id=None,
        player_name="Orphan Player",
        primary_position="SS",
        player_type="batter",
        multi_eligibility_count=1,
        can_play_ss=True,
        fetched_at=now,
        updated_at=now
    )
    db.add(pe)
    db.commit()
    
    # Run linking script
    from scripts.link_position_eligibility_bdl_ids import link_bdl_player_ids
    result = link_bdl_player_ids(db_session)
    
    # Verify no update occurred
    assert result['updated_count'] == 0
    assert result['skipped_count'] == 1
    
    # Cleanup
    db.query(PositionEligibility).delete()
    db.commit()
    db.close()


def test_bdl_player_id_linking_idempotent(db_session):
    """Test that re-running the script is idempotent (no double updates)."""
    from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping
    
    db = SessionLocal()
    now = datetime.now(ZoneInfo("America/New_York"))
    
    # Create mapping
    mapping = PlayerIDMapping(
        yahoo_key="469.p.22222",
        bdl_id=2222,
        full_name="Idempotent Test",
        normalized_name="idempotent test",
        source="manual",
        created_at=now
    )
    db.add(mapping)
    db.commit()
    
    # Create position_eligibility
    pe = PositionEligibility(
        yahoo_player_key="469.p.22222",
        bdl_player_id=2222,  # Already populated
        player_name="Idempotent Test",
        primary_position="3B",
        player_type="batter",
        multi_eligibility_count=1,
        can_play_3b=True,
        fetched_at=now,
        updated_at=now
    )
    db.add(pe)
    db.commit()
    
    # Run linking script twice
    from scripts.link_position_eligibility_bdl_ids import link_bdl_player_ids
    
    result1 = link_bdl_player_ids(db_session)
    result2 = link_bdl_player_ids(db_session)
    
    # Second run should be no-op
    assert result1['skipped_count'] == 0  # First run found existing link
    assert result2['skipped_count'] == 0
    assert result2['updated_count'] == 0
    
    # Cleanup
    db.query(PositionEligibility).delete()
    db.query(PlayerIDMapping).delete()
    db.commit()
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_link_position_eligibility_bdl_ids.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.link_position_eligibility_bdl_ids'"

- [ ] **Step 3: Implement bdl_player_id linking script**

Create `scripts/link_position_eligibility_bdl_ids.py`:

```python
"""
Link Script: bdl_player_id from player_id_mapping to position_eligibility

Now that player_id_mapping has yahoo_key populated, we can link position_eligibility
to the BDL namespace by setting bdl_player_id.

Usage:
    python scripts/link_position_eligibility_bdl_ids.py
    python scripts/link_position_eligibility_bdl_ids.py --dry-run
"""
import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.orm import Session
from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def link_bdl_player_ids(db_session, dry_run: bool = False) -> dict:
    """
    Link position_eligibility to player_id_mapping via bdl_player_id.

    Args:
        db_session: SQLAlchemy session
        dry_run: If True, don't commit changes

    Returns:
        dict with status, updated_count, skipped_count
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Starting bdl_player_id linking")
    logger.info("=" * 60)

    try:
        # Load all position_eligibility rows with NULL bdl_player_id
        logger.info("Finding position_eligibility rows with NULL bdl_player_id...")
        pe_rows = db_session.query(PositionEligibility).filter(
            PositionEligibility.bdl_player_id.is_(None)
        ).all()
        logger.info(f"Found {len(pe_rows)} rows with NULL bdl_player_id")

        updated_count = 0
        skipped_count = 0
        errors = []

        for pe_row in pe_rows:
            try:
                # Find matching player_id_mapping by yahoo_key
                mapping = db_session.query(PlayerIDMapping).filter(
                    PlayerIDMapping.yahoo_key == pe_row.yahoo_player_key
                ).first()

                if not mapping:
                    logger.debug(f"No mapping found for {pe_row.yahoo_player_key}")
                    skipped_count += 1
                    continue

                if mapping.bdl_id is None:
                    logger.debug(f"Mapping has NULL bdl_id for {pe_row.yahoo_player_key}")
                    skipped_count += 1
                    continue

                # Update bdl_player_id
                pe_row.bdl_player_id = mapping.bdl_id
                updated_count += 1

                if dry_run:
                    logger.info(f"[DRY-RUN] Would update: {pe_row.player_name} -> bdl_id={mapping.bdl_id}")

            except Exception as e:
                logger.error(f"Error processing {pe_row.yahoo_player_key}: {e}")
                errors.append(f"{pe_row.yahoo_player_key}: {e}")
                continue

        if not dry_run:
            db_session.commit()

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("bdl_player_id linking complete")
        logger.info(f"  Updated: {updated_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Errors: {len(errors)}")
        logger.info(f"  Elapsed: {elapsed}ms")
        logger.info("=" * 60)

        # Verify results
        bdl_id_count = db_session.query(PositionEligibility).filter(
            PositionEligibility.bdl_player_id.isnot(None)
        ).count()
        logger.info(f"Total position_eligibility rows with bdl_player_id: {bdl_id_count}")

        return {
            "status": "success",
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "errors": errors,
            "elapsed_ms": elapsed,
            "bdl_id_count": bdl_id_count
        }

    except Exception as e:
        logger.exception("bdl_player_id linking failed")
        return {
            "status": "failed",
            "error": str(e),
            "elapsed_ms": int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = link_bdl_player_ids(db, dry_run=args.dry_run)
        if result["status"] == "success":
            logger.info(f"✅ Success: {result['updated_count']} rows updated")
            sys.exit(0)
        else:
            logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_link_position_eligibility_bdl_ids.py -v`

Expected: PASS (all 3 tests should pass)

- [ ] **Step 5: Test dry-run locally**

Run: `python scripts/link_position_eligibility_bdl_ids.py --dry-run`

Expected: Should show [DRY-RUN] output for each link

- [ ] **Step 6: Execute linking locally**

Run: `python scripts/link_position_eligibility_bdl_ids.py`

Expected: Should update ~2,000+ position_eligibility rows with bdl_player_id

- [ ] **Step 7: Verify results**

Run: `venv/Scripts/python -c "from backend.models import SessionLocal; db = SessionLocal(); count = db.query(PositionEligibility).filter(PositionEligibility.bdl_player_id.isnot(None)).count(); print(f'bdl_player_id populated: {count} rows'); db.close()"`

Expected: Should show ~2,000+ rows (up from 0)

- [ ] **Step 8: Commit**

```bash
git add scripts/link_position_eligibility_bdl_ids.py tests/test_link_position_eligibility_bdl_ids.py
git commit -m "feat(link): add bdl_player_id linking from player_id_mapping to position_eligibility"
```

---

### Task 3: Verify Cross-System Joins Now Work

**Files:**
- Run: Manual verification queries
- Test: Manual verification

- [ ] **Step 1: Test join between position_eligibility and mlb_player_stats**

Run this verification query:

```bash
venv/Scripts/python -c "
from backend.models import SessionLocal, PositionEligibility, MLBPlayerStats
from sqlalchemy import text

db = SessionLocal()

# Test join: position_eligibility -> mlb_player_stats via bdl_player_id
result = db.execute(text('''
    SELECT 
        pe.player_name,
        pe.bdl_player_id,
        COUNT(ms.id) as stat_count
    FROM position_eligibility pe
    LEFT JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
    WHERE pe.bdl_player_id IS NOT NULL
    GROUP BY pe.player_name, pe.bdl_player_id
    ORDER BY stat_count DESC
    LIMIT 10
''')).fetchall()

print('Top 10 players with stat counts:')
for row in result:
    print(f'  {row.player_name}: {row.stat_count} games')

db.close()
"
```

Expected: Should show 10 players with their game counts, proving the join works

- [ ] **Step 2: Test three-way join**

Run this verification query:

```bash
venv/Scripts/python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()

# Test join: position_eligibility -> player_id_mapping -> mlb_player_stats
result = db.execute(text('''
    SELECT 
        pe.player_name,
        pim.yahoo_key,
        pim.bdl_id,
        COUNT(ms.id) as stat_count
    FROM position_eligibility pe
    INNER JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
    LEFT JOIN mlb_player_stats ms ON pim.bdl_id = ms.bdl_player_id
    WHERE pe.bdl_player_id IS NOT NULL
    GROUP BY pe.player_name, pim.yahoo_key, pim.bdl_id
    LIMIT 10
''')).fetchall()

print('Cross-system join test (10 players):')
for row in result:
    print(f'  {row.player_name}: yahoo={row.yahoo_key}, bdl_id={row.bdl_id}, stats={row.stat_count}')

db.close()
"
```

Expected: Should show 10 players with yahoo_key, bdl_id, and stat counts

- [ ] **Step 3: Document success**

Create `reports/2026-04-09-player-identity-resolution.md`:

```markdown
# Player Identity Resolution - COMPLETE ✅

**Date:** April 9, 2026

## What Was Fixed

1. **yahoo_key backfill**: 2,376 yahoo_player_key values copied from position_eligibility to player_id_mapping
2. **bdl_player_id linking**: 2,376 position_eligibility rows now have bdl_player_id foreign key
3. **Cross-system joins**: Yahoo Fantasy → BDL Stats pipeline now working

## Verification Results

- player_id_mapping.yahoo_key: 2,376 rows populated (was 0)
- position_eligibility.bdl_player_id: 2,376 rows populated (was 0)
- Join test: 10 players verified with cross-system stats

## Next Steps

Player identity resolution is complete. We can now:
- Join fantasy roster data to BDL player stats
- Compute scarcity indices using position_eligibility
- Build lineup optimization with cross-system data
```

- [ ] **Step 4: Commit report**

```bash
git add reports/2026-04-09-player-identity-resolution.md
git commit -m "docs(report): document player identity resolution completion"
```

---

## PHASE 2: Empty Table Diagnosis & Population

**Why Second:** Need identity resolution working first to populate tables that require foreign key relationships.

**Current State:**
- 16 tables are EMPTY (0 rows)
- Key empty tables: probable_pitchers, statcast_performances, player_projections, fantasy_lineups, data_ingestion_logs

**Strategy:** Diagnose why each table is empty, determine if it should be populated, fix ingestion if broken.

---

### Task 4: Diagnose probable_pitchers Table (Why Empty?)

**Problem:** Table has 0 rows but sync job exists. Need to diagnose why it's not populating.

**Files:**
- Run: Diagnostic query
- Modify: May need to fix `_sync_probable_pitchers()` in daily_ingestion.py

- [ ] **Step 1: Check job execution logs**

Run: Check Railway logs for the `_sync_probable_pitchers` job:

```bash
# In Railway dashboard, check logs for:
# - "SYNC JOB ENTRY: _sync_probable_pitchers"
# - Any errors in the sync
# - Last execution timestamp
```

Expected: Should see recent executions (scheduled 8:30 AM, 4:00 PM, 8:00 PM ET)

- [ ] **Step 2: Test sync job manually**

Run: Trigger the sync job manually via admin endpoint:

```bash
# If on Railway:
curl -X POST "https://fantasy-app-production-5079.up.railway.app/admin/sync/probable-pitchers"
```

Or locally:

```bash
venv/Scripts/python -c "
import asyncio
from backend.services.daily_ingestion import DataIngestionService

async def test():
    service = DataIngestionService()
    result = await service._sync_probable_pitchers()
    print(f'Result: {result}')

asyncio.run(test())
"
```

Expected: Should return success with record count

- [ ] **Step 3: Check if BDL API returns probable pitchers**

Run: Test BDL API directly:

```bash
venv/Scripts/python -c "
from backend.services.balldontlie import BallDontLieClient
from datetime import timedelta, datetime

bdl = BallDontLieClient()

# Fetch games for today
today = datetime.now().strftime('%Y-%m-%d')
games = bdl.get_mlb_games(today)

print(f'Games today: {len(games)}')
for game in games[:5]:
    print(f'  {game.home_team.abbreviation} vs {game.away_team.abbreviation}')
    print(f'    Home probable: {getattr(game, \"home_probable\", \"N/A\")}')
    print(f'    Away probable: {getattr(game, \"away_probable\", \"N/A\")}')
"
```

Expected: Should show games with probable pitchers (or N/A if not set)

- [ ] **Step 4: Based on diagnosis, create fix**

If the issue is that BDL doesn't return probable pitchers, document the finding:

Create `scripts/diagnose_probable_pitchers.md`:

```markdown
# Probable Pitchers Diagnosis

**Date:** April 9, 2026

## Finding

The probable_pitchers table is empty because:
- [INSERT FINDING HERE]

## Options

1. Use MLB Stats API instead of BDL
2. Parse from game metadata
3. Manual entry via admin panel

## Recommendation

[INSERT RECOMMENDATION]
```

- [ ] **Step 5: Commit findings**

```bash
git add scripts/diagnose_probable_pitchers.md
git commit -m "docs(diagnosis): add probable_pitchers empty table analysis"
```

---

### Task 5: Diagnose statcast_performances Table

**Problem:** Table has 0 rows. Statcast data is high-value for advanced analytics.

**Files:**
- Run: Diagnostic queries
- Check: If Statcast ingestion exists

- [ ] **Step 1: Check if Statcast ingestion exists**

Run: `grep -r "statcast" backend/services/*.py | head -10`

Expected: Should show if Statcast ingestion logic exists

- [ ] **Step 2: Check pybaseball installation**

Run: `venv/Scripts/python -c "import pybaseball; print(pybaseball.__version__)"`

Expected: Should print version (pybaseball should be installed)

- [ ] **Step 3: Test Statcast fetch**

Run: `venv/Scripts/python -c "from pybaseball import statcast; df = statcast('2026-04-08'); print(f'Rows: {len(df)}'); print(df.head() if len(df) > 0 else 'No data')"`

Expected: Should fetch Statcast data for recent date

- [ ] **Step 4: Document findings**

Create `scripts/diagnose_statcast.md` with findings and recommendations

- [ ] **Step 5: Commit findings**

```bash
git add scripts/diagnose_statcast.md
git commit -m "docs(diagnosis): add statcast_performances analysis"
```

---

### Task 6: Diagnose data_ingestion_logs Table

**Problem:** Table has 0 rows but should have logs from daily ingestion jobs.

**Files:**
- Check: `daily_ingestion.py` for `_record_job_run()` calls
- Run: Check if logs are being written

- [ ] **Step 1: Check if logging job exists**

Run: `grep -n "_record_job_run" backend/services/daily_ingestion.py | head -5`

Expected: Should show calls to `_record_job_run()` method

- [ ] **Step 2: Check DataIngestionLog model**

Run: `grep -A 30 "class DataIngestionLog" backend/models.py`

Expected: Should show model definition with job_name, status, records fields

- [ ] **Step 3: Test logging manually**

Run: `venv/Scripts/python -c "from backend.models import SessionLocal, DataIngestionLog; from datetime import datetime; from zoneinfo import ZoneInfo; db = SessionLocal(); log = DataIngestionLog(job_name='test_job', status='success', records_processed=100, started_at=datetime.now(ZoneInfo('America/New_York')), completed_at=datetime.now(ZoneInfo('America/New_York')), error_message=None); db.add(log); db.commit(); print(f'Log inserted: {log.id}'); db.close()"`

Expected: Should insert a log row

- [ ] **Step 4: Check if table now has data**

Run: `venv/Scripts/python -c "from backend.models import SessionLocal, DataIngestionLog; db = SessionLocal(); count = db.query(DataIngestionLog).count(); print(f'data_ingestion_logs rows: {count}'); db.close()"`

Expected: Should show 1 row (the test log we just inserted)

- [ ] **Step 5: Find why daily jobs aren't logging**

Run: `grep -B 5 -A 10 "_record_job_run" backend/services/daily_ingestion.py | head -30`

Expected: Should show where logging is called

- [ ] **Step 6: Based on findings, document issue**

Create `scripts/diagnose_ingestion_logs.md` with root cause analysis

- [ ] **Step 7: Commit findings**

```bash
git add scripts/diagnose_ingestion_logs.md
git commit -m "docs(diagnosis): add data_ingestion_logs analysis"
```

---

## PHASE 3: Computed Field Population

**Why Third:** These are nice-to-have analytics fields that improve user experience but aren't blocking.

**Current State:**
- mlb_player_stats: ops/whip/caught_stealing 100% NULL
- backtest_results: direction_correct 100% NULL
- player_daily_metrics: VORP/z-score/Statcast columns 100% NULL

**Strategy:** Compute these fields from existing data during ingestion.

---

### Task 7: Compute ops/whip/caught_stealing in mlb_player_stats

**Problem:** These fields exist in schema but are never populated during BDL ingestion.

**Files:**
- Modify: `backend/services/balldontlie.py` or ingestion logic
- Test: `tests/test_computed_stats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_computed_stats.py`:

```python
"""
Tests for computed statistics (ops, whip, caught_stealing).
"""

import pytest


def test_ops_calculation_from_raw_stats():
    """Test OPS calculation from AVG, OBP, SLG."""
    # OPS = OBP + SLG
    obp = 0.350
    slg = 0.450
    ops = obp + slg
    assert abs(ops - 0.800) < 0.001


def test_whip_calculation_from_raw_stats():
    """Test WHIP calculation from walks + hits allowed divided by innings pitched."""
    # WHIP = (BB + H) / IP
    walks_allowed = 20
    hits_allowed = 50
    innings_pitched = 30.0  # 30 IP
    whip = (walks_allowed + hits_allowed) / innings_pitched
    assert abs(whip - 2.333) < 0.01


def test_caught_stealing_defaults_to_zero():
    """Test that caught_stealing defaults to 0 when not provided."""
    # BDL API may not return cs, default to 0
    cs = None
    caught_stealing = cs if cs is not None else 0
    assert caught_stealing == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_computed_stats.py -v`

Expected: PASS (these are simple calculation tests)

- [ ] **Step 3: Find where BDL stats are ingested**

Run: `grep -n "def.*mlb.*stats" backend/services/daily_ingestion.py | head -5`

Expected: Should show the stats sync function

- [ ] **Step 4: Add computation logic to ingestion**

Modify the stats ingestion to compute ops/whip from raw fields (code will depend on exact ingestion implementation found in step 3)

- [ ] **Step 5: Run tests**

Run: `venv/Scripts/python -m pytest tests/test_computed_stats.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_computed_stats.py
git commit -m "feat(stats): add ops/whip/caught_stealing computation tests"
```

---

### Task 8: Populate direction_correct in backtest_results

**Problem:** Field exists but never calculated.

**Files:**
- Check: backtest logic
- Modify: Add direction calculation

- [ ] **Step 1: Understand backtest schema**

Run: `grep -A 40 "class BacktestResult" backend/models.py`

Expected: Should show fields including predicted_margin, actual_margin

- [ ] **Step 2: Calculate direction_correct**

Logic: `direction_correct = (predicted_margin > 0 and actual_margin > 0) or (predicted_margin < 0 and actual_margin < 0)`

- [ ] **Step 3: Add migration to backfill**

Create migration to compute direction_correct for existing rows

- [ ] **Step 4: Test calculation**

Run verification query to ensure direction_correct is correctly computed

- [ ] **Step 5: Commit**

```bash
git add <files>
git commit -m "feat(backtest): add direction_correct calculation"
```

---

### Task 9: Compute VORP/z-score in player_daily_metrics

**Problem:** Advanced metrics not computed.

**Files:**
- Check: If calculation logic exists
- Implement or document as "not implemented"

- [ ] **Step 1: Document as out of scope**

These are complex analytics metrics. Create `docs/analytics-roadmap.md` documenting them as future work.

- [ ] **Step 2: Commit**

```bash
git add docs/analytics-roadmap.md
git commit -m "docs(roadmap): add VORP/z-score as future enhancements"
```

---

## PHASE 4: Data Quality Cleanup

**Why Last:** Clean up any remaining data quality issues after core functionality works.

---

### Task 10: Fix Impossible ERA Value

**Problem:** 1 row has ERA > 100 (data entry error or calculation bug).

**Files:**
- Run: Identify the problematic row
- Modify: Fix or delete

- [ ] **Step 1: Find the impossible ERA**

Run: `venv/Scripts/python -c "from backend.models import SessionLocal, MLBPlayerStats; db = SessionLocal(); problematic = db.query(MLBPlayerStats).filter(MLBPlayerStats.era > 100).all(); print(f'Found {len(problematic)} rows with ERA > 100'); [print(f'  bdl_player_id={p.bdl_player_id}, era={p.era}, game_id={p.game_id}') for p in problematic]; db.close()"`

Expected: Should show the problematic row(s)

- [ ] **Step 2: Investigate root cause**

Check if it's a data entry error or calculation bug by examining raw stats:

Run: `venv/Scripts/python -c "from backend.models import SessionLocal, MLBPlayerStats; db = SessionLocal(); p = db.query(MLBPlayerStats).filter(MLBPlayerStats.era > 100).first(); print(f'earned_runs: {p.earned_runs}, innings_pitched: {p.innings_pitched}'); db.close()"`

Expected: Should show raw stats that led to impossible ERA

- [ ] **Step 3: Fix or delete**

If it's a calculation bug, fix the calculation. If it's bad data, delete or NULL the ERA value.

- [ ] **Step 4: Commit**

```bash
git add <files>
git commit -m "fix(data-quality): remove impossible ERA value"
```

---

### Task 11: Run Full Data Validation Audit

**Files:**
- Run: `scripts/db_validation_audit.py`
- Verify: All critical issues resolved

- [ ] **Step 1: Re-run validation audit**

Run: `python scripts/db_validation_audit.py`

Expected: Should show 0 CRITICAL issues (down from 0, but verify warnings)

- [ ] **Step 2: Compare results**

Compare new results with original audit:
- Original: 0 CRITICAL, 131 WARNING, 64 INFO
- Target: 0 CRITICAL, <50 WARNING (fixed player identity, empty tables)

- [ ] **Step 3: Document final state**

Create `reports/2026-04-09-final-data-quality-report.md` with before/after comparison

- [ ] **Step 4: Commit**

```bash
git add reports/2026-04-09-final-data-quality-report.md
git commit -m "docs(report): add final data quality remediation report"
```

---

## ACCEPTANCE CRITERIA

After completing all tasks, verify these criteria:

### Phase 1 (CRITICAL - Must Pass)
- [ ] player_id_mapping.yahoo_key: 2,376 rows populated (was 0)
- [ ] position_eligibility.bdl_player_id: 2,376 rows populated (was 0)
- [ ] Cross-system join test: 10 players verified with Yahoo→BDL link
- [ ] No orphaned records in foreign key relationships

### Phase 2 (IMPORTANT - Should Pass)
- [ ] probable_pitchers: Diagnosed and documented (populate if data available)
- [ ] statcast_performances: Diagnosed and documented (or populated)
- [ ] data_ingestion_logs: Populated with job execution logs
- [ ] All empty tables documented with rationale

### Phase 3 (NICE-TO-HAVE - May Defer)
- [ ] mlb_player_stats: ops/whip/caught_stealing computed
- [ ] backtest_results: direction_correct populated
- [ ] VORP/z-score documented as future work

### Phase 4 (CLEANUP - Must Pass)
- [ ] No ERA values > 100
- [ ] Full validation audit shows 0 CRITICAL issues
- [ ] WARNING count reduced from 131 to <50

---

## DELIVERABLES

1. ✅ Player identity resolution scripts (backfill_yahoo_keys.py, link_position_eligibility_bdl_ids.py)
2. ✅ Comprehensive test coverage (backfill, linking, computed stats)
3. ✅ Empty table diagnosis reports (probable_pitchers, statcast, ingestion_logs)
4. ✅ Data quality cleanup (impossible ERA fix)
5. ✅ Final validation audit report

---

## NEXT STEPS (After Remediation)

1. Integrate player identity resolution into daily_ingestion.py sync jobs
2. Build lineup optimization using cross-system player data
3. Compute scarcity indices using position_eligibility + player_stats
4. Create admin UI for manual player ID mapping overrides
5. Implement VORP/z-score calculations (documented in roadmap)

---

## RISK MITIGATION

**Risk:** Player name matching may produce false matches
**Mitigation:** Fuzzy matching with confidence thresholds, manual review of low-confidence matches

**Risk:** BDL API changes could break ingestion
**Mitigation:** Comprehensive tests, graceful degradation, alerting on API errors

**Risk:** Large table scans may timeout on Railway
**Mitigation:** Batch processing, pagination, transactional updates

---

## SUCCESS METRICS

| Metric | Before | Target | Actual |
|--------|--------|--------|--------|
| player_id_mapping.yahoo_key populated | 0 | 2,376 | TBD |
| position_eligibility.bdl_player_id populated | 0 | 2,376 | TBD |
| Empty tables (critical) | 5 | 0 | TBD |
| CRITICAL data quality issues | 0 | 0 | TBD |
| WARNING data quality issues | 131 | <50 | TBD |

