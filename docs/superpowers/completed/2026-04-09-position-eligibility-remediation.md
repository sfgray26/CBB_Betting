# ? COMPLETED � COMPLETE

> **Status:** COMPLETED (April 11, 2026)
> **Original Location:** docs/superpowers/plans/2026-04-09-position-eligibility-remediation.md
> **Moved To:** docs/superpowers/completed/2026-04-09-position-eligibility-remediation.md
> **Archive Reason:** All tasks in this plan have been successfully executed

---

# Position Eligibility Data Quality Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix position_eligibility data pipeline (currently 0 rows) and ensure data quality with comprehensive verification

**Architecture:** Database schema verification → Yahoo API client fixes → Backfill script validation → Data quality verification → Test coverage

**Tech Stack:** Python 3.11, SQLAlchemy, PostgreSQL, Yahoo Fantasy API, pytest

---

## PRE-REQUISITE: Database Schema Audit

Before any code changes, verify the actual database schema matches the model definition.

**Files:**
- Run: Railway shell commands
- Test: Manual verification

- [ ] **Step 1: Check actual database schema**

Run this command on Railway to inspect the current position_eligibility table structure:

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()

# 1. Check if table exists and row count
try:
    count = db.execute(text('SELECT COUNT(*) FROM position_eligibility')).scalar()
    print(f'Current row count: {count}')
except Exception as e:
    print(f'Table check error: {e}')

# 2. Get column definitions
print('\n=== CURRENT SCHEMA ===')
result = db.execute(text('''
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_name = 'position_eligibility'
    ORDER BY ordinal_position
'''))
for row in result:
    print(f'{row.column_name}: {row.data_type} (nullable={row.is_nullable})')

# 3. Check constraints
print('\n=== CONSTRAINTS ===')
result = db.execute(text('''
    SELECT constraint_name, constraint_type
    FROM information_schema.table_constraints
    WHERE table_name = 'position_eligibility'
'''))
for row in result:
    print(f'{row.constraint_name}: {row.constraint_type}')

# 4. Check indexes
print('\n=== INDEXES ===')
result = db.execute(text('''
    SELECT indexname, indexdef
    FROM pg_indexes
    WHERE tablename = 'position_eligibility'
'''))
for row in result:
    print(f'{row.indexname}')

db.close()
"
```

Expected: Should show yahoo_player_key column exists with nullable=False, and unique constraint `_pe_yahoo_uc` exists.

---

## TASK 1: Database Schema Migration (if needed)

**Files:**
- Create: `scripts/migrations/v10_verify_position_eligibility_schema.py`
- Modify: None
- Test: Manual verification

- [ ] **Step 1: Create schema verification script**

Write `scripts/migrations/v10_verify_position_eligibility_schema.py`:

```python
"""
Migration v10: Verify position_eligibility schema matches model definition.

This migration ensures the database schema matches the SQLAlchemy model:
- yahoo_player_key: String(50), nullable=False, unique, indexed
- bdl_player_id: Integer, nullable=True, indexed
- All can_play_* flags: Boolean, nullable=False, default=False
- primary_position, player_type, multi_eligibility_count columns present
- Unique constraint on yahoo_player_key (_pe_yahoo_uc)

If schema drift is detected, this script will:
1. Add missing columns
2. Fix nullable constraints
3. Add missing indexes/constraints
4. NOT delete any existing data
"""

from sqlalchemy import text
from backend.models import engine
import sys

def check_and_fix_schema():
    """Verify schema and apply fixes if needed."""
    with engine.connect() as conn:
        print("=== SCHEMA VERIFICATION ===\n")

        # Check if table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'position_eligibility'
            )
        """)).scalar()

        if not result:
            print("❌ Table does not exist - creating from model")
            conn.execute(text("""
                CREATE TABLE position_eligibility (
                    id SERIAL PRIMARY KEY,
                    yahoo_player_key VARCHAR(50) NOT NULL,
                    bdl_player_id INTEGER,
                    player_name VARCHAR(100),
                    first_name VARCHAR(50),
                    last_name VARCHAR(50),
                    can_play_c BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_1b BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_2b BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_3b BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_ss BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_lf BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_cf BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_rf BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_of BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_dh BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_util BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_sp BOOLEAN NOT NULL DEFAULT FALSE,
                    can_play_rp BOOLEAN NOT NULL DEFAULT FALSE,
                    primary_position VARCHAR(10),
                    player_type VARCHAR(10) NOT NULL DEFAULT 'batter',
                    scarcity_rank INTEGER,
                    league_rostered_pct FLOAT,
                    multi_eligibility_count INTEGER NOT NULL DEFAULT 0,
                    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """))
            conn.commit()
            print("✅ Table created")
        else:
            print("✅ Table exists - checking schema...")

        # Check yahoo_player_key column
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'position_eligibility'
                AND column_name = 'yahoo_player_key'
            )
        """)).scalar()

        if not result:
            print("❌ Missing yahoo_player_key - adding...")
            conn.execute(text("""
                ALTER TABLE position_eligibility
                ADD COLUMN yahoo_player_key VARCHAR(50)
            """))
            conn.commit()
            print("✅ Added yahoo_player_key column")
        else:
            print("✅ yahoo_player_key exists")

        # Check unique constraint
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.table_constraints
                WHERE table_name = 'position_eligibility'
                AND constraint_name = '_pe_yahoo_uc'
            )
        """)).scalar()

        if not result:
            print("❌ Missing unique constraint - adding...")
            conn.execute(text("""
                ALTER TABLE position_eligibility
                ADD CONSTRAINT _pe_yahoo_uc UNIQUE (yahoo_player_key)
            """))
            conn.commit()
            print("✅ Added unique constraint")
        else:
            print("✅ Unique constraint exists")

        # Check index on yahoo_player_key
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE tablename = 'position_eligibility'
                AND indexname = 'idx_pe_yahoo_player_key'
            )
        """)).scalar()

        if not result:
            print("❌ Missing index - adding...")
            conn.execute(text("""
                CREATE INDEX idx_pe_yahoo_player_key
                ON position_eligibility(yahoo_player_key)
            """))
            conn.commit()
            print("✅ Added index")
        else:
            print("✅ Index exists")

        print("\n=== SCHEMA VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    check_and_fix_schema()
```

- [ ] **Step 2: Run schema verification locally**

Run: `python scripts/migrations/v10_verify_position_eligibility_schema.py`

Expected: Should report all checks pass with ✅ markers

- [ ] **Step 3: Run schema verification on Railway**

Run: `railway run python scripts/migrations/v10_verify_position_eligibility_schema.py`

Expected: Should apply any missing schema elements to production database

- [ ] **Step 4: Commit**

```bash
git add scripts/migrations/v10_verify_position_eligibility_schema.py
git commit -m "feat(migration): add v10 schema verification for position_eligibility"
```

---

## TASK 2: Yahoo API Client Diagnostics

**Files:**
- Create: `scripts/diagnose_yahoo_api.py`
- Modify: `backend/fantasy_baseball/yahoo_client_resilient.py` (if bugs found)
- Test: Manual execution

- [ ] **Step 1: Create diagnostic script**

Write `scripts/diagnose_yahoo_api.py`:

```python
"""
Diagnostic script to test Yahoo Fantasy API integration.

This script tests:
1. OAuth token validity
2. League accessibility
3. Roster data structure
4. Player count sanity check

Usage:
    python scripts/diagnose_yahoo_api.py
"""

import os
import sys
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def diagnose_yahoo_api():
    """Run diagnostics on Yahoo Fantasy API integration."""
    print("=" * 60)
    print("YAHOO FANTASY API DIAGNOSTICS")
    print("=" * 60)

    try:
        # Initialize client
        print("\n[1] Initializing Yahoo client...")
        yahoo = YahooFantasyClient()
        print(f"✅ Client initialized")
        print(f"   League key: {yahoo.league_key}")
        print(f"   Game key: {yahoo.game_key}")

        if not yahoo.league_key:
            print("❌ ERROR: No league_key configured")
            print("   Set YAHOO_LEAGUE_ID environment variable")
            return {"status": "failed", "error": "No league_key"}

        # Test league metadata
        print(f"\n[2] Fetching league metadata for {yahoo.league_key}...")
        league_data = yahoo._get(f"league/{yahoo.league_key}/teams")
        print(f"✅ League data fetched")
        print(f"   Response type: {type(league_data).__name__}")

        # Test roster fetch
        print(f"\n[3] Fetching rosters from league...")
        all_players = yahoo.get_league_rosters(league_key=yahoo.league_key, include_team_key=True)
        print(f"✅ Roster data fetched")
        print(f"   Total players returned: {len(all_players)}")

        if len(all_players) == 0:
            print("❌ WARNING: Zero players returned")
            print("   Possible causes:")
            print("   - League ID is incorrect")
            print("   - League has not drafted yet")
            print("   - League is empty (no teams)")
            return {"status": "warning", "error": "Zero players"}

        # Sanity check: sample first 3 players
        print(f"\n[4] Sample player data (first 3):")
        for i, player in enumerate(all_players[:3], 1):
            print(f"\n   Player {i}:")
            print(f"     player_key: {player.get('player_key', 'MISSING')}")
            print(f"     name: {player.get('name', 'MISSING')}")
            print(f"     positions: {player.get('positions', 'MISSING')}")
            print(f"     team_key: {player.get('team_key', 'MISSING')}")

        # Count players with valid position data
        valid_positions = sum(1 for p in all_players if p.get('positions'))
        print(f"\n[5] Position data quality:")
        print(f"   Players with positions: {valid_positions}/{len(all_players)}")
        print(f"   Coverage: {100 * valid_positions // len(all_players) if all_players else 0}%")

        # Count multi-eligible players
        multi_eligible = sum(1 for p in all_players if len(p.get('positions', [])) > 1)
        print(f"   Multi-eligible players: {multi_eligible}")

        # Expected: ~750 players for 30 teams × 25 players
        if len(all_players) < 100:
            print(f"\n❌ WARNING: Only {len(all_players)} players")
            print(f"   Expected: ~750 (30 teams × 25 players)")
            print(f"   This suggests a Yahoo league configuration issue")
        elif len(all_players) > 2000:
            print(f"\n❌ WARNING: Too many players ({len(all_players)})")
            print(f"   Expected: ~750")
            print(f"   This suggests duplicates or wrong data structure")
        else:
            print(f"\n✅ Player count looks reasonable ({len(all_players)})")

        print("\n" + "=" * 60)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 60)

        return {
            "status": "success",
            "player_count": len(all_players),
            "valid_positions": valid_positions,
            "multi_eligible": multi_eligible,
        }

    except Exception as e:
        logger.exception("Diagnostics failed")
        print(f"\n❌ ERROR: {e}")
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    result = diagnose_yahoo_api()
    print(f"\nResult: {result['status']}")
    sys.exit(0 if result['status'] in ['success', 'warning'] else 1)
```

- [ ] **Step 2: Run diagnostics locally**

Run: `python scripts/diagnose_yahoo_api.py`

Expected: Should show player count, validate position data, identify any Yahoo API issues

- [ ] **Step 3: Run diagnostics on Railway**

Run: `railway run python scripts/diagnose_yahoo_api.py`

Expected: Should test production Yahoo integration, identify if league ID is correct

- [ ] **Step 4: Based on diagnostic results, document findings**

Create `docs/yahoo_api_diagnostic_results.md` with findings. If issues found, proceed to Task 3.

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_yahoo_api.py
git commit -m "feat(diagnostics): add Yahoo API diagnostic script"
```

---

## TASK 3: Fix Backfill Script Data Quality Issues

**Files:**
- Modify: `scripts/backfill_positions.py`
- Test: `tests/test_backfill_positions.py` (new)

- [ ] **Step 1: Add defensive logging to backfill script**

Add enhanced error handling and logging to `scripts/backfill_positions.py`:

Add this function after line 86 (after `determine_player_type`):

```python
def validate_player_data(player_data: dict) -> tuple[bool, str]:
    """
    Validate player data before upsert.

    Returns (is_valid, error_reason).
    """
    if not player_data:
        return False, "Empty player data"

    player_key = player_data.get("player_key")
    if not player_key:
        return False, "Missing player_key"

    name = player_data.get("name")
    if not name:
        return False, f"Missing name for player_key={player_key}"

    positions = player_data.get("positions", [])
    if not positions:
        return False, f"No positions for {name} ({player_key})"

    # Validate position strings are expected format
    valid_positions = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF",
                      "DH", "SP", "RP", "P", "UTIL", "Util"}
    for pos in positions:
        if pos.upper() not in valid_positions:
            return False, f"Invalid position '{pos}' for {name}"

    return True, ""


def log_quality_stats(player_data: dict, flags: dict):
    """Log quality metrics for debugging."""
    positions = player_data.get("positions", [])
    name = player_data.get("name", "Unknown")

    active_flags = sum(1 for v in flags.values() if v)
    logger.debug("%s: %d positions -> %d active flags",
                 name, len(positions), active_flags)

    # Warn if suspicious
    if len(positions) > 0 and active_flags == 0:
        logger.warning("%s: Has %d positions but 0 active flags - BUG!",
                       name, len(positions))
```

Now update the main loop in `backfill_position_eligibility` function (replace lines 123-180):

```python
        for player_data in all_players:
            player_key = player_data.get("player_key")
            if not player_key or player_key in seen_keys:
                if player_key in seen_keys:
                    logger.warning("Duplicate player_key: %s - skipping", player_key)
                skipped += 1
                continue
            seen_keys.add(player_key)

            # Validate player data before processing
            is_valid, error_reason = validate_player_data(player_data)
            if not is_valid:
                logger.debug("Skipping invalid player: %s", error_reason)
                skipped += 1
                continue

            name = player_data.get("name", "Unknown")
            positions = player_data.get("positions", [])

            # Build flags for ALL positions in one dict
            flags = build_position_flags(positions)

            # Log quality stats for debugging
            log_quality_stats(player_data, flags)

            primary = determine_primary_position(positions)
            ptype = determine_player_type(positions)

            # Count meaningful positions (exclude Util)
            meaningful = [p for p in positions if p.upper() not in ("UTIL",)]
            multi_count = len(meaningful)

            if dry_run:
                flag_str = ", ".join(k.replace("can_play_", "").upper()
                                     for k, v in flags.items() if v)
                logger.info("[DRY-RUN] %s (%s): %s | primary=%s type=%s count=%d",
                            name, player_key, flag_str, primary, ptype, multi_count)
                created += 1
                continue

            # Upsert: ON CONFLICT (yahoo_player_key) DO UPDATE
            stmt = pg_insert(PositionEligibility.__table__).values(
                yahoo_player_key=player_key,
                bdl_player_id=None,
                player_name=name,
                first_name=name.get("first", "") if isinstance(name, dict) else "",
                last_name=name.get("last", "") if isinstance(name, dict) else "",
                primary_position=primary,
                player_type=ptype,
                multi_eligibility_count=multi_count,
                fetched_at=now,
                updated_at=now,
                **flags,
            ).on_conflict_do_update(
                constraint="_pe_yahoo_uc",
                set_={
                    "player_name": name,
                    "first_name": name.get("first", "") if isinstance(name, dict) else "",
                    "last_name": name.get("last", "") if isinstance(name, dict) else "",
                    "primary_position": primary,
                    "player_type": ptype,
                    "multi_eligibility_count": multi_count,
                    "updated_at": now,
                    **flags,
                },
            )
            db.execute(stmt)
            created += 1
```

- [ ] **Step 2: Test backfill script locally**

Run: `python scripts/backfill_positions.py --dry-run`

Expected: Should show dry-run output with player counts, no errors

- [ ] **Step 3: Create unit tests for backfill functions**

Write `tests/test_backfill_positions.py`:

```python
"""
Tests for position eligibility backfill script.

Tests cover:
- Position flag building logic
- Primary position determination
- Player type classification
- Data validation
"""

import pytest
from scripts.backfill_positions import (
    build_position_flags,
    determine_primary_position,
    determine_player_type,
    validate_player_data,
)


class TestBuildPositionFlags:
    """Tests for build_position_flags function."""

    def test_single_position_batter(self):
        """Single position batter should set only that flag."""
        positions = ["1B"]
        flags = build_position_flags(positions)

        assert flags["can_play_1b"] is True
        assert flags["can_play_2b"] is False
        assert flags["can_play_of"] is False

    def test_outfield_position_sets_of_flag(self):
        """Any OF position (LF/CF/RF) should set can_play_of."""
        for pos in ["LF", "CF", "RF"]:
            flags = build_position_flags([pos])
            assert flags["can_play_of"] is True, f"{pos} should set OF flag"

    def test_multi_eligible_player(self):
        """Player with multiple positions should set all flags."""
        positions = ["2B", "SS", "OF"]
        flags = build_position_flags(positions)

        assert flags["can_play_2b"] is True
        assert flags["can_play_ss"] is True
        assert flags["can_play_of"] is True
        assert flags["can_play_1b"] is False

    def test_pitcher_positions(self):
        """Pitcher positions should set correct flags."""
        for pos in ["SP", "RP"]:
            flags = build_position_flags([pos])
            assert flags[f"can_play_{pos.lower()}"] is True

    def test_util_position(self):
        """UTIL position should set util flag."""
        flags = build_position_flags(["UTIL"])
        assert flags["can_play_util"] is True


class TestDeterminePrimaryPosition:
    """Tests for determine_primary_position function."""

    def test_uses_scarcity_priority(self):
        """Should pick most scarce position first."""
        # C is most scarce
        assert determine_primary_position(["1B", "C"]) == "C"
        # SS > 2B
        assert determine_primary_position(["2B", "SS"]) == "SS"
        # CF > LF
        assert determine_primary_position(["LF", "CF"]) == "CF"

    def test_fallback_to_first_position(self):
        """If no position in priority list, use first."""
        assert determine_primary_position(["DH"]) == "DH"

    def test_empty_positions(self):
        """Empty position list should return DH."""
        assert determine_primary_position([]) == "DH"


class TestDeterminePlayerType:
    """Tests for determine_player_type function."""

    def test_batter_only(self):
        """Only batter positions should classify as batter."""
        assert determine_player_type(["1B", "2B", "3B"]) == "batter"

    def test_pitcher_only(self):
        """Only pitcher positions should classify as pitcher."""
        assert determine_player_type(["SP"]) == "pitcher"
        assert determine_player_type(["RP"]) == "pitcher"

    def test_two_way_player(self):
        """Both batter and pitcher positions should classify as two_way."""
        assert determine_player_type(["1B", "SP"]) == "two_way"
        assert determine_player_type(["OF", "RP"]) == "two_way"


class TestValidatePlayerData:
    """Tests for validate_player_data function."""

    def test_valid_player_data(self):
        """Complete player data should pass validation."""
        player_data = {
            "player_key": "469.p.12345",
            "name": "Test Player",
            "positions": ["1B", "2B"]
        }
        is_valid, error = validate_player_data(player_data)
        assert is_valid is True
        assert error == ""

    def test_missing_player_key(self):
        """Missing player_key should fail validation."""
        player_data = {
            "name": "Test Player",
            "positions": ["1B"]
        }
        is_valid, error = validate_player_data(player_data)
        assert is_valid is False
        assert "player_key" in error.lower()

    def test_missing_name(self):
        """Missing name should fail validation."""
        player_data = {
            "player_key": "469.p.12345",
            "positions": ["1B"]
        }
        is_valid, error = validate_player_data(player_data)
        assert is_valid is False
        assert "name" in error.lower()

    def test_empty_positions(self):
        """Empty positions list should fail validation."""
        player_data = {
            "player_key": "469.p.12345",
            "name": "Test Player",
            "positions": []
        }
        is_valid, error = validate_player_data(player_data)
        assert is_valid is False
        assert "position" in error.lower()

    def test_invalid_position_string(self):
        """Invalid position should fail validation."""
        player_data = {
            "player_key": "469.p.12345",
            "name": "Test Player",
            "positions": ["XX"]  # Invalid position code
        }
        is_valid, error = validate_player_data(player_data)
        assert is_valid is False
        assert "invalid" in error.lower()
```

- [ ] **Step 4: Run tests**

Run: `venv/Scripts/python -m pytest tests/test_backfill_positions.py -v`

Expected: All 15+ tests should pass

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_positions.py tests/test_backfill_positions.py
git commit -m "feat(backfill): add data validation and unit tests for position eligibility"
```

---

## TASK 4: Create Data Quality Verification Script

**Files:**
- Create: `scripts/verify_data_quality.py`
- Test: `tests/test_verify_data_quality.py` (new)

- [ ] **Step 1: Write verification script**

Write `scripts/verify_data_quality.py`:

```python
"""
Comprehensive data quality verification for position_eligibility table.

Run after backfill to ensure data integrity.

Usage:
    python scripts/verify_data_quality.py
    python scripts/verify_data_quality.py --fix
"""

import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")
from backend.models import SessionLocal, PositionEligibility
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_quality_checks(db_session) -> dict:
    """
    Run all data quality checks and return report.

    Returns dict with check results and errors found.
    """
    report = {
        'timestamp': datetime.now(ZoneInfo("America/New_York")).isoformat(),
        'total_rows': 0,
        'null_yahoo_keys': 0,
        'duplicate_yahoo_keys': 0,
        'null_player_names': 0,
        'zero_eligibility': 0,
        'multi_eligible_players': 0,
        'multi_count_mismatches': 0,
        'primary_position_distribution': {},
        'player_type_distribution': {},
        'errors': [],
        'warnings': [],
        'passed': False
    }

    # Check 1: Total rows
    report['total_rows'] = db_session.query(PositionEligibility).count()
    logger.info("Total rows: %d", report['total_rows'])

    if report['total_rows'] == 0:
        report['errors'].append("CRITICAL: Table is empty - backfill may not have run")
        return report
    elif report['total_rows'] < 100:
        report['warnings'].append(f"Low row count: {report['total_rows']} (expected ~750)")
    elif report['total_rows'] > 2000:
        report['warnings'].append(f"High row count: {report['total_rows']} (expected ~750, possible duplicates)")

    # Check 2: NULL yahoo_player_key (CRITICAL)
    null_keys = db_session.query(PositionEligibility).filter(
        PositionEligibility.yahoo_player_key.is_(None)
    ).count()
    report['null_yahoo_keys'] = null_keys
    if null_keys > 0:
        report['errors'].append(f"CRITICAL: {null_keys} rows have NULL yahoo_player_key")

    # Check 3: Duplicate yahoo_player_key (CRITICAL)
    dupes = db_session.execute(text("""
        SELECT yahoo_player_key, COUNT(*) as cnt
        FROM position_eligibility
        GROUP BY yahoo_player_key
        HAVING COUNT(*) > 1
    """)).fetchall()
    report['duplicate_yahoo_keys'] = len(dupes)
    if dupes:
        report['errors'].append(f"CRITICAL: {len(dupes)} yahoo_player_keys are duplicated")
        for d in dupes[:5]:
            report['errors'].append(f"  - {d.yahoo_player_key}: {d.cnt} rows")

    # Check 4: NULL player names
    null_names = db_session.query(PositionEligibility).filter(
        PositionEligibility.player_name.is_(None)
    ).count()
    report['null_player_names'] = null_names
    if null_names > 0:
        report['warnings'].append(f"WARNING: {null_names} rows have NULL player_name")

    # Check 5: Players with zero position eligibility (all flags False)
    zero_elig = db_session.execute(text("""
        SELECT COUNT(*) FROM position_eligibility
        WHERE can_play_c = false AND can_play_1b = false
          AND can_play_2b = false AND can_play_3b = false
          AND can_play_ss = false AND can_play_lf = false
          AND can_play_cf = false AND can_play_rf = false
          AND can_play_of = false AND can_play_dh = false
          AND can_play_sp = false AND can_play_rp = false
    """)).scalar()
    report['zero_eligibility'] = zero_elig
    if zero_elig > 0:
        report['errors'].append(f"ERROR: {zero_elig} players have zero position eligibility (all flags False)")

    # Check 6: Multi-eligibility count
    multi = db_session.query(PositionEligibility).filter(
        PositionEligibility.multi_eligibility_count > 1
    ).count()
    report['multi_eligible_players'] = multi
    logger.info("Multi-eligible players: %d", multi)

    # Check 7: multi_eligibility_count matches actual flags
    mismatches = db_session.execute(text("""
        SELECT yahoo_player_key, player_name, multi_eligibility_count,
               (CASE WHEN can_play_c THEN 1 ELSE 0 END +
                CASE WHEN can_play_1b THEN 1 ELSE 0 END +
                CASE WHEN can_play_2b THEN 1 ELSE 0 END +
                CASE WHEN can_play_3b THEN 1 ELSE 0 END +
                CASE WHEN can_play_ss THEN 1 ELSE 0 END +
                CASE WHEN can_play_lf THEN 1 ELSE 0 END +
                CASE WHEN can_play_cf THEN 1 ELSE 0 END +
                CASE WHEN can_play_rf THEN 1 ELSE 0 END +
                CASE WHEN can_play_dh THEN 1 ELSE 0 END +
                CASE WHEN can_play_sp THEN 1 ELSE 0 END +
                CASE WHEN can_play_rp THEN 1 ELSE 0 END) as actual_count
        FROM position_eligibility
        HAVING multi_eligibility_count != actual_count
    """)).fetchall()
    report['multi_count_mismatches'] = len(mismatches)
    if mismatches:
        report['errors'].append(f"ERROR: {len(mismatches)} players have mismatched multi_eligibility_count")
        for m in mismatches[:5]:
            report['errors'].append(f"  - {m.player_name}: stored={m.multi_eligibility_count}, actual={m.actual_count}")

    # Check 8: Primary position distribution
    primary_dist = db_session.execute(text("""
        SELECT primary_position, COUNT(*) as cnt
        FROM position_eligibility
        GROUP BY primary_position
        ORDER BY cnt DESC
    """)).fetchall()
    report['primary_position_distribution'] = {p.primary_position: p.cnt for p in primary_dist}
    logger.info("Primary position distribution: %s", report['primary_position_distribution'])

    # Check 9: Player type distribution
    type_dist = db_session.execute(text("""
        SELECT player_type, COUNT(*) as cnt
        FROM position_eligibility
        GROUP BY player_type
        ORDER BY cnt DESC
    """)).fetchall()
    report['player_type_distribution'] = {p.player_type: p.cnt for p in type_dist}
    logger.info("Player type distribution: %s", report['player_type_distribution'])

    # Final pass/fail determination
    report['passed'] = len(report['errors']) == 0

    return report


def print_report(report: dict):
    """Print formatted quality report."""
    print("\n" + "=" * 60)
    print("POSITION ELIGIBILITY DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Timestamp: {report['timestamp']}\n")

    print("SUMMARY:")
    print(f"  Total rows: {report['total_rows']}")
    print(f"  Multi-eligible players: {report['multi_eligible_players']}")
    print(f"  NULL yahoo_player_key: {report['null_yahoo_keys']}")
    print(f"  Duplicate yahoo_player_key: {report['duplicate_yahoo_keys']}")
    print(f"  NULL player_name: {report['null_player_names']}")
    print(f"  Zero eligibility: {report['zero_eligibility']}")
    print(f"  Multi-count mismatches: {report['multi_count_mismatches']}")

    if report['errors']:
        print(f"\n❌ ERRORS ({len(report['errors'])}):")
        for error in report['errors']:
            print(f"  {error}")

    if report['warnings']:
        print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"  {warning}")

    print(f"\nPrimary position distribution:")
    for pos, count in report['primary_position_distribution'].items():
        print(f"  {pos or 'NULL'}: {count}")

    print(f"\nPlayer type distribution:")
    for ptype, count in report['player_type_distribution'].items():
        print(f"  {ptype}: {count}")

    print("\n" + "=" * 60)
    if report['passed']:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ QUALITY CHECKS FAILED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues found")
    args = parser.parse_args()

    db = SessionLocal()

    try:
        report = run_quality_checks(db)
        print_report(report)

        if args.fix and report['duplicate_yahoo_keys'] > 0:
            print("\nRemoving duplicates...")
            # Delete duplicate rows, keeping the one with highest ID
            db.execute(text("""
                DELETE FROM position_eligibility
                WHERE id NOT IN (
                    SELECT MAX(id) FROM position_eligibility
                    GROUP BY yahoo_player_key
                )
            """))
            db.commit()
            print("Duplicates removed. Re-run verification to confirm.")

        sys.exit(0 if report['passed'] else 1)

    finally:
        db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create tests for verification script**

Write `tests/test_verify_data_quality.py`:

```python
"""
Tests for data quality verification script.
"""

import pytest
from scripts.verify_data_quality import run_quality_checks
from backend.models import SessionLocal, PositionEligibility
from datetime import datetime
from zoneinfo import ZoneInfo


@pytest.fixture
def clean_db():
    """Provide clean database session."""
    db = SessionLocal()
    yield db
    # Cleanup: delete all test data
    db.query(PositionEligibility).delete()
    db.commit()
    db.close()


class TestRunQualityChecks:
    """Tests for run_quality_checks function."""

    def test_empty_table_fails(self, clean_db):
        """Empty table should fail with critical error."""
        report = run_quality_checks(clean_db)
        assert report['passed'] is False
        assert report['total_rows'] == 0
        assert any("empty" in e.lower() for e in report['errors'])

    def test_single_valid_player_passes(self, clean_db):
        """Single valid player should pass all checks."""
        now = datetime.now(ZoneInfo("America/New_York"))
        player = PositionEligibility(
            yahoo_player_key="469.p.test1",
            player_name="Test Player",
            primary_position="1B",
            player_type="batter",
            multi_eligibility_count=1,
            can_play_1b=True,
            fetched_at=now,
            updated_at=now
        )
        clean_db.add(player)
        clean_db.commit()

        report = run_quality_checks(clean_db)
        assert report['passed'] is True
        assert report['total_rows'] == 1

    def test_null_yahoo_key_fails(self, clean_db):
        """NULL yahoo_player_key should fail."""
        now = datetime.now(ZoneInfo("America/New_York"))
        player = PositionEligibility(
            yahoo_player_key=None,  # NULL
            player_name="Test",
            primary_position="1B",
            player_type="batter",
            multi_eligibility_count=1,
            can_play_1b=True,
            fetched_at=now,
            updated_at=now
        )
        clean_db.add(player)
        clean_db.commit()

        report = run_quality_checks(clean_db)
        assert report['passed'] is False
        assert report['null_yahoo_keys'] == 1

    def test_duplicate_yahoo_key_fails(self, clean_db):
        """Duplicate yahoo_player_key should fail."""
        now = datetime.now(ZoneInfo("America/New_York"))

        player1 = PositionEligibility(
            yahoo_player_key="469.p.dup",
            player_name="Player 1",
            primary_position="1B",
            player_type="batter",
            multi_eligibility_count=1,
            can_play_1b=True,
            fetched_at=now,
            updated_at=now
        )

        player2 = PositionEligibility(
            yahoo_player_key="469.p.dup",  # Duplicate key
            player_name="Player 2",
            primary_position="2B",
            player_type="batter",
            multi_eligibility_count=1,
            can_play_2b=True,
            fetched_at=now,
            updated_at=now
        )

        clean_db.add(player1)
        clean_db.add(player2)
        clean_db.commit()

        report = run_quality_checks(clean_db)
        assert report['passed'] is False
        assert report['duplicate_yahoo_keys'] == 1

    def test_zero_eligibility_fails(self, clean_db):
        """Player with all flags False should fail."""
        now = datetime.now(ZoneInfo("America/New_York"))
        player = PositionEligibility(
            yahoo_player_key="469.p.zero",
            player_name="Zero Eligibility",
            primary_position="DH",
            player_type="batter",
            multi_eligibility_count=1,  # Claims 1 position but all flags False
            can_play_c=False,
            can_play_1b=False,
            can_play_2b=False,
            can_play_3b=False,
            can_play_ss=False,
            can_play_lf=False,
            can_play_cf=False,
            can_play_rf=False,
            can_play_of=False,
            can_play_dh=False,
            can_play_sp=False,
            can_play_rp=False,
            fetched_at=now,
            updated_at=now
        )
        clean_db.add(player)
        clean_db.commit()

        report = run_quality_checks(clean_db)
        assert report['passed'] is False
        assert report['zero_eligibility'] == 1

    def test_multi_count_mismatch_fails(self, clean_db):
        """Mismatched multi_eligibility_count should fail."""
        now = datetime.now(ZoneInfo("America/New_York"))
        player = PositionEligibility(
            yahoo_player_key="469.p.multi",
            player_name="Multi Player",
            primary_position="1B",
            player_type="batter",
            multi_eligibility_count=5,  # Claims 5 positions
            can_play_1b=True,
            can_play_2b=True,  # Only 2 actual positions
            fetched_at=now,
            updated_at=now
        )
        clean_db.add(player)
        clean_db.commit()

        report = run_quality_checks(clean_db)
        assert report['passed'] is False
        assert report['multi_count_mismatches'] == 1
```

- [ ] **Step 3: Run verification script tests**

Run: `venv/Scripts/python -m pytest tests/test_verify_data_quality.py -v`

Expected: All tests should pass

- [ ] **Step 4: Test verification script against empty table**

Run: `railway run python scripts/verify_data_quality.py`

Expected: Should show empty table error (this is expected before backfill)

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_data_quality.py tests/test_verify_data_quality.py
git commit -m "feat(verification): add data quality verification script with tests"
```

---

## TASK 5: Execute Backfill and Verify

**Files:**
- Run: `scripts/backfill_positions.py` on Railway
- Run: `scripts/verify_data_quality.py` on Railway
- Test: Manual verification

- [ ] **Step 1: Run Yahoo API diagnostics**

Run: `railway run python scripts/diagnose_yahoo_api.py`

Expected: Should show player count and identify if Yahoo league is accessible

**If diagnostics show zero players:**
- STOP - Yahoo league configuration issue
- User must verify league ID and ensure league has teams drafted
- Do NOT proceed with backfill until Yahoo API returns players

- [ ] **Step 2: Run backfill in dry-run mode**

Run: `railway run python scripts/backfill_positions.py --dry-run`

Expected: Should show dry-run output with player positions, no database writes

- [ ] **Step 3: Run backfill for real**

Run: `railway run python scripts/backfill_positions.py`

Expected: Should insert ~750 rows, show multi-eligible players

- [ ] **Step 4: Run data quality verification**

Run: `railway run python scripts/verify_data_quality.py`

Expected: Should show ✅ ALL CHECKS PASSED with ~750 rows

- [ ] **Step 5: Document results**

Create `reports/2026-04-09-position-eligibility-backfill-report.md` with:
- Rows inserted
- Multi-eligible player count
- Quality check results
- Any issues found and fixed

---

## ACCEPTANCE CRITERIA VERIFICATION

After completing all tasks, verify these criteria:

- [ ] **AC1:** position_eligibility table has ~750 rows (not 0, not 2000+)
- [ ] **AC2:** Zero NULL values in yahoo_player_key column
- [ ] **AC3:** Zero duplicate yahoo_player_key values
- [ ] **AC4:** Each player has exactly ONE row with ALL their positions
- [ ] **AC5:** multi_eligibility_count matches actual number of can_play_* flags set to TRUE
- [ ] **AC6:** No rows with all position flags set to FALSE
- [ ] **AC7:** primary_position set to most valuable/defensive position
- [ ] **AC8:** player_type correctly identifies "batter", "pitcher", or "two_way"
- [ ] **AC9:** All unit tests pass (pytest)
- [ ] **AC10:** Data quality verification passes with zero critical errors

---

## EDGE CASES HANDLED

1. **5+ position eligible players**: Flag system handles unlimited positions
2. **Two-way players (Ohtani)**: player_type="two_way" when both batter and pitcher flags present
3. **Unexpected position codes**: validate_player_data() rejects invalid position strings
4. **Mid-season player additions**: Upsert logic adds new players, updates existing
5. **Backfill fails halfway**: Transaction rolls back, no partial data committed

---

## DELIVERABLES

1. ✅ Schema verification script: `scripts/migrations/v10_verify_position_eligibility_schema.py`
2. ✅ Yahoo API diagnostics: `scripts/diagnose_yahoo_api.py`
3. ✅ Enhanced backfill script: Updated `scripts/backfill_positions.py` with validation
4. ✅ Data quality verification: `scripts/verify_data_quality.py`
5. ✅ Comprehensive test coverage: `tests/test_backfill_positions.py` and `tests/test_verify_data_quality.py`
6. ✅ Execution report: `reports/2026-04-09-position-eligibility-backfill-report.md`

---

## NEXT STEPS (After Acceptance)

1. Integrate position_eligibility sync into daily_ingestion.py
2. Add bdl_player_id mapping via player_id_mapping table
3. Calculate scarcity_rank based on league roster percentages
4. Create frontend endpoints for position eligibility queries

