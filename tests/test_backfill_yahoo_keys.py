"""
Tests for yahoo_key backfill from position_eligibility to player_id_mapping.

Uses SQLite in-memory for fast, isolated testing.
"""
import pytest
from datetime import datetime, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Import models
from backend.models import Base, PositionEligibility, PlayerIDMapping


@pytest.fixture(scope="module")
def engine():
    """SQLite in-memory engine for backfill tests."""
    # Patch SQLite compiler to render JSONB as TEXT (SQLite has no JSONB type)
    from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
    if not hasattr(SQLiteTypeCompiler, "_jsonb_patched"):
        SQLiteTypeCompiler.visit_JSONB = lambda self, type_, **kw: "TEXT"
        SQLiteTypeCompiler._jsonb_patched = True

    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    # Create all tables (SQLite-compatible schema)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    """Fresh database session for each test."""
    with Session(engine) as session:
        yield session
        session.rollback()


def test_yahoo_key_backfill_populates_mappings(db):
    """Test that yahoo_keys from position_eligibility populate player_id_mapping."""
    now = datetime.now(ZoneInfo("America/New_York"))

    # Create test position_eligibility rows
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
    db.flush()

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

    # Run backfill
    from scripts.backfill_yahoo_keys import backfill_yahoo_keys
    result = backfill_yahoo_keys(db)

    # Verify yahoo_keys populated
    assert result['updated_count'] == 2

    # Refresh from DB to see committed changes
    db.refresh(mapping1)
    db.refresh(mapping2)

    assert mapping1.yahoo_key == "469.p.12345"
    assert mapping2.yahoo_key == "469.p.67890"


def test_yahoo_key_backfill_handles_name_mismatches(db):
    """Test that name mismatches are handled gracefully."""
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
    db.flush()

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
    result = backfill_yahoo_keys(db)

    # Verify no update occurred for mismatched name
    assert result['updated_count'] == 0
    assert result['skipped_count'] == 1

    # Verify yahoo_key is still NULL
    db.refresh(mapping)
    assert mapping.yahoo_key is None


def test_yahoo_key_backfill_fuzzy_matching(db):
    """Test that fuzzy name matching works for minor variations."""
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
    db.flush()

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
    result = backfill_yahoo_keys(db)

    # Fuzzy match should work
    assert result['updated_count'] == 1

    db.refresh(mapping)
    assert mapping.yahoo_key == "469.p.55555"


def test_yahoo_key_backfill_skips_already_populated(db):
    """Test that rows with existing yahoo_key are not overwritten."""
    now = datetime.now(ZoneInfo("America/New_York"))

    # Create position_eligibility
    pe = PositionEligibility(
        yahoo_player_key="469.p.11111",
        player_name="Already Mapped",
        primary_position="SS",
        player_type="batter",
        multi_eligibility_count=1,
        can_play_ss=True,
        fetched_at=now,
        updated_at=now
    )
    db.add(pe)
    db.flush()

    # Create mapping with existing yahoo_key
    mapping = PlayerIDMapping(
        bdl_id=1111,
        yahoo_key="469.p.00000",  # Already has a key
        yahoo_id=None,
        mlbam_id=None,
        full_name="Already Mapped",
        normalized_name="already mapped",
        source="manual",
        created_at=now
    )
    db.add(mapping)
    db.commit()

    # Run backfill
    from scripts.backfill_yahoo_keys import backfill_yahoo_keys
    result = backfill_yahoo_keys(db)

    # Should skip - yahoo_key already exists
    assert result['updated_count'] == 0

    # Verify original yahoo_key unchanged
    db.refresh(mapping)
    assert mapping.yahoo_key == "469.p.00000"
