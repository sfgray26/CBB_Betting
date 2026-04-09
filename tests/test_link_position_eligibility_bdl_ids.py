"""
Tests for linking position_eligibility to player_id_mapping via bdl_player_id.

Uses SQLite in-memory for fast, isolated testing.
"""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Import models
from backend.models import Base, PositionEligibility, PlayerIDMapping


@pytest.fixture(scope="module")
def engine():
    """SQLite in-memory engine for linking tests."""
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


def test_bdl_player_id_linking_populates_foreign_keys(db):
    """Test that bdl_player_id is populated from player_id_mapping."""
    from backend.models import PositionEligibility, PlayerIDMapping

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
    db.flush()

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
    db.flush()

    # Run linking script
    from scripts.link_position_eligibility_bdl_ids import link_bdl_player_ids
    result = link_bdl_player_ids(db)

    # Verify bdl_player_id populated
    assert result['updated_count'] == 1

    db.refresh(pe)
    assert pe.bdl_player_id == 1111


def test_bdl_player_id_linking_handles_missing_mappings(db):
    """Test that position_eligibility without matching mapping are skipped."""
    from backend.models import PositionEligibility

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
    db.flush()

    # Run linking script
    from scripts.link_position_eligibility_bdl_ids import link_bdl_player_ids
    result = link_bdl_player_ids(db)

    # Verify no update occurred
    assert result['updated_count'] == 0
    assert result['skipped_count'] == 1


def test_bdl_player_id_linking_idempotent(db):
    """Test that re-running the script is idempotent (no double updates)."""
    from backend.models import PositionEligibility, PlayerIDMapping

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
    db.flush()

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
    db.flush()

    # Run linking script twice
    from scripts.link_position_eligibility_bdl_ids import link_bdl_player_ids

    result1 = link_bdl_player_ids(db)
    result2 = link_bdl_player_ids(db)

    # Second run should be no-op (first run skipped the already-linked row)
    assert result1['skipped_count'] == 1  # First run skipped the row with existing bdl_player_id
    assert result1['updated_count'] == 0
    assert result2['skipped_count'] == 1  # Second run also skipped it
    assert result2['updated_count'] == 0
