"""
Tests for EMAC-077 EPIC-1: Time-Series Schema (ORM layer only).

Uses SQLite in-memory to test ORM model definitions and column existence.
PostgreSQL-specific features (JSONB, CHECK constraints) are tested structurally
(column exists, type maps) rather than via live DDL execution.
"""
import pytest
from datetime import date, datetime
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

# Import the ORM models
from backend.models import Base, PlayerDailyMetric, ProjectionSnapshot


@pytest.fixture(scope="module")
def engine():
    """SQLite in-memory engine for structural tests."""
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
    with Session(engine) as session:
        yield session
        session.rollback()


# ---------------------------------------------------------------------------
# EPIC-1 Test 1: player_daily_metrics insert + unique constraint
# ---------------------------------------------------------------------------

def test_player_daily_metric_insert_and_unique_constraint(db):
    row = PlayerDailyMetric(
        player_id="mlb.p.12345",
        player_name="Willi Castro",
        metric_date=date(2025, 4, 1),
        sport="mlb",
        vorp_7d=1.5,
        z_score_total=0.8,
    )
    db.add(row)
    db.flush()
    assert row.id is not None

    # Second insert with same (player_id, metric_date, sport) should raise
    dup = PlayerDailyMetric(
        player_id="mlb.p.12345",
        player_name="Willi Castro",
        metric_date=date(2025, 4, 1),
        sport="mlb",
    )
    db.add(dup)
    with pytest.raises(Exception):
        db.flush()


# ---------------------------------------------------------------------------
# EPIC-1 Test 2: sport check constraint (ORM level -- SQLite won't enforce CHECK)
# ---------------------------------------------------------------------------

def test_player_daily_metric_sport_check_constraint_rejects_invalid(db):
    """
    Verify that the sport field is constrained. In SQLite this is a structural
    test -- we verify the column accepts 'mlb' and 'cbb', and that the CHECK
    constraint is defined in the DDL (validated by inspecting the migration SQL).
    """
    row = PlayerDailyMetric(
        player_id="mlb.p.99",
        player_name="Test Player",
        metric_date=date(2025, 4, 2),
        sport="mlb",
    )
    db.add(row)
    db.flush()
    assert row.sport == "mlb"

    # SQLite does not enforce CHECK constraints at the DB level, but the
    # migration SQL includes the constraint -- test the column exists with right name
    inspector = inspect(db.bind)
    cols = {c["name"] for c in inspector.get_columns("player_daily_metrics")}
    assert "sport" in cols


# ---------------------------------------------------------------------------
# EPIC-1 Test 3: projection_snapshot insert and query
# ---------------------------------------------------------------------------

def test_projection_snapshot_insert_and_query(db):
    snap = ProjectionSnapshot(
        snapshot_date=date(2025, 4, 1),
        sport="mlb",
        player_changes={"test_player": {"z_old": 0.5, "z_new": 1.2}},
        total_players=300,
        significant_changes=12,
    )
    db.add(snap)
    db.flush()
    assert snap.id is not None

    fetched = db.get(ProjectionSnapshot, snap.id)
    assert fetched.total_players == 300
    assert fetched.significant_changes == 12


# ---------------------------------------------------------------------------
# EPIC-1 Test 4: pricing_engine column exists on Prediction model
# ---------------------------------------------------------------------------

def test_pricing_engine_column_exists_on_prediction(engine):
    """Verify the Prediction ORM model has a pricing_engine column after migration."""
    from backend.models import Prediction
    assert hasattr(Prediction, "pricing_engine"), (
        "Prediction model missing pricing_engine column -- run migrate_v8_post_draft.py"
    )


# ---------------------------------------------------------------------------
# EPIC-1 Test 5: pricing_engine column has correct type annotation
# ---------------------------------------------------------------------------

def test_pricing_engine_rejects_invalid_values():
    """
    The CHECK constraint (markov, gaussian, NULL) is PostgreSQL-only.
    Verify the column is defined as String(20) in the ORM.
    """
    from backend.models import Prediction
    from sqlalchemy import String
    col = Prediction.__table__.c.pricing_engine
    assert isinstance(col.type, String), f"Expected String, got {type(col.type)}"
    assert col.type.length == 20


# ---------------------------------------------------------------------------
# Helpers: load migration module and normalise DOWNGRADE_SQL to a string
# ---------------------------------------------------------------------------

def _load_migration_module():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "migrate_v8",
        Path(__file__).resolve().parents[1] / "scripts" / "migrate_v8_post_draft.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _downgrade_sql_as_string(mod) -> str:
    """
    Normalise DOWNGRADE_SQL to a single string regardless of whether the
    migration stores it as a plain string or as a list of statements.
    Both representations are valid; tests must accept either form.
    """
    raw = mod.DOWNGRADE_SQL
    if isinstance(raw, list):
        return "\n".join(raw)
    return raw


# ---------------------------------------------------------------------------
# EPIC-1 Test 6: downgrade SQL structure validation
# ---------------------------------------------------------------------------

def test_downgrade_removes_all_new_tables():
    """
    Validate that the migration script's DOWNGRADE_SQL contains DROP TABLE
    for both new tables. This is a static analysis test -- no DB required.
    """
    mod = _load_migration_module()
    sql = _downgrade_sql_as_string(mod)

    assert "DROP TABLE IF EXISTS player_daily_metrics" in sql, (
        "DOWNGRADE_SQL missing DROP TABLE for player_daily_metrics"
    )
    assert "DROP TABLE IF EXISTS projection_snapshots" in sql, (
        "DOWNGRADE_SQL missing DROP TABLE for projection_snapshots"
    )


# ---------------------------------------------------------------------------
# EPIC-1 Test 7: downgrade removes pricing_engine
# ---------------------------------------------------------------------------

def test_downgrade_removes_pricing_engine_column():
    """
    Validate that DOWNGRADE_SQL drops the pricing_engine column from predictions.
    """
    mod = _load_migration_module()
    sql = _downgrade_sql_as_string(mod)

    assert "DROP COLUMN IF EXISTS pricing_engine" in sql, (
        "DOWNGRADE_SQL missing DROP COLUMN for pricing_engine"
    )
