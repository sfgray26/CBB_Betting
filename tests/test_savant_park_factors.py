"""Tests for Baseball Savant park factor snapshot support."""

import tempfile
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import backend.fantasy_baseball.ballpark_factors as ballpark_factors
from backend.fantasy_baseball.ballpark_factors import get_park_factor, park_adjusted_era
from backend.fantasy_baseball.savant_park_factors import (
    load_savant_park_factor_snapshot,
    savant_index_to_factor,
)
from backend.models import ParkFactor


def test_savant_index_to_factor_converts_100_scale():
    assert savant_index_to_factor("100") == 1.0
    assert savant_index_to_factor("125") == 1.25
    assert savant_index_to_factor("83") == 0.83
    assert savant_index_to_factor(None) == 1.0


def test_savant_snapshot_contains_core_venue_fields():
    snapshot = load_savant_park_factor_snapshot()

    coors = next(row for row in snapshot if row["team"] == "COL")
    assert coors["park_name"] == "Coors Field"
    assert coors["season"] == 2025
    assert coors["rolling_years"] == 3
    assert coors["run_factor"] == 1.25
    assert coors["woba_factor"] == 1.12

    tmobile = next(row for row in snapshot if row["team"] == "SEA")
    assert tmobile["run_factor"] == 0.83
    assert tmobile["so_factor"] == 1.16


def test_get_park_factor_supports_savant_columns_from_db():
    temp_db = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
    temp_db.close()
    db_path = Path(temp_db.name)

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    ParkFactor.__table__.create(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    db = SessionLocal()
    db.add(
        ParkFactor(
            park_name="SEA",
            team="SEA",
            run_factor=0.83,
            hr_factor=0.93,
            era_factor=0.83,
            woba_factor=0.91,
            xwobacon_factor=0.99,
            so_factor=1.16,
        )
    )
    db.commit()

    try:
        assert get_park_factor("SEA", "woba", _db_session=db) == 0.91
        assert get_park_factor("SEA", "xwobacon", _db_session=db) == 0.99
        assert get_park_factor("SEA", "so", _db_session=db) == 1.16
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_park_adjusted_era_increases_in_hitter_park():
    original_cache = dict(ballpark_factors._park_factor_cache)

    try:
        get_park_factor.cache_clear()
        ballpark_factors._park_factor_cache = {("era", "COL"): 1.25}
        assert park_adjusted_era(4.00, "COL") == 5.0
    finally:
        ballpark_factors._park_factor_cache = original_cache
        get_park_factor.cache_clear()
