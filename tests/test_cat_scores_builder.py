"""
Tests for backend/services/cat_scores_builder.py

Verifies:
  - compute_cat_scores populates cat_scores in-place
  - classify_player correctly identifies batters and pitchers
  - run_backfill populates cat_scores for rows that have none
"""
import json
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> Session:
    """In-memory SQLite DB with a minimal player_projections + statcast_performances schema."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE player_projections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                player_name TEXT,
                team TEXT,
                positions TEXT,
                player_type TEXT,
                r INTEGER, hr INTEGER, rbi INTEGER, sb INTEGER,
                avg REAL, slg REAL, ops REAL,
                era REAL, whip REAL, k_per_nine REAL,
                w INTEGER, l INTEGER, qs INTEGER,
                hr_pit INTEGER, k_pit INTEGER, nsv INTEGER,
                cat_scores TEXT,
                updated_at TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE statcast_performances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                team TEXT,
                game_date TEXT
            )
        """))
        conn.commit()
    return Session(engine)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_compute_cat_scores_populates_keys():
    from backend.services.cat_scores_builder import compute_cat_scores, BATTER_WEIGHTS

    players = [
        {"proj": {"r": 80, "h": 150, "hr": 25, "rbi": 90, "k_bat": 120,
                  "tb": 260, "avg": 0.275, "ops": 0.850, "nsb": 10}},
        {"proj": {"r": 60, "h": 130, "hr": 15, "rbi": 70, "k_bat": 100,
                  "tb": 200, "avg": 0.250, "ops": 0.760, "nsb": 5}},
    ]
    compute_cat_scores(players, BATTER_WEIGHTS)
    for p in players:
        assert "cat_scores" in p
        assert set(p["cat_scores"].keys()) == set(BATTER_WEIGHTS.keys())


def test_classify_player_batter():
    from backend.services.cat_scores_builder import classify_player
    assert classify_player(["OF"], None, 20, 80) == "batter"
    assert classify_player("1B,OF", None, 10, 50) == "batter"


def test_classify_player_pitcher():
    from backend.services.cat_scores_builder import classify_player
    assert classify_player(["SP"], 3.50, 0, 0) == "pitcher"
    assert classify_player(["RP"], None, 0, 0) == "pitcher"


def test_run_backfill_populates_cat_scores():
    """Given 1 PlayerProjection row with cat_scores=None, run_backfill writes a non-empty dict."""
    from backend.services.cat_scores_builder import run_backfill

    db = _make_db()
    db.execute(text(
        "INSERT INTO player_projections "
        "(player_id, player_name, team, positions, r, hr, rbi, sb, avg, slg, ops, "
        " era, whip, k_per_nine, w, l, qs, hr_pit, k_pit, nsv, cat_scores) "
        "VALUES ('592450', 'Mookie Betts', 'LAD', 'OF', "
        "100, 30, 100, 20, 0.290, 0.520, 0.890, NULL, NULL, NULL, "
        "NULL, NULL, NULL, NULL, NULL, NULL, NULL)"
    ))
    db.commit()

    result = run_backfill(db)
    assert result["status"] == "success"
    assert result["cat_scores_updated"] >= 1

    row = db.execute(text("SELECT cat_scores FROM player_projections WHERE player_id='592450'")).fetchone()
    assert row is not None
    parsed = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    assert isinstance(parsed, dict)
    assert len(parsed) >= 1, "cat_scores should have at least 1 category key after backfill"

    db.close()
