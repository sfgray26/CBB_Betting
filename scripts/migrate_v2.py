"""
Migration v2: add closing_lines and alerts tables.

Usage:
    python scripts/migrate_v2.py

This script is safe to run multiple times — it uses CREATE TABLE IF NOT EXISTS.
Alternatively, simply re-run scripts/init_db.py which calls
Base.metadata.create_all() and will create any missing tables automatically.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from backend.models import engine, Base

# Import models so their metadata is registered before create_all()
from backend.models import ClosingLine, DBAlert  # noqa: F401


DDL_CLOSING_LINES = """
CREATE TABLE IF NOT EXISTS closing_lines (
    id              SERIAL PRIMARY KEY,
    game_id         INTEGER NOT NULL REFERENCES games(id),
    captured_at     TIMESTAMP NOT NULL DEFAULT NOW(),
    spread          FLOAT,
    spread_odds     INTEGER,
    total           FLOAT,
    total_odds      INTEGER,
    moneyline_home  INTEGER,
    moneyline_away  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_closing_lines_game
    ON closing_lines (game_id);

CREATE INDEX IF NOT EXISTS idx_closing_lines_captured
    ON closing_lines (captured_at);
"""

DDL_ALERTS = """
CREATE TABLE IF NOT EXISTS alerts (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    alert_type      VARCHAR(50) NOT NULL,
    severity        VARCHAR(20) NOT NULL,
    message         TEXT NOT NULL,
    threshold       FLOAT,
    current_value   FLOAT,
    acknowledged    BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_created
    ON alerts (created_at);

CREATE INDEX IF NOT EXISTS idx_alerts_type
    ON alerts (alert_type);

CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged
    ON alerts (acknowledged);
"""


def migrate():
    print("Running CBB Edge v2 migration...")

    # Use SQLAlchemy's create_all for the canonical approach
    Base.metadata.create_all(bind=engine)
    print("  ✅ SQLAlchemy create_all() complete (closing_lines + alerts tables created/verified)")

    # Also run raw DDL as belt-and-suspenders for the indexes
    with engine.connect() as conn:
        conn.execute(text(DDL_CLOSING_LINES))
        conn.execute(text(DDL_ALERTS))
        conn.commit()
    print("  ✅ Indexes created/verified")
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
