"""
Migration v5: create team_profiles table with defensive four-factor columns.

Usage:
    python scripts/migrate_v5.py

This script is safe to run multiple times — it checks whether the table and
each column already exist before making any changes.

What this creates
-----------------
team_profiles table:
    id             SERIAL PRIMARY KEY
    team_name      VARCHAR NOT NULL
    season_year    INTEGER NOT NULL
    source         VARCHAR NOT NULL DEFAULT 'barttorvik'

    -- Efficiency margins (same scale as KenPom AdjEM, approx -30 to +30)
    adj_oe         FLOAT   -- Adjusted Offensive Efficiency
    adj_de         FLOAT   -- Adjusted Defensive Efficiency
    adj_em         FLOAT   -- Adjusted Efficiency Margin (AdjOE - AdjDE)

    -- Offensive four factors
    pace           FLOAT   -- Possessions per 40 min
    efg_pct        FLOAT   -- Effective FG% (offensive)
    to_pct         FLOAT   -- Turnover rate (offensive; lower is better)
    ft_rate        FLOAT   -- FT attempts / FGA (offensive)
    three_par      FLOAT   -- 3PA / FGA (offensive)

    -- Defensive four factors (previously absent — caused Markov "blind defense")
    def_efg_pct    FLOAT   -- Opponent eFG% allowed
    def_to_pct     FLOAT   -- Opponent TO rate forced
    def_ft_rate    FLOAT   -- Opponent FT rate allowed
    def_three_par  FLOAT   -- Opponent 3PA rate allowed

    fetched_at     TIMESTAMP

Unique constraint:
    (team_name, season_year, source)  -- allows multiple sources per team/season

Integration:
    Populated nightly by RatingsService.save_team_profiles(db).
    Read by the matchup engine profile cache so the Markov simulator uses
    real per-team defensive stats instead of D1-average defaults.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from backend.models import engine


def table_exists(conn, table: str) -> bool:
    result = conn.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = :t"
    ), {"t": table})
    return result.fetchone() is not None


def column_exists(conn, table: str, column: str) -> bool:
    result = conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = :t AND column_name = :c"
    ), {"t": table, "c": column})
    return result.fetchone() is not None


def constraint_exists(conn, constraint_name: str) -> bool:
    result = conn.execute(text(
        "SELECT constraint_name FROM information_schema.table_constraints "
        "WHERE constraint_name = :c"
    ), {"c": constraint_name})
    return result.fetchone() is not None


def run_migration():
    with engine.connect() as conn:

        # ----------------------------------------------------------------
        # Step 1: Create the team_profiles table if it doesn't exist
        # ----------------------------------------------------------------
        if table_exists(conn, "team_profiles"):
            print("Table team_profiles already exists — checking columns.")
        else:
            conn.execute(text("""
                CREATE TABLE team_profiles (
                    id          SERIAL PRIMARY KEY,
                    team_name   VARCHAR NOT NULL,
                    season_year INTEGER NOT NULL,
                    source      VARCHAR NOT NULL DEFAULT 'barttorvik',

                    adj_oe      FLOAT,
                    adj_de      FLOAT,
                    adj_em      FLOAT,

                    pace        FLOAT,
                    efg_pct     FLOAT,
                    to_pct      FLOAT,
                    ft_rate     FLOAT,
                    three_par   FLOAT,

                    def_efg_pct   FLOAT,
                    def_to_pct    FLOAT,
                    def_ft_rate   FLOAT,
                    def_three_par FLOAT,

                    fetched_at  TIMESTAMP DEFAULT NOW()
                )
            """))
            print("Created table team_profiles")

        # ----------------------------------------------------------------
        # Step 2: Add any missing columns (idempotent for partial runs)
        # ----------------------------------------------------------------
        _columns = [
            ("adj_oe",       "FLOAT"),
            ("adj_de",       "FLOAT"),
            ("adj_em",       "FLOAT"),
            ("pace",         "FLOAT"),
            ("efg_pct",      "FLOAT"),
            ("to_pct",       "FLOAT"),
            ("ft_rate",      "FLOAT"),
            ("three_par",    "FLOAT"),
            ("def_efg_pct",  "FLOAT"),
            ("def_to_pct",   "FLOAT"),
            ("def_ft_rate",  "FLOAT"),
            ("def_three_par","FLOAT"),
            ("fetched_at",   "TIMESTAMP DEFAULT NOW()"),
        ]
        for col_name, col_type in _columns:
            if column_exists(conn, "team_profiles", col_name):
                print(f"  Column team_profiles.{col_name} already exists — skipping.")
            else:
                conn.execute(text(
                    f"ALTER TABLE team_profiles ADD COLUMN {col_name} {col_type}"
                ))
                print(f"  Added team_profiles.{col_name}")

        # ----------------------------------------------------------------
        # Step 3: Add indexes and unique constraint
        # ----------------------------------------------------------------
        _indexes = [
            ("ix_team_profiles_team_name",   "team_profiles (team_name)"),
            ("ix_team_profiles_season_year",  "team_profiles (season_year)"),
        ]
        for idx_name, idx_cols in _indexes:
            result = conn.execute(text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'team_profiles' AND indexname = :n"
            ), {"n": idx_name})
            if result.fetchone():
                print(f"  Index {idx_name} already exists — skipping.")
            else:
                conn.execute(text(f"CREATE INDEX {idx_name} ON {idx_cols}"))
                print(f"  Created index {idx_name}")

        constraint_name = "_team_season_source_uc"
        if constraint_exists(conn, constraint_name):
            print(f"  Constraint {constraint_name} already exists — skipping.")
        else:
            conn.execute(text(
                "ALTER TABLE team_profiles "
                "ADD CONSTRAINT _team_season_source_uc "
                "UNIQUE (team_name, season_year, source)"
            ))
            print(f"  Created unique constraint {constraint_name}")

        conn.commit()
        print("\nMigration v5 complete.")
        print("\nNext step: run the nightly analysis or manually call")
        print("  RatingsService().save_team_profiles(db)")
        print("to populate the table with current BartTorvik data.")


if __name__ == "__main__":
    run_migration()
