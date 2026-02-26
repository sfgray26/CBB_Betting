"""
Migration v6: Defensive four-factor columns — D1 average defaults + backfill.

=============================================================================
HOW TO RUN THIS MIGRATION
=============================================================================

1. Activate the virtual environment (if not already active):

       Windows:  venv\\Scripts\\activate
       Linux/Mac: source venv/bin/activate

2. Make sure PostgreSQL is running and DATABASE_URL is configured:

       # Either set it in .env or export directly:
       export DATABASE_URL=postgresql://postgres@127.0.0.1:5432/cbb_edge

3. Run the migration:

       python scripts/migrate_v6.py

4. Clear the ratings cache so fresh defensive stats are fetched:

       # Option A — restart the backend (simplest):
       uvicorn backend.main:app --reload

       # Option B — warm-restart via admin endpoint (no downtime):
       curl -X POST http://localhost:8000/admin/run-analysis \\
            -H "X-API-Key: <your-api-key>"

       # Option C — invalidate the in-process cache directly (Python REPL):
       from backend.services.ratings import get_ratings_service
       get_ratings_service().cache = {}
       get_ratings_service().cache_timestamp = None

5. Verify the migration succeeded:

       psql $DATABASE_URL -c "\\d team_profiles"
       psql $DATABASE_URL -c "SELECT COUNT(*) FROM team_profiles WHERE def_efg_pct IS NOT NULL;"

=============================================================================
WHAT THIS MIGRATION DOES
=============================================================================

The team_profiles table was created in v5 but the defensive four-factor
columns were given NULL defaults.  When the Markov simulator starts, it
reads these columns via TeamSimProfile and gets None — then falls back to
its own hardcoded defaults (D1 averages), which is correct, but any team
with a NULL def_efg_pct causes analysis.py to log a warning and switch to
heuristic mode, locking the entire game into the Gaussian CI path.

This migration:
  1. Creates the team_profiles table if it doesn't exist yet (safe to run
     even if v5 was never applied).
  2. Ensures every defensive column exists with the correct FLOAT type.
  3. Sets PostgreSQL DEFAULT constraints to D1 averages so future INSERTs
     automatically get real baselines instead of NULL.
  4. Backfills any existing NULL rows with the D1 average so the Markov
     engine immediately has valid inputs without waiting for the next
     nightly BartTorvik scrape.

D1 averages used (2024-25 season):
    def_efg_pct   = 0.505   (opponent effective FG% allowed)
    def_to_pct    = 0.175   (opponent turnover rate forced)
    def_ft_rate   = 0.280   (opponent free-throw rate allowed)
    def_three_par = 0.360   (opponent 3-point attempt rate)
    pace          = 68.0    (possessions per 40 min)
    efg_pct       = 0.505   (offensive effective FG%)
    to_pct        = 0.175   (offensive turnover rate)
    ft_rate       = 0.320   (offensive free-throw rate)
    three_par     = 0.360   (offensive 3-point attempt rate)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from backend.models import engine

# ---------------------------------------------------------------------------
# D1 average defaults (decimal scale, matching TeamSimProfile expectations)
# ---------------------------------------------------------------------------
D1 = {
    "def_efg_pct":   0.505,
    "def_to_pct":    0.175,
    "def_ft_rate":   0.280,
    "def_three_par": 0.360,
    "pace":          68.0,
    "efg_pct":       0.505,
    "to_pct":        0.175,
    "ft_rate":       0.320,
    "three_par":     0.360,
}


def table_exists(conn, table: str) -> bool:
    r = conn.execute(text(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = :t"
    ), {"t": table})
    return r.fetchone() is not None


def column_exists(conn, table: str, column: str) -> bool:
    r = conn.execute(text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = :t AND column_name = :c"
    ), {"t": table, "c": column})
    return r.fetchone() is not None


def constraint_exists(conn, name: str) -> bool:
    r = conn.execute(text(
        "SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = :c"
    ), {"c": name})
    return r.fetchone() is not None


def run_migration() -> None:
    with engine.connect() as conn:

        # ----------------------------------------------------------------
        # Step 1 — Create table (idempotent: skipped if already present)
        # ----------------------------------------------------------------
        if not table_exists(conn, "team_profiles"):
            conn.execute(text(f"""
                CREATE TABLE team_profiles (
                    id            SERIAL PRIMARY KEY,
                    team_name     VARCHAR  NOT NULL,
                    season_year   INTEGER  NOT NULL,
                    source        VARCHAR  NOT NULL DEFAULT 'barttorvik',
                    adj_oe        FLOAT,
                    adj_de        FLOAT,
                    adj_em        FLOAT,
                    pace          FLOAT    DEFAULT {D1['pace']},
                    efg_pct       FLOAT    DEFAULT {D1['efg_pct']},
                    to_pct        FLOAT    DEFAULT {D1['to_pct']},
                    ft_rate       FLOAT    DEFAULT {D1['ft_rate']},
                    three_par     FLOAT    DEFAULT {D1['three_par']},
                    def_efg_pct   FLOAT    DEFAULT {D1['def_efg_pct']},
                    def_to_pct    FLOAT    DEFAULT {D1['def_to_pct']},
                    def_ft_rate   FLOAT    DEFAULT {D1['def_ft_rate']},
                    def_three_par FLOAT    DEFAULT {D1['def_three_par']},
                    fetched_at    TIMESTAMP DEFAULT NOW()
                )
            """))
            print("[ OK ] Created table team_profiles")
        else:
            print("[SKIP] Table team_profiles already exists")

        # ----------------------------------------------------------------
        # Step 2 — Add any missing columns with D1 defaults
        # ----------------------------------------------------------------
        # Full column spec: (column_name, sql_type, default_value_or_None)
        _columns = [
            ("adj_oe",        "FLOAT",     None),
            ("adj_de",        "FLOAT",     None),
            ("adj_em",        "FLOAT",     None),
            ("pace",          "FLOAT",     D1["pace"]),
            ("efg_pct",       "FLOAT",     D1["efg_pct"]),
            ("to_pct",        "FLOAT",     D1["to_pct"]),
            ("ft_rate",       "FLOAT",     D1["ft_rate"]),
            ("three_par",     "FLOAT",     D1["three_par"]),
            ("def_efg_pct",   "FLOAT",     D1["def_efg_pct"]),
            ("def_to_pct",    "FLOAT",     D1["def_to_pct"]),
            ("def_ft_rate",   "FLOAT",     D1["def_ft_rate"]),
            ("def_three_par", "FLOAT",     D1["def_three_par"]),
            ("fetched_at",    "TIMESTAMP", None),
        ]

        for col_name, col_type, default in _columns:
            if column_exists(conn, "team_profiles", col_name):
                # Column exists — ensure the DEFAULT constraint is set
                if default is not None:
                    conn.execute(text(
                        f"ALTER TABLE team_profiles "
                        f"ALTER COLUMN {col_name} SET DEFAULT {default}"
                    ))
                    print(f"[ OK ] Set DEFAULT {default} on team_profiles.{col_name}")
                else:
                    print(f"[SKIP] Column team_profiles.{col_name} already exists (no default)")
            else:
                if default is not None:
                    conn.execute(text(
                        f"ALTER TABLE team_profiles "
                        f"ADD COLUMN {col_name} {col_type} DEFAULT {default}"
                    ))
                else:
                    conn.execute(text(
                        f"ALTER TABLE team_profiles ADD COLUMN {col_name} {col_type}"
                    ))
                print(f"[ OK ] Added column team_profiles.{col_name}"
                      + (f" (default={default})" if default is not None else ""))

        # ----------------------------------------------------------------
        # Step 3 — Backfill existing NULL rows with D1 averages
        # ----------------------------------------------------------------
        print("\nBackfilling NULL defensive stats with D1 averages...")
        for col_name, d1_val in D1.items():
            result = conn.execute(text(
                f"UPDATE team_profiles "
                f"SET {col_name} = {d1_val} "
                f"WHERE {col_name} IS NULL"
            ))
            rows_updated = result.rowcount
            if rows_updated:
                print(f"[ OK ] Backfilled {rows_updated} rows in team_profiles.{col_name} → {d1_val}")
            else:
                print(f"[SKIP] team_profiles.{col_name}: no NULL rows to backfill")

        # ----------------------------------------------------------------
        # Step 4 — Indexes and unique constraint
        # ----------------------------------------------------------------
        for idx_name, idx_cols in [
            ("ix_team_profiles_team_name",  "team_profiles (team_name)"),
            ("ix_team_profiles_season_year", "team_profiles (season_year)"),
        ]:
            r = conn.execute(text(
                "SELECT 1 FROM pg_indexes "
                "WHERE tablename = 'team_profiles' AND indexname = :n"
            ), {"n": idx_name})
            if r.fetchone():
                print(f"[SKIP] Index {idx_name} already exists")
            else:
                conn.execute(text(f"CREATE INDEX {idx_name} ON {idx_cols}"))
                print(f"[ OK ] Created index {idx_name}")

        uc_name = "_team_season_source_uc"
        if constraint_exists(conn, uc_name):
            print(f"[SKIP] Unique constraint {uc_name} already exists")
        else:
            conn.execute(text(
                "ALTER TABLE team_profiles "
                "ADD CONSTRAINT _team_season_source_uc "
                "UNIQUE (team_name, season_year, source)"
            ))
            print(f"[ OK ] Created unique constraint {uc_name}")

        conn.commit()

    print("""
=============================================================================
Migration v6 complete.
=============================================================================

NEXT STEPS:
  1. Restart the backend to clear the in-process ratings cache:
       uvicorn backend.main:app --reload

  2. Trigger a fresh BartTorvik scrape to populate real defensive stats:
       curl -X POST http://localhost:8000/admin/run-analysis \\
            -H "X-API-Key: <your-api-key>"

  OR: call RatingsService().save_team_profiles(db) in a Python shell.

  3. Verify real values are stored:
       psql $DATABASE_URL -c \\
         "SELECT team_name, def_efg_pct, def_to_pct FROM team_profiles LIMIT 5;"

  Until the BartTorvik scrape runs, all teams will use D1 averages
  (def_efg_pct=0.505, def_to_pct=0.175) which is safe and correct.
=============================================================================
""")


if __name__ == "__main__":
    run_migration()
