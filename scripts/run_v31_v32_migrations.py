#!/usr/bin/env python
"""
Quick migration runner for V31 and V32 database schema updates.
Run this within the Railway environment (e.g., via railway exec or similar).
"""
import os
import sys
from sqlalchemy import create_engine, text

def run_migrations():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in environment")
        sys.exit(1)

    engine = create_engine(db_url)

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        # V31: player_rolling_stats columns
        prs_columns = [
            ("w_runs", "V31 Decay-weighted runs scored"),
            ("w_tb", "V31 Decay-weighted total bases"),
            ("w_qs", "V31 Decay-weighted quality starts"),
        ]

        print("=== V31: Adding player_rolling_stats columns ===")
        for col_name, comment in prs_columns:
            try:
                conn.execute(text(f"""
                    ALTER TABLE player_rolling_stats
                    ADD COLUMN IF NOT EXISTS {col_name} DOUBLE PRECISION
                """))
                print(f"  ✓ {col_name} added")
            except Exception as e:
                print(f"  ✗ {col_name} failed: {e}")

        # V32: player_scores columns
        ps_columns = [
            "z_r", "z_h", "z_tb", "z_k_b", "z_ops", "z_k_p", "z_qs"
        ]

        print("\n=== V32: Adding player_scores columns ===")
        for col_name in ps_columns:
            try:
                conn.execute(text(f"""
                    ALTER TABLE player_scores
                    ADD COLUMN IF NOT EXISTS {col_name} DOUBLE PRECISION
                """))
                print(f"  ✓ {col_name} added")
            except Exception as e:
                print(f"  ✗ {col_name} failed: {e}")

        # Verification
        print("\n=== Verification ===")
        prs_check = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'player_rolling_stats'
            AND column_name IN ('w_runs', 'w_tb', 'w_qs')
            ORDER BY column_name
        """)).fetchall()
        print(f"player_rolling_stats: {[r[0] for r in prs_check]}")

        ps_check = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'player_scores'
            AND column_name IN ('z_r', 'z_h', 'z_tb', 'z_k_b', 'z_ops', 'z_k_p', 'z_qs')
            ORDER BY column_name
        """)).fetchall()
        print(f"player_scores: {[r[0] for r in ps_check]}")

    print("\n=== Migrations complete ===")

if __name__ == "__main__":
    run_migrations()
