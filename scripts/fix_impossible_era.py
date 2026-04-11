#!/usr/bin/env python3
"""
Fix impossible ERA values in mlb_player_stats table.

ERA should be between 0 and 100. Values outside this range are data errors
and should be set to NULL.
"""

from sqlalchemy import text
from backend.models import SessionLocal


def main():
    print("Finding impossible ERA values (ERA < 0 or ERA > 100)...")

    db = SessionLocal()

    try:
        # Find all impossible ERAs
        result = db.execute(text("""
            SELECT COUNT(*) as count
            FROM mlb_player_stats
            WHERE era IS NOT NULL AND (era < 0 OR era > 100)
        """)).fetchone()

        impossible_count = result.count
        print(f"Found {impossible_count} rows with impossible ERA")

        if impossible_count == 0:
            print("No impossible ERAs found. Exiting.")
            return

        # Show sample of impossible ERAs
        print("\nSample of impossible ERA values:")
        samples = db.execute(text("""
            SELECT
                bdl_player_id,
                era,
                earned_runs,
                innings_pitched,
                game_date,
                opponent_team
            FROM mlb_player_stats
            WHERE era IS NOT NULL AND (era < 0 OR era > 100)
            ORDER BY ABS(era) DESC
            LIMIT 5
        """)).fetchall()

        for sample in samples:
            calc_era = (sample.earned_runs / sample.innings_pitched * 9) if sample.innings_pitched and sample.innings_pitched > 0 else None
            print(f"  Player {sample.bdl_player_id}: ERA={sample.era}, "
                  f"ER={sample.earned_runs}, IP={sample.innings_pitched}, "
                  f"Calc ERA={calc_era:.2f if calc_era else 'N/A'}, "
                  f"Date={sample.game_date}")

        # Confirm before fixing
        print(f"\nThis will set ERA to NULL for {impossible_count} rows.")
        response = input("Proceed? (yes/no): ").strip().lower()

        if response != "yes":
            print("Aborted.")
            return

        # Fix the impossible ERAs
        print("\nSetting impossible ERAs to NULL...")
        result = db.execute(text("""
            UPDATE mlb_player_stats
            SET era = NULL
            WHERE era IS NOT NULL AND (era < 0 OR era > 100)
        """))

        db.commit()
        print(f"Updated {result.rowcount} rows")

        # Verify the fix
        remaining = db.execute(text("""
            SELECT COUNT(*) as count
            FROM mlb_player_stats
            WHERE era IS NOT NULL AND (era < 0 OR era > 100)
        """)).fetchone()

        print(f"\nVerification: {remaining.count} impossible ERAs remaining")

        if remaining.count == 0:
            print("SUCCESS: All impossible ERAs have been fixed")
        else:
            print("WARNING: Some impossible ERAs remain")

    except Exception as e:
        db.rollback()
        print(f"ERROR: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
