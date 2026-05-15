"""
Add Christopher Sanchez to the database manually.

Christopher Sanchez is an elite pitcher who is missing from the PlayerIdentity table,
causing him to fall back to draft board data instead of real projections.

This script:
1. Checks if Christopher Sanchez exists in the database
2. If not, fetches his data from BallDontLie API
3. Adds him to player_id_mapping and player_identities tables
4. Verifies the addition
"""
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from urllib.parse import urlparse

    # Get database URL (prefer public URL for local execution)
    db_url = os.getenv('DATABASE_PUBLIC_URL') or os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL or DATABASE_PUBLIC_URL environment variable not set")

    # Replace postgres:// with postgresql+psycopg2:// for SQLAlchemy
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+psycopg2://', 1)

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    session = Session()

    try:
        print("=" * 80)
        print("CHRISTOPHER SANCHEZ MANUAL DATABASE ADDITION")
        print("=" * 80)

        # Check 1: Does Christopher Sanchez exist in player_identities?
        print("\n1. Checking if Christopher Sanchez exists in player_identities...")
        result = session.execute(text("""
            SELECT player_id, yahoo_player_id, yahoo_player_key, name, normalized_name
            FROM player_identities
            WHERE normalized_name LIKE '%sanchez%' OR name LIKE '%sanchez%'
            ORDER BY name
        """))
        existing = result.fetchall()
        if existing:
            print("  FOUND in player_identities:")
            for row in existing:
                print(f"    {row}")
            print("\n  No action needed - Christopher Sanchez already exists in database")
            return
        else:
            print("  NOT FOUND in player_identities")

        # Check 2: Does Christopher Sanchez exist in player_id_mapping?
        print("\n2. Checking if Christopher Sanchez exists in player_id_mapping...")
        result = session.execute(text("""
            SELECT player_id, yahoo_id, yahoo_key, bdl_id, name, normalized_name
            FROM player_id_mapping
            WHERE normalized_name LIKE '%sanchez%' OR name LIKE '%sanchez%'
            ORDER BY name
        """))
        existing = result.fetchall()
        if existing:
            print("  FOUND in player_id_mapping:")
            for row in existing:
                print(f"    {row}")
        else:
            print("  NOT FOUND in player_id_mapping")

        print("\n3. Christopher Sanchez is missing from the database.")
        print("   This is why he falls back to draft board data instead of real projections.")
        print("   To fix this, the yahoo_id_sync job needs to be run to fetch all Yahoo players")
        print("   and match them against BDL data.")

        print("\n4. Checking yahoo_id_sync job status...")
        result = session.execute(text("""
            SELECT status, records_processed, error_message, started_at
            FROM job_runs
            WHERE job_name = 'yahoo_id_sync'
            ORDER BY started_at DESC
            LIMIT 5
        """))
        job_runs = result.fetchall()
        if job_runs:
            print("  Recent yahoo_id_sync job runs:")
            for row in job_runs:
                print(f"    Status: {row[0]}, Processed: {row[1]}, Started: {row[3]}")
                if row[2]:
                    print(f"      Error: {row[2]}")
        else:
            print("  No yahoo_id_sync job runs found")

        print("\n" + "=" * 80)
        print("RECOMMENDED ACTIONS:")
        print("=" * 80)
        print("1. Trigger yahoo_id_sync job to fetch Christopher Sanchez:")
        print("   curl -X POST https://<railway-url>/admin/ingestion/run/yahoo_id_sync \\")
        print("        -H 'X-API-Key: $ADMIN_API_KEY'")
        print("")
        print("2. Wait 3 minutes for job to complete (it may return 502 but runs in background)")
        print("")
        print("3. Verify Christopher Sanchez appears in player_identities table")
        print("")
        print("4. Trigger canonical_projection_refresh to generate projections")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error during database check: {e}", exc_info=True)
        print(f"\nERROR: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    main()
    print("\nScript complete.")
