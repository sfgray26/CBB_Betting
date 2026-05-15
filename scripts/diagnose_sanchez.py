"""
Diagnostic script to investigate Christopher Sanchez data pipeline issue.

Run on Railway: railway run python scripts/diagnose_sanchez.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from urllib.parse import urlparse

def get_database_url():
    """Get DATABASE_URL from environment with fallback."""
    # Prefer DATABASE_PUBLIC_URL for local development (resolves from local machine)
    # Fall back to DATABASE_URL for Railway execution (internal hostname)
    db_url = os.getenv('DATABASE_PUBLIC_URL') or os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL or DATABASE_PUBLIC_URL environment variable not set")
    return db_url

def main():
    print("=" * 80)
    print("CHRISTOPHER SANCHEZ DATA PIPELINE DIAGNOSTIC")
    print("=" * 80)

    db_url = get_database_url()
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Query 1: Check player_id_mapping
        print("\n1. Christopher Sanchez in player_id_mapping")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT player_id, yahoo_id, yahoo_key, bdl_id, name, normalized_name
            FROM player_id_mapping
            WHERE normalized_name LIKE '%sanchez%' OR name LIKE '%sanchez%'
            ORDER BY name
        """))
        rows = result.fetchall()
        if rows:
            for r in rows:
                print(f"  {r}")
        else:
            print("  NOT FOUND")

        # Query 2: Check player_identities (raw Yahoo sync)
        print("\n2. Christopher Sanchez in player_identities")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT player_id, yahoo_player_id, yahoo_player_key, name, normalized_name
            FROM player_identities
            WHERE normalized_name LIKE '%sanchez%' OR name LIKE '%sanchez%'
            ORDER BY name
        """))
        rows = result.fetchall()
        if rows:
            for r in rows:
                print(f"  {r}")
        else:
            print("  NOT FOUND")

        # Query 3: Check canonical_projections
        print("\n3. Christopher Sanchez in canonical_projections")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT player_id, name, projection_source, player_type
            FROM canonical_projections
            WHERE name LIKE '%sanchez%'
            ORDER BY name
        """))
        rows = result.fetchall()
        if rows:
            for r in rows:
                print(f"  {r}")
        else:
            print("  NOT FOUND")

        # Query 4: Check yahoo_id_sync job status
        print("\n4. yahoo_id_sync job status (last 5 runs)")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT status, records_processed, error_message, started_at
            FROM job_runs
            WHERE job_name = 'yahoo_id_sync'
            ORDER BY started_at DESC
            LIMIT 5
        """))
        rows = result.fetchall()
        for r in rows:
            print(f"  Status: {r[0]}, Processed: {r[1]}, Error: {r[2]}, Started: {r[3]}")

        # Query 5: Row counts
        print("\n5. Table row counts")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM player_id_mapping) as player_id_mapping,
                (SELECT COUNT(*) FROM player_identities) as player_identities,
                (SELECT COUNT(*) FROM canonical_projections) as canonical_projections
        """))
        r = result.fetchone()
        print(f"  player_id_mapping: {r[0]}")
        print(f"  player_identities: {r[1]}")
        print(f"  canonical_projections: {r[2]}")

        # Query 6: Check if there are any players with 'Sanchez' that we can find
        print("\n6. All players with 'Sanchez' in name")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT name, player_id, player_type, projection_source
            FROM canonical_projections
            WHERE name ILIKE '%sanchez%'
            ORDER BY name
        """))
        rows = result.fetchall()
        if rows:
            for r in rows:
                print(f"  {r}")
        else:
            print("  NOT FOUND")

        # Query 7: Check for any players named Christopher
        print("\n7. All players with first name 'Christopher'")
        print("-" * 80)
        result = conn.execute(text("""
            SELECT name, player_id, player_type, projection_source
            FROM canonical_projections
            WHERE name ILIKE 'Christopher %'
            ORDER BY name
        """))
        rows = result.fetchall()
        if rows:
            print(f"  Found {len(rows)} players")
            for r in rows[:10]:  # Show first 10
                print(f"  {r}")
            if len(rows) > 10:
                print(f"  ... and {len(rows) - 10} more")
        else:
            print("  NOT FOUND")

if __name__ == "__main__":
    main()
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
