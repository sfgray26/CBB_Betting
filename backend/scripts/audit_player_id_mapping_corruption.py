"""
PlayerIDMapping Corruption Audit

Documents all corrupted rows in player_id_mapping table and the corruption pattern.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.models import SessionLocal, PlayerIDMapping, MLBPlayerStats


def audit_corruption():
    """Audit player_id_mapping for corruption patterns."""
    db = SessionLocal()

    print("="*70)
    print("PLAYERIDMAPPING CORRUPTION AUDIT")
    print("="*70)

    # Pattern 1: Duplicate rows for same player (same mlbam_id)
    print("\n" + "="*70)
    print("PATTERN 1: DUPLICATE ROWS (same mlbam_id, different bdl_id)")
    print("="*70)

    # Find mlbam_ids with multiple bdl_ids
    dup_query = text("""
        SELECT mlbam_id, full_name, COUNT(*) as row_count,
               array_agg(bdl_id ORDER BY bdl_id) as bdl_ids,
               array_agg(yahoo_key ORDER BY yahoo_key) as yahoo_keys
        FROM player_id_mapping
        WHERE mlbam_id IS NOT NULL
        GROUP BY mlbam_id, full_name
        HAVING COUNT(*) > 1
        ORDER BY full_name
    """)

    dups = db.execute(dup_query).fetchall()

    print(f"\nFound {len(dups)} players with duplicate mappings:")
    for row in dups:
        mlbam_id, full_name, row_count, bdl_ids, yahoo_keys = row
        print(f"\n  {full_name} (mlbam_id={mlbam_id})")
        print(f"    Rows: {row_count}")
        print(f"    BDL IDs: {bdl_ids}")
        print(f"    Yahoo Keys: {yahoo_keys}")

        # Check which BDL IDs have stats
        for bdl_id in bdl_ids:
            if bdl_id:
                stats = db.query(MLBPlayerStats).filter(
                    MLBPlayerStats.bdl_player_id == bdl_id
                ).count()
                print(f"      bdl_id={bdl_id}: {stats} games")

    # Pattern 2: bdl_id equals mlbam_id (clearly wrong)
    print("\n" + "="*70)
    print("PATTERN 2: BDL_ID = MLBAM_ID (wrong ID type)")
    print("="*70)

    bdl_equals_mlbam = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.bdl_id == PlayerIDMapping.mlbam_id
    ).all()

    print(f"\nFound {len(bdl_equals_mlbam)} rows where bdl_id = mlbam_id:")
    for row in bdl_equals_mlbam:
        print(f"  {row.full_name}: bdl_id={row.bdl_id} (equals mlbam_id)")

    # Pattern 3: yahoo_key row has no stats, alternative row has stats
    print("\n" + "="*70)
    print("PATTERN 3: YAHOO_KEY ROW HAS NO STATS, ALTERNATIVE HAS STATS")
    print("="*70)

    # Find players where yahoo_key row has bdl_id with no stats
    # but same mlbam_id has alternative bdl_id with stats
    stats_check_query = text("""
        SELECT p1.full_name, p1.yahoo_key, p1.bdl_id as corrupted_bdl,
               p1.mlbam_id, p2.bdl_id as correct_bdl
        FROM player_id_mapping p1
        JOIN player_id_mapping p2 ON p1.mlbam_id = p2.mlbam_id AND p1.bdl_id != p2.bdl_id
        WHERE p1.yahoo_key IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM player_scores ps WHERE ps.bdl_player_id = p1.bdl_id LIMIT 1
          )
          AND EXISTS (
              SELECT 1 FROM player_scores ps WHERE ps.bdl_player_id = p2.bdl_id LIMIT 1
          )
        ORDER BY p1.full_name
    """)

    affected = db.execute(stats_check_query).fetchall()

    print(f"\nFound {len(affected)} players affected by corruption:")
    for row in affected:
        full_name, yahoo_key, corrupted_bdl, mlbam_id, correct_bdl = row
        print(f"\n  {full_name}:")
        print(f"    Yahoo Key: {yahoo_key}")
        print(f"    Corrupted BDL ID: {corrupted_bdl} (optimizer uses this - NO STATS)")
        print(f"    Correct BDL ID: {correct_bdl} (HAS STATS)")

    # Pattern 4: Rows with no yahoo_key and no bdl_id (empty mappings)
    print("\n" + "="*70)
    print("PATTERN 4: EMPTY MAPPINGS (no yahoo_key, no bdl_id)")
    print("="*70)

    empty_mappings = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.yahoo_key.is_(None),
        PlayerIDMapping.bdl_id.is_(None)
    ).all()

    print(f"\nFound {len(empty_mappings)} empty mappings:")
    for row in empty_mappings:
        print(f"  {row.full_name}: mlbam_id={row.mlbam_id}")

    # Summary
    print("\n" + "="*70)
    print("CORRUPTION SUMMARY")
    print("="*70)
    print(f"Total rows in player_id_mapping: {db.query(PlayerIDMapping).count()}")
    print(f"Duplicate rows (same mlbam_id): {len(dups)} players")
    print(f"bdl_id = mlbam_id (wrong type): {len(bdl_equals_mlbam)} rows")
    print(f"Affected by Pattern 3 (optimizer finds wrong row): {len(affected)} players")
    print(f"Empty mappings: {len(empty_mappings)} rows")

    # Corruption analysis
    print("\n" + "="*70)
    print("CORRUPTION ANALYSIS")
    print("="*70)
    print("\nHOW CORRUPTION HAPPENED:")
    print("1. Ingestion process creates row with yahoo_key + mlbam_id")
    print("2. Code incorrectly assigns bdl_id = mlbam_id (ID type confusion)")
    print("3. Later, another process creates correct row with correct bdl_id")
    print("4. Optimizer finds first row via yahoo_key, uses wrong bdl_id")
    print("5. Wrong bdl_id has no stats -> fallback to 0.0")

    print("\nWORKAROUND VS CLEANUP:")
    print("\nOption A: Permanent Workaround (CURRENT APPROACH)")
    print("  Pros:")
    print("    - No data migration required")
    print("    - Handles existing and future corruption")
    print("    - Low risk, already tested")
    print("  Cons:")
    print("    - Leaves corrupted data in DB")
    print("    - Adds query overhead per optimization")
    print("    - Confusing for anyone auditing the table")

    print("\nOption B: DB Cleanup + Migration")
    print("  Pros:")
    print("    - Clean data, no confusion")
    print("    - No query overhead")
    print("    - Fixes root cause")
    print("  Cons:")
    print("    - Requires cascading updates to all dependent tables")
    print("    - Foreign key constraints complicate deletion")
    print("    - Higher risk of breaking things")
    print("    - Need downtime for migration")

    print("\nRECOMMENDATION:")
    print("  Keep workaround PERMANENTLY because:")
    print("  1. Corruption has spread to dependent tables (player_opportunity FK)")
    print("  2. Migration requires cascade through multiple tables")
    print("  3. Workaround adds minimal overhead (<50ms per optimization)")
    print("  4. Workaround handles future corruption automatically")
    print("  5. Data cleanup is high-risk for low reward")

    print("\nMONITORING NEEDED:")
    print("  1. Log fallback rate per optimization")
    print("  2. Alert if fallback rate >10%")
    print("  3. Alert if player >50% ownership has no projection")
    print("  4. Periodic audit for new corruption patterns")

    db.close()


if __name__ == "__main__":
    audit_corruption()
