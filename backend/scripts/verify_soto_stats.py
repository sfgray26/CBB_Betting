"""
Verify Juan Soto stats - check if he has stats under different BDL ID.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.models import SessionLocal, MLBPlayerStats, PlayerIDMapping


def main():
    db = SessionLocal()

    print("="*60)
    print("JUAN SOTO DATA VERIFICATION")
    print("="*60)

    # Find all Juan Soto mappings
    mappings = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.full_name.ilike("%juan soto%")
    ).all()

    print(f"\nFound {len(mappings)} Juan Soto mappings:")
    for m in mappings:
        print(f"\n  Mapping:")
        print(f"    full_name: {m.full_name}")
        print(f"    yahoo_key: {m.yahoo_key}")
        print(f"    bdl_id: {m.bdl_id}")
        print(f"    mlbam_id: {m.mlbam_id}")

        # Check stats for each BDL ID
        if m.bdl_id:
            stats = db.query(MLBPlayerStats).filter(
                MLBPlayerStats.bdl_player_id == m.bdl_id
            ).count()
            print(f"    MLBPlayerStats rows: {stats}")

            if stats > 0:
                latest = db.query(MLBPlayerStats).filter(
                    MLBPlayerStats.bdl_player_id == m.bdl_id
                ).order_by(MLBPlayerStats.game_date.desc()).limit(1).first()
                print(f"    Latest game: {latest.game_date}")
                print(f"    Recent stats: AB={latest.ab}, H={latest.hits}, HR={latest.home_runs}")

    # Check if ANY stats exist for Juan Soto by name in raw_payload
    print(f"\n" + "="*60)
    print("Checking raw_payload for 'Soto' mentions...")
    result = db.execute(text("""
        SELECT bdl_player_id, COUNT(*) as cnt
        FROM mlb_player_stats
        WHERE raw_payload::text ILIKE '%Soto%'
        GROUP BY bdl_player_id
        LIMIT 10
    """)).fetchall()

    print(f"Found {len(result)} BDL IDs with 'Soto' in stats:")
    for row in result:
        bdl_id, cnt = row
        print(f"  BDL ID {bdl_id}: {cnt} games")

        # Check who this BDL ID belongs to
        mapping = db.query(PlayerIDMapping).filter(
            PlayerIDMapping.bdl_id == bdl_id
        ).first()
        if mapping:
            print(f"    -> Belongs to: {mapping.full_name}")

    db.close()


if __name__ == "__main__":
    main()
