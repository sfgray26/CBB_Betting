"""
Check NO_GAME_DATA players - verify if they have stats under correct BDL ID.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.models import SessionLocal, MLBPlayerStats, PlayerIDMapping


def main():
    db = SessionLocal()

    print("="*60)
    print("NO_GAME_DATA PLAYER VERIFICATION")
    print("="*60)

    # NO_GAME_DATA players - have MLBAM ID but no BDL ID
    players = [
        ("Jordan Walker", 691023),
        ("Carson Benge", 701807),
        ("Munetaka Murakami", 808959),
    ]

    for name, mlbam_id in players:
        print(f"\n{'='*60}")
        print(f"Checking: {name} (MLBAM ID: {mlbam_id})")
        print(f"{'='*60}")

        # Find all mappings
        mappings = db.query(PlayerIDMapping).filter(
            PlayerIDMapping.mlbam_id == mlbam_id
        ).all()

        print(f"Found {len(mappings)} mappings:")
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

        # Check if stats exist under MLBAM ID (in case BDL ID = MLBAM ID)
        print(f"\n  Checking if stats exist under mlbam_id={mlbam_id}...")
        stats_by_mlbam = db.query(MLBPlayerStats).filter(
            MLBPlayerStats.bdl_player_id == mlbam_id
        ).count()
        print(f"    Stats under bdl_player_id={mlbam_id}: {stats_by_mlbam}")

        # Also search by name in raw_payload
        result = db.execute(text("""
            SELECT bdl_player_id, COUNT(*) as cnt
            FROM mlb_player_stats
            WHERE raw_payload::text ILIKE :name_pattern
            GROUP BY bdl_player_id
            LIMIT 5
        """), {"name_pattern": f"%{name.split()[0]}%{name.split()[1]}%" if ' ' in name else f"%{name}%"}).fetchall()

        if result:
            print(f"\n  Found {len(result)} BDL IDs with '{name}' in stats:")
            for row in result:
                bdl_id, cnt = row
                print(f"    BDL ID {bdl_id}: {cnt} games")

                # Check who this belongs to
                mapping = db.query(PlayerIDMapping).filter(
                    PlayerIDMapping.bdl_id == bdl_id
                ).first()
                if mapping:
                    print(f"      -> Belongs to: {mapping.full_name}")

    db.close()


if __name__ == "__main__":
    main()
