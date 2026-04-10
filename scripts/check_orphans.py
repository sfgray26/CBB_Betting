"""Quick diagnostic script to check orphaned position_eligibility records"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.models import SessionLocal, PlayerIDMapping, PositionEligibility
from sqlalchemy import func

def main():
    db = SessionLocal()

    try:
        # Count orphaned records
        total_elig = db.query(func.count(PositionEligibility.id)).scalar()
        linked = db.query(func.count(PositionEligibility.id)).join(
            PlayerIDMapping, PositionEligibility.bdl_player_id == PlayerIDMapping.bdl_id
        ).scalar()
        orphan_count = total_elig - linked

        print(f'Total position_eligibility records: {total_elig}')
        print(f'Linked to player_id_mapping: {linked}')
        print(f'Orphaned records: {orphan_count}')

        # Sample some orphan names
        orphan_players = db.query(PositionEligibility).outerjoin(
            PlayerIDMapping, PositionEligibility.bdl_player_id == PlayerIDMapping.bdl_id
        ).filter(PlayerIDMapping.bdl_id.is_(None)).limit(10).all()

        print('\nSample orphaned player_name values:')
        for p in orphan_players:
            print(f'  {p.player_name} (yahoo_key: {p.yahoo_player_key})')

        # Check if there are PlayerIDMapping records
        mapping_count = db.query(func.count(PlayerIDMapping.id)).scalar()
        print(f'\nTotal player_id_mapping records: {mapping_count}')

        # Sample some mapping names
        mappings = db.query(PlayerIDMapping).limit(10).all()
        print('Sample player_id_mapping names:')
        for m in mappings:
            print(f'  {m.full_name} (bdl_id: {m.bdl_id})')

    finally:
        db.close()

if __name__ == "__main__":
    main()
