"""
Quick script to check database state after sync job execution.
Run with: railway run -- python scripts/check_db_state.py
"""

from backend.models import SessionLocal, PlayerIDMapping, PositionEligibility, ProbablePitcherSnapshot
from sqlalchemy import func

db = SessionLocal()

print('=== DATABASE STATE AFTER JOB EXECUTION ===')
print()

player_id_count = db.query(PlayerIDMapping).count()
print(f'PlayerIDMapping: {player_id_count:,} total rows')

if player_id_count > 0:
    non_null_yahoo = db.query(PlayerIDMapping).filter(PlayerIDMapping.yahoo_id.isnot(None)).count()
    non_null_mlbam = db.query(PlayerIDMapping).filter(PlayerIDMapping.mlbam_id.isnot(None)).count()
    print(f'  non-null yahoo_id: {non_null_yahoo:,} ({non_null_yahoo*100/player_id_count:.1f}%)')
    print(f'  non-null mlbam_id: {non_null_mlbam:,} ({non_null_mlbam*100/player_id_count:.1f}%)')

print()
position_count = db.query(PositionEligibility).count()
print(f'PositionEligibility: {position_count:,} rows')

pitcher_count = db.query(ProbablePitcherSnapshot).count()
print(f'ProbablePitcherSnapshot: {pitcher_count:,} rows')

if player_id_count > 0:
    print()
    print('=== SAMPLE PLAYER ID MAPPINGS ===')
    samples = db.query(PlayerIDMapping).limit(5).all()
    for i, p in enumerate(samples, 1):
        print(f'{i}. BDL:{p.bdl_id} | Name:{p.full_name} | MLBAM:{p.mlbam_id} | Yahoo:{p.yahoo_id}')

db.close()
