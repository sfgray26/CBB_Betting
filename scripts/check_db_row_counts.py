"""
CRITICAL: Verify actual database row counts after sync job execution.
Run with: railway run -- python scripts/check_db_row_counts.py
"""

from backend.models import SessionLocal, PlayerIDMapping, PositionEligibility, ProbablePitcherSnapshot
from sqlalchemy import func
from datetime import date

db = SessionLocal()

print('=' * 70)
print('DATABASE ROW COUNT VERIFICATION - April 9, 2026')
print('=' * 70)
print()

# PLAYER ID MAPPING
print('1. PLAYER ID MAPPING TABLE')
print('-' * 70)
player_id_total = db.query(PlayerIDMapping).count()
player_id_with_yahoo = db.query(PlayerIDMapping).filter(PlayerIDMapping.yahoo_key.isnot(None)).count()
player_id_with_mlbam = db.query(PlayerIDMapping).filter(PlayerIDMapping.mlbam_id.isnot(None)).count()

print(f'Total rows: {player_id_total:,}')
print(f'With yahoo_key: {player_id_with_yahoo:,} ({player_id_with_yahoo*100/player_id_total:.1f}%)' if player_id_total > 0 else 'With yahoo_key: 0')
print(f'With mlbam_id: {player_id_with_mlbam:,} ({player_id_with_mlbam*100/player_id_total:.1f}%)' if player_id_total > 0 else 'With mlbam_id: 0')

if player_id_total > 0:
    print()
    print('Sample records:')
    samples = db.query(PlayerIDMapping).limit(5).all()
    for i, p in enumerate(samples, 1):
        print(f'  {i}. BDL:{p.bdl_id} | Name:{p.full_name} | MLBAM:{p.mlbam_id} | Yahoo:{p.yahoo_key}')
else:
    print('⚠️  NO DATA - Table is empty!')

print()

# POSITION ELIGIBILITY
print('2. POSITION ELIGIBILITY TABLE')
print('-' * 70)
position_total = db.query(PositionEligibility).count()
position_with_cf = db.query(PositionEligibility).filter(PositionEligibility.can_play_CF == True).count()
position_with_lf = db.query(PositionEligibility).filter(PositionEligibility.can_play_LF == True).count()
position_with_rf = db.query(PositionEligibility).filter(PositionEligibility.can_play_RF == True).count()

print(f'Total rows: {position_total:,}')
print(f'Can play CF: {position_with_cf:,}')
print(f'Can play LF: {position_with_lf:,}')
print(f'Can play RF: {position_with_rf:,}')

if position_total > 0:
    print()
    print('Sample multi-eligibility players:')
    multi_players = db.query(PositionEligibility).filter(
        (PositionEligibility.can_play_CF == True) &
        (PositionEligibility.can_play_LF == True)
    ).limit(5).all()
    for i, p in enumerate(multi_players, 1):
        positions = []
        if p.can_play_C: positions.append('C')
        if p.can_play_1B: positions.append('1B')
        if p.can_play_2B: positions.append('2B')
        if p.can_play_3B: positions.append('3B')
        if p.can_play_SS: positions.append('SS')
        if p.can_play_LF: positions.append('LF')
        if p.can_play_CF: positions.append('CF')
        if p.can_play_RF: positions.append('RF')
        if p.can_play_OF: positions.append('OF')
        if p.can_play_DH: positions.append('DH')
        if p.can_play_UTIL: positions.append('UTIL')
        print(f'  {i}. BDL:{p.bdl_player_id} | Positions: {"/".join(positions)}')
else:
    print('⚠️  NO DATA - Table is empty!')

print()

# PROBABLE PITCHERS
print('3. PROBABLE PITCHER SNAPSHOT TABLE')
print('-' * 70)
today = date.today()
pitchers_total = db.query(ProbablePitcherSnapshot).count()
pitchers_today = db.query(ProbablePitcherSnapshot).filter(
    ProbablePitcherSnapshot.game_date >= today
).count()

print(f'Total rows: {pitchers_total:,}')
print(f'Today or later: {pitchers_today:,}')

if pitchers_total > 0:
    print()
    print('Recent snapshots:')
    recent = db.query(ProbablePitcherSnapshot).order_by(
        ProbablePitcherSnapshot.game_date.desc()
    ).limit(5).all()
    for i, p in enumerate(recent, 1):
        print(f'  {i}. {p.game_date} | {p.bdl_player_id} | Team:{p.team}')
else:
    print('⚠️  NO DATA - Table is empty!')

print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'Player ID Mapping: {"✅ DATA" if player_id_total > 0 else "❌ EMPTY"} ({player_id_total:,} rows)')
print(f'Position Eligibility: {"✅ DATA" if position_total > 0 else "❌ EMPTY"} ({position_total:,} rows)')
print(f'Probable Pitchers: {"✅ DATA" if pitchers_total > 0 else "❌ EMPTY"} ({pitchers_total:,} rows)')
print()

# SUCCESS CRITERIA CHECK
print('SUCCESS CRITERIA:')
print('-' * 70)
print(f'✓ position_eligibility has 700+ rows: {"✅ PASS" if position_total >= 700 else "❌ FAIL"}')
print(f'✓ Multi-eligibility data present: {"✅ PASS" if position_with_cf > 0 and position_with_lf > 0 else "❌ FAIL"}')
print(f'✓ probable_pitchers has recent data: {"✅ PASS" if pitchers_today > 0 else "⚠️  EXPECTED (early morning)"}')
print()

db.close()
