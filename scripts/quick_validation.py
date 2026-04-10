"""Quick validation check for Railway."""
import sys
sys.path.insert(0, ".")

from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()

try:
    # Check key tables
    tables = ['player_id_mapping', 'position_eligibility', 'mlb_player_stats']

    for table in tables:
        result = db.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
        print(f'{table}: {result:,} rows')

    # Check yahoo_key population
    result = db.execute(text('SELECT COUNT(yahoo_key) FROM player_id_mapping')).scalar()
    print(f'player_id_mapping.yahoo_key populated: {result:,}')

    # Check bdl_player_id population
    result = db.execute(text('SELECT COUNT(bdl_player_id) FROM position_eligibility')).scalar()
    print(f'position_eligibility.bdl_player_id populated: {result:,}')

    print('✅ Validation complete')

finally:
    db.close()
