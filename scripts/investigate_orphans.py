#!/usr/bin/env python
"""Investigate orphaned position_eligibility records."""

import sys
import os

# Add backend to path for Railway
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.models import SessionLocal
from sqlalchemy import text

def main():
    try:
        db = SessionLocal()

        # Count total orphans
        count_result = db.execute(text('''
            SELECT COUNT(DISTINCT pe.id)
            FROM position_eligibility pe
            LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
            WHERE pe.yahoo_player_key IS NOT NULL AND pim.yahoo_key IS NULL
        ''')).scalar()
        print(f'Total orphaned records: {count_result}')

        # Get sample orphans
        result = db.execute(text('''
            SELECT pe.id, pe.player_name, pe.yahoo_player_key, pe.primary_position
            FROM position_eligibility pe
            LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
            WHERE pe.yahoo_player_key IS NOT NULL AND pim.yahoo_key IS NULL
            LIMIT 15
        ''')).fetchall()

        print(f'\nSample orphaned records:')
        for row in result:
            print(f'  ID={row.id} | {row.player_name} | {row.yahoo_player_key} | {row.primary_position}')

        # Get player_id_mapping sample for comparison
        print(f'\nSample player_id_mapping records:')
        mapping_result = db.execute(text('''
            SELECT id, yahoo_key, bdl_player_name, bdl_player_id
            FROM player_id_mapping
            WHERE yahoo_key IS NOT NULL AND bdl_player_name IS NOT NULL
            LIMIT 10
        ''')).fetchall()
        for row in mapping_result:
            print(f'  ID={row.id} | {row.bdl_player_name} | yahoo_key={row.yahoo_key} | bdl_id={row.bdl_player_id}')

        db.close()

    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
