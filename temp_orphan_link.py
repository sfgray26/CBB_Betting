#!/usr/bin/env python
"""Temporary script to execute orphan linking on Railway"""
from backend.models import SessionLocal
from backend.fantasy_baseball.orphan_linker import link_orphaned_records

db = SessionLocal()
print('Starting orphan linking...')
print('Before: Checking orphan count...')

# Count before
from sqlalchemy import text
before = db.execute(text('''
    SELECT COUNT(*) FROM position_eligibility pe
    LEFT JOIN player_id_mapping pim ON pe.bdl_player_id = pim.id
    WHERE pe.yahoo_player_key IS NOT NULL AND pe.bdl_player_id IS NULL
''')).scalar()
print(f'Orphans before linking: {before}')

# Execute linking
result = link_orphaned_records(db, dry_run=False, verbose=False)
print(f'Linked: {result["linked_count"]}')
print(f'Remaining: {result["remaining_count"]}')
print(f'Success rate: {result["success_rate"]:.1f}%')

# Sample linked records
sample = db.execute(text('''
    SELECT pe.player_name, pim.full_name, pe.bdl_player_id
    FROM position_eligibility pe
    JOIN player_id_mapping pim ON pe.bdl_player_id = pim.id
    WHERE pe.bdl_player_id IS NOT NULL
    ORDER BY pe.id DESC
    LIMIT 5
''')).fetchall()
print('\nSample linked records:')
for row in sample:
    print(f'  {row.player_name} -> {row.full_name}')

db.close()
