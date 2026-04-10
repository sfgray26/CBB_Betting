from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()

result = db.execute(text('''
    SELECT
        pe.player_name,
        pe.bdl_player_id,
        COUNT(ms.id) as stat_count
    FROM position_eligibility pe
    LEFT JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
    WHERE pe.bdl_player_id IS NOT NULL
    GROUP BY pe.player_name, pe.bdl_player_id
    ORDER BY stat_count DESC
    LIMIT 10
''')).fetchall()

print('Top 10 players with stat counts:')
for row in result:
    print(f'  {row.player_name}: {row.stat_count} games')

db.close()
