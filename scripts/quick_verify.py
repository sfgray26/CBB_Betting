import sys; sys.path.insert(0, '.')
from backend.models import SessionLocal
from sqlalchemy import text
from datetime import datetime
from zoneinfo import ZoneInfo

db = SessionLocal()

# Check 1: Count remaining numeric names
r = db.execute(text('SELECT COUNT(*) FROM player_projections WHERE player_name ~ \'^[0-9]+$\''))
print(f'Remaining numeric names: {r.scalar()}')

# Check 2: Test datetime subtraction (the crash fix)
r = db.execute(text('SELECT MAX(computed_at) FROM player_daily_metrics WHERE metric_date IS NOT NULL'))
latest = r.scalar()
if latest:
    now = datetime.now(ZoneInfo('America/New_York'))
    age_days = (now - latest).days
    print(f'Latest metrics: {latest}, Age: {age_days} days')
    print('DateTime fix: PASSED')
else:
    print('No metrics data found')

# Check 3: Sample orphan players
r = db.execute(text('''
    SELECT DISTINCT player_name, COUNT(*) as cnt
    FROM player_projections
    WHERE player_name ~ \'^[0-9]+\'
    GROUP BY player_name
    ORDER BY cnt DESC
    LIMIT 5
'''))
print('Sample orphan players (player_id: row_count):')
for row in r:
    print(f'  {row[0]}: {row[1]} rows')

db.close()
