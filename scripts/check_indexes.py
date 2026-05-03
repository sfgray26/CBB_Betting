from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    indexes = conn.execute(text('''
      SELECT tablename, indexname
      FROM pg_indexes
      WHERE tablename IN ('player_projections', 'player_scores', 'statcast_performances')
      ORDER BY tablename, indexname
    ''')).fetchall()

    print(f"{'TABLE':30} {'INDEX':40}")
    print("-" * 70)
    for t in indexes:
        print(f"{t[0]:30} {t[1]:40}")
