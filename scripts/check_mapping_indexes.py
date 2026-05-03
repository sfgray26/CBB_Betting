from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    indexes = conn.execute(text('''
      SELECT tablename, indexname
      FROM pg_indexes
      WHERE tablename = 'player_id_mapping'
      ORDER BY tablename, indexname
    ''')).fetchall()

    for t in indexes:
        print(f"{t[0]:30} {t[1]:40}")
