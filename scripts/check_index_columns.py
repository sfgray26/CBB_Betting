from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    indexes = conn.execute(text('''
      SELECT
          t.relname AS table_name,
          i.relname AS index_name,
          a.attname AS column_name
      FROM
          pg_class t,
          pg_class i,
          pg_index ix,
          pg_attribute a
      WHERE
          t.oid = ix.indrelid
          AND i.oid = ix.indexrelid
          AND a.attrelid = t.oid
          AND a.attnum = ANY(ix.indkey)
          AND t.relkind = 'r'
          AND t.relname IN ('player_projections', 'player_scores', 'statcast_performances')
      ORDER BY
          t.relname,
          i.relname;
    ''')).fetchall()

    print(f"{'TABLE':25} {'INDEX':40} {'COLUMN'}")
    print("-" * 80)
    for t in indexes:
        print(f"{t[0]:25} {t[1]:40} {t[2]}")
