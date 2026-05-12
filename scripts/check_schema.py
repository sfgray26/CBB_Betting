import os
import sqlalchemy
from sqlalchemy import text

def check_schema():
    url = os.environ.get('DATABASE_URL').replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        print('--- Schema and Table Check ---')
        sql = "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('information_schema', 'pg_catalog') ORDER BY table_schema, table_name"
        rows = conn.execute(text(sql)).fetchall()
        for r in rows:
            print(f'{r[0]} | {r[1]}')

if __name__ == "__main__":
    check_schema()
