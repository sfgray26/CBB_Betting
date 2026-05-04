from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    tables = conn.execute(text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'")).fetchall()
    print("Tables in public schema:")
    for t in tables:
        print(f" - {t[0]}")
