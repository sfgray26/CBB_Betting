import os
from sqlalchemy import create_engine, text

e = create_engine(os.environ["DATABASE_URL"])
with e.connect() as c:
    # List tables with row counts
    r = c.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
    ))
    tables = [row[0] for row in r]
    print(f"Tables ({len(tables)}):")
    for t in tables:
        ct = c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        print(f"  {t}: {ct} rows")
