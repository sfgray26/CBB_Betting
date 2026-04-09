"""Quick database table audit - run on Railway with: railway run python scripts/audit_db_tables.py"""
import os
from sqlalchemy import create_engine, inspect, text

# Get DATABASE_URL from Railway environment
db_url = os.environ.get('DATABASE_URL')
if not db_url:
    print("ERROR: DATABASE_URL not found")
    exit(1)

# Fix postgres:// to postgresql+psycopg2:// for SQLAlchemy
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql+psycopg2://', 1)

engine = create_engine(db_url)

# Get all tables
inspector = inspect(engine)
tables = sorted(inspector.get_table_names())

print("=" * 80)
print("RAILWAY DATABASE TABLE AUDIT")
print("=" * 80)
print(f"\nTotal tables: {len(tables)}\n")

# Check each table
with engine.connect() as conn:
    for table in tables:
        try:
            result = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
            count = result.scalar()

            if count == 0:
                print(f"⚠️  {table:50s} EMPTY")
            else:
                print(f"✓ {table:50s} {count:>10,} rows")
        except Exception as e:
            print(f"❌ {table:50s} ERROR: {str(e)[:40]}")

print("\n" + "=" * 80)
