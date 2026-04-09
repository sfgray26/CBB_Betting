"""Check for multiple databases in PostgreSQL instance"""
import os
from sqlalchemy import create_engine, text
from urllib.parse import urlparse

# Get DATABASE_URL from Railway environment
db_url = os.environ.get('DATABASE_URL')
if not db_url:
    print("ERROR: DATABASE_URL not found")
    exit(1)

# Fix postgres:// to postgresql+psycopg2:// for SQLAlchemy
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql+psycopg2://', 1)

# Connect to postgres default database to list all databases
# Extract connection info
parsed = urlparse(db_url)
# Connect to 'postgres' database to list all databases
admin_url = f"postgresql+psycopg2://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}/postgres"
engine = create_engine(admin_url)

print("=" * 80)
print("POSTGRESQL DATABASE LISTING")
print("=" * 80)

with engine.connect() as conn:
    # List all databases
    result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname"))
    databases = [row[0] for row in result]

    print(f"\nFound {len(databases)} databases:")
    for db in databases:
        print(f"  - {db}")

    # Check which databases have tables
    print("\n" + "=" * 80)
    print("TABLE COUNTS PER DATABASE")
    print("=" * 80)

    for db_name in databases:
        if db_name in ['postgres', 'template0', 'template1']:
            continue

        try:
            db_url_specific = f"postgresql+psycopg2://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}/{db_name}"
            engine_db = create_engine(db_url_specific)

            with engine_db.connect() as conn_db:
                try:
                    result = conn_db.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"))
                    table_count = result.scalar()
                    print(f"{db_name:30s} {table_count} tables")

                    # List tables if database has tables
                    if table_count > 0:
                        result = conn_db.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"))
                        tables = [row[0] for row in result]

                        # Check for our target tables
                        target_tables = ['position_eligibility', 'probable_pitchers']
                        found_targets = [t for t in target_tables if t in tables]
                        if found_targets:
                            print(f"  ✓ Found target tables: {found_targets}")

                except Exception as e:
                    print(f"{db_name:30s} ERROR: {str(e)[:50]}")
        except Exception as e:
            print(f"{db_name:30s} Cannot connect: {str(e)[:50]}")

print("\n" + "=" * 80)
print("CURRENT DATABASE_URL:")
print("=" * 80)
print(f"Database: {parsed.path[1:]}")  # Remove leading slash
print(f"Host: {parsed.hostname}")
print(f"Port: {parsed.port}")
print(f"User: {parsed.username}")
