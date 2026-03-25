#!/usr/bin/env python3
"""
Apply OpenClaw v4.0 database migration.

Usage:
    python scripts/migrations/apply_openclaw_migration.py
    
    # With specific database URL:
    DATABASE_URL=postgresql://... python scripts/migrations/apply_openclaw_migration.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment."""
    url = os.getenv('DATABASE_URL')
    if not url:
        # Try to load from .env file
        env_path = project_root / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith('DATABASE_URL='):
                        url = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
    
    if not url:
        logger.error("DATABASE_URL not found in environment or .env file")
        sys.exit(1)
    
    return url


def apply_migration():
    """Apply the SQL migration."""
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)
    
    db_url = get_database_url()
    migration_path = Path(__file__).parent / 'v8_openclaw_monitoring.sql'
    
    if not migration_path.exists():
        logger.error(f"Migration file not found: {migration_path}")
        sys.exit(1)
    
    logger.info(f"Applying migration: {migration_path.name}")
    
    # Read migration SQL
    with open(migration_path) as f:
        sql = f.read()
    
    # Connect and execute
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = False
        cursor = conn.cursor()
        
        # Execute migration
        cursor.execute(sql)
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN (
                'model_performance_metrics',
                'vulnerability_reports', 
                'learning_journal',
                'roadmap_state'
            )
        """)
        
        created_tables = [row[0] for row in cursor.fetchall()]
        
        conn.commit()
        
        logger.info(f"✅ Migration applied successfully!")
        logger.info(f"   Created/verified tables: {', '.join(created_tables)}")
        
        # Show current Guardian status
        cursor.execute("""
            SELECT value, metadata 
            FROM model_performance_metrics 
            WHERE sport = 'system' AND metric_type = 'guardian_status'
        """)
        row = cursor.fetchone()
        if row:
            import json
            metadata = json.loads(row[1])
            logger.info(f"   Guardian freeze: ACTIVE until {metadata.get('freeze_until', 'N/A')}")
            logger.info(f"   Current phase: {metadata.get('phase', 'N/A')}")
        
        cursor.close()
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    apply_migration()
