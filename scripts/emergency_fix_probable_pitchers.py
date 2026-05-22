#!/usr/bin/env python3
"""
EMERGENCY FIX: Probable Pitchers Pipeline

Root Cause: Missing database constraint causing upserts to fail silently
Solution: Create constraint, run backfill, trigger sync
"""

import asyncio
import sys
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/backend')

from sqlalchemy import text
from backend.models import SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def emergency_fix():
    """Fix the probable_pitchers pipeline."""
    db = SessionLocal()
    
    try:
        logger.info("=" * 60)
        logger.info("EMERGENCY FIX: Probable Pitchers Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Check current state
        logger.info("\n1. Checking current probable_pitchers state...")
        result = db.execute(text("SELECT COUNT(*) FROM probable_pitchers")).scalar()
        logger.info(f"   Current count: {result} rows")
        
        if result > 0:
            today = db.execute(text("SELECT COUNT(*) FROM probable_pitchers WHERE game_date = CURRENT_DATE")).scalar()
            logger.info(f"   Today's games: {today}")
        
        # Step 2: Check if constraint exists
        logger.info("\n2. Checking database constraint...")
        constraint = db.execute(text("""
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = '_pp_date_team_uc'
              AND table_name = 'probable_pitchers'
        """)).fetchone()
        
        if constraint:
            logger.info("   ✓ Constraint exists")
        else:
            logger.info("   ✗ Constraint MISSING - creating now...")
            try:
                db.execute(text("""
                    ALTER TABLE probable_pitchers
                    ADD CONSTRAINT _pp_date_team_uc UNIQUE (game_date, team)
                """))
                db.commit()
                logger.info("   ✓ Constraint created successfully")
            except Exception as e:
                db.rollback()
                logger.error(f"   ✗ Failed to create constraint: {e}")
                return False
        
        # Step 3: Check table structure
        logger.info("\n3. Checking table structure...")
        columns = db.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'probable_pitchers'
            ORDER BY ordinal_position
        """)).fetchall()
        
        logger.info(f"   Found {len(columns)} columns:")
        for col in columns[:10]:  # Show first 10
            logger.info(f"     - {col[0]}: {col[1]}")
        
        # Step 4: Run sync
        logger.info("\n4. Running probable pitchers sync...")
        logger.info("   (This will fetch data from MLB Stats API)")
        
        from backend.services.daily_ingestion import DailyIngestionService
        service = DailyIngestionService(db)
        
        try:
            result = await service._sync_probable_pitchers()
            logger.info(f"\n   Sync result: {result}")
        except Exception as e:
            logger.error(f"   ✗ Sync failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 5: Verify fix
        logger.info("\n5. Verifying fix...")
        final_count = db.execute(text("SELECT COUNT(*) FROM probable_pitchers")).scalar()
        today_count = db.execute(text("SELECT COUNT(*) FROM probable_pitchers WHERE game_date = CURRENT_DATE")).scalar()
        
        logger.info(f"   Final count: {final_count} rows")
        logger.info(f"   Today's games: {today_count}")
        
        if final_count > 0:
            logger.info("\n" + "=" * 60)
            logger.info("✓ FIX SUCCESSFUL - Data restored!")
            logger.info("=" * 60)
            return True
        else:
            logger.error("\n" + "=" * 60)
            logger.error("✗ FIX FAILED - Still no data")
            logger.error("=" * 60)
            return False
            
    except Exception as e:
        logger.error(f"\n✗ Emergency fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = asyncio.run(emergency_fix())
    sys.exit(0 if success else 1)
