#!/usr/bin/env python3
"""
Database initialization script
Creates all tables and optionally seeds test data
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from backend.models import Base, engine, SessionLocal
from backend.models import Game, Prediction, BetLog, ModelParameter, PerformanceSnapshot
from datetime import datetime
import logging
from sqlalchemy import text, inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database(drop_existing: bool = False):
    """
    Initialize database tables
    
    Args:
        drop_existing: If True, drops all tables first (DANGER: data loss!)
    """
    logger.info("🔧 Initializing CBB Edge database...")
    
    if drop_existing:
        logger.warning("⚠️  Dropping all existing tables!")
        response = input("Are you sure? This will delete all data. Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            logger.info("Aborted.")
            return
        
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Existing tables dropped")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created successfully")
    
    # List created tables
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logger.info(f"📋 Tables: {', '.join(tables)}")
    
    return True


def seed_test_data():
    """Add test data for development"""
    logger.info("🌱 Seeding test data...")
    
    db = SessionLocal()
    
    try:
        # Add initial model parameters
        params = [
            ModelParameter(
                parameter_name="base_sd",
                parameter_value=11.0,
                reason="initial_configuration",
                changed_by="auto"
            ),
            ModelParameter(
                parameter_name="home_advantage",
                parameter_value=3.09,
                reason="sagarin_2026_average",
                changed_by="auto"
            ),
        ]
        
        db.add_all(params)
        db.commit()
        
        logger.info("✅ Test data seeded")
    
    except Exception as e:
        logger.error(f"❌ Error seeding data: {e}")
        db.rollback()
    
    finally:
        db.close()


def check_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize CBB Edge database")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables (DANGER!)")
    parser.add_argument("--seed", action="store_true", help="Seed test data")
    parser.add_argument("--check", action="store_true", help="Only check connection")
    
    args = parser.parse_args()
    
    if args.check:
        check_connection()
    else:
        if check_connection():
            init_database(drop_existing=args.drop)
            
            if args.seed:
                seed_test_data()
            
            logger.info("🎉 Database initialization complete!")
        else:
            logger.error("Cannot initialize database - connection failed")
            sys.exit(1)
