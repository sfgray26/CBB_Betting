#!/usr/bin/env python
"""
Migration v9: Live Data Pipeline Tables

Adds tables for:
- Statcast daily performance data
- Bayesian-updated player projections
- Pattern detection alerts
- Data ingestion audit logs

Run: python scripts/migrate_v9_live_data.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text
from backend.models import Base, DATABASE_URL


def migrate_up():
    """Create new tables for live data pipeline."""
    print("Running migration v9: Live Data Pipeline Tables")
    print("=" * 60)
    
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Table 1: Statcast Performances
        print("\n1. Creating statcast_performances table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS statcast_performances (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                player_name VARCHAR(100) NOT NULL,
                team VARCHAR(10) NOT NULL,
                game_date DATE NOT NULL,
                pa INTEGER DEFAULT 0,
                ab INTEGER DEFAULT 0,
                h INTEGER DEFAULT 0,
                doubles INTEGER DEFAULT 0,
                triples INTEGER DEFAULT 0,
                hr INTEGER DEFAULT 0,
                r INTEGER DEFAULT 0,
                rbi INTEGER DEFAULT 0,
                bb INTEGER DEFAULT 0,
                so INTEGER DEFAULT 0,
                hbp INTEGER DEFAULT 0,
                sb INTEGER DEFAULT 0,
                cs INTEGER DEFAULT 0,
                exit_velocity_avg FLOAT DEFAULT 0.0,
                launch_angle_avg FLOAT DEFAULT 0.0,
                hard_hit_pct FLOAT DEFAULT 0.0,
                barrel_pct FLOAT DEFAULT 0.0,
                xba FLOAT DEFAULT 0.0,
                xslg FLOAT DEFAULT 0.0,
                xwoba FLOAT DEFAULT 0.0,
                avg FLOAT DEFAULT 0.0,
                obp FLOAT DEFAULT 0.0,
                slg FLOAT DEFAULT 0.0,
                ops FLOAT DEFAULT 0.0,
                woba FLOAT DEFAULT 0.0,
                ip FLOAT DEFAULT 0.0,
                er INTEGER DEFAULT 0,
                k_pit INTEGER DEFAULT 0,
                bb_pit INTEGER DEFAULT 0,
                pitches INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, game_date)
            )
        """))
        
        # Indexes for statcast_performances
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_statcast_player_date 
            ON statcast_performances(player_id, game_date)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_statcast_game_date 
            ON statcast_performances(game_date)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_statcast_team 
            ON statcast_performances(team)
        """))
        print("   ✓ Created statcast_performances table")
        
        # Table 2: Player Projections
        print("\n2. Creating player_projections table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS player_projections (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL UNIQUE,
                player_name VARCHAR(100) NOT NULL,
                team VARCHAR(10),
                positions JSONB,
                woba FLOAT DEFAULT 0.320,
                avg FLOAT DEFAULT 0.250,
                obp FLOAT DEFAULT 0.320,
                slg FLOAT DEFAULT 0.400,
                ops FLOAT DEFAULT 0.720,
                xwoba FLOAT DEFAULT 0.320,
                hr INTEGER DEFAULT 15,
                r INTEGER DEFAULT 65,
                rbi INTEGER DEFAULT 65,
                sb INTEGER DEFAULT 5,
                era FLOAT DEFAULT 4.00,
                whip FLOAT DEFAULT 1.30,
                k_per_nine FLOAT DEFAULT 8.5,
                bb_per_nine FLOAT DEFAULT 3.0,
                shrinkage FLOAT DEFAULT 1.0,
                data_quality_score FLOAT DEFAULT 0.0,
                sample_size INTEGER DEFAULT 0,
                prior_source VARCHAR(50) DEFAULT 'steamer',
                update_method VARCHAR(50) DEFAULT 'prior',
                cat_scores JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_projections_player 
            ON player_projections(player_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_projections_team 
            ON player_projections(team)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_projections_updated 
            ON player_projections(updated_at)
        """))
        print("   ✓ Created player_projections table")
        
        # Table 3: Pattern Detection Alerts
        print("\n3. Creating pattern_detection_alerts table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pattern_detection_alerts (
                id SERIAL PRIMARY KEY,
                pattern_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                confidence FLOAT DEFAULT 0.5,
                player_id VARCHAR(50),
                player_name VARCHAR(100),
                team VARCHAR(10),
                game_date DATE NOT NULL,
                title VARCHAR(200) NOT NULL,
                description TEXT,
                betting_implication TEXT,
                detection_data JSONB DEFAULT '{}',
                data_sources JSONB DEFAULT '[]',
                is_active BOOLEAN DEFAULT TRUE,
                resolved_at TIMESTAMP,
                resolution_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alerted_at TIMESTAMP
            )
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_alerts_pattern_type 
            ON pattern_detection_alerts(pattern_type)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_alerts_date 
            ON pattern_detection_alerts(game_date)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_alerts_player 
            ON pattern_detection_alerts(player_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_alerts_team 
            ON pattern_detection_alerts(team)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_alerts_active 
            ON pattern_detection_alerts(is_active, game_date)
        """))
        print("   ✓ Created pattern_detection_alerts table")
        
        # Table 4: Data Ingestion Logs
        print("\n4. Creating data_ingestion_logs table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS data_ingestion_logs (
                id SERIAL PRIMARY KEY,
                job_type VARCHAR(50) NOT NULL,
                target_date DATE NOT NULL,
                status VARCHAR(20) NOT NULL,
                records_processed INTEGER DEFAULT 0,
                records_failed INTEGER DEFAULT 0,
                processing_time_seconds FLOAT,
                validation_errors INTEGER DEFAULT 0,
                validation_warnings INTEGER DEFAULT 0,
                data_quality_score FLOAT,
                error_details JSONB DEFAULT '[]',
                warning_details JSONB DEFAULT '[]',
                summary_stats JSONB DEFAULT '{}',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                stack_trace TEXT
            )
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_logs_job_type 
            ON data_ingestion_logs(job_type)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_logs_date 
            ON data_ingestion_logs(target_date)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_logs_status 
            ON data_ingestion_logs(status)
        """))
        print("   ✓ Created data_ingestion_logs table")
        
        conn.commit()
    
    print("\n" + "=" * 60)
    print("Migration v9 completed successfully!")
    print("\nNew tables created:")
    print("  - statcast_performances")
    print("  - player_projections")
    print("  - pattern_detection_alerts")
    print("  - data_ingestion_logs")
    print("\nNext steps:")
    print("  1. Run daily ingestion: python -m backend.fantasy_baseball.statcast_ingestion")
    print("  2. Verify data quality in logs")
    print("  3. Integrate with scheduler")


def migrate_down():
    """Rollback migration - drop the new tables."""
    print("Rolling back migration v9: Dropping live data tables")
    
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data_ingestion_logs CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS pattern_detection_alerts CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS player_projections CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS statcast_performances CASCADE"))
        conn.commit()
    
    print("Rollback complete. Tables dropped.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migration v9: Live Data Pipeline")
    parser.add_argument("--downgrade", action="store_true", help="Rollback migration")
    
    args = parser.parse_args()
    
    if args.downgrade:
        migrate_down()
    else:
        migrate_up()
