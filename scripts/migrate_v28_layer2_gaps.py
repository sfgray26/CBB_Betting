#!/usr/bin/env python3
"""
Migration v28: Layer 2 Gap Closure

1. Add missing constraint _pp_date_team_uc to probable_pitchers table
2. Create weather_forecasts table for canonical weather persistence
3. Create park_factors table for canonical park factor persistence
4. Create deployment_version table for version tracking

Run: railway run python scripts/migrate_v28_layer2_gaps.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import create_engine, text
from datetime import datetime, UTC
import subprocess


def get_git_info():
    """Get git commit SHA and timestamp for deployment fingerprint."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL
        ).decode().strip()

        timestamp = subprocess.check_output(
            ['git', 'show', '-s', '--format=%ci', 'HEAD'],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return sha, timestamp
    except Exception:
        return 'unknown', 'unknown'


def migrate_db(engine):
    """Apply all migrations."""

    with engine.begin() as conn:
        print("Applying v28 Layer 2 Gap Closure migration...")

        # 1. Add missing constraint to probable_pitchers
        print("\n1. Adding constraint _pp_date_team_uc to probable_pitchers...")

        # Check if constraint already exists
        check_constraint = text("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'probable_pitchers'
            AND constraint_name = '_pp_date_team_uc'
        """)
        result = conn.execute(check_constraint).fetchone()

        if result and result[0]:
            print("   Constraint already exists, skipping.")
        else:
            # Remove duplicates if any exist first
            conn.execute(text("""
                DELETE FROM probable_pitchers p1
                USING probable_pitchers p2
                WHERE p1.id < p2.id
                AND p1.game_date = p2.game_date
                AND p1.team = p2.team
            """))

            # Add the constraint
            conn.execute(text("""
                ALTER TABLE probable_pitchers
                ADD CONSTRAINT _pp_date_team_uc
                UNIQUE (game_date, team)
            """))
            print("   Constraint added successfully.")

        # 2. Create weather_forecasts table
        print("\n2. Creating weather_forecasts table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weather_forecasts (
                id SERIAL PRIMARY KEY,
                game_date DATE NOT NULL,
                park_name VARCHAR(100) NOT NULL,
                forecast_date DATE NOT NULL DEFAULT CURRENT_DATE,
                temperature_high FLOAT,
                temperature_low FLOAT,
                humidity INTEGER,
                wind_speed FLOAT,
                wind_direction VARCHAR(10),
                precipitation_probability INTEGER,
                conditions VARCHAR(100),
                fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (game_date, park_name, forecast_date)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_weather_game_date
            ON weather_forecasts (game_date)
        """))
        print("   Table created.")

        # 3. Create park_factors table
        print("\n3. Creating park_factors table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS park_factors (
                id SERIAL PRIMARY KEY,
                park_name VARCHAR(100) NOT NULL UNIQUE,
                hr_factor FLOAT NOT NULL DEFAULT 1.0,
                run_factor FLOAT NOT NULL DEFAULT 1.0,
                hits_factor FLOAT NOT NULL DEFAULT 1.0,
                era_factor FLOAT NOT NULL DEFAULT 1.0,
                whip_factor FLOAT NOT NULL DEFAULT 1.0,
                data_source VARCHAR(50),
                season INTEGER,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("   Table created.")

        # 4. Insert default park factors if table is empty
        print("\n4. Seeding default park factors...")
        result = conn.execute(text("SELECT COUNT(*) FROM park_factors")).fetchone()
        if result[0] == 0:
            default_factors = [
                ('Yankee Stadium', 1.02, 1.01, 1.00, 0.99, 1.00, 'fangraphs', 2025),
                ('Dodger Stadium', 0.95, 0.97, 0.98, 1.01, 1.00, 'fangraphs', 2025),
                ('Coors Field', 1.25, 1.15, 1.10, 1.10, 1.05, 'fangraphs', 2025),
                ('Fenway Park', 1.08, 1.05, 1.03, 1.02, 1.01, 'fangraphs', 2025),
                ('Wrigley Field', 1.05, 1.04, 1.03, 1.01, 1.01, 'fangraphs', 2025),
                ('Oracle Park', 0.92, 0.95, 0.96, 0.98, 0.99, 'fangraphs', 2025),
                ('Truist Park', 0.99, 1.00, 0.99, 1.00, 1.00, 'fangraphs', 2025),
                ('Petco Park', 0.94, 0.96, 0.97, 1.00, 0.99, 'fangraphs', 2025),
                ('Citizens Bank Park', 1.09, 1.06, 1.04, 1.02, 1.01, 'fangraphs', 2025),
                ('Great American Ball Park', 1.15, 1.08, 1.05, 1.03, 1.02, 'fangraphs', 2025),
                ('American Family Field', 1.05, 1.03, 1.02, 0.99, 1.00, 'fangraphs', 2025),
                ('PNC Park', 0.97, 0.98, 0.98, 0.99, 1.00, 'fangraphs', 2025),
                ('LoanDepot Park', 0.96, 0.97, 0.97, 1.01, 1.00, 'fangraphs', 2025),
                ('Citi Field', 0.95, 0.97, 0.97, 1.00, 1.00, 'fangraphs', 2025),
                ('Nationals Park', 1.00, 1.00, 1.00, 1.00, 1.00, 'fangraphs', 2025),
                ('Tropicana Field', 0.94, 0.96, 0.97, 1.00, 0.99, 'fangraphs', 2025),
                ('Busch Stadium', 1.00, 1.00, 1.00, 1.00, 1.00, 'fangraphs', 2025),
                ('Comerica Park', 1.02, 1.02, 1.01, 1.00, 1.00, 'fangraphs', 2025),
                ('Kauffman Stadium', 1.00, 1.00, 1.00, 1.00, 1.00, 'fangraphs', 2025),
                ('Target Field', 0.98, 0.99, 0.99, 1.01, 1.00, 'fangraphs', 2025),
                ('Globe Life Field', 0.98, 0.99, 0.99, 1.00, 1.00, 'fangraphs', 2025),
                ('Angel Stadium', 0.98, 0.99, 0.99, 1.00, 1.00, 'fangraphs', 2025),
                ('Oakland Coliseum', 0.97, 0.98, 0.98, 1.01, 1.00, 'fangraphs', 2025),
                ('Rogers Centre', 1.03, 1.02, 1.01, 1.00, 1.00, 'fangraphs', 2025),
                ('T-Mobile Park', 0.94, 0.96, 0.97, 1.01, 1.00, 'fangraphs', 2025),
                ('Progressive Field', 1.02, 1.02, 1.01, 1.00, 1.00, 'fangraphs', 2025),
                ('Guaranteed Rate Field', 1.07, 1.05, 1.04, 1.01, 1.01, 'fangraphs', 2025),
            ]
            for park in default_factors:
                conn.execute(text("""
                    INSERT INTO park_factors
                    (park_name, hr_factor, run_factor, hits_factor, era_factor, whip_factor, data_source, season)
                    VALUES (:park_name, :hr, :run, :hits, :era, :whip, :source, :season)
                """), {
                    'park_name': park[0], 'hr': park[1], 'run': park[2], 'hits': park[3],
                    'era': park[4], 'whip': park[5], 'source': park[6], 'season': park[7]
                })
            print(f"   Inserted {len(default_factors)} default park factors.")
        else:
            print("   Park factors already seeded.")

        # 5. Create deployment_version table for /admin/version endpoint
        print("\n5. Creating deployment_version table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS deployment_version (
                id SERIAL PRIMARY KEY,
                git_commit_sha VARCHAR(100) NOT NULL UNIQUE,
                git_commit_date VARCHAR(50),
                build_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                app_version VARCHAR(50) DEFAULT 'dev',
                deployed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("   Table created.")

        # 6. Populate deployment_version
        print("\n6. Populating deployment_version...")
        sha, commit_timestamp = get_git_info()
        conn.execute(text("""
            INSERT INTO deployment_version (git_commit_sha, git_commit_date, app_version)
            VALUES (:sha, :commit_date, 'dev')
            ON CONFLICT (git_commit_sha)
            DO UPDATE SET build_timestamp = CURRENT_TIMESTAMP
        """), {'sha': sha, 'commit_date': commit_timestamp})
        print(f"   Deployment version recorded: SHA={sha[:12]}...")

    print("\n✓ Migration v28 complete!")


def main():
    """Run migration."""
    from backend.core.database import get_db_url

    engine = create_engine(get_db_url())

    print("Starting Layer 2 Gap Closure migration...")
    migrate_db(engine)

    print("\nVerifying migration...")

    with engine.begin() as conn:
        # Verify constraint exists
        result = conn.execute(text("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_name = 'probable_pitchers' AND constraint_name = '_pp_date_team_uc'
        """)).fetchone()
        print(f"✓ Constraint _pp_date_team_uc: {'EXISTS' if result else 'MISSING'}")

        # Verify tables exist
        for table in ['weather_forecasts', 'park_factors', 'deployment_version']:
            result = conn.execute(text("""
                SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :tbl)
            """), {'tbl': table}).fetchone()
            print(f"✓ Table {table}: {'EXISTS' if result[0] else 'MISSING'}")


if __name__ == "__main__":
    main()
