#!/usr/bin/env python3
"""
DIAGNOSE: Why scoreboard shows all zeros

This script checks:
1. Yahoo API connectivity
2. Database player_scores table
3. canonical_projections availability
4. category mappings
"""

import sys
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/backend')

import os
os.environ['ENVIRONMENT'] = 'development'

from sqlalchemy import text
from backend.models import SessionLocal
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def diagnose():
    db = SessionLocal()
    
    print("=" * 70)
    print("SCOREBOARD DIAGNOSTIC REPORT")
    print("=" * 70)
    
    try:
        # 1. Check Yahoo configuration
        print("\n1. YAHOO CONFIGURATION")
        print("-" * 70)
        yahoo_configured = bool(os.getenv('YAHOO_CLIENT_ID')) and bool(os.getenv('YAHOO_CLIENT_SECRET'))
        print(f"   Yahoo OAuth configured: {yahoo_configured}")
        if not yahoo_configured:
            print("   ❌ CRITICAL: Yahoo OAuth not configured - can't fetch live stats")
        
        # 2. Check player_scores table
        print("\n2. PLAYER_SCORES TABLE (Direct Yahoo Stats)")
        print("-" * 70)
        try:
            count = db.execute(text("SELECT COUNT(*) FROM player_scores")).scalar()
            print(f"   Total records: {count}")
            
            if count > 0:
                latest = db.execute(text("""
                    SELECT MAX(scoring_period_id), MAX(updated_at) 
                    FROM player_scores
                """)).fetchone()
                print(f"   Latest scoring period: {latest[0]}")
                print(f"   Last updated: {latest[1]}")
                
                # Check sample data
                sample = db.execute(text("""
                    SELECT player_name, team, stat_1, stat_2, stat_3
                    FROM player_scores
                    LIMIT 3
                """)).fetchall()
                print(f"\n   Sample data:")
                for s in sample:
                    print(f"     {s[0]} ({s[1]}): {s[2]}, {s[3]}, {s[4]}")
            else:
                print("   ❌ CRITICAL: player_scores table is EMPTY")
                print("   This means Yahoo stats are not being ingested")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        # 3. Check canonical_projections
        print("\n3. CANONICAL_PROJECTIONS TABLE")
        print("-" * 70)
        try:
            count = db.execute(text("SELECT COUNT(*) FROM canonical_projections")).scalar()
            print(f"   Total records: {count}")
            
            latest = db.execute(text("""
                SELECT MAX(projection_date), COUNT(*) 
                FROM canonical_projections 
                WHERE projection_date = CURRENT_DATE
            """)).fetchone()
            print(f"   Today's projections: {latest[1]} records")
            print(f"   Latest projection date: {latest[0]}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        # 4. Check probable_pitchers
        print("\n4. PROBABLE_PITCHERS TABLE")
        print("-" * 70)
        try:
            count = db.execute(text("SELECT COUNT(*) FROM probable_pitchers")).scalar()
            print(f"   Total records: {count}")
            
            today = db.execute(text("""
                SELECT COUNT(*) FROM probable_pitchers WHERE game_date = CURRENT_DATE
            """)).scalar()
            print(f"   Today's games: {today}")
            
            if today == 0 and count > 0:
                latest = db.execute(text("""
                    SELECT MAX(game_date) FROM probable_pitchers
                """)).scalar()
                print(f"   Latest game date: {latest}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        # 5. Check data_fetches (ingestion tracking)
        print("\n5. DATA_FETCHES (Ingestion Status)")
        print("-" * 70)
        try:
            fetches = db.execute(text("""
                SELECT table_name, last_fetched_at, status, record_count
                FROM data_fetches
                ORDER BY last_fetched_at DESC
                LIMIT 10
            """)).fetchall()
            
            if fetches:
                print(f"   Recent fetches:")
                for f in fetches:
                    status_icon = "✅" if f[2] == 'success' else "❌"
                    print(f"     {status_icon} {f[0]}: {f[3]} records at {f[1]}")
            else:
                print("   No fetch history found")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        # 6. Check player_id_mapping
        print("\n6. PLAYER_ID_MAPPING (Player Identity)")
        print("-" * 70)
        try:
            count = db.execute(text("SELECT COUNT(*) FROM player_id_mapping")).scalar()
            print(f"   Total mappings: {count}")
            
            if count > 0:
                yahoo_mapped = db.execute(text("""
                    SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NOT NULL
                """)).scalar()
                bdl_mapped = db.execute(text("""
                    SELECT COUNT(*) FROM player_id_mapping WHERE bdl_id IS NOT NULL
                """)).scalar()
                print(f"   With Yahoo keys: {yahoo_mapped}")
                print(f"   With BDL IDs: {bdl_mapped}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        # 7. Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        issues = []
        if not yahoo_configured:
            issues.append("Yahoo OAuth not configured - can't fetch live data")
        # player_scores check would go here if we could query
        
        if issues:
            print("\n❌ CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("\n✅ No critical configuration issues found")
            print("\nℹ️  The data exists in the database.")
            print("   The issue is likely in the scoreboard aggregation logic or Yahoo API response parsing.")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print("""
1. Check Yahoo API response in logs:
   grep -i "scoreboard.*raw data" /var/log/app.log

2. Verify player_scores data exists:
   SELECT COUNT(*) FROM player_scores;

3. Check if Yahoo stats are being parsed correctly:
   - The stat IDs might not be mapping to category codes
   - The category_contracts.py mapping might be wrong

4. Test Yahoo API directly:
   curl -H "Authorization: Bearer $TOKEN" \\
     https://fantasysports.yahooapis.com/fantasy/v2/league/

5. Check if the issue is in category math:
   - Look for division by zero
   - Check for None values being treated as 0
        """)
        
    finally:
        db.close()

if __name__ == "__main__":
    diagnose()
