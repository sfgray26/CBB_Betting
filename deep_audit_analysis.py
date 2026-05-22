#!/usr/bin/env python3
"""
Deep audit analysis - investigate specific pipeline failures
"""

import os
import sys
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

os.chdir('/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
_ET = ZoneInfo("America/New_York")

def run_query(query, params=None):
    """Execute a query and return results."""
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        return [dict(row._mapping) for row in result]

def analyze_all_tables():
    """Get comprehensive info on all tables."""
    print("=" * 80)
    print("DETAILED TABLE ANALYSIS")
    print("=" * 80)
    
    # Get all tables
    tables = run_query("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    
    table_names = [t['table_name'] for t in tables]
    print(f"\nTotal tables found: {len(table_names)}\n")
    
    issues = []
    
    for table in table_names:
        try:
            # Get row count
            count_result = run_query(f"SELECT COUNT(*) as cnt FROM {table}")
            row_count = count_result[0]['cnt'] if count_result else 0
            
            # Try to get latest timestamp
            # Find timestamp columns
            cols = run_query(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}' 
                AND data_type IN ('timestamp without time zone', 'timestamp with time zone', 'date', 'datetime')
                ORDER BY ordinal_position
            """)
            
            latest = None
            timestamp_col = None
            for col in cols:
                col_name = col['column_name']
                try:
                    result = run_query(f"SELECT MAX({col_name}) as latest FROM {table}")
                    if result and result[0]['latest']:
                        latest = result[0]['latest']
                        timestamp_col = col_name
                        break
                except:
                    continue
            
            # Check if table has data but no timestamps
            has_data_but_no_timestamps = row_count > 0 and latest is None and len(cols) > 0
            
            # Determine status
            if row_count == 0:
                status = "EMPTY"
                issues.append(f"EMPTY: {table}")
            elif latest:
                days_old = (datetime.now(_ET) - latest.replace(tzinfo=_ET) if latest.tzinfo is None else datetime.now(_ET) - latest).days
                if days_old > 30:
                    status = f"STALE ({days_old} days)"
                    issues.append(f"STALE: {table} ({days_old} days, latest: {latest})")
                elif days_old > 7:
                    status = f"WARNING ({days_old} days)"
                else:
                    status = "OK"
            else:
                status = "NO_TIMESTAMP"
                if row_count > 0:
                    issues.append(f"NO_TIMESTAMP: {table} ({row_count} rows)")
            
            print(f"{table:40s} | Rows: {row_count:8d} | {timestamp_col or 'N/A':20s} | {status}")
            
        except Exception as e:
            print(f"{table:40s} | ERROR: {str(e)[:50]}")
            issues.append(f"ERROR: {table} - {str(e)[:50]}")
    
    return issues

def check_critical_fields():
    """Check for nulls in critical fields."""
    print("\n\n" + "=" * 80)
    print("NULL CHECKS IN CRITICAL FIELDS")
    print("=" * 80)
    
    checks = [
        ("games", "external_id"),
        ("games", "home_team"),
        ("games", "away_team"),
        ("predictions", "game_id"),
        ("predictions", "verdict"),
        ("player_projections", "player_id"),
        ("player_id_mappings", "player_id"),
        ("mlb_teams", "team_name"),
        ("mlb_teams", "yahoo_team_id"),
        ("fantasy_lineups", "yahoo_league_id"),
        ("simulation_results", "player_id"),
        ("decision_results", "player_id"),
    ]
    
    issues = []
    for table, column in checks:
        try:
            result = run_query(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) as nulls
                FROM {table}
            """)
            if result:
                total = result[0]['total']
                nulls = result[0]['nulls'] or 0
                if total > 0:
                    pct = (nulls / total) * 100
                    status = "OK" if nulls == 0 else "WARNING" if pct < 10 else "CRITICAL"
                    print(f"  {status}: {table}.{column} - {nulls}/{total} null ({pct:.1f}%)")
                    if nulls > 0:
                        issues.append(f"NULLS: {table}.{column} has {nulls} nulls ({pct:.1f}%)")
        except Exception as e:
            print(f"  ERROR: {table}.{column} - {str(e)[:50]}")
    
    return issues

def check_data_consistency():
    """Check for data consistency issues."""
    print("\n\n" + "=" * 80)
    print("DATA CONSISTENCY CHECKS")
    print("=" * 80)
    
    issues = []
    
    # Check games without predictions
    try:
        result = run_query("""
            SELECT COUNT(*) as cnt FROM games g
            LEFT JOIN predictions p ON g.id = p.game_id
            WHERE p.id IS NULL
        """)
        games_no_preds = result[0]['cnt'] if result else 0
        print(f"  Games without predictions: {games_no_preds}")
        if games_no_preds > 0:
            issues.append(f"ORPHAN: {games_no_preds} games have no predictions")
    except Exception as e:
        print(f"  ERROR checking games without predictions: {e}")
    
    # Check predictions without games
    try:
        result = run_query("""
            SELECT COUNT(*) as cnt FROM predictions p
            LEFT JOIN games g ON p.game_id = g.id
            WHERE g.id IS NULL
        """)
        preds_no_games = result[0]['cnt'] if result else 0
        print(f"  Predictions without games: {preds_no_games}")
        if preds_no_games > 0:
            issues.append(f"ORPHAN: {preds_no_games} predictions have no games")
    except Exception as e:
        print(f"  ERROR checking predictions without games: {e}")
    
    # Check for duplicate mappings
    try:
        result = run_query("""
            SELECT player_id, COUNT(*) as cnt 
            FROM player_id_mappings 
            GROUP BY player_id 
            HAVING COUNT(*) > 1
        """)
        dups = len(result)
        print(f"  Duplicate player_id_mappings: {dups}")
        if dups > 0:
            issues.append(f"DUPLICATES: {dups} duplicate player_id_mappings")
    except Exception as e:
        print(f"  ERROR checking duplicate mappings: {e}")
    
    # Check bet_logs without outcomes that should have them
    try:
        result = run_query("""
            SELECT COUNT(*) as cnt FROM bet_logs bl
            JOIN games g ON bl.game_id = g.id
            WHERE g.completed = true AND bl.outcome IS NULL
        """)
        bets_no_outcome = result[0]['cnt'] if result else 0
        print(f"  Completed games with bets missing outcomes: {bets_no_outcome}")
        if bets_no_outcome > 0:
            issues.append(f"INCOMPLETE: {bets_no_outcome} bets missing outcomes for completed games")
    except Exception as e:
        print(f"  ERROR checking bet outcomes: {e}")
    
    return issues

def check_scheduler_status():
    """Check ingestion and scheduler status."""
    print("\n\n" + "=" * 80)
    print("SCHEDULER AND INGESTION STATUS")
    print("=" * 80)
    
    issues = []
    
    # Check data_fetches by source over last 30 days
    try:
        result = run_query("""
            SELECT 
                data_source,
                COUNT(*) as total,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
                MAX(fetch_time) as latest,
                MIN(fetch_time) as earliest
            FROM data_fetches
            WHERE fetch_time >= NOW() - INTERVAL '30 days'
            GROUP BY data_source
            ORDER BY latest DESC NULLS LAST
        """)
        
        print("\n  Data fetches in last 30 days:")
        for row in result:
            src = row['data_source']
            total = row['total']
            success = row['success']
            failed = row['failed']
            latest = row['latest']
            
            if latest:
                days_ago = (datetime.now(_ET) - latest.replace(tzinfo=_ET)).days
            else:
                days_ago = "N/A"
            
            status = "OK" if failed == 0 and days_ago != "N/A" and days_ago <= 1 else "WARNING" if failed < success else "CRITICAL"
            print(f"    {status}: {src} - {total} fetches ({success}✓/{failed}✗) latest: {days_ago} days ago")
            
            if failed > 0:
                issues.append(f"FETCH_FAILURES: {src} has {failed} failures in 30 days")
            if days_ago != "N/A" and days_ago > 7:
                issues.append(f"STALE_FETCH: {src} last fetched {days_ago} days ago")
    except Exception as e:
        print(f"  ERROR checking data_fetches: {e}")
    
    # Check recent errors
    try:
        result = run_query("""
            SELECT data_source, fetch_time, error_message
            FROM data_fetches
            WHERE NOT success AND fetch_time >= NOW() - INTERVAL '7 days'
            ORDER BY fetch_time DESC
            LIMIT 10
        """)
        
        if result:
            print("\n  Recent failures:")
            for row in result:
                print(f"    ✗ {row['data_source']} at {row['fetch_time']}: {str(row['error_message'])[:60]}")
    except Exception as e:
        print(f"  ERROR checking recent failures: {e}")
    
    return issues

def check_projection_quality():
    """Check projection data quality."""
    print("\n\n" + "=" * 80)
    print("PROJECTION DATA QUALITY")
    print("=" * 80)
    
    issues = []
    
    # Check player_projections
    try:
        result = run_query("""
            SELECT 
                projection_type,
                COUNT(*) as count,
                MAX(fetched_at) as latest,
                MIN(fetched_at) as earliest
            FROM player_projections
            GROUP BY projection_type
        """)
        
        print("\n  Player projections by type:")
        for row in result:
            ptype = row['projection_type']
            count = row['count']
            latest = row['latest']
            
            if latest:
                days_ago = (datetime.now(_ET) - latest.replace(tzinfo=_ET)).days
            else:
                days_ago = "N/A"
            
            print(f"    - {ptype}: {count} rows (latest: {days_ago} days ago)")
            
            if days_ago != "N/A" and days_ago > 7:
                issues.append(f"STALE_PROJECTION: {ptype} projections {days_ago} days old")
    except Exception as e:
        print(f"  ERROR checking player_projections: {e}")
    
    # Check for projections with missing key fields
    try:
        result = run_query("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN projected_value IS NULL THEN 1 ELSE 0 END) as missing_value,
                SUM(CASE WHEN projected_points IS NULL THEN 1 ELSE 0 END) as missing_points
            FROM player_projections
        """)
        
        if result:
            row = result[0]
            print(f"\n  Projection completeness:")
            print(f"    Total: {row['total']}")
            print(f"    Missing projected_value: {row['missing_value']}")
            print(f"    Missing projected_points: {row['missing_points']}")
            
            if row['missing_value'] and row['missing_value'] > 0:
                issues.append(f"NULLS: {row['missing_value']} projections missing projected_value")
    except Exception as e:
        print(f"  ERROR checking projection completeness: {e}")
    
    return issues

def generate_detailed_report():
    """Generate detailed audit report."""
    all_issues = []
    
    all_issues.extend(analyze_all_tables())
    all_issues.extend(check_critical_fields())
    all_issues.extend(check_data_consistency())
    all_issues.extend(check_scheduler_status())
    all_issues.extend(check_projection_quality())
    
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL ISSUES")
    print("=" * 80)
    
    critical = [i for i in all_issues if 'CRITICAL' in i or 'STALE' in i or 'EMPTY' in i]
    warnings = [i for i in all_issues if 'WARNING' in i or 'NO_TIMESTAMP' in i]
    others = [i for i in all_issues if i not in critical and i not in warnings]
    
    print(f"\nCritical/High Priority ({len(critical)}):")
    for issue in critical:
        print(f"  ✗ {issue}")
    
    print(f"\nWarnings ({len(warnings)}):")
    for issue in warnings:
        print(f"  ⚠ {issue}")
    
    print(f"\nOther Issues ({len(others)}):")
    for issue in others:
        print(f"  • {issue}")
    
    print("\n" + "=" * 80)
    
    return all_issues

if __name__ == "__main__":
    print("Starting deep audit analysis...\n")
    issues = generate_detailed_report()
    
    # Save summary
    with open('/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/DATA_QUALITY_AUDIT_DETAILED.md', 'w') as f:
        f.write("# Detailed Data Quality Audit Summary\n\n")
        f.write(f"Generated: {datetime.now(_ET).strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
        f.write(f"## Issues Found: {len(issues)}\n\n")
        for issue in issues:
            f.write(f"- {issue}\n")
    
    print("\n✓ Detailed analysis complete!")
    print("  Detailed issues saved to: DATA_QUALITY_AUDIT_DETAILED.md")
