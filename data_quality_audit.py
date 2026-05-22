#!/usr/bin/env python3
"""
DATA QUALITY AUDIT SCRIPT - P0 Production Incident Investigation
Comprehensive database audit to identify data gaps and pipeline failures.
"""

import os
import sys
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

# Set up paths
os.chdir('/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')

# Load environment
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, inspect, text, func, Table, MetaData
from sqlalchemy.orm import sessionmaker

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/cbb_edge")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_ET = ZoneInfo("America/New_York")

def get_db_session():
    return SessionLocal()

def get_all_tables():
    """Get all tables in the database."""
    inspector = inspect(engine)
    return sorted(inspector.get_table_names())

def get_table_columns(table_name):
    """Get columns for a specific table."""
    inspector = inspect(engine)
    return inspector.get_columns(table_name)

def get_foreign_keys(table_name):
    """Get foreign keys for a table."""
    inspector = inspect(engine)
    return inspector.get_foreign_keys(table_name)

def count_rows(table_name):
    """Count rows in a table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()

def get_recent_data_count(table_name, date_column='created_at', days=7):
    """Count records from last N days."""
    try:
        cutoff = datetime.now(_ET) - timedelta(days=days)
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE {date_column} >= :cutoff
            """), {"cutoff": cutoff})
            return result.scalar()
    except Exception as e:
        return f"ERROR: {str(e)[:50]}"

def get_latest_timestamp(table_name, date_column='created_at'):
    """Get the latest timestamp in a table."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT MAX({date_column}) FROM {table_name}
            """))
            return result.scalar()
    except Exception as e:
        return None

def check_nulls_in_critical_fields(table_name, columns):
    """Check for null values in critical fields."""
    null_counts = {}
    with engine.connect() as conn:
        for col in columns:
            try:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL
                """))
                null_counts[col] = result.scalar()
            except:
                null_counts[col] = "N/A"
    return null_counts

def check_foreign_key_integrity():
    """Check for orphaned records in foreign key relationships."""
    issues = []
    
    # Define known FK relationships to check
    relationships = [
        ("predictions", "game_id", "games", "id"),
        ("bet_logs", "game_id", "games", "id"),
        ("bet_logs", "prediction_id", "predictions", "id"),
        ("closing_lines", "game_id", "games", "id"),
        ("fantasy_draft_picks", "session_id", "fantasy_draft_sessions", "id"),
        ("player_daily_metrics", "player_id", "player_id_mappings", "player_id"),
        ("mlb_game_logs", "team_id", "mlb_teams", "id"),
        ("mlb_player_stats", "player_id", "player_id_mappings", "player_id"),
        ("mlb_player_stats", "game_id", "mlb_game_logs", "game_id"),
        ("player_rolling_stats", "player_id", "player_id_mappings", "player_id"),
        ("player_scores", "player_id", "player_id_mappings", "player_id"),
        ("player_momentum", "player_id", "player_id_mappings", "player_id"),
        ("simulation_results", "player_id", "player_id_mappings", "player_id"),
        ("decision_results", "player_id", "player_id_mappings", "player_id"),
        ("decision_results", "simulation_result_id", "simulation_results", "id"),
        ("backtest_results", "decision_id", "decision_results", "id"),
        ("position_eligibilities", "player_id", "player_id_mappings", "player_id"),
        ("probable_pitcher_snapshots", "player_id", "player_id_mappings", "player_id"),
        ("statcast_performances", "player_id", "player_id_mappings", "player_id"),
        ("statcast_batter_metrics", "player_id", "player_id_mappings", "player_id"),
        ("statcast_pitcher_metrics", "player_id", "player_id_mappings", "player_id"),
        ("savant_pitch_quality_scores", "player_id", "player_id_mappings", "player_id"),
        ("player_projections", "player_id", "player_id_mappings", "player_id"),
        ("pattern_detection_alerts", "player_id", "player_id_mappings", "player_id"),
    ]
    
    with engine.connect() as conn:
        for child_table, child_col, parent_table, parent_col in relationships:
            try:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM {child_table} c
                    LEFT JOIN {parent_table} p ON c.{child_col} = p.{parent_col}
                    WHERE c.{child_col} IS NOT NULL AND p.{parent_col} IS NULL
                """))
                orphaned = result.scalar()
                if orphaned > 0:
                    issues.append({
                        'child_table': child_table,
                        'child_col': child_col,
                        'parent_table': parent_table,
                        'parent_col': parent_col,
                        'orphaned_count': orphaned
                    })
            except Exception as e:
                # Table might not exist
                pass
    
    return issues

def analyze_data_freshness():
    """Analyze data freshness across key tables."""
    freshness = []
    
    # Define tables with their date columns
    tables_to_check = [
        ("games", "game_date"),
        ("predictions", "created_at"),
        ("bet_logs", "timestamp"),
        ("data_fetches", "fetch_time"),
        ("closing_lines", "captured_at"),
        ("mlb_game_logs", "game_date"),
        ("mlb_player_stats", "fetched_at"),
        ("player_daily_metrics", "date"),
        ("player_rolling_stats", "calculated_at"),
        ("statcast_performances", "fetched_at"),
        ("statcast_batter_metrics", "fetched_at"),
        ("statcast_pitcher_metrics", "fetched_at"),
        ("player_projections", "fetched_at"),
        ("simulation_results", "simulated_at"),
        ("decision_results", "created_at"),
        ("fantasy_lineups", "for_date"),
        ("probable_pitcher_snapshots", "snapshot_date"),
    ]
    
    for table_name, date_col in tables_to_check:
        try:
            latest = get_latest_timestamp(table_name, date_col)
            row_count = count_rows(table_name)
            
            if latest:
                days_old = (datetime.now(_ET) - latest.replace(tzinfo=_ET) if latest.tzinfo is None else latest).days
            else:
                days_old = None
                
            freshness.append({
                'table': table_name,
                'date_column': date_col,
                'total_rows': row_count,
                'latest_record': latest,
                'days_since_update': days_old
            })
        except Exception as e:
            freshness.append({
                'table': table_name,
                'date_column': date_col,
                'total_rows': 'ERROR',
                'latest_record': None,
                'days_since_update': None,
                'error': str(e)[:50]
            })
    
    return freshness

def check_projection_data():
    """Check projection data specifically."""
    projections = {}
    
    with engine.connect() as conn:
        # Check player_projections
        try:
            result = conn.execute(text("""
                SELECT projection_type, COUNT(*) as count,
                       MAX(fetched_at) as latest
                FROM player_projections
                GROUP BY projection_type
            """))
            projections['by_type'] = [dict(row._mapping) for row in result]
        except Exception as e:
            projections['by_type_error'] = str(e)[:50]
        
        # Check projection cache
        try:
            result = conn.execute(text("""
                SELECT COUNT(*) as count, MAX(created_at) as latest
                FROM projection_cache_entries
            """))
            row = result.fetchone()
            if row:
                projections['cache'] = dict(row._mapping)
        except Exception as e:
            projections['cache_error'] = str(e)[:50]
        
        # Check projection snapshots
        try:
            result = conn.execute(text("""
                SELECT COUNT(*) as count, MAX(snapshot_date) as latest
                FROM projection_snapshots
            """))
            row = result.fetchone()
            if row:
                projections['snapshots'] = dict(row._mapping)
        except Exception as e:
            projections['snapshots_error'] = str(e)[:50]
    
    return projections

def check_statcast_data():
    """Check Statcast data freshness and completeness."""
    statcast = {}
    
    tables = [
        'statcast_performances',
        'statcast_batter_metrics', 
        'statcast_pitcher_metrics',
        'savant_pitch_quality_scores'
    ]
    
    for table in tables:
        try:
            statcast[table] = {
                'total_rows': count_rows(table),
                'latest': get_latest_timestamp(table, 'fetched_at')
            }
        except Exception as e:
            statcast[table] = {'error': str(e)[:50]}
    
    return statcast

def check_player_mappings():
    """Check player ID mapping integrity."""
    mappings = {}
    
    with engine.connect() as conn:
        try:
            # Check total mappings
            result = conn.execute(text("SELECT COUNT(*) FROM player_id_mappings"))
            mappings['total'] = result.scalar()
            
            # Check mappings by source
            result = conn.execute(text("""
                SELECT 
                    CASE 
                        WHEN savant_id IS NOT NULL THEN 'savant'
                        WHEN yahoo_id IS NOT NULL THEN 'yahoo'
                        WHEN fangraphs_id IS NOT NULL THEN 'fangraphs'
                        ELSE 'other'
                    END as source,
                    COUNT(*) as count
                FROM player_id_mappings
                GROUP BY 1
            """))
            mappings['by_source'] = [dict(row._mapping) for row in result]
            
            # Check for null IDs
            result = conn.execute(text("""
                SELECT 
                    SUM(CASE WHEN savant_id IS NULL THEN 1 ELSE 0 END) as null_savant,
                    SUM(CASE WHEN yahoo_id IS NULL THEN 1 ELSE 0 END) as null_yahoo,
                    SUM(CASE WHEN fangraphs_id IS NULL THEN 1 ELSE 0 END) as null_fangraphs,
                    SUM(CASE WHEN mlbam_id IS NULL THEN 1 ELSE 0 END) as null_mlbam
                FROM player_id_mappings
            """))
            row = result.fetchone()
            if row:
                mappings['null_counts'] = dict(row._mapping)
                
        except Exception as e:
            mappings['error'] = str(e)[:100]
    
    return mappings

def check_ingestion_logs():
    """Check data ingestion logs for failures."""
    logs = {}
    
    with engine.connect() as conn:
        try:
            # Recent fetches by source
            result = conn.execute(text("""
                SELECT 
                    data_source,
                    COUNT(*) as total,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
                    MAX(fetch_time) as latest_fetch
                FROM data_fetches
                WHERE fetch_time >= NOW() - INTERVAL '7 days'
                GROUP BY data_source
            """))
            logs['recent_by_source'] = [dict(row._mapping) for row in result]
            
            # Recent failures
            result = conn.execute(text("""
                SELECT 
                    data_source,
                    fetch_time,
                    error_message
                FROM data_fetches
                WHERE NOT success AND fetch_time >= NOW() - INTERVAL '3 days'
                ORDER BY fetch_time DESC
                LIMIT 20
            """))
            logs['recent_failures'] = [dict(row._mapping) for row in result]
            
        except Exception as e:
            logs['error'] = str(e)[:100]
    
    return logs

def check_predictions_and_bets():
    """Check predictions and betting data."""
    data = {}
    
    with engine.connect() as conn:
        try:
            # Predictions summary
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT game_id) as unique_games,
                    MAX(created_at) as latest_prediction,
                    SUM(CASE WHEN verdict != 'PASS' THEN 1 ELSE 0 END) as bet_recommendations
                FROM predictions
            """))
            row = result.fetchone()
            if row:
                data['predictions'] = dict(row._mapping)
            
            # Predictions by date
            result = conn.execute(text("""
                SELECT 
                    prediction_date,
                    COUNT(*) as count
                FROM predictions
                WHERE prediction_date >= CURRENT_DATE - INTERVAL '14 days'
                GROUP BY prediction_date
                ORDER BY prediction_date DESC
            """))
            data['predictions_by_date'] = [dict(row._mapping) for row in result]
            
            # Bet logs
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 0 THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending,
                    MAX(timestamp) as latest_bet
                FROM bet_logs
            """))
            row = result.fetchone()
            if row:
                data['bet_logs'] = dict(row._mapping)
                
        except Exception as e:
            data['error'] = str(e)[:100]
    
    return data

def check_simulation_and_decisions():
    """Check simulation results and decision data."""
    data = {}
    
    with engine.connect() as conn:
        try:
            # Simulation results
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_simulations,
                    MAX(simulated_at) as latest_simulation
                FROM simulation_results
            """))
            row = result.fetchone()
            if row:
                data['simulations'] = dict(row._mapping)
            
            # Decision results
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_decisions,
                    MAX(created_at) as latest_decision
                FROM decision_results
            """))
            row = result.fetchone()
            if row:
                data['decisions'] = dict(row._mapping)
                
        except Exception as e:
            data['error'] = str(e)[:100]
    
    return data

def generate_audit_report():
    """Generate comprehensive data quality audit report."""
    report = []
    report.append("=" * 80)
    report.append("DATA QUALITY AUDIT REPORT")
    report.append(f"Generated: {datetime.now(_ET).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    report.append("=" * 80)
    report.append("")
    
    # 1. Database Overview
    report.append("-" * 80)
    report.append("1. DATABASE OVERVIEW")
    report.append("-" * 80)
    tables = get_all_tables()
    report.append(f"Total tables: {len(tables)}")
    report.append("")
    
    # 2. Data Freshness
    report.append("-" * 80)
    report.append("2. DATA FRESHNESS ANALYSIS")
    report.append("-" * 80)
    freshness = analyze_data_freshness()
    
    stale_tables = []
    empty_tables = []
    
    for item in freshness:
        table = item['table']
        rows = item['total_rows']
        latest = item['latest_record']
        days = item.get('days_since_update')
        
        if rows == 0 or (isinstance(rows, str) and 'ERROR' in str(rows)):
            empty_tables.append(table)
        elif days is not None and days > 3:
            stale_tables.append((table, days, latest))
        
        status = "✓" if days is not None and days <= 1 else "⚠" if days is not None and days <= 3 else "✗"
        report.append(f"  {status} {table}:")
        report.append(f"      Rows: {rows}")
        report.append(f"      Latest: {latest}")
        if days is not None:
            report.append(f"      Days since update: {days}")
        report.append("")
    
    # 3. Critical Stale Data
    report.append("-" * 80)
    report.append("3. CRITICAL STALE/DATA GAPS (>3 days since update)")
    report.append("-" * 80)
    if stale_tables:
        for table, days, latest in sorted(stale_tables, key=lambda x: -x[1]):
            report.append(f"  ✗ {table}: {days} days stale (latest: {latest})")
    else:
        report.append("  ✓ No stale tables detected")
    report.append("")
    
    # 4. Empty Tables
    report.append("-" * 80)
    report.append("4. EMPTY OR ERROR TABLES")
    report.append("-" * 80)
    if empty_tables:
        for table in empty_tables:
            report.append(f"  ✗ {table}: NO DATA")
    else:
        report.append("  ✓ All tables have data")
    report.append("")
    
    # 5. Foreign Key Integrity
    report.append("-" * 80)
    report.append("5. FOREIGN KEY INTEGRITY")
    report.append("-" * 80)
    fk_issues = check_foreign_key_integrity()
    if fk_issues:
        report.append(f"  Found {len(fk_issues)} orphaned record issues:")
        for issue in fk_issues:
            report.append(f"    ✗ {issue['child_table']}.{issue['child_col']} -> {issue['parent_table']}.{issue['parent_col']}")
            report.append(f"      Orphaned records: {issue['orphaned_count']}")
    else:
        report.append("  ✓ No foreign key integrity issues detected")
    report.append("")
    
    # 6. Projection Data
    report.append("-" * 80)
    report.append("6. PROJECTION DATA STATUS")
    report.append("-" * 80)
    projections = check_projection_data()
    if 'by_type' in projections:
        report.append("  Projections by type:")
        for p in projections['by_type']:
            report.append(f"    - {p.get('projection_type', 'unknown')}: {p.get('count', 0)} rows (latest: {p.get('latest')})")
    if 'cache' in projections:
        cache = projections['cache']
        report.append(f"  Projection cache: {cache.get('count', 0)} entries (latest: {cache.get('latest')})")
    if 'snapshots' in projections:
        snap = projections['snapshots']
        report.append(f"  Projection snapshots: {snap.get('count', 0)} entries (latest: {snap.get('latest')})")
    report.append("")
    
    # 7. Statcast Data
    report.append("-" * 80)
    report.append("7. STATCAST DATA STATUS")
    report.append("-" * 80)
    statcast = check_statcast_data()
    for table, info in statcast.items():
        if 'error' in info:
            report.append(f"  ✗ {table}: {info['error']}")
        else:
            latest = info.get('latest')
            days = (datetime.now(_ET) - latest.replace(tzinfo=_ET)).days if latest else None
            status = "✓" if days is not None and days <= 1 else "⚠" if days is not None and days <= 3 else "✗"
            report.append(f"  {status} {table}: {info.get('total_rows', 0)} rows (latest: {latest})")
    report.append("")
    
    # 8. Player ID Mappings
    report.append("-" * 80)
    report.append("8. PLAYER ID MAPPINGS")
    report.append("-" * 80)
    mappings = check_player_mappings()
    if 'total' in mappings:
        report.append(f"  Total mappings: {mappings['total']}")
    if 'by_source' in mappings:
        report.append("  By source:")
        for m in mappings['by_source']:
            report.append(f"    - {m.get('source', 'unknown')}: {m.get('count', 0)}")
    if 'null_counts' in mappings:
        report.append("  Null ID counts (potential gaps):")
        for key, val in mappings['null_counts'].items():
            if val and val > 0:
                report.append(f"    ⚠ {key}: {val} nulls")
    report.append("")
    
    # 9. Ingestion Logs
    report.append("-" * 80)
    report.append("9. INGESTION PIPELINE STATUS (Last 7 Days)")
    report.append("-" * 80)
    logs = check_ingestion_logs()
    if 'recent_by_source' in logs:
        for log in logs['recent_by_source']:
            source = log.get('data_source', 'unknown')
            total = log.get('total', 0)
            success = log.get('successful', 0)
            failed = log.get('failed', 0)
            latest = log.get('latest_fetch')
            status = "✓" if failed == 0 else "⚠" if failed < success else "✗"
            report.append(f"  {status} {source}: {total} fetches ({success} success, {failed} failed) latest: {latest}")
    
    if 'recent_failures' in logs and logs['recent_failures']:
        report.append("\n  Recent Failures:")
        for f in logs['recent_failures'][:5]:
            report.append(f"    ✗ {f.get('data_source')} at {f.get('fetch_time')}: {f.get('error_message', 'Unknown')[:60]}")
    report.append("")
    
    # 10. Predictions and Betting
    report.append("-" * 80)
    report.append("10. PREDICTIONS AND BETTING DATA")
    report.append("-" * 80)
    preds = check_predictions_and_bets()
    if 'predictions' in preds:
        p = preds['predictions']
        report.append(f"  Total predictions: {p.get('total_predictions', 0)}")
        report.append(f"  Unique games: {p.get('unique_games', 0)}")
        report.append(f"  Bet recommendations: {p.get('bet_recommendations', 0)}")
        report.append(f"  Latest prediction: {p.get('latest_prediction')}")
    
    if 'predictions_by_date' in preds:
        report.append("\n  Predictions by date (last 14 days):")
        for p in preds['predictions_by_date']:
            report.append(f"    - {p.get('prediction_date')}: {p.get('count', 0)} predictions")
    
    if 'bet_logs' in preds:
        b = preds['bet_logs']
        report.append(f"\n  Bet logs: {b.get('total_bets', 0)} total ({b.get('wins', 0)} wins, {b.get('losses', 0)} losses, {b.get('pending', 0)} pending)")
        report.append(f"  Latest bet: {b.get('latest_bet')}")
    report.append("")
    
    # 11. Simulations and Decisions
    report.append("-" * 80)
    report.append("11. SIMULATION AND DECISION DATA")
    report.append("-" * 80)
    sims = check_simulation_and_decisions()
    if 'simulations' in sims:
        s = sims['simulations']
        report.append(f"  Total simulations: {s.get('total_simulations', 0)}")
        report.append(f"  Latest simulation: {s.get('latest_simulation')}")
    if 'decisions' in sims:
        d = sims['decisions']
        report.append(f"  Total decisions: {d.get('total_decisions', 0)}")
        report.append(f"  Latest decision: {d.get('latest_decision')}")
    report.append("")
    
    # 12. Summary and Recommendations
    report.append("-" * 80)
    report.append("12. SUMMARY AND RECOMMENDATIONS")
    report.append("-" * 80)
    
    critical_issues = len(stale_tables) + len(empty_tables) + len(fk_issues)
    
    report.append(f"\n  CRITICAL ISSUES FOUND: {critical_issues}")
    
    if stale_tables:
        report.append("\n  IMMEDIATE ACTION REQUIRED:")
        report.append("  - The following tables have stale data (>3 days):")
        for table, days, _ in stale_tables[:10]:
            report.append(f"    • {table} ({days} days stale)")
    
    if empty_tables:
        report.append("\n  EMPTY TABLES (verify ingestion pipelines):")
        for table in empty_tables[:10]:
            report.append(f"    • {table}")
    
    if fk_issues:
        report.append("\n  DATA INTEGRITY ISSUES:")
        report.append("  - Foreign key violations detected, run data cleanup")
    
    # Check if we have any recent data at all
    recent_predictions = [p for p in (preds.get('predictions_by_date', [])) 
                          if p.get('prediction_date') and p.get('prediction_date') >= date.today() - timedelta(days=3)]
    
    if not recent_predictions:
        report.append("\n  ⚠⚠⚠ CRITICAL: No recent predictions found! ⚠⚠⚠")
        report.append("      The prediction pipeline may be completely stalled.")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF AUDIT REPORT")
    report.append("=" * 80)
    
    return "\n".join(report), {
        'tables': tables,
        'stale_tables': stale_tables,
        'empty_tables': empty_tables,
        'fk_issues': fk_issues,
        'freshness': freshness,
        'projections': projections,
        'statcast': statcast,
        'mappings': mappings,
        'ingestion': logs,
        'predictions': preds,
        'simulations': sims
    }

if __name__ == "__main__":
    print("Starting comprehensive data quality audit...")
    print("Connecting to database...")
    
    try:
        report, data = generate_audit_report()
        
        # Write to file
        output_path = '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/DATA_QUALITY_AUDIT.md'
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Audit complete! Report written to: {output_path}")
        print("\n" + "=" * 80)
        print(report)
        
    except Exception as e:
        print(f"✗ Audit failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
