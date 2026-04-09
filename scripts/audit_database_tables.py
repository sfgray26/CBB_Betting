"""
Database Table Audit Script

Audits all tables in Railway database to identify:
1. Tables with expected data vs empty tables
2. Orphaned schema (tables created but never populated)
3. Data pipeline issues (tables that should have data but don't)

Usage:
    railway run python scripts/audit_database_tables.py
"""
import os
import sys

# Get DATABASE_URL from environment
db_url = os.environ.get('DATABASE_URL')
if not db_url:
    print("ERROR: DATABASE_URL environment variable not set")
    print("This script must be run via: railway run python scripts/audit_database_tables.py")
    sys.exit(1)

from sqlalchemy import inspect, text, create_engine
from urllib.parse import urlparse

# Fix Railway's DATABASE_URL format for SQLAlchemy
# Railway provides: postgres://user:pass@host:port/db
# SQLAlchemy needs: postgresql+psycopg2://user:pass@host:port/db
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql+psycopg2://', 1)

engine = create_engine(db_url)


def get_table_categories():
    """Categorize tables by domain and expected state."""
    return {
        # CBB Betting (Active/Frozen)
        'cbb_betting_active': {
            'tables': ['Game', 'Prediction', 'BetLog', 'ClosingLine', 'TeamProfile'],
            'expected_state': 'CBB season closed - archival data expected',
            'can_be_empty': False
        },
        'cbb_betting_support': {
            'tables': ['ModelParameter', 'PerformanceSnapshot', 'DataFetch', 'DBAlert'],
            'expected_state': 'Should have historical data',
            'can_be_empty': False
        },

        # Fantasy Baseball (Active)
        'fantasy_core': {
            'tables': ['MLBPlayerStats', 'PlayerIDMapping', 'PlayerRollingStats',
                      'PlayerScore', 'PlayerMomentum', 'PlayerProjection'],
            'expected_state': 'ACTIVE - Should have 2026 season data',
            'can_be_empty': False
        },
        'fantasy_schedule': {
            'tables': ['MLBTeam', 'MLBGameLog', 'MLBOddsSnapshot'],
            'expected_state': 'ACTIVE - Should have 2026 season data',
            'can_be_empty': False
        },
        'fantasy_h2h': {
            'tables': ['PositionEligibility', 'ProbablePitcherSnapshot'],
            'expected_state': 'ACTIVE - Populated by migrations v25/v26',
            'can_be_empty': False
        },

        # Ingestion & Pipeline (Active)
        'ingestion': {
            'tables': ['DataIngestionLog', 'DailySnapshot'],
            'expected_state': 'ACTIVE - Should have recent entries',
            'can_be_empty': False
        },

        # Statcast (Active)
        'statcast': {
            'tables': ['StatcastPerformance'],
            'expected_state': 'ACTIVE - May be sparse on off-days',
            'can_be_empty': True  # Off-day or lag is acceptable
        },

        # Draft & Lineup (Historical/Testing)
        'draft': {
            'tables': ['FantasyDraftSession', 'FantasyDraftPick', 'FantasyLineup'],
            'expected_state': 'Historical - Can be empty if no drafts run',
            'can_be_empty': True
        },

        # Backtesting (Historical)
        'backtest': {
            'tables': ['SimulationResult', 'DecisionResult', 'BacktestResult'],
            'expected_state': 'Historical - Can be empty if no backtests run',
            'can_be_empty': True
        },

        # Optimization & Cache (Support)
        'optimization': {
            'tables': ['DecisionExplanation', 'PlayerValuationCache'],
            'expected_state': 'Support tables - can be sparse',
            'can_be_empty': True
        },
        'cache': {
            'tables': ['ProjectionSnapshot', 'ProjectionCacheEntry', 'PlayerDailyMetric'],
            'expected_state': 'Cache tables - can be empty',
            'can_be_empty': True
        },

        # Pattern Detection (Experimental)
        'experimental': {
            'tables': ['PatternDetectionAlert'],
            'expected_state': 'Experimental - can be empty',
            'can_be_empty': True
        },

        # User Data (Optional)
        'user': {
            'tables': ['UserPreferences'],
            'expected_state': 'User settings - can be empty',
            'can_be_empty': True
        },
    }


def audit_tables():
    """Audit all tables and generate report."""
    inspector = inspect(engine)
    all_tables = sorted(inspector.get_table_names())

    categories = get_table_categories()
    table_to_category = {}
    for category, config in categories.items():
        for table in config['tables']:
            table_to_category[table] = category

    print("=" * 80)
    print("DATABASE TABLE AUDIT REPORT")
    print("=" * 80)
    print(f"\nTotal tables in schema: {len(all_tables)}\n")

    # Audit each table
    results = []
    empty_tables = []
    populated_tables = []
    missing_from_models = []
    critical_issues = []

    for table in all_tables:
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                count = result.scalar()

                category = table_to_category.get(table, 'uncategorized')
                category_info = categories.get(category, {})

                is_empty = count == 0
                can_be_empty = category_info.get('can_be_empty', False)
                expected_state = category_info.get('expected_state', 'Unknown')

                result_data = {
                    'table': table,
                    'count': count,
                    'category': category,
                    'is_empty': is_empty,
                    'can_be_empty': can_be_empty,
                    'expected_state': expected_state
                }

                results.append(result_data)

                if is_empty:
                    empty_tables.append(result_data)
                    if not can_be_empty:
                        critical_issues.append(result_data)
                else:
                    populated_tables.append(result_data)

        except Exception as e:
            print(f"ERROR auditing {table}: {str(e)[:100]}")
            missing_from_models.append({'table': table, 'error': str(e)[:100]})

    # Print summary
    print(f"Populated tables: {len(populated_tables)}")
    print(f"Empty tables: {len(empty_tables)}")
    print(f"Critical issues (should have data): {len(critical_issues)}")
    print(f"Missing from models/can't audit: {len(missing_from_models)}")

    # Detailed report by category
    print("\n" + "=" * 80)
    print("DETAILED REPORT BY CATEGORY")
    print("=" * 80)

    for category in sorted(categories.keys()):
        config = categories[category]
        category_tables = [t for t in all_tables if t in config['tables']]

        if not category_tables:
            continue

        print(f"\n[{category.upper()}]")
        print(f"Expected: {config['expected_state']}")

        for table in sorted(category_tables):
            table_result = next((r for r in results if r['table'] == table), None)
            if table_result:
                count_str = f"{table_result['count']:,} rows" if not table_result['is_empty'] else "EMPTY"
                status_icon = "✓" if not table_result['is_empty'] else ("!" if not table_result['can_be_empty'] else "○")
                print(f"  {status_icon} {table:40s} {count_str}")

    # Critical issues section
    if critical_issues:
        print("\n" + "=" * 80)
        print("⚠️  CRITICAL ISSUES - Tables that should have data but are empty")
        print("=" * 80)

        for issue in critical_issues:
            print(f"\n{issue['table']}:")
            print(f"  Category: {issue['category']}")
            print(f"  Expected: {issue['expected_state']}")
            print(f"  Current: EMPTY")

    # Uncategorized/Unknown tables
    uncategorized = [t for t in all_tables if t not in table_to_category]
    if uncategorized:
        print("\n" + "=" * 80)
        print("❓ UNCATEGORIZED TABLES")
        print("=" * 80)

        for table in sorted(uncategorized):
            table_result = next((r for r in results if r['table'] == table), None)
            if table_result:
                count_str = f"{table_result['count']:,} rows" if not table_result['is_empty'] else "EMPTY"
                print(f"  {table:40s} {count_str}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if critical_issues:
        print("\n🔴 HIGH PRIORITY:")
        for issue in critical_issues:
            print(f"  - Investigate {issue['table']} ({issue['category']})")
            print(f"    Reason: {issue['expected_state']}")

    if empty_tables:
        expected_empty = [t for t in empty_tables if t['can_be_empty']]
        if expected_empty:
            print(f"\n✅ {len(expected_empty)} tables are appropriately empty (cache/historical/experimental)")

    print(f"\n📊 SUMMARY:")
    print(f"  Total tables: {len(all_tables)}")
    print(f"  Populated: {len(populated_tables)} ({len(populated_tables)/len(all_tables)*100:.1f}%)")
    print(f"  Empty (expected): {len([t for t in empty_tables if t['can_be_empty']])}")
    print(f"  Empty (concerning): {len(critical_issues)}")


if __name__ == "__main__":
    audit_tables()
