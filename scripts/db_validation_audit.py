"""
Database Validation Audit — Comprehensive Data Quality Check

Checks ALL tables for:
  1. Row counts (empty tables flagged)
  2. NULL values in important columns
  3. Orphaned foreign keys
  4. Duplicate detection on unique-ish columns
  5. Data freshness (stale data)
  6. Value range issues (negative counts, invalid percentages, etc.)

Usage:
    python scripts/db_validation_audit.py
    python scripts/db_validation_audit.py --table mlb_game_log   # Single table
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sqlalchemy import text, inspect
from backend.models import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NOW = datetime.now(ZoneInfo("America/New_York"))
ISSUES = []  # Collected across all checks


def add_issue(severity: str, table: str, detail: str):
    """Record an issue. severity: CRITICAL / WARNING / INFO"""
    ISSUES.append({"severity": severity, "table": table, "detail": detail})


def run_query(db, sql: str, params: dict = None):
    """Execute raw SQL and return all rows."""
    result = db.execute(text(sql), params or {})
    return result.fetchall()


def check_row_counts(db):
    """Phase 1: Row counts for every table."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: TABLE ROW COUNTS")
    logger.info("=" * 70)

    inspector = inspect(db.bind)
    tables = sorted(inspector.get_table_names())

    counts = {}
    for tbl in tables:
        if tbl == "alembic_version":
            continue
        row = run_query(db, f'SELECT COUNT(*) FROM "{tbl}"')
        cnt = row[0][0] if row else 0
        counts[tbl] = cnt

    # Display
    logger.info("%-40s %10s %s", "Table", "Rows", "Status")
    logger.info("-" * 70)
    for tbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        status = ""
        if cnt == 0:
            status = "⚠ EMPTY"
            add_issue("WARNING", tbl, f"Table is EMPTY (0 rows)")
        elif cnt < 5:
            status = "⚠ LOW"
            add_issue("INFO", tbl, f"Very few rows ({cnt})")
        logger.info("%-40s %10d %s", tbl, cnt, status)

    return counts


def check_nulls(db, table_counts: dict):
    """Phase 2: NULL counts for every column in populated tables."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: NULL VALUE AUDIT")
    logger.info("=" * 70)

    inspector = inspect(db.bind)

    for tbl, cnt in sorted(table_counts.items()):
        if cnt == 0:
            continue

        columns = inspector.get_columns(tbl)
        null_report = []

        for col in columns:
            col_name = col["name"]
            nullable = col.get("nullable", True)

            row = run_query(db, f'SELECT COUNT(*) FROM "{tbl}" WHERE "{col_name}" IS NULL')
            null_count = row[0][0] if row else 0

            if null_count > 0:
                pct = (null_count / cnt) * 100
                severity = "CRITICAL" if not nullable else ("WARNING" if pct > 50 else "INFO")
                null_report.append((col_name, null_count, pct, nullable, severity))

                if pct > 25 or not nullable:
                    add_issue(severity, tbl,
                              f"Column '{col_name}' has {null_count} NULLs ({pct:.1f}%) "
                              f"[nullable={nullable}]")

        if null_report:
            logger.info("")
            logger.info("Table: %s (%d rows)", tbl, cnt)
            logger.info("  %-30s %8s %8s %10s", "Column", "NULLs", "% NULL", "Nullable?")
            logger.info("  " + "-" * 60)
            for col_name, null_count, pct, nullable, severity in null_report:
                flag = "" if nullable else " ← NOT NULLABLE!"
                logger.info("  %-30s %8d %7.1f%% %10s%s",
                            col_name, null_count, pct, "yes" if nullable else "NO", flag)


def check_foreign_keys(db):
    """Phase 3: Orphaned foreign key references."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: FOREIGN KEY INTEGRITY")
    logger.info("=" * 70)

    fk_checks = [
        # (child_table, child_col, parent_table, parent_col, description)
        ("predictions", "game_id", "games", "id", "Predictions → Games"),
        ("bet_logs", "game_id", "games", "id", "BetLogs → Games"),
        ("bet_logs", "prediction_id", "predictions", "id", "BetLogs → Predictions"),
        ("closing_lines", "game_id", "games", "id", "ClosingLines → Games"),
        ("fantasy_draft_picks", "session_id", "fantasy_draft_sessions", "id", "DraftPicks → DraftSession"),
        ("mlb_game_log", "home_team_id", "mlb_team", "team_id", "MLBGameLog.home → MLBTeam"),
        ("mlb_game_log", "away_team_id", "mlb_team", "team_id", "MLBGameLog.away → MLBTeam"),
        ("mlb_odds_snapshot", "game_id", "mlb_game_log", "game_id", "MLBOdds → MLBGameLog"),
        ("mlb_player_stats", "game_id", "mlb_game_log", "game_id", "MLBPlayerStats → MLBGameLog"),
        ("decision_explanations", "decision_id", "decision_results", "id", "Explanations → Decisions"),
    ]

    for child_tbl, child_col, parent_tbl, parent_col, desc in fk_checks:
        try:
            # Check if both tables exist and have rows
            child_count = run_query(db, f'SELECT COUNT(*) FROM "{child_tbl}"')[0][0]
            if child_count == 0:
                logger.info("  %-40s SKIP (child empty)", desc)
                continue

            orphan_sql = f'''
                SELECT COUNT(*) FROM "{child_tbl}" c
                WHERE c."{child_col}" IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1 FROM "{parent_tbl}" p WHERE p."{parent_col}" = c."{child_col}"
                )
            '''
            orphan_count = run_query(db, orphan_sql)[0][0]

            if orphan_count > 0:
                pct = (orphan_count / child_count) * 100
                logger.info("  %-40s %d orphans (%.1f%%)", desc, orphan_count, pct)
                add_issue("CRITICAL" if pct > 10 else "WARNING", child_tbl,
                          f"FK {desc}: {orphan_count} orphaned rows ({pct:.1f}%)")
            else:
                logger.info("  %-40s OK", desc)
        except Exception as e:
            logger.info("  %-40s ERROR: %s", desc, str(e)[:60])


def check_duplicates(db, table_counts: dict):
    """Phase 4: Duplicate detection on key columns."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: DUPLICATE DETECTION")
    logger.info("=" * 70)

    dup_checks = [
        # (table, columns_list, description)
        ("position_eligibility", ["yahoo_player_key"], "PE by yahoo_player_key"),
        ("player_id_mapping", ["yahoo_key"], "ID mapping by yahoo_key"),
        ("mlb_game_log", ["game_id"], "MLB games by game_id"),
        ("mlb_team", ["abbreviation"], "MLB teams by abbreviation"),
        ("games", ["external_id"], "CBB games by external_id"),
        ("player_projections", ["player_id"], "Projections by player_id"),
        ("daily_snapshots", ["as_of_date"], "Daily snapshots by date"),
        ("statcast_performances", ["player_id", "game_date"], "Statcast by player+date"),
        ("player_rolling_stats", ["bdl_player_id", "as_of_date", "window_days"], "Rolling stats key"),
        ("player_scores", ["bdl_player_id", "as_of_date", "window_days"], "Scores key"),
        ("player_momentum", ["bdl_player_id", "as_of_date"], "Momentum key"),
    ]

    for tbl, cols, desc in dup_checks:
        if table_counts.get(tbl, 0) == 0:
            logger.info("  %-40s SKIP (empty)", desc)
            continue

        try:
            col_list = ", ".join(f'"{c}"' for c in cols)
            # Exclude NULLs from duplicate check
            where_clause = " AND ".join(f'"{c}" IS NOT NULL' for c in cols)

            dup_sql = f'''
                SELECT {col_list}, COUNT(*) as cnt
                FROM "{tbl}"
                WHERE {where_clause}
                GROUP BY {col_list}
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
                LIMIT 5
            '''
            dups = run_query(db, dup_sql)

            if dups:
                total_dups = sum(row[-1] - 1 for row in dups)
                logger.info("  %-40s %d duplicate groups found!", desc, len(dups))
                for row in dups[:3]:
                    logger.info("    Key=%s, Count=%d", row[:-1], row[-1])
                add_issue("WARNING", tbl, f"Duplicates: {len(dups)} groups ({desc})")
            else:
                logger.info("  %-40s OK (no duplicates)", desc)
        except Exception as e:
            logger.info("  %-40s ERROR: %s", desc, str(e)[:60])


def check_data_freshness(db, table_counts: dict):
    """Phase 5: Data freshness — when was each table last updated?"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 5: DATA FRESHNESS")
    logger.info("=" * 70)

    freshness_checks = [
        # (table, date_column, expected_freshness_days, description)
        ("position_eligibility", "updated_at", 7, "Position eligibility"),
        ("mlb_game_log", "game_date", 3, "MLB game log"),
        ("mlb_odds_snapshot", "fetched_at", 3, "MLB odds"),
        ("mlb_player_stats", "game_date", 3, "MLB player stats"),
        ("statcast_performances", "game_date", 7, "Statcast"),
        ("player_rolling_stats", "as_of_date", 3, "Rolling stats"),
        ("player_scores", "as_of_date", 3, "Player scores"),
        ("player_momentum", "as_of_date", 3, "Player momentum"),
        ("data_ingestion_logs", "started_at", 1, "Ingestion logs"),
        ("daily_snapshots", "as_of_date", 3, "Daily snapshots"),
        ("player_projections", "updated_at", 7, "Projections"),
        ("probable_pitchers", "game_date", 3, "Probable pitchers"),
        ("data_fetches", "fetch_time", 3, "Data fetches"),
        ("fantasy_lineups", "lineup_date", 3, "Fantasy lineups"),
    ]

    now_naive = NOW.replace(tzinfo=None)

    for tbl, date_col, max_days, desc in freshness_checks:
        if table_counts.get(tbl, 0) == 0:
            logger.info("  %-30s SKIP (empty)", desc)
            continue

        try:
            # Use a fresh connection to avoid transaction cascade
            fresh_db = SessionLocal()
            row = run_query(fresh_db, f'SELECT MAX("{date_col}") FROM "{tbl}"')
            fresh_db.close()
            latest = row[0][0] if row and row[0][0] else None

            if latest is None:
                logger.info("  %-30s No dates found", desc)
                add_issue("WARNING", tbl, f"No valid dates in '{date_col}'")
                continue

            # Handle date vs datetime, strip tz for safe subtraction
            if hasattr(latest, 'date'):
                latest_naive = latest.replace(tzinfo=None) if latest.tzinfo else latest
                days_old = (now_naive - latest_naive).days
                latest_str = latest.strftime("%Y-%m-%d %H:%M")
            else:
                days_old = (now_naive.date() - latest).days
                latest_str = str(latest)

            status = "OK" if days_old <= max_days else f"⚠ {days_old}d old (max {max_days}d)"
            logger.info("  %-30s Latest: %s  %s", desc, latest_str, status)

            if days_old > max_days:
                add_issue("WARNING", tbl,
                          f"Stale data: latest '{date_col}' is {days_old} days old (threshold: {max_days}d)")
        except Exception as e:
            logger.info("  %-30s ERROR: %s", desc, str(e)[:80])


def check_value_ranges(db_unused, table_counts: dict):
    """Phase 6: Value range validation — catch impossible/suspicious values."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 6: VALUE RANGE CHECKS")
    logger.info("=" * 70)

    range_checks = [
        # (table, condition, description, severity)
        ("position_eligibility", "multi_eligibility_count < 0 OR multi_eligibility_count > 10",
         "PE: impossible eligibility count", "CRITICAL"),
        ("position_eligibility", "player_type NOT IN ('batter', 'pitcher', 'two_way')",
         "PE: invalid player_type", "CRITICAL"),
        ("mlb_player_stats", "ab < 0 OR runs < 0 OR hits < 0 OR hr < 0",
         "MLB stats: negative counting stats", "CRITICAL"),
        ("mlb_player_stats", "era < 0 OR era > 100",
         "MLB stats: impossible ERA", "WARNING"),
        ("player_scores", "score_0_100 < 0 OR score_0_100 > 100",
         "Scores: out of 0-100 range", "CRITICAL"),
        ("player_momentum", "signal NOT IN ('SURGING', 'HOT', 'STABLE', 'COLD', 'COLLAPSING')",
         "Momentum: invalid signal value", "WARNING"),
        ("statcast_performances", "pa < 0 OR ab < 0 OR h < 0",
         "Statcast: negative counting stats", "CRITICAL"),
        ("mlb_game_log", "home_score < 0 OR away_score < 0",
         "Game log: negative scores", "CRITICAL"),
        ("simulation_results", "n_simulations <= 0",
         "Simulations: invalid n_simulations", "WARNING"),
    ]

    for tbl, condition, desc, severity in range_checks:
        if table_counts.get(tbl, 0) == 0:
            logger.info("  %-50s SKIP (empty)", desc)
            continue

        try:
            fresh_db = SessionLocal()
            row = run_query(fresh_db, f'SELECT COUNT(*) FROM "{tbl}" WHERE {condition}')
            fresh_db.close()
            cnt = row[0][0] if row else 0

            if cnt > 0:
                logger.info("  %-50s %d violations!", desc, cnt)
                add_issue(severity, tbl, f"{desc}: {cnt} rows")
            else:
                logger.info("  %-50s OK", desc)
        except Exception as e:
            logger.info("  %-50s ERROR: %s", desc, str(e)[:60])


def check_mlb_specific(db_unused, table_counts: dict):
    """Phase 7: MLB-specific cross-table validation."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 7: MLB CROSS-TABLE VALIDATION")
    logger.info("=" * 70)

    # 7a: Teams with zero games
    if table_counts.get("mlb_team", 0) > 0:
        try:
            db = SessionLocal()
            sql = '''
                SELECT t.abbreviation, t.name
                FROM mlb_team t
                LEFT JOIN mlb_game_log g ON t.team_id = g.home_team_id OR t.team_id = g.away_team_id
                WHERE g.game_id IS NULL
            '''
            no_games = run_query(db, sql)
            if no_games:
                logger.info("  Teams with ZERO games: %d", len(no_games))
                for row in no_games[:5]:
                    logger.info("    %s (%s)", row[1], row[0])
                add_issue("WARNING", "mlb_team", f"{len(no_games)} teams have no games")
            else:
                logger.info("  All teams have games: OK")
            db.close()
        except Exception as e:
            logger.info("  Teams check ERROR: %s", str(e)[:80])

    # 7b: Games with no player stats
    if table_counts.get("mlb_game_log", 0) > 0 and table_counts.get("mlb_player_stats", 0) > 0:
        try:
            db = SessionLocal()
            sql = '''
                SELECT COUNT(*) FROM mlb_game_log g
                WHERE g.status = 'Final'
                AND NOT EXISTS (
                    SELECT 1 FROM mlb_player_stats ps WHERE ps.game_id = g.game_id
                )
            '''
            cnt = run_query(db, sql)[0][0]
            if cnt > 0:
                logger.info("  Final games with NO player stats: %d", cnt)
                add_issue("WARNING", "mlb_game_log",
                          f"{cnt} completed games have no player stats")
            else:
                logger.info("  All completed games have stats: OK")
            db.close()
        except Exception as e:
            logger.info("  Games-stats check ERROR: %s", str(e)[:80])

    # 7c: Position eligibility vs player_id_mapping coverage
    if table_counts.get("position_eligibility", 0) > 0 and table_counts.get("player_id_mapping", 0) > 0:
        try:
            db = SessionLocal()
            sql = '''
                SELECT COUNT(*) FROM position_eligibility pe
                WHERE pe.yahoo_player_key IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1 FROM player_id_mapping m WHERE m.yahoo_key = pe.yahoo_player_key
                )
            '''
            cnt = run_query(db, sql)[0][0]
            total = table_counts["position_eligibility"]
            pct = (cnt / total * 100) if total else 0
            logger.info("  PE players NOT in player_id_mapping: %d / %d (%.1f%%)", cnt, total, pct)
            if pct > 50:
                add_issue("WARNING", "player_id_mapping",
                          f"{cnt}/{total} PE players ({pct:.1f}%) have no ID mapping")
            db.close()
        except Exception as e:
            logger.info("  PE-mapping check ERROR: %s", str(e)[:80])

    # 7d: Player stats with no player mapping
    if table_counts.get("mlb_player_stats", 0) > 0:
        try:
            db = SessionLocal()
            sql = '''
                SELECT COUNT(DISTINCT bdl_player_id) FROM mlb_player_stats
                WHERE bdl_player_id NOT IN (
                    SELECT COALESCE(bdl_id, -1) FROM player_id_mapping WHERE bdl_id IS NOT NULL
                )
            '''
            cnt = run_query(db, sql)[0][0]
            total = run_query(db, 'SELECT COUNT(DISTINCT bdl_player_id) FROM mlb_player_stats')[0][0]
            logger.info("  Player stats bdl_ids NOT in mapping: %d / %d", cnt, total)
            if cnt > total * 0.3:
                add_issue("WARNING", "mlb_player_stats",
                          f"{cnt}/{total} distinct bdl_player_ids have no ID mapping")
            db.close()
        except Exception as e:
            logger.info("  Stats-mapping check ERROR: %s", str(e)[:80])


def print_summary():
    """Print final issue summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 70)

    if not ISSUES:
        logger.info("  No issues found! Database is in good shape.")
        return

    # Group by severity
    by_severity = defaultdict(list)
    for issue in ISSUES:
        by_severity[issue["severity"]].append(issue)

    for sev in ["CRITICAL", "WARNING", "INFO"]:
        items = by_severity.get(sev, [])
        if items:
            logger.info("")
            logger.info("  [%s] — %d issues", sev, len(items))
            logger.info("  " + "-" * 50)
            for item in items:
                logger.info("    %-30s %s", item["table"], item["detail"])

    logger.info("")
    logger.info("Total: %d CRITICAL, %d WARNING, %d INFO",
                len(by_severity.get("CRITICAL", [])),
                len(by_severity.get("WARNING", [])),
                len(by_severity.get("INFO", [])))


def run_audit(target_table: str = None):
    """Run the full audit."""
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 70)
    logger.info("DATABASE VALIDATION AUDIT")
    logger.info("Started: %s", t0.strftime("%Y-%m-%d %H:%M:%S ET"))
    logger.info("=" * 70)

    db = SessionLocal()

    try:
        # Phase 1: Row counts
        counts = check_row_counts(db)

        if target_table:
            counts = {k: v for k, v in counts.items() if k == target_table}

        # Phase 2: NULL audit
        check_nulls(db, counts)

        # Phase 3: Foreign key integrity
        check_foreign_keys(db)

        # Phase 4: Duplicates
        check_duplicates(db, counts)

        # Phase 5: Data freshness
        check_data_freshness(db, counts)

        # Phase 6: Value ranges
        check_value_ranges(db, counts)

        # Phase 7: MLB cross-table
        check_mlb_specific(db, counts)

        # Summary
        print_summary()

    finally:
        db.close()

    elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds())
    logger.info("")
    logger.info("Audit completed in %ds", elapsed)

    return ISSUES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database validation audit")
    parser.add_argument("--table", type=str, help="Audit a single table only")
    args = parser.parse_args()

    issues = run_audit(target_table=args.table)

    critical = [i for i in issues if i["severity"] == "CRITICAL"]
    if critical:
        logger.error("AUDIT FAILED: %d critical issues found", len(critical))
        sys.exit(1)
    else:
        logger.info("AUDIT PASSED (no critical issues)")
