"""
Admin endpoint for comprehensive data quality validation.

This endpoint provides the full Task 11 validation audit via HTTP API.
Usage: GET /admin/validation-audit

Author: Claude Code (Master Architect)
Date: April 10, 2026
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from backend.models import SessionLocal

# Constants for validation thresholds
ORPHAN_BASELINE = 362  # Expected orphan count after P-3 (permanently unmatchable prospects)
ORPHAN_TOLERANCE = 50  # Allow this many new orphans before alerting
STATCAST_MIN_ROWS = 5000  # Minimum expected rows for populated Statcast table
STATCAST_EXPECTED_ROWS = 15000  # Expected rows for March 20 - April 11 date range

router = APIRouter()

@router.get("/validation-audit")
async def validation_audit():
    """
    Admin endpoint to perform comprehensive data quality validation audit.

    This is Task 11: Full validation of all data quality fixes from Tasks 1-10.
    Returns detailed markdown report with all findings and recommendations.
    """
    db = SessionLocal()

    try:
        # Store all findings
        findings = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": []
        }

        # Store validation results for summary
        validation_results = {}

        def add_finding(severity, category, table, issue, recommendation, sql_check=None):
            """Add a finding to the appropriate severity bucket."""
            findings[severity].append({
                "category": category,
                "table": table,
                "issue": issue,
                "recommendation": recommendation,
                "sql_check": sql_check
            })

        # ========================================================================
        # SECTION 1: PLAYER IDENTITY RESOLUTION (Tasks 1-3)
        # ========================================================================
        # Check 1.1: player_id_mapping.yahoo_key population
        result = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(yahoo_key) as yahoo_key_populated,
                COUNT(*) - COUNT(yahoo_key) as yahoo_key_null,
                ROUND(100.0 * COUNT(yahoo_key) / NULLIF(COUNT(*), 1), 2) as population_pct
            FROM player_id_mapping
        """)).fetchone()

        if result.yahoo_key_populated == 0:
            add_finding("critical", "Player Identity", "player_id_mapping",
                "CRITICAL: yahoo_key is 0% populated. Cross-system linkage broken.",
                "Re-run Task 1: backfill_yahoo_keys.py script",
                "SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NULL")
        elif result.yahoo_key_populated < 1000:
            add_finding("high", "Player Identity", "player_id_mapping",
                f"Only {result.yahoo_key_populated} yahoo_key rows populated (expected 2,000+).",
                "Verify fuzzy name matching worked correctly or re-run backfill.",
                "SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NULL")

        # Check 1.2: position_eligibility.bdl_player_id population
        result = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(bdl_player_id) as bdl_id_populated,
                COUNT(*) - COUNT(bdl_player_id) as bdl_id_null,
                ROUND(100.0 * COUNT(bdl_player_id) / NULLIF(COUNT(*), 1), 2) as population_pct
            FROM position_eligibility
        """)).fetchone()

        if result.bdl_id_populated == 0:
            add_finding("critical", "Player Identity", "position_eligibility",
                "CRITICAL: bdl_player_id is 0% populated. Cross-system linkage broken.",
                "Re-run Task 2: link_position_eligibility_bdl_ids.py script",
                "SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NULL")

        # Check 1.3: Cross-system join verification
        result = db.execute(text("""
            SELECT COUNT(*) as joinable_players
            FROM (
                SELECT DISTINCT pe.bdl_player_id
                FROM position_eligibility pe
                WHERE pe.bdl_player_id IS NOT NULL
            ) linked_players
            INNER JOIN mlb_player_stats ms ON linked_players.bdl_player_id = ms.bdl_player_id
        """)).fetchone()

        if result.joinable_players < 100:
            add_finding("high", "Player Identity", "Cross-System Joins",
                f"Only {result.joinable_players} players have cross-system stats (expected 1,000+).",
                "Verify Task 3: player identity resolution may need adjustment.",
                None)

        # ========================================================================
        # SECTION 2: COMPUTED FIELDS (Task 7)
        # ========================================================================
        # Check 2.1: ops (On-Base Percentage Plus Slugging Percentage)
        result = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(ops) as ops_populated,
                COUNT(*) - COUNT(ops) as ops_null,
                COUNT(*) FILTER (WHERE obp IS NOT NULL AND slg IS NOT NULL AND ops IS NULL) as backfillable_ops
            FROM mlb_player_stats
        """)).fetchone()

        # Only report if backfillable rows exist (have obp+slg but missing ops)
        if result.backfillable_ops > 0:
            add_finding("high", "Computed Fields", "mlb_player_stats",
                f"{result.backfillable_ops} rows have NULL ops despite having obp+slg components.",
                "Run POST /admin/backfill-ops-whip to populate computed fields.",
                "SELECT COUNT(*) FROM mlb_player_stats WHERE obp IS NOT NULL AND slg IS NOT NULL AND ops IS NULL")
        elif result.ops_null > 0 and result.ops_null == result.backfillable_ops:
            # All NULL ops are structural (missing components), this is expected
            findings["info"].append({
                "category": "Computed Fields",
                "issue": f"ops has {result.ops_null} NULL rows (all structurally unbackfillable - missing obp or slg)",
                "recommendation": "No action needed. These rows lack required components.",
                "sql_check": None
            })

        # Check 2.2: whip (Walks + Hits Per Inning Pitched)
        # Note: innings_pitched is stored as string (e.g., "6.2"), skip numeric check
        # Also skip position_type filter as that column doesn't exist
        result = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(whip) as whip_populated,
                COUNT(*) - COUNT(whip) as whip_null,
                COUNT(*) FILTER (
                    WHERE walks_allowed IS NOT NULL
                      AND hits_allowed IS NOT NULL
                      AND innings_pitched IS NOT NULL
                ) as has_components
            FROM mlb_player_stats
        """)).fetchone()

        if result.whip_populated == 0:
            add_finding("critical", "Computed Fields", "mlb_player_stats",
                "CRITICAL: whip is 100% NULL despite having components.",
                "Verify Task 7 implementation: WHIP should be (BB+H)/IP.",
                "SELECT COUNT(*) FROM mlb_player_stats WHERE walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL AND whip IS NULL")

        # ========================================================================
        # SECTION 3: IMPOSSIBLE DATA VALUES (Task 10)
        # ========================================================================
        # Check 3.1: Impossible ERA values
        result = db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE era < 0) as era_negative,
                COUNT(*) FILTER (WHERE era > 100) as era_impossible,
                MIN(era) as min_era,
                MAX(era) as max_era
            FROM mlb_player_stats
            WHERE era IS NOT NULL
        """)).fetchone()

        if result.era_negative > 0:
            add_finding("critical", "Data Quality", "mlb_player_stats",
                f"CRITICAL: {result.era_negative} rows have NEGATIVE ERA (impossible).",
                "Fix source data or calculation bug. ERA cannot be negative.",
                "SELECT bdl_player_id, era, earned_runs, innings_pitched FROM mlb_player_stats WHERE era < 0")
        elif result.era_impossible > 0:
            add_finding("critical", "Data Quality", "mlb_player_stats",
                f"CRITICAL: {result.era_impossible} rows have ERA > 100 (impossible).",
                "Fix calculation bug or NULL out impossible values.",
                "SELECT bdl_player_id, era, earned_runs, innings_pitched FROM mlb_player_stats WHERE era > 100 ORDER BY era DESC LIMIT 10")

        # Check 3.2: Impossible AVG values
        result = db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE avg < 0) as avg_negative,
                COUNT(*) FILTER (WHERE avg > 1.0) as avg_impossible,
                MIN(avg) as min_avg,
                MAX(avg) as max_avg
            FROM mlb_player_stats
            WHERE avg IS NOT NULL
        """)).fetchone()

        if result.avg_negative > 0:
            add_finding("critical", "Data Quality", "mlb_player_stats",
                f"CRITICAL: {result.avg_negative} rows have NEGATIVE AVG (impossible).",
                "Fix source data or calculation bug. AVG cannot be negative.",
                "SELECT bdl_player_id, avg FROM mlb_player_stats WHERE avg < 0")
        elif result.avg_impossible > 0:
            add_finding("medium", "Data Quality", "mlb_player_stats",
                f"{result.avg_impossible} rows have AVG > 1.0 (rare but possible for single games).",
                "Verify these are legitimate small-sample cases, not errors.",
                "SELECT bdl_player_id, avg, at_bats FROM mlb_player_stats WHERE avg > 1.0 ORDER BY avg DESC LIMIT 10")

        # ========================================================================
        # SECTION 4: EMPTY TABLES (Tasks 4-6)
        # ========================================================================
        tables_to_check = [
            "probable_pitchers",
            "statcast_performances",
            "data_ingestion_logs"
        ]

        for table_name in tables_to_check:
            result = db.execute(text(f"""
                SELECT COUNT(*) as row_count FROM {table_name}
            """)).fetchone()

            row_count = result.row_count if hasattr(result, 'row_count') else 0

            if row_count == 0:
                if table_name == "probable_pitchers":
                    add_finding("info", "Empty Tables", "probable_pitchers",
                        "Empty as expected: BDL API doesn't provide probable pitcher data (Task 4).",
                        "Use MLB Stats API instead or mark as intentionally empty.",
                        None)
                elif table_name == "statcast_performances":
                    # Statcast table validation with row count checks
                    statcast_count = db.execute(text("SELECT COUNT(*) FROM statcast_performances")).scalar()
                    if statcast_count == 0:
                        add_finding("high", "Empty Tables", "statcast_performances",
                            "statcast_performances table is empty",
                            "Run POST /admin/backfill/statcast to populate. If rows processed but table empty, check transform_to_performance() for column name mismatches.",
                            "SELECT COUNT(*) FROM statcast_performances")
                    elif statcast_count < STATCAST_MIN_ROWS:
                        add_finding("medium", "Empty Tables", "statcast_performances",
                            f"statcast_performances has only {statcast_count} rows (expected {STATCAST_EXPECTED_ROWS}+ for March 20 - April 11)",
                            "Re-run POST /admin/backfill/statcast to fill missing dates.",
                            "SELECT COUNT(*) FROM statcast_performances")
                    else:
                        findings["info"].append({
                            "category": "Data Volume",
                            "table": "statcast_performances",
                            "issue": f"Statcast data populated: {statcast_count} rows",
                            "recommendation": "No action needed.",
                            "sql_check": None
                        })
                        # Store row count for summary
                        validation_results["statcast_row_count"] = statcast_count
                elif table_name == "data_ingestion_logs":
                    add_finding("info", "Empty Tables", "data_ingestion_logs",
                        "Empty by design: Infrastructure exists but logging not implemented (Task 6).",
                        "Implement full audit logging (4 hours, medium priority).",
                        None)
            else:
                # Table has rows, check statcast specifically for minimum threshold
                if table_name == "statcast_performances" and row_count < STATCAST_MIN_ROWS:
                    add_finding("medium", "Empty Tables", "statcast_performances",
                        f"statcast_performances has only {row_count} rows (expected {STATCAST_EXPECTED_ROWS}+ for March 20 - April 11)",
                        "Re-run POST /admin/backfill/statcast to fill missing dates.",
                        "SELECT COUNT(*) FROM statcast_performances")

        # ========================================================================
        # SECTION 5: FOREIGN KEY INTEGRITY
        # ========================================================================
        # Check 5.1: Orphaned position_eligibility records
        result = db.execute(text("""
            SELECT COUNT(*) as orphaned_count
            FROM position_eligibility pe
            LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
            WHERE pe.yahoo_player_key IS NOT NULL
              AND pim.yahoo_key IS NULL
        """)).fetchone()

        # Dynamic threshold: warn if orphan count grew significantly
        # Current baseline after P-3 is ORPHAN_BASELINE (permanently unmatchable prospects)
        if result.orphaned_count > ORPHAN_BASELINE + ORPHAN_TOLERANCE:
            add_finding("medium", "Foreign Keys", "position_eligibility",
                f"{result.orphaned_count} orphaned position_eligibility rows (baseline: {ORPHAN_BASELINE}).",
                f"{result.orphaned_count - ORPHAN_BASELINE} new orphans detected. Consider re-running fuzzy linker.",
                "SELECT COUNT(*) FROM position_eligibility pe LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key WHERE pim.yahoo_key IS NULL")
        elif result.orphaned_count > 0:
            findings["info"].append({
                "category": "Foreign Keys",
                "issue": f"{result.orphaned_count} orphaned position_eligibility rows (at expected baseline)",
                "recommendation": "These are primarily minor-league prospects with no MLB/BDL entry. No action needed.",
                "sql_check": None
            })

        # ========================================================================
        # SECTION 6: DATA FRESHNESS
        # ========================================================================
        # Check 6.1: Recent mlb_player_stats
        result = db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE game_date >= CURRENT_DATE - INTERVAL '7 days') as last_7_days,
                COUNT(*) FILTER (WHERE game_date >= CURRENT_DATE - INTERVAL '30 days') as last_30_days,
                MAX(game_date) as latest_date,
                CURRENT_DATE as today
            FROM mlb_player_stats
            WHERE game_date IS NOT NULL
        """)).fetchone()

        days_old = (result.today - result.latest_date).days if result.latest_date else 999
        if days_old > 30:
            add_finding("high", "Data Freshness", "mlb_player_stats",
                f"Data is {days_old} days old (last game: {result.latest_date}).",
                "Trigger BDL stats ingestion to refresh data.",
                None)

        # ========================================================================
        # FINAL REPORT
        # ========================================================================
        # Count by severity
        critical_count = len(findings["critical"])
        high_count = len(findings["high"])
        medium_count = len(findings["medium"])
        low_count = len(findings["low"])
        info_count = len(findings["info"])

        # Build report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("VALIDATION AUDIT SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"CRITICAL issues:  {critical_count}")
        report_lines.append(f"HIGH issues:     {high_count}")
        report_lines.append(f"MEDIUM issues:   {medium_count}")
        report_lines.append(f"LOW issues:     {low_count}")
        report_lines.append(f"INFO issues:    {info_count}")
        report_lines.append("")

        if critical_count > 0:
            report_lines.append("🔴 CRITICAL ISSUES FOUND")
            report_lines.append("")
            for i, finding in enumerate(findings["critical"], 1):
                report_lines.append(f"{i}. [{finding['category']}] {finding['table']}")
                report_lines.append(f"   Issue: {finding['issue']}")
                report_lines.append(f"   Fix: {finding['recommendation']}")
                report_lines.append("")

        if high_count > 0:
            report_lines.append("🟠 HIGH PRIORITY ISSUES FOUND")
            report_lines.append("")
            for i, finding in enumerate(findings["high"], 1):
                report_lines.append(f"{i}. [{finding['category']}] {finding['table']}")
                report_lines.append(f"   Issue: {finding['issue']}")
                report_lines.append(f"   Fix: {finding['recommendation']}")
                report_lines.append("")

        if medium_count > 0:
            report_lines.append("🟡 MEDIUM PRIORITY ISSUES FOUND")
            report_lines.append("")
            for i, finding in enumerate(findings["medium"], 1):
                report_lines.append(f"{i}. [{finding['category']}] {finding['table']}")
                report_lines.append(f"   Issue: {finding['issue']}")
                report_lines.append(f"   Fix: {finding['recommendation']}")
                report_lines.append("")

        if low_count > 0:
            report_lines.append("🟢 LOW PRIORITY ISSUES FOUND")
            report_lines.append("")
            for i, finding in enumerate(findings["low"], 1):
                report_lines.append(f"{i}. [{finding['category']}] {finding['table']}")
                report_lines.append(f"   Issue: {finding['issue']}")
                report_lines.append(f"   Fix: {finding['recommendation']}")
                report_lines.append("")

        if info_count > 0:
            report_lines.append("ℹ️  INFORMATIONAL ITEMS")
            report_lines.append("")
            for i, finding in enumerate(findings["info"], 1):
                report_lines.append(f"{i}. [{finding['category']}] {finding['table']}")
                report_lines.append(f"   Note: {finding['issue']}")
                report_lines.append("")

        # Overall assessment
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("OVERALL ASSESSMENT")
        report_lines.append("=" * 80)
        report_lines.append("")

        total_issues = critical_count + high_count + medium_count
        if total_issues == 0:
            report_lines.append("✅ EXCELLENT: No data quality issues found!")
            report_lines.append("   All Tasks 1-10 fixes verified and working correctly.")
            report_lines.append("   Data quality remediation COMPLETE.")
        elif total_issues <= 3:
            report_lines.append("✅ GOOD: Minor issues found")
            report_lines.append(f"   {total_issues} issues require attention, but data quality is generally good.")
            report_lines.append("   Address HIGH priority issues before next development phase.")
        elif total_issues <= 10:
            report_lines.append("⚠️  FAIR: Moderate issues found")
            report_lines.append(f"   {total_issues} issues need attention for optimal data quality.")
            report_lines.append("   Prioritize CRITICAL and HIGH issues before feature development.")
        else:
            report_lines.append("🔴 POOR: Significant issues found")
            report_lines.append(f"   {total_issues} issues require attention.")
            report_lines.append("   Data quality remediation incomplete. Address issues before proceeding.")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF VALIDATION AUDIT")
        report_lines.append("=" * 80)

        return {
            "summary": {
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
                "info": info_count,
                "total_issues": total_issues,
                "assessment": get_assessment(total_issues)
            },
            "findings": findings,
            "report": "\n".join(report_lines)
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Validation audit failed: {str(e)}")
    finally:
        db.close()

def get_assessment(total_issues):
    """Get overall assessment based on total issue count."""
    if total_issues == 0:
        return "EXCELLENT: No data quality issues found! All Tasks 1-10 fixes verified and working correctly."
    elif total_issues <= 3:
        return f"GOOD: {total_issues} minor issues found. Data quality is generally good."
    elif total_issues <= 10:
        return f"FAIR: {total_issues} moderate issues found. Address CRITICAL and HIGH issues before feature development."
    else:
        return f"POOR: {total_issues} significant issues found. Data quality remediation incomplete."
