#!/usr/bin/env python
"""
DevOps DB health snapshot — quick overview of table sizes, freshness, and anomalies.

Usage (local):
    python scripts/devops/db_health.py

Usage (Railway production):
    railway ssh python scripts/devops/db_health.py

Output: JSON health report.
"""
import json
import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from backend.models import engine


def main() -> None:
    report = {
        "generated_at": date.today().isoformat(),
        "tables": {},
        "anomalies": [],
    }

    tables_to_check = [
        "player_id_mapping",
        "position_eligibility",
        "mlb_player_stats",
        "statcast_performances",
        "player_rolling_stats",
        "player_scores",
        "simulation_results",
        "decision_results",
        "probable_pitchers",
        "data_ingestion_logs",
    ]

    with engine.connect() as conn:
        for table in tables_to_check:
            try:
                row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                report["tables"][table] = {"row_count": row_count}
            except Exception as exc:
                report["tables"][table] = {"row_count": None, "error": str(exc)}

        # Check player_id_mapping duplicates
        try:
            dupe_count = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT bdl_id FROM player_id_mapping
                        WHERE bdl_id IS NOT NULL
                        GROUP BY bdl_id HAVING COUNT(*) > 1
                    ) t
                    """
                )
            ).scalar()
            if dupe_count:
                report["anomalies"].append(
                    {
                        "severity": "high",
                        "table": "player_id_mapping",
                        "message": f"{dupe_count} duplicate bdl_id(s) found",
                    }
                )
        except Exception as exc:
            report["anomalies"].append(
                {
                    "severity": "warning",
                    "table": "player_id_mapping",
                    "message": f"Could not check duplicates: {exc}",
                }
            )

        # Check freshness of rolling stats
        try:
            latest_date = conn.execute(
                text("SELECT MAX(metric_date) FROM player_rolling_stats")
            ).scalar()
            report["tables"]["player_rolling_stats"]["latest_date"] = str(latest_date) if latest_date else None
        except Exception:
            pass

        # Check freshness of player_scores
        try:
            latest_date = conn.execute(
                text("SELECT MAX(score_date) FROM player_scores")
            ).scalar()
            report["tables"]["player_scores"]["latest_date"] = str(latest_date) if latest_date else None
        except Exception:
            pass

        # Check decision_results volume (expected: 13 per day)
        try:
            decision_count_7d = conn.execute(
                text("SELECT COUNT(*) FROM decision_results WHERE as_of_date >= CURRENT_DATE - INTERVAL '7 days'")
            ).scalar()
            report["tables"]["decision_results"]["count_7d"] = decision_count_7d
        except Exception:
            pass

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
