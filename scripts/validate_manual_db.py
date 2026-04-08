#!/usr/bin/env python3
"""
Manual DB Validation Script — CBB Edge Analyzer

Connects to a real PostgreSQL database (read-only replica recommended)
and runs full pipeline layer validation, emitting structured JSON artifacts
for human review.

Usage:
    DATABASE_URL=<url> ARTIFACTS_ENABLED=true python scripts/validate_manual_db.py

Environment variables:
    DATABASE_URL       — Postgres connection string (required).
                         Must be externally reachable — *.railway.internal
                         hostnames are NOT accessible from GitHub Actions runners.
    ARTIFACTS_ENABLED  — 'true' to write artifacts to ARTIFACT_DIR (default: true)
    ARTIFACT_DIR       — base directory for artifacts (default: artifacts)
    MAX_ARTIFACT_ROWS  — max rows per layer file (default: 500)
    VALIDATION_SEASON  — season year (default: current year).
                         Empty string is treated the same as "not set".
    VALIDATION_SAMPLE  — players/records to sample per layer (default: 50)
    REDIS_URL          — Redis connection string (optional; only used if the
                         validated code paths exercise Redis).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, inspect, text

# ============================================================================
# Configuration — resolve env vars with safe fallbacks
# ============================================================================

DATABASE_URL: str | None = os.environ.get("DATABASE_URL")

_TZ = ZoneInfo("America/New_York")

# Use `or` (not `??`/ternary) so that an empty string falls through to the
# default, preventing the "Season: NaN" class of bug.
_season_raw = os.environ.get("VALIDATION_SEASON") or str(datetime.now(_TZ).year)
try:
    SEASON = int(_season_raw)
except ValueError:
    SEASON = datetime.now(_TZ).year

SAMPLE_SIZE = int(os.environ.get("VALIDATION_SAMPLE") or "50")
ARTIFACTS_ENABLED = (os.environ.get("ARTIFACTS_ENABLED", "true").lower() == "true")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
MAX_ARTIFACT_ROWS = int(os.environ.get("MAX_ARTIFACT_ROWS") or "500")

# ============================================================================
# Early guard — fail fast with a clear error rather than a cryptic connection
# timeout that wastes the whole workflow budget.
# ============================================================================

if not DATABASE_URL:
    print("❌ DATABASE_URL environment variable is required.")
    print()
    print("Set DATABASE_URL to a PostgreSQL connection string for a read-only")
    print("replica or staging environment. Example:")
    print("  postgresql://user:pass@host:5432/dbname?sslmode=require")
    sys.exit(1)

# Railway-internal hostnames are only accessible from within the Railway
# private network. They cannot be reached from GitHub Actions runners.
if ".railway.internal" in DATABASE_URL:
    print("❌ DATABASE_URL contains a Railway-internal hostname (*.railway.internal).")
    print()
    print("Railway-internal hostnames are NOT accessible from GitHub Actions runners.")
    print("Replace DATABASE_URL with an externally reachable PostgreSQL URL:")
    print("  • Railway public domain:  postgresql://user:pass@<app>.railway.app:5432/db")
    print("  • External read-replica:  postgresql://user:pass@<host>:5432/db?sslmode=require")
    sys.exit(1)

# ============================================================================
# Helpers
# ============================================================================

_run_ts = datetime.now(_TZ).strftime("%Y-%m-%d_%H-%M-%S")
_artifact_base = Path(ARTIFACT_DIR) / f"run-{_run_ts}"
_layer_results: list[dict[str, Any]] = []


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def header(title: str) -> None:
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def write_artifact(name: str, data: Any) -> None:
    """Write a JSON artifact file, capped at MAX_ARTIFACT_ROWS for lists."""
    if not ARTIFACTS_ENABLED:
        return
    _artifact_base.mkdir(parents=True, exist_ok=True)
    path = _artifact_base / name
    if isinstance(data, list) and len(data) > MAX_ARTIFACT_ROWS:
        data = data[:MAX_ARTIFACT_ROWS]
    path.write_text(json.dumps(data, indent=2, default=str))
    log(f"  📄 {path}")


# ============================================================================
# Database connection
# NOTE: engine is created in main() after all early-exit guards have passed.
# ============================================================================

engine: Any = None  # set by main() after validation guards


# ============================================================================
# Validation layers
# ============================================================================

def validate_ingestion() -> dict[str, Any]:
    """Layer 1 — Ingestion: check game-level and player-level ingest tables."""
    header("LAYER 1: INGESTION")

    errors: list[str] = []
    warnings: list[str] = []

    with engine.connect() as conn:
        inspector = inspect(engine)
        all_tables = set(inspector.get_table_names())

        # Core ingestion tables
        required = {"games", "data_ingestion_log"}
        missing = required - all_tables
        if missing:
            errors.append(f"Missing ingestion tables: {sorted(missing)}")

        game_count = 0
        log_count = 0

        if "games" in all_tables:
            row = conn.execute(
                text("SELECT COUNT(*) FROM games WHERE season = :s"),
                {"s": SEASON},
            ).scalar()
            game_count = row or 0
            log(f"Games (season {SEASON}):             {game_count}")
            if game_count == 0:
                warnings.append(f"No games found for season {SEASON}")

        if "data_ingestion_log" in all_tables:
            row = conn.execute(
                text("SELECT COUNT(*) FROM data_ingestion_log"),
            ).scalar()
            log_count = row or 0
            log(f"DataIngestionLog rows:             {log_count}")

        # Sample recent games
        sample: list[dict] = []
        if "games" in all_tables:
            rows = conn.execute(
                text(
                    "SELECT id, home_team, away_team, game_date, season "
                    "FROM games "
                    "WHERE season = :s "
                    "ORDER BY game_date DESC "
                    "LIMIT :lim"
                ),
                {"s": SEASON, "lim": SAMPLE_SIZE},
            ).mappings().all()
            sample = [dict(r) for r in rows]

    valid = len(errors) == 0
    log(f"Validation: {'✅ PASS' if valid else '❌ FAIL'}")
    for e in errors:
        log(f"  ERROR: {e}")
    for w in warnings:
        log(f"  WARN:  {w}")

    write_artifact(
        "ingestion-records.json",
        sample,
    )
    write_artifact(
        "ingestion-summary.json",
        {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "stats": {"game_count": game_count, "log_count": log_count},
        },
    )

    return {"layer": "ingestion", "valid": valid, "errors": errors, "warnings": warnings}


def validate_predictions() -> dict[str, Any]:
    """Layer 2 — Predictions: check model output tables."""
    header("LAYER 2: PREDICTIONS")

    errors: list[str] = []
    warnings: list[str] = []

    with engine.connect() as conn:
        inspector = inspect(engine)
        all_tables = set(inspector.get_table_names())

        pred_count = 0
        bet_count = 0
        sample: list[dict] = []

        if "predictions" in all_tables:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE EXTRACT(year FROM created_at) = :y"
                ),
                {"y": SEASON},
            ).scalar()
            pred_count = row or 0
            log(f"Predictions (season {SEASON}):        {pred_count}")
            if pred_count == 0:
                warnings.append(f"No predictions found for season {SEASON}")

            rows = conn.execute(
                text(
                    "SELECT id, home_team, away_team, predicted_winner, confidence, created_at "
                    "FROM predictions "
                    "WHERE EXTRACT(year FROM created_at) = :y "
                    "ORDER BY created_at DESC "
                    "LIMIT :lim"
                ),
                {"y": SEASON, "lim": SAMPLE_SIZE},
            ).mappings().all()
            sample = [dict(r) for r in rows]
        else:
            warnings.append("Table 'predictions' not found")

        if "bet_logs" in all_tables:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM bet_logs "
                    "WHERE EXTRACT(year FROM placed_at) = :y"
                ),
                {"y": SEASON},
            ).scalar()
            bet_count = row or 0
            log(f"BetLogs (season {SEASON}):            {bet_count}")

    valid = len(errors) == 0
    log(f"Validation: {'✅ PASS' if valid else '❌ FAIL'}")
    for e in errors:
        log(f"  ERROR: {e}")
    for w in warnings:
        log(f"  WARN:  {w}")

    write_artifact("predictions-records.json", sample)
    write_artifact(
        "predictions-summary.json",
        {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "stats": {"pred_count": pred_count, "bet_count": bet_count},
        },
    )

    return {"layer": "predictions", "valid": valid, "errors": errors, "warnings": warnings}


def validate_fantasy() -> dict[str, Any]:
    """Layer 3 — Fantasy: check fantasy-specific data tables."""
    header("LAYER 3: FANTASY DATA")

    errors: list[str] = []
    warnings: list[str] = []

    with engine.connect() as conn:
        inspector = inspect(engine)
        all_tables = set(inspector.get_table_names())

        metric_count = 0
        projection_count = 0
        sample: list[dict] = []

        if "player_daily_metrics" in all_tables:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM player_daily_metrics "
                    "WHERE season = :s"
                ),
                {"s": SEASON},
            ).scalar()
            metric_count = row or 0
            log(f"PlayerDailyMetrics (season {SEASON}): {metric_count}")
            if metric_count == 0:
                warnings.append(f"No player daily metrics found for season {SEASON}")

            rows = conn.execute(
                text(
                    "SELECT player_id, metric_date, season "
                    "FROM player_daily_metrics "
                    "WHERE season = :s "
                    "ORDER BY metric_date DESC "
                    "LIMIT :lim"
                ),
                {"s": SEASON, "lim": SAMPLE_SIZE},
            ).mappings().all()
            sample = [dict(r) for r in rows]
        else:
            warnings.append("Table 'player_daily_metrics' not found")

        if "player_projections" in all_tables:
            row = conn.execute(
                text("SELECT COUNT(*) FROM player_projections"),
            ).scalar()
            projection_count = row or 0
            log(f"PlayerProjections (all time):      {projection_count}")
        else:
            warnings.append("Table 'player_projections' not found")

    valid = len(errors) == 0
    log(f"Validation: {'✅ PASS' if valid else '❌ FAIL'}")
    for e in errors:
        log(f"  ERROR: {e}")
    for w in warnings:
        log(f"  WARN:  {w}")

    write_artifact("fantasy-records.json", sample)
    write_artifact(
        "fantasy-summary.json",
        {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "stats": {"metric_count": metric_count, "projection_count": projection_count},
        },
    )

    return {"layer": "fantasy", "valid": valid, "errors": errors, "warnings": warnings}


def validate_mlb() -> dict[str, Any]:
    """Layer 4 — MLB: check MLB-specific data tables."""
    header("LAYER 4: MLB DATA")

    errors: list[str] = []
    warnings: list[str] = []

    with engine.connect() as conn:
        inspector = inspect(engine)
        all_tables = set(inspector.get_table_names())

        game_log_count = 0
        player_stats_count = 0
        sample: list[dict] = []

        if "mlb_game_logs" in all_tables:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM mlb_game_logs "
                    "WHERE season = :s"
                ),
                {"s": SEASON},
            ).scalar()
            game_log_count = row or 0
            log(f"MLBGameLogs (season {SEASON}):        {game_log_count}")
            if game_log_count == 0:
                warnings.append(f"No MLB game logs found for season {SEASON}")

            rows = conn.execute(
                text(
                    "SELECT id, player_mlbam_id, game_date, season "
                    "FROM mlb_game_logs "
                    "WHERE season = :s "
                    "ORDER BY game_date DESC "
                    "LIMIT :lim"
                ),
                {"s": SEASON, "lim": SAMPLE_SIZE},
            ).mappings().all()
            sample = [dict(r) for r in rows]
        else:
            warnings.append("Table 'mlb_game_logs' not found")

        if "mlb_player_stats" in all_tables:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM mlb_player_stats "
                    "WHERE season = :s"
                ),
                {"s": SEASON},
            ).scalar()
            player_stats_count = row or 0
            log(f"MLBPlayerStats (season {SEASON}):    {player_stats_count}")

    valid = len(errors) == 0
    log(f"Validation: {'✅ PASS' if valid else '❌ FAIL'}")
    for e in errors:
        log(f"  ERROR: {e}")
    for w in warnings:
        log(f"  WARN:  {w}")

    write_artifact("mlb-records.json", sample)
    write_artifact(
        "mlb-summary.json",
        {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "stats": {"game_log_count": game_log_count, "player_stats_count": player_stats_count},
        },
    )

    return {"layer": "mlb", "valid": valid, "errors": errors, "warnings": warnings}


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    global engine

    # Create the database engine here — after all early-exit guards at module
    # level have already validated DATABASE_URL (non-empty, not Railway-internal).
    # This makes the guard dependency explicit and avoids a module-level engine
    # that could be reached with an invalid URL if guards are ever restructured.
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)  # type: ignore[arg-type]

    log("")
    log("=" * 70)
    log("DB-BACKED MANUAL VALIDATION RUN")
    log(f"Season: {SEASON}  |  Sample: {SAMPLE_SIZE} records per layer")
    log(f"Artifacts: {'✅ enabled → ' + ARTIFACT_DIR if ARTIFACTS_ENABLED else '⛔ disabled'}")
    log("=" * 70)

    # Write run metadata artifact
    write_artifact(
        "run-metadata.json",
        {
            "run_timestamp": _run_ts,
            "season": SEASON,
            "sample_size": SAMPLE_SIZE,
            "artifacts_enabled": ARTIFACTS_ENABLED,
            "artifact_dir": ARTIFACT_DIR,
            "max_artifact_rows": MAX_ARTIFACT_ROWS,
        },
    )

    # Run all layers
    layer_fns = [validate_ingestion, validate_predictions, validate_fantasy, validate_mlb]
    for fn in layer_fns:
        try:
            result = fn()
            _layer_results.append(result)
        except Exception as exc:
            layer_name = fn.__name__.replace("validate_", "")
            log(f"\n❌ Validation run failed at layer '{layer_name}': {exc}")
            write_artifact(
                "validation-report.json",
                {
                    "overall_valid": False,
                    "failed_layer": layer_name,
                    "error": str(exc),
                    "layers": _layer_results,
                },
            )
            raise SystemExit(1) from exc

    # Overall report
    overall_valid = all(r["valid"] for r in _layer_results)
    write_artifact(
        "validation-report.json",
        {
            "overall_valid": overall_valid,
            "season": SEASON,
            "layers": _layer_results,
        },
    )

    header("VALIDATION COMPLETE")
    for r in _layer_results:
        status = "✅ PASS" if r["valid"] else "❌ FAIL"
        log(f"  {r['layer']:20s} {status}")
    log("")
    if overall_valid:
        log("✅ All layers passed.")
    else:
        log("❌ One or more layers failed. See artifacts for details.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
