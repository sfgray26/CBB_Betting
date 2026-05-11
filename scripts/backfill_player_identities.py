#!/usr/bin/env python
"""
Backfill player_identities from player_id_mapping.

Usage
-----
    python scripts/backfill_player_identities.py --dry-run
    python scripts/backfill_player_identities.py
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _require_database_url() -> str:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    return db_url


def _source_columns(conn) -> dict[str, str | None]:
    rows = conn.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'player_id_mapping'
            ORDER BY ordinal_position
            """
        )
    ).fetchall()
    columns = {row[0] for row in rows}

    if "full_name" in columns:
        name_column = "full_name"
    elif "player_name" in columns:
        name_column = "player_name"
    else:
        raise RuntimeError("player_id_mapping is missing both full_name and player_name")

    return {
        "name_column": name_column,
        "yahoo_guid_column": "yahoo_key" if "yahoo_key" in columns else None,
        "fangraphs_column": "fangraphs_id" if "fangraphs_id" in columns else None,
    }


def _sql_statements(source: dict[str, str | None]) -> list[str]:
    name_column = source["name_column"]
    yahoo_guid_expr = source["yahoo_guid_column"] or "NULL"
    fangraphs_expr = source["fangraphs_column"] or "NULL"

    return [
        f"""
        INSERT INTO player_identities (
            yahoo_guid,
            yahoo_id,
            mlbam_id,
            fangraphs_id,
            full_name,
            normalized_name,
            active
        )
        SELECT
            {yahoo_guid_expr},
            yahoo_id,
            mlbam_id,
            {fangraphs_expr},
            {name_column},
            lower(trim({name_column})),
            TRUE
        FROM player_id_mapping pim
        WHERE yahoo_id IS NOT NULL
          AND {name_column} IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM player_identities pi
              WHERE pi.yahoo_id = pim.yahoo_id
                 OR ({yahoo_guid_expr} IS NOT NULL AND pi.yahoo_guid = {yahoo_guid_expr})
                 OR (pim.mlbam_id IS NOT NULL AND pi.mlbam_id = pim.mlbam_id)
                 OR ({fangraphs_expr} IS NOT NULL AND pi.fangraphs_id = {fangraphs_expr})
          )
        ON CONFLICT DO NOTHING
        """,
        f"""
        INSERT INTO player_identities (
            yahoo_guid,
            yahoo_id,
            mlbam_id,
            fangraphs_id,
            full_name,
            normalized_name,
            active
        )
        SELECT
            {yahoo_guid_expr},
            yahoo_id,
            mlbam_id,
            {fangraphs_expr},
            {name_column},
            lower(trim({name_column})),
            TRUE
        FROM player_id_mapping pim
        WHERE yahoo_id IS NULL
          AND {yahoo_guid_expr} IS NOT NULL
          AND {name_column} IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM player_identities pi
              WHERE pi.yahoo_guid = {yahoo_guid_expr}
                 OR (pim.mlbam_id IS NOT NULL AND pi.mlbam_id = pim.mlbam_id)
                 OR ({fangraphs_expr} IS NOT NULL AND pi.fangraphs_id = {fangraphs_expr})
          )
        ON CONFLICT DO NOTHING
        """,
        f"""
        INSERT INTO player_identities (
            yahoo_guid,
            yahoo_id,
            mlbam_id,
            fangraphs_id,
            full_name,
            normalized_name,
            active
        )
        SELECT
            {yahoo_guid_expr},
            yahoo_id,
            mlbam_id,
            {fangraphs_expr},
            {name_column},
            lower(trim({name_column})),
            TRUE
        FROM player_id_mapping pim
        WHERE yahoo_id IS NULL
          AND {yahoo_guid_expr} IS NULL
          AND pim.mlbam_id IS NOT NULL
          AND {name_column} IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM player_identities pi
              WHERE pi.mlbam_id = pim.mlbam_id
                 OR ({fangraphs_expr} IS NOT NULL AND pi.fangraphs_id = {fangraphs_expr})
          )
        ON CONFLICT DO NOTHING
        """,
    ]


def run(dry_run: bool = False) -> None:
    engine = create_engine(_require_database_url())

    with engine.connect() as conn:
        source = _source_columns(conn)
        statements = _sql_statements(source)

    print("=== Backfill player_identities from player_id_mapping ===")
    print(
        "Source columns:",
        {
            "name_column": source["name_column"],
            "yahoo_guid_column": source["yahoo_guid_column"],
            "fangraphs_column": source["fangraphs_column"],
        },
    )

    if dry_run:
        for idx, statement in enumerate(statements, start=1):
            print(f"\n--- Statement {idx} ---")
            print(statement.strip())
        return

    inserted_total = 0
    with engine.begin() as conn:
        for idx, statement in enumerate(statements, start=1):
            result = conn.execute(text(statement))
            rowcount = result.rowcount if result.rowcount is not None else 0
            inserted_total += max(rowcount, 0)
            print(f"Statement {idx}: inserted {rowcount} rows")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM player_identities")).scalar()
        with_yahoo_guid = conn.execute(
            text("SELECT COUNT(*) FROM player_identities WHERE yahoo_guid IS NOT NULL")
        ).scalar()
        with_yahoo_id = conn.execute(
            text("SELECT COUNT(*) FROM player_identities WHERE yahoo_id IS NOT NULL")
        ).scalar()
        with_mlbam = conn.execute(
            text("SELECT COUNT(*) FROM player_identities WHERE mlbam_id IS NOT NULL")
        ).scalar()
        with_fangraphs = conn.execute(
            text("SELECT COUNT(*) FROM player_identities WHERE fangraphs_id IS NOT NULL")
        ).scalar()

    print("\n=== Verification ===")
    print(f"Inserted this run: {inserted_total}")
    print(f"player_identities total: {total}")
    print(f"with yahoo_guid: {with_yahoo_guid}")
    print(f"with yahoo_id: {with_yahoo_id}")
    print(f"with mlbam_id: {with_mlbam}")
    print(f"with fangraphs_id: {with_fangraphs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill player_identities from player_id_mapping")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
