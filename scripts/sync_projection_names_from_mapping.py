"""
Sync numeric player names in player_projections from player_id_mapping.

Finds rows in player_projections where player_name matches '^[0-9]+$' (pure
numeric — leftover BDL player-ID values never replaced with real names), looks
up the canonical name in player_id_mapping, and updates player_projections.

Usage:
  python scripts/sync_projection_names_from_mapping.py          # dry-run
  python scripts/sync_projection_names_from_mapping.py --execute # commit changes

Run on Railway:
  railway run python scripts/sync_projection_names_from_mapping.py --execute
"""
import argparse
import os
import sys

import psycopg2
import psycopg2.extras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Commit changes; omit for dry-run",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(database_url)
    conn.autocommit = False

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Find projection rows with purely numeric names
            cur.execute(
                """
                SELECT pp.id, pp.player_name, pp.bdl_player_id
                FROM player_projections pp
                WHERE pp.player_name ~ '^[0-9]+$'
                ORDER BY pp.player_name
                """
            )
            numeric_rows = cur.fetchall()

            if not numeric_rows:
                print("No numeric player names found in player_projections — nothing to do.")
                return

            print(f"Found {len(numeric_rows)} row(s) with numeric names:")

            updated = 0
            skipped = 0

            for row in numeric_rows:
                proj_id = row["id"]
                numeric_name = row["player_name"]
                bdl_id = row["bdl_player_id"]

                # Try to find canonical name in player_id_mapping
                # Match on bdl_id if available, otherwise fall back to numeric name as bdl_id
                lookup_bdl_id = bdl_id if bdl_id else numeric_name

                cur.execute(
                    """
                    SELECT full_name
                    FROM player_id_mapping
                    WHERE bdl_id = %s
                      AND full_name IS NOT NULL
                      AND full_name !~ '^[0-9]+$'
                    LIMIT 1
                    """,
                    (str(lookup_bdl_id),),
                )
                mapping_row = cur.fetchone()

                if mapping_row is None:
                    print(
                        f"  SKIP  id={proj_id} name={numeric_name!r} bdl_id={lookup_bdl_id}"
                        " — no matching entry in player_id_mapping"
                    )
                    skipped += 1
                    continue

                canonical_name = mapping_row["full_name"]
                print(
                    f"  {'UPDATE' if args.execute else 'WOULD UPDATE'}"
                    f"  id={proj_id} {numeric_name!r} -> {canonical_name!r}"
                )

                if args.execute:
                    cur.execute(
                        "UPDATE player_projections SET player_name = %s WHERE id = %s",
                        (canonical_name, proj_id),
                    )
                    updated += 1

            if args.execute:
                conn.commit()
                print(f"\nCommitted: {updated} updated, {skipped} skipped.")
            else:
                conn.rollback()
                print(
                    f"\nDry-run complete: {len(numeric_rows) - skipped} would be updated,"
                    f" {skipped} would be skipped."
                    "\nRe-run with --execute to commit."
                )

    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
