#!/usr/bin/env python
"""
DevOps DB query runner — execute arbitrary SQL against the app database.

Usage (local):
    python scripts/devops/db_query.py "SELECT COUNT(*) FROM player_id_mapping"

Usage (Railway production):
    railway ssh python scripts/devops/db_query.py "SELECT COUNT(*) FROM player_id_mapping"

Output: JSON array of objects (one per row).
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from backend.models import engine


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/devops/db_query.py <SQL_QUERY>", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1].strip()
    if not query:
        print("Error: empty query", file=sys.stderr)
        sys.exit(1)

    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = [dict(row) for row in result.mappings()]

    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
