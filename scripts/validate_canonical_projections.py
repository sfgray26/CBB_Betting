"""
Shadow validation for ProjectionAssemblyService.

Simulates the canonical projection assembly pipeline without writing any rows.
Connects to the DB, loads the player board, and checks:
  - Identity resolution coverage (normalized_name match in player_identities)
  - Statcast coverage (mlbam_id present in statcast_batter/pitcher_metrics)
  - Top-N identity misses by player name

Usage:
    # Against Railway DB (public proxy)
    $env:DATABASE_URL = "postgresql://postgres:<pw>@junction.proxy.rlwy.net:45402/railway"
    .\\venv\\Scripts\\python scripts/validate_canonical_projections.py

    # Show more miss details
    .\\venv\\Scripts\\python scripts/validate_canonical_projections.py --top-misses 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text

from backend.fantasy_baseball.id_resolution_service import _normalize_name
from backend.fantasy_baseball.player_board import get_board


def _load_identity_index(engine):
    """Return set of normalized_name strings from player_identities."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT normalized_name FROM player_identities WHERE active = true"
        )).fetchall()
    return {r[0] for r in rows}


def _load_statcast_mlbam_set(engine):
    """Return set of mlbam_ids that have any statcast metrics row."""
    with engine.connect() as conn:
        batters = conn.execute(text(
            "SELECT player_id FROM statcast_batter_metrics"
        )).fetchall()
        pitchers = conn.execute(text(
            "SELECT player_id FROM statcast_pitcher_metrics"
        )).fetchall()
    return {r[0] for r in batters} | {r[0] for r in pitchers}


def _load_mlbam_by_name(engine):
    """Return dict of normalized_name → mlbam_id from player_identities."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT normalized_name, mlbam_id FROM player_identities WHERE active = true AND mlbam_id IS NOT NULL"
        )).fetchall()
    return {r[0]: r[1] for r in rows}


def main():
    parser = argparse.ArgumentParser(description="Validate canonical projection coverage")
    parser.add_argument(
        "--top-misses",
        type=int,
        default=10,
        help="Number of identity-miss player names to display (default: 10)",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set.")
        sys.exit(1)

    print(f"Connecting to DB...")
    engine = create_engine(db_url, pool_pre_ping=True)

    print("Loading player_identities index...")
    identity_index = _load_identity_index(engine)
    name_to_mlbam = _load_mlbam_by_name(engine)
    print(f"  → {len(identity_index)} active identity rows loaded")

    print("Loading statcast mlbam coverage set...")
    statcast_mlbam = _load_statcast_mlbam_set(engine)
    print(f"  → {len(statcast_mlbam)} players with statcast metrics")

    print("Loading player board...")
    board = get_board()
    print(f"  → {len(board)} players on board")

    if not board:
        print("ERROR: Board returned 0 players. Check DB connection and player_projections table.")
        sys.exit(1)

    # Simulate assembly pass
    total = len(board)
    resolved = 0
    statcast_covered = 0
    misses = []

    for player in board:
        name = player.get("name", "")
        normalized = _normalize_name(name)

        if normalized in identity_index:
            resolved += 1
            mlbam_id = name_to_mlbam.get(normalized)
            if mlbam_id and mlbam_id in statcast_mlbam:
                statcast_covered += 1
        else:
            misses.append(name)

    miss_count = total - resolved
    resolved_pct = resolved / total * 100 if total else 0.0
    statcast_pct = statcast_covered / total * 100 if total else 0.0
    miss_pct = miss_count / total * 100 if total else 0.0

    # Per-type breakdown
    batters = [p for p in board if p.get("type") == "batter"]
    pitchers = [p for p in board if p.get("type") == "pitcher"]
    batter_resolved = sum(
        1 for p in batters if _normalize_name(p.get("name", "")) in identity_index
    )
    pitcher_resolved = sum(
        1 for p in pitchers if _normalize_name(p.get("name", "")) in identity_index
    )

    print()
    print("=" * 60)
    print("  CANONICAL PROJECTION SHADOW VALIDATION REPORT")
    print("=" * 60)
    print(f"  Board total:          {total:>6}")
    print(f"  Identity resolved:    {resolved:>6}  ({resolved_pct:.1f}%)")
    print(f"  Statcast adjusted:    {statcast_covered:>6}  ({statcast_pct:.1f}%)")
    print(f"  Identity misses:      {miss_count:>6}  ({miss_pct:.1f}%)")
    print()
    print(f"  Batters:  {len(batters)} total, {batter_resolved} resolved ({batter_resolved/len(batters)*100:.1f}%)" if batters else "  Batters:  0")
    print(f"  Pitchers: {len(pitchers)} total, {pitcher_resolved} resolved ({pitcher_resolved/len(pitchers)*100:.1f}%)" if pitchers else "  Pitchers: 0")
    print()

    if miss_count > 0:
        top_n = min(args.top_misses, miss_count)
        print(f"  TOP {top_n} IDENTITY MISSES (of {miss_count}):")
        for i, name in enumerate(misses[:top_n], 1):
            normalized = _normalize_name(name)
            print(f"    {i:>3}. {name!r:35} (normalized: {normalized!r})")
    else:
        print("  All board players resolved — 0 identity misses!")

    print("=" * 60)
    print()

    # Gate: warn if resolution is below 80%
    if resolved_pct < 80.0:
        print(f"WARNING: Identity resolution below 80% ({resolved_pct:.1f}%). Do NOT enable CANONICAL_PROJECTION_V1.")
        sys.exit(2)
    elif resolved_pct < 95.0:
        print(f"NOTICE: Identity resolution {resolved_pct:.1f}% — above threshold but under 95%. Review misses above.")
    else:
        print(f"OK: Identity resolution {resolved_pct:.1f}% — safe to consider enabling CANONICAL_PROJECTION_V1.")


if __name__ == "__main__":
    main()
