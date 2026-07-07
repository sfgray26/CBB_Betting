"""
Check if MISSING_SCORE players have MLB game logs.

If they have game logs but no rolling stats, ingestion failed.
If they have no game logs, they haven't played (minors/injured).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.models import SessionLocal, MLBPlayerStats, PlayerRollingStats


def check_player_game_log(db, player_name: str, bdl_id: int, mlbam_id: int):
    """Check if a player has MLB game logs and rolling stats."""
    print(f"\n{'='*60}")
    print(f"Checking: {player_name}")
    print(f"BDL ID: {bdl_id}, MLBAM ID: {mlbam_id}")
    print(f"{'='*60}")

    # Check MLB player stats (by bdl_player_id)
    stats_count = 0
    if bdl_id:
        stats_count = db.query(MLBPlayerStats).filter(
            MLBPlayerStats.bdl_player_id == bdl_id
        ).count()
        print(f"MLB Player Stats: {stats_count} rows")

        if stats_count > 0:
            latest = db.query(MLBPlayerStats).filter(
                MLBPlayerStats.bdl_player_id == bdl_id
            ).order_by(MLBPlayerStats.game_date.desc()).limit(1).first()
            if latest:
                print(f"  Latest game: {latest.game_date}")

    # Check rolling stats
    rolling_count = 0
    if bdl_id:
        rolling_count = db.query(PlayerRollingStats).filter(
            PlayerRollingStats.bdl_player_id == bdl_id
        ).count()
        print(f"Rolling Stats: {rolling_count} rows")

        if rolling_count > 0:
            latest = db.query(PlayerRollingStats).filter(
                PlayerRollingStats.bdl_player_id == bdl_id
            ).order_by(PlayerRollingStats.as_of_date.desc()).limit(1).first()
            if latest:
                print(f"  Latest as_of: {latest.as_of_date}, window={latest.window_days}d")

    # Diagnosis
    print(f"\nDiagnosis:")
    if stats_count == 0:
        print(f"  -> NO game data - player likely in minors or hasn't played")
        return "NO_GAME_DATA"
    elif rolling_count == 0:
        print(f"  -> HAS game logs but NO rolling stats - ingestion pipeline failure")
        return "INGESTION_FAILURE"
    else:
        print(f"  -> HAS both game logs and rolling stats")
        return "HAS_DATA"


def main():
    db = SessionLocal()

    try:
        # MISSING_SCORE players
        players = [
            ("Dillon Dingler", 693307, 693307),
            ("Tyler Soderstrom", 691016, 691016),
            ("Juan Soto", 665742, 665742),
        ]

        print("="*60)
        print("MISSING_SCORE PLAYER GAME LOG CHECK")
        print("="*60)

        for name, bdl_id, mlbam_id in players:
            diagnosis = check_player_game_log(db, name, bdl_id, mlbam_id)

    finally:
        db.close()


if __name__ == "__main__":
    main()
