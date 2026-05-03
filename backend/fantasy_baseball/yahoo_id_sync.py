"""Sync Yahoo player IDs from fantasy API to player_id_mapping table."""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text
from backend.models import SessionLocal
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

logger = logging.getLogger(__name__)

ADVISORY_LOCK_ID = 100_034


def _try_advisory_lock(db, lock_id: int) -> bool:
    """
    Try to acquire a PostgreSQL advisory lock.

    Returns True if lock acquired, False if already held.
    """
    result = db.execute(text(f"SELECT pg_try_advisory_lock({lock_id})")).scalar()
    return bool(result)


def _release_advisory_lock(db, lock_id: int):
    """Release a PostgreSQL advisory lock."""
    db.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))


def sync_yahoo_player_ids() -> int:
    """
    Fetch all players from Yahoo fantasy league and map to BDL IDs.
    Returns count of players synced.
    """
    client = YahooFantasyClient()
    db = SessionLocal()

    try:
        # Get all players from league
        league_key = "mlb.l.72586"

        logger.info(f"Fetching players from {league_key}")

        # Use league rosters endpoint
        players_data = client.get_league_rosters(league_key)

        if not players_data:
            logger.warning(f"No players returned from {league_national_key}")
            return 0

        synced = 0
        for player in players_data:
            # Extract Yahoo player key
            player_key = player.get('player_key', '')
            if not player_key:
                continue

            # Extract Yahoo ID (last part of key: 12345)
            yahoo_id = player_key.split('.')[-1] if player_key else None

            # Get BDL player_id from name lookup
            name = player.get('name', '')
            team = player.get('editorial_team_abbr', '')

            if not name or not yahoo_id:
                continue

            # Try to find existing BDL player_id
            player_id = _lookup_bdl_id(db, name, team)

            if player_id:
                # Upsert to player_id_mapping
                db.execute(text('''
                  INSERT INTO player_id_mapping (yahoo_id, player_id, updated_at)
                  VALUES (:yahoo_id, :player_id, :updated_at)
                  ON CONFLICT (player_id) DO UPDATE
                  SET yahoo_id = EXCLUDED.yahoo_id,
                      updated_at = EXCLUDED.updated_at
                '''), {
                    'yahoo_id': yahoo_id,
                    'player_id': player_id,
                    'updated_at': datetime.now(ZoneInfo("America/New_York"))
                })

                synced += 1

        db.commit()
        logger.info(f"Synced {synced} Yahoo player IDs")
        return synced

    except Exception as e:
        db.rollback()
        logger.error(f"Yahoo ID sync failed: {e}", exc_info=True)
        raise
    finally:
        db.close()

def _lookup_bdl_id(db, name: str, team: str | None) -> int | None:
    """Lookup BDL player_id by name and team."""
    # Try exact name match first
    result = db.execute(text('''
      SELECT player_id FROM mlb_player_stats
      WHERE name = :name
      LIMIT 1
    '''), {'name': name}).fetchone()

    if result:
        return result[0]

    # Try name + team match
    if team:
        result = db.execute(text('''
          SELECT player_id FROM mlb_player_stats
          WHERE name = :name AND team = :team
          LIMIT 1
        '''), {'name': name, 'team': team}).fetchone()

        if result:
            return result[0]

    # Not found - return None (will be filled by BDL sync job)
    return None

def run_yahoo_id_sync_job():
    """Run Yahoo ID sync with advisory lock."""
    db = SessionLocal()

    try:
        if _try_advisory_lock(db, ADVISORY_LOCK_ID):
            try:
                logger.info("Starting Yahoo ID sync job")
                count = sync_yahoo_player_ids()
                logger.info(f"Yahoo ID sync complete: {count} players")
                return count
            finally:
                _release_advisory_lock(db, ADVISORY_LOCK_ID)
        else:
            logger.warning("Yahoo ID sync already running")
            return 0
    finally:
        db.close()

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    count = run_yahoo_id_sync_job()
    print(f"Synced {count} Yahoo player IDs")
    sys.exit(0 if count > 0 else 1)
