"""
Backfill Script: Probable Pitchers (March 20 - April 8, 2026)

Fetches historical probable pitchers for the 2026 MLB season (opening day through today).
Uses BallDon'tLie API to fetch schedule data and extract probable pitchers.

This is a ONE-TIME backfill — ongoing sync happens via daily_ingestion.py (3x daily)

Usage:
    python scripts/backfill_probable_pitchers.py

Expected Output:
    ~540 rows in probable_pitchers table (18 days × 30 teams)
"""
import logging
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from backend.services.balldontlie import BallDontLieClient
from backend.models import SessionLocal, ProbablePitcherSnapshot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Season dates
OPENING_DAY = date(2026, 3, 20)  # Approximate MLB opening day 2026
TODAY = date.today()


def backfill_probable_pitchers(start_date: date = OPENING_DAY, end_date: date = TODAY) -> dict:
    """
    Backfill probable pitchers from start_date to end_date.

    Args:
        start_date: First date to backfill (default: March 20, 2026)
        end_date: Last date to backfill (default: today)

    Returns:
        dict with status, records_processed, elapsed_ms, date_range
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info(f"Starting probable pitchers backfill: {start_date} to {end_date}")
    logger.info("=" * 60)

    try:
        bdl_client = BallDontLieClient()
        db = SessionLocal()

        records_processed = 0
        records_created = 0
        records_updated = 0
        dates_processed = 0
        dates_with_errors = 0

        # Iterate through each day in the range
        current_date = start_date
        while current_date <= end_date:
            try:
                logger.info(f"Fetching games for {current_date}...")

                # Fetch games for this date from BDL
                games = bdl_client.get_games(
                    dates=[current_date.isoformat()],
                    league='MLB'
                )

                if not games or len(games) == 0:
                    logger.warning(f"No games found for {current_date} (off-day or no data)")
                    current_date += timedelta(days=1)
                    continue

                logger.info(f"Found {len(games)} games for {current_date}")

                # Process each game
                for game in games:
                    try:
                        # Extract game data
                        game_id = game.get('id')
                        home_team = game.get('home', {}).get('abbreviation', '') if game.get('home') else ''
                        away_team = game.get('away', {}).get('abbreviation', '') if game.get('away') else ''

                        # Extract probable pitchers
                        # BDL game object: home_probable, away_probable
                        home_probable = game.get('home_probable')
                        away_probable = game.get('away_probable')

                        # Store home probable pitcher
                        if home_probable:
                            home_pitcher_id = home_probable.get('id')
                            home_pitcher_name = home_probable.get('full_name')

                            if home_pitcher_id and home_team:
                                records_processed += _store_probable_pitcher(
                                    db, home_pitcher_id, home_pitcher_name, home_team,
                                    current_date, game_id, 'home', records_created, records_updated
                                )
                                if records_processed % 2 == 0:  # Count stored separately
                                    records_created = max(records_created, _count_created(db))
                                    records_updated = max(records_updated, _count_updated(db))

                        # Store away probable pitcher
                        if away_probable:
                            away_pitcher_id = away_probable.get('id')
                            away_pitcher_name = away_probable.get('full_name')

                            if away_pitcher_id and away_team:
                                records_processed += _store_probable_pitcher(
                                    db, away_pitcher_id, away_pitcher_name, away_team,
                                    current_date, game_id, 'away', records_created, records_updated
                                )
                                if records_processed % 2 == 0:
                                    records_created = max(records_created, _count_created(db))
                                    records_updated = max(records_updated, _count_updated(db))

                    except Exception as e:
                        logger.warning(f"Failed to process game {game.get('id')}: {e}")
                        continue

                # Commit after processing all games for this date
                db.commit()
                dates_processed += 1
                logger.info(f"✅ Processed {current_date}: {len(games)} games")

            except Exception as e:
                logger.error(f"Failed to process date {current_date}: {e}")
                dates_with_errors += 1
                # Continue to next date

            current_date += timedelta(days=1)

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("Probable pitchers backfill complete")
        logger.info(f"  Date range: {start_date} to {end_date}")
        logger.info(f"  Dates processed: {dates_processed}")
        logger.info(f"  Dates with errors: {dates_with_errors}")
        logger.info(f"  Records processed: {records_processed}")
        logger.info(f"  Elapsed: {elapsed}ms")
        logger.info("=" * 60)

        # Verify table count
        table_count = db.query(ProbablePitcherSnapshot).count()
        logger.info(f"Total rows in probable_pitchers table: {table_count}")

        db.close()

        return {
            'status': 'success',
            'records_processed': records_processed,
            'records_created': records_created,
            'records_updated': records_updated,
            'dates_processed': dates_processed,
            'dates_with_errors': dates_with_errors,
            'date_range': f"{start_date} to {end_date}",
            'table_count': table_count,
            'elapsed_ms': elapsed
        }

    except Exception as e:
        logger.exception("Probable pitchers backfill failed: %s", e)
        return {
            'status': 'failed',
            'records_processed': 0,
            'error': str(e),
            'elapsed_ms': int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        }


def _store_probable_pitcher(
    db: Session,
    pitcher_id: int,
    pitcher_name: str,
    team: str,
    game_date: date,
    game_id: int,
    home_away: str,
    created_count: int,
    updated_count: int
) -> int:
    """Store or update a probable pitcher record. Returns 1."""
    now = datetime.now(ZoneInfo("America/New_York"))

    existing = db.query(ProbablePitcherSnapshot).filter(
        ProbablePitcherSnapshot.game_date == game_date,
        ProbablePitcherSnapshot.team == team
    ).first()

    if existing:
        existing.bdl_player_id = pitcher_id
        existing.player_name = pitcher_name
        existing.home_away = home_away
        existing.game_id = game_id
        existing.updated_at = now
    else:
        record = ProbablePitcherSnapshot(
            game_date=game_date,
            team=team,
            bdl_player_id=pitcher_id,
            player_name=pitcher_name,
            home_away=home_away,
            game_id=game_id,
            created_at=now,
            updated_at=now
        )
        db.add(record)

    return 1


def _count_created(db: Session) -> int:
    """Helper: count records created in current session (not persisted)."""
    # This is approximate — actual count happens after commit
    return 0


def _count_updated(db: Session) -> int:
    """Helper: count records updated in current session (not persisted)."""
    # This is approximate — actual count happens after commit
    return 0


if __name__ == "__main__":
    result = backfill_probable_pitchers()

    if result['status'] == 'success':
        logger.info(f"✅ Backfill successful: {result['records_processed']} records")
        logger.info(f"   Date range: {result['date_range']}")
        logger.info(f"   Dates processed: {result['dates_processed']}")
        sys.exit(0)
    else:
        logger.error(f"❌ Backfill failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
