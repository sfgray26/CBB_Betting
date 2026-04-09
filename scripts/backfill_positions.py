"""
Backfill Script: Position Eligibility (Current Snapshot)

Fetches CURRENT position eligibility from Yahoo Fantasy API for all 30 MLB teams.
This creates a baseline snapshot of position data (CF/LF/RF granularity).

Note: Yahoo API does NOT expose historical position data — only current snapshot.
Ongoing updates will track changes over time via daily_ingestion.py

Usage:
    python scripts/backfill_positions.py

Expected Output:
    ~750 rows in position_eligibility table (30 teams × ~25 players with multi-eligibility)
"""
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
from backend.models import SessionLocal, PositionEligibility

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# All 30 MLB team league keys (will be resolved dynamically)
MLB_TEAMS = [
    'lal',  # Los Angeles Angels
    'bal',  # Baltimore Orioles
    'bos',  # Boston Red Sox
    'cws',  # Chicago White Sox
    'cle',  # Cleveland Guardians
    'det',  # Detroit Tigers
    'hou',  # Houston Astros
    'kc',   # Kansas City Royals
    'laa',  # Los Angeles Angels (alternate)
    'mia',  # Miami Marlins
    'mil',  # Milwaukee Brewers
    'min',  # Minnesota Twins
    'nyy',  # New York Yankees
    'oak',  # Oakland Athletics
    'sea',  # Seattle Mariners
    'tbr',  # Tampa Bay Rays
    'tex',  # Texas Rangers
    'tor',  # Toronto Blue Jays
    'ari',  # Arizona Diamondbacks
    'atl',  # Atlanta Braves
    'chc',  # Chicago Cubs
    'cin',  # Cincinnati Reds
    'col',  # Colorado Rockies
    'lad',  # Los Angeles Dodgers
    'nym',  # New York Mets
    'phi',  # Philadelphia Phillies
    'pit',  # Pittsburgh Pirates
    'sd',   # San Diego Padres
    'sfg',  # San Francisco Giants
    'stl',  # St. Louis Cardinals
    'was',  # Washington Nationals
]


def backfill_position_eligibility() -> dict:
    """
    Fetch current position eligibility from Yahoo Fantasy API for all MLB teams.

    Returns:
        dict with status, records_processed, elapsed_ms
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Starting position eligibility backfill (current snapshot)")
    logger.info("=" * 60)

    try:
        yahoo_client = YahooFantasyClient()
        db = SessionLocal()

        # Get user's league to find team keys
        # Note: This assumes the user has access to at least one fantasy baseball league
        logger.info("Fetching user's fantasy leagues...")

        # Get the user's team key from their configured league
        team_key = yahoo_client.get_my_team_key()
        if not team_key:
            logger.error("Could not determine user's team key")
            logger.error("This script requires at least one Yahoo Fantasy Baseball league")
            return {
                'status': 'failed',
                'records_processed': 0,
                'error': 'No Yahoo Fantasy team key found',
                'elapsed_ms': int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
            }

        # Fetch all rosters from the league (all 30 teams)
        logger.info(f"Fetching all team rosters from league {yahoo_client.league_key}...")

        records_processed = 0
        records_created = 0
        records_updated = 0
        teams_processed = 0

        # Get all team rosters from the league
        all_rosters = yahoo_client.get_league_rosters(yahoo_client.league_key)
        if not all_rosters:
            logger.error("Failed to fetch league rosters")
            return {
                'status': 'failed',
                'records_processed': 0,
                'error': 'Failed to fetch league rosters',
                'elapsed_ms': int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
            }

        # Process each team roster
        for roster_entry in all_rosters:
            try:
                team_key = roster_entry.get('team_key')
                team_name = roster_entry.get('name', 'Unknown')

                if not team_key:
                    continue

                logger.info(f"Processing {team_name} ({team_key})...")

                # Get roster for this team
                roster = yahoo_client.get_team_roster(team_key)
                if not roster:
                    logger.warning(f"No roster data for {team_name}")
                    continue

                # Process each player on roster
                players_data = roster.get('players', {}).get('player', [])
                if not players_data:
                    logger.warning(f"No players on {team_name} roster")
                    continue

                logger.info(f"Found {len(players_data)} players on {team_name} roster")

                for player_data in players_data:
                    try:
                        # Extract player info
                        player_info = player_data if isinstance(player_data, dict) else {}
                        player_key = player_info.get('player_key', '')

                        if not player_key:
                            continue

                        # Extract player name
                        name = player_info.get('name', {}).get('full', '')
                        first_name = player_info.get('name', {}).get('first', '')
                        last_name = player_info.get('name', {}).get('last', '')

                        # Extract position eligibility
                        # Yahoo returns position data as: {'position': 'C'}, {'position': '1B'}, etc.
                        # OR in eligible_positions array
                        positions = player_info.get('eligible_positions', [])
                        if not positions:
                            # Try alternate format
                            position_data = player_info.get('position', {})
                            if position_data:
                                positions = [position_data]

                        if not positions:
                            logger.debug(f"No position data for {name}")
                            continue

                        # Process each position
                        for pos_data in positions:
                            if isinstance(pos_data, dict):
                                pos_type = pos_data.get('position', '')
                            elif isinstance(pos_data, str):
                                pos_type = pos_data
                            else:
                                continue

                            if not pos_type:
                                continue

                            # Map Yahoo position to database columns
                            can_play_flags = _map_position_to_flags(pos_type)

                            # Check if record exists
                            # Note: We don't have BDL player ID yet — will be updated later
                            existing = db.query(PositionEligibility).filter(
                                PositionEligibility.yahoo_player_key == player_key,
                                PositionEligibility.position_type == pos_type
                            ).first()

                            now = datetime.now(ZoneInfo("America/New_York"))

                            if existing:
                                existing.player_name = name
                                existing.first_name = first_name
                                existing.last_name = last_name
                                existing.updated_at = now
                                records_updated += 1
                            else:
                                record = PositionEligibility(
                                    yahoo_player_key=player_key,
                                    bdl_player_id=None,  # Will be populated via player_id_mapping
                                    player_name=name,
                                    first_name=first_name,
                                    last_name=last_name,
                                    position_type=pos_type,
                                    **can_play_flags,
                                    created_at=now,
                                    updated_at=now
                                )
                                db.add(record)
                                records_created += 1

                            records_processed += 1

                    except Exception as e:
                        logger.warning(f"Failed to process player: {e}")
                        continue

                # Commit after each team
                db.commit()
                teams_processed += 1
                logger.info(f"✅ Processed {team_name}: {len(players_data)} players")

            except Exception as e:
                logger.error(f"Failed to process team: {e}")
                continue

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("Position eligibility backfill complete")
        logger.info(f"  Teams processed: {teams_processed}")
        logger.info(f"  Records processed: {records_processed}")
        logger.info(f"  Records created: {records_created}")
        logger.info(f"  Records updated: {records_updated}")
        logger.info(f"  Elapsed: {elapsed}ms")
        logger.info("=" * 60)

        # Verify table count
        table_count = db.query(PositionEligibility).count()
        logger.info(f"Total rows in position_eligibility table: {table_count}")

        db.close()

        return {
            'status': 'success',
            'records_processed': records_processed,
            'records_created': records_created,
            'records_updated': records_updated,
            'teams_processed': teams_processed,
            'table_count': table_count,
            'elapsed_ms': elapsed
        }

    except Exception as e:
        logger.exception("Position eligibility backfill failed: %s", e)
        return {
            'status': 'failed',
            'records_processed': 0,
            'error': str(e),
            'elapsed_ms': int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        }


def _map_position_to_flags(position: str) -> dict:
    """
    Map Yahoo position abbreviation to can_play_* flags.

    Yahoo positions:
    - C, 1B, 2B, 3B, SS, LF, CF, RF, OF, DH, UTIL

    Returns dict with can_play_* flags set to True.
    """
    flags = {
        'can_play_c': False,
        'can_play_1b': False,
        'can_play_2b': False,
        'can_play_3b': False,
        'can_play_ss': False,
        'can_play_lf': False,
        'can_play_cf': False,
        'can_play_rf': False,
        'can_play_of': False,
        'can_play_dh': False,
        'can_play_util': False,
    }

    position_upper = position.upper()

    if position_upper == 'C':
        flags['can_play_c'] = True
    elif position_upper == '1B':
        flags['can_play_1b'] = True
    elif position_upper == '2B':
        flags['can_play_2b'] = True
    elif position_upper == '3B':
        flags['can_play_3b'] = True
    elif position_upper == 'SS':
        flags['can_play_ss'] = True
    elif position_upper == 'LF':
        flags['can_play_lf'] = True
        flags['can_play_of'] = True
    elif position_upper == 'CF':
        flags['can_play_cf'] = True
        flags['can_play_of'] = True
    elif position_upper == 'RF':
        flags['can_play_rf'] = True
        flags['can_play_of'] = True
    elif position_upper == 'OF':
        flags['can_play_of'] = True
    elif position_upper == 'DH':
        flags['can_play_dh'] = True
    elif position_upper == 'UTIL':
        flags['can_play_util'] = True

    return flags


if __name__ == "__main__":
    result = backfill_position_eligibility()

    if result['status'] == 'success':
        logger.info(f"✅ Backfill successful: {result['records_processed']} records")
        logger.info(f"   Teams processed: {result['teams_processed']}")
        sys.exit(0)
    else:
        logger.error(f"❌ Backfill failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
