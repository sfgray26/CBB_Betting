"""
Backfill Script: Statcast Performances (March 20 - April 8, 2026)

Fetches historical Statcast data from Baseball Savant for the 2026 MLB season.
Uses CSV export API with date-specific queries to fetch batter and pitcher data.

This is a ONE-TIME backfill — ongoing sync happens via daily_ingestion.py (every 6 hours)

Usage:
    python scripts/backfill_statcast.py

Expected Output:
    ~20,000 rows in statcast_performances table (18 days × 750 players)

Note: This script can take 10-20 minutes to run due to API rate limits and data volume.
"""
import logging
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.models import SessionLocal, StatcastPerformance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Season dates
OPENING_DAY = date(2026, 3, 20)  # Approximate MLB opening day 2026
TODAY = date.today()

# Baseball Savant CSV export API
BASEBALL_SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


def backfill_statcast(start_date: date = OPENING_DAY, end_date: date = TODAY) -> dict:
    """
    Backfill Statcast data from start_date to end_date.

    Args:
        start_date: First date to backfill (default: March 20, 2026)
        end_date: Last date to backfill (default: today)

    Returns:
        dict with status, records_processed, elapsed_ms, date_range
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info(f"Starting Statcast backfill: {start_date} to {end_date}")
    logger.info("=" * 60)
    logger.warning("This may take 10-20 minutes due to API rate limits")

    try:
        db = SessionLocal()

        total_records = 0
        dates_processed = 0
        dates_with_errors = 0
        dates_with_no_data = 0

        # Iterate through each day in the range
        current_date = start_date
        while current_date <= end_date:
            try:
                logger.info(f"Fetching Statcast data for {current_date}...")

                # Fetch both batters and pitchers for this date
                date_records = _fetch_statcast_day(current_date, db)

                if date_records == 0:
                    logger.warning(f"No Statcast data for {current_date} (off-day or API issue)")
                    dates_with_no_data += 1
                else:
                    logger.info(f"✅ Fetched {date_records} records for {current_date}")
                    dates_processed += 1
                    total_records += date_records

                # Be respectful of the API — add small delay between dates
                time.sleep(1)

            except Exception as e:
                logger.error(f"Failed to process date {current_date}: {e}")
                dates_with_errors += 1

            current_date += timedelta(days=1)

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("Statcast backfill complete")
        logger.info(f"  Date range: {start_date} to {end_date}")
        logger.info(f"  Dates processed: {dates_processed}")
        logger.info(f"  Dates with no data: {dates_with_no_data}")
        logger.info(f"  Dates with errors: {dates_with_errors}")
        logger.info(f"  Total records: {total_records}")
        logger.info(f"  Elapsed: {elapsed}ms")
        logger.info("=" * 60)

        # Verify table count
        table_count = db.query(StatcastPerformance).count()
        logger.info(f"Total rows in statcast_performances table: {table_count}")

        db.close()

        return {
            'status': 'success',
            'records_processed': total_records,
            'dates_processed': dates_processed,
            'dates_with_no_data': dates_with_no_data,
            'dates_with_errors': dates_with_errors,
            'date_range': f"{start_date} to {end_date}",
            'table_count': table_count,
            'elapsed_ms': elapsed
        }

    except Exception as e:
        logger.exception("Statcast backfill failed: %s", e)
        return {
            'status': 'failed',
            'records_processed': 0,
            'error': str(e),
            'elapsed_ms': int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        }


def _fetch_statcast_day(target_date: date, db: Session) -> int:
    """
    Fetch and store Statcast data for a single date (both batters and pitchers).

    Returns number of records stored, or 0 if no data available.
    """
    records_stored = 0

    # Fetch batter data
    batter_df = _fetch_by_player_type(target_date, 'batter')
    if batter_df is not None and len(batter_df) > 0:
        batter_df = batter_df.copy()
        batter_df['_statcast_player_type'] = 'batter'
        records_stored += _store_performances(batter_df, db, target_date)

    # Fetch pitcher data
    pitcher_df = _fetch_by_player_type(target_date, 'pitcher')
    if pitcher_df is not None and len(pitcher_df) > 0:
        pitcher_df = pitcher_df.copy()
        pitcher_df['_statcast_player_type'] = 'pitcher'
        records_stored += _store_performances(pitcher_df, db, target_date)

    return records_stored


def _fetch_by_player_type(target_date: date, player_type: str):
    """Fetch Statcast data for a specific date and player type from Baseball Savant."""
    logger.info(f"Fetching Statcast {player_type} data for {target_date}")

    params = {
        'all': 'true',
        'hfGT': 'R|',
        'hfSea': f'{target_date.year}|',
        'player_type': player_type,
        'game_date_gt': (target_date - timedelta(days=1)).isoformat(),
        'game_date_lt': (target_date + timedelta(days=1)).isoformat(),
        'group_by': 'name-date',
        'sort_col': 'pitches',
        'player_event_sort': 'api_p_release_speed',
        'sort_order': 'desc',
        'type': 'details',
    }

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(
            BASEBALL_SAVANT_URL,
            params=params,
            headers=headers,
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(f"Statcast API returned {response.status_code} for {player_type} on {target_date}")
            return None

        from io import StringIO
        text_content = response.text
        df = pd.read_csv(StringIO(text_content))

        logger.info(f"Statcast {player_type}: {len(df)} rows fetched ({len(text_content)} bytes)")
        return df

    except Exception as e:
        logger.exception(f"Failed to fetch Statcast {player_type} data for {target_date}: {e}")
        return None


def _store_performances(df: pd.DataFrame, db: Session, target_date: date) -> int:
    """Transform and upert Statcast performances to database."""
    from backend.fantasy_baseball.statcast_ingestion import StatcastIngestionAgent

    # Reuse the transformation logic from statcast_ingestion.py
    agent = StatcastIngestionAgent()
    performances = agent.transform_to_performance(df)

    # Store using upsert logic
    rows_upserted = 0
    now = datetime.now(ZoneInfo("America/New_York"))

    for perf in performances:
        try:
            stmt = pg_insert(StatcastPerformance.__table__).values(
                player_id=perf.player_id,
                player_name=perf.player_name,
                team=perf.team,
                game_date=perf.game_date,
                pa=perf.pa,
                ab=perf.ab,
                h=perf.h,
                doubles=perf.doubles,
                triples=perf.triples,
                hr=perf.hr,
                r=perf.r,
                rbi=perf.rbi,
                bb=perf.bb,
                so=perf.so,
                hbp=perf.hbp,
                sb=perf.sb,
                cs=perf.cs,
                exit_velocity_avg=perf.exit_velocity_avg,
                launch_angle_avg=perf.launch_angle_avg,
                hard_hit_pct=perf.hard_hit_pct,
                barrel_pct=perf.barrel_pct,
                xba=perf.xba,
                xslg=perf.xslg,
                xwoba=perf.xwoba,
                woba=perf.woba,
                avg=perf.avg,
                obp=perf.obp,
                slg=perf.slg,
                ops=perf.ops,
                ip=perf.ip,
                er=perf.er,
                k_pit=perf.k_pit,
                bb_pit=perf.bb_pit,
                pitches=perf.pitches,
                created_at=now,
            ).on_conflict_do_update(
                constraint='uq_player_date',
                set_=dict(
                    player_name=perf.player_name,
                    team=perf.team,
                    pa=perf.pa,
                    ab=perf.ab,
                    h=perf.h,
                    doubles=perf.doubles,
                    triples=perf.triples,
                    hr=perf.hr,
                    r=perf.r,
                    rbi=perf.rbi,
                    bb=perf.bb,
                    so=perf.so,
                    hbp=perf.hbp,
                    sb=perf.sb,
                    cs=perf.cs,
                    exit_velocity_avg=perf.exit_velocity_avg,
                    launch_angle_avg=perf.launch_angle_avg,
                    hard_hit_pct=perf.hard_hit_pct,
                    barrel_pct=perf.barrel_pct,
                    xba=perf.xba,
                    xslg=perf.xslg,
                    xwoba=perf.xwoba,
                    woba=perf.woba,
                    avg=perf.avg,
                    obp=perf.obp,
                    slg=perf.slg,
                    ops=perf.ops,
                    ip=perf.ip,
                    er=perf.er,
                    k_pit=perf.k_pit,
                    bb_pit=perf.bb_pit,
                    pitches=perf.pitches,
                ),
            )
            db.execute(stmt)
            rows_upserted += 1
        except Exception as e:
            logger.warning(f"Failed to upsert performance for {perf.player_name}: {e}")
            continue

    db.commit()
    logger.info(f"Stored {rows_upserted} {target_date} performances")
    return rows_upserted


if __name__ == "__main__":
    result = backfill_statcast()

    if result['status'] == 'success':
        logger.info(f"✅ Backfill successful: {result['records_processed']} records")
        logger.info(f"   Date range: {result['date_range']}")
        logger.info(f"   Dates processed: {result['dates_processed']}")
        logger.info(f"   Dates with no data: {result['dates_with_no_data']}")
        sys.exit(0)
    else:
        logger.error(f"❌ Backfill failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
