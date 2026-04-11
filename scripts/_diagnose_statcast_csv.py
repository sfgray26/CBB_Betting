"""Diagnostic: Capture actual Baseball Savant CSV columns for 2026 season."""
import logging
import pandas as pd
from datetime import date, timedelta
from io import StringIO
import requests

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

BASEBALL_SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

def fetch_csv_columns(target_date: date, player_type: str) -> list[str]:
    """Fetch CSV for given date/player_type and return sorted column names."""
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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    response = requests.get(BASEBALL_SAVANT_URL, params=params, headers=headers, timeout=60)
    if response.status_code != 200:
        logger.error(f"HTTP {response.status_code} for {player_type} on {target_date}")
        return []

    df = pd.read_csv(StringIO(response.text))
    logger.info(f"\n{'='*60}")
    logger.info(f"{player_type.upper()} CSV for {target_date}: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"{'='*60}")
    logger.info("Columns (sorted):")
    for col in sorted(df.columns):
        sample_val = df[col].iloc[0] if len(df) > 0 else None
        logger.info(f"  {col:40s} = {repr(sample_val)[:50]}")
    logger.info(f"{'='*60}\n")
    return sorted(df.columns)

# Fetch April 9, 2026 (recent game day)
test_date = date(2026, 4, 9)
batter_cols = fetch_csv_columns(test_date, 'batter')
pitcher_cols = fetch_csv_columns(test_date, 'pitcher')

# Check for expected columns
logger.info("DIAGNOSTIC SUMMARY:")
logger.info(f"  'player_id' in batter columns: {('player_id' in batter_cols)}")
logger.info(f"  'player_id' in pitcher columns: {('player_id' in pitcher_cols)}")
logger.info(f"  'player_name' in batter columns: {('player_name' in batter_cols)}")
logger.info(f"  'player_name' in pitcher columns: {('player_name' in pitcher_cols)}")
logger.info(f"  'last_name' in batter columns: {('last_name' in batter_cols)}")
logger.info(f"  'first_name' in batter columns: {('first_name' in batter_cols)}")