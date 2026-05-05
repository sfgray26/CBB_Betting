"""
Backfill Savant Pitch Quality scores from statcast_pitcher_metrics.

This computes an inactive-by-default research/waiver signal. It does not enable
waiver or projection feature flags.

Run:
    railway run python scripts/backfill_savant_pitch_quality.py
or:
    DATABASE_URL=<url> python scripts/backfill_savant_pitch_quality.py
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import Json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.fantasy_baseball.savant_pitch_quality import (  # noqa: E402
    SavantPitcherInput,
    score_pitcher_population,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


def get_db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("DATABASE_URL="):
                        url = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not url:
        logger.error("DATABASE_URL not found in environment or .env")
        sys.exit(1)
    return url


def _fetch_pitchers(cur, season: int, as_of_date) -> list[SavantPitcherInput]:
    cur.execute(
        """
        SELECT
            mlbam_id,
            player_name,
            team,
            season,
            xera,
            xwoba,
            barrel_percent_allowed,
            hard_hit_percent_allowed,
            avg_exit_velocity_allowed,
            k_percent,
            bb_percent,
            k_9,
            whiff_percent,
            ip,
            CAST(COALESCE(ip, 0) * 16 AS integer) AS pitches,
            era,
            whip
        FROM statcast_pitcher_metrics
        WHERE season = %s
        """,
        (season,),
    )
    rows = cur.fetchall()
    pitchers: list[SavantPitcherInput] = []
    for row in rows:
        (
            mlbam_id,
            player_name,
            team,
            row_season,
            xera,
            xwoba,
            barrel_percent_allowed,
            hard_hit_percent_allowed,
            avg_exit_velocity_allowed,
            k_percent,
            bb_percent,
            k_9,
            whiff_percent,
            ip,
            pitches,
            era,
            whip,
        ) = row
        pitchers.append(
            SavantPitcherInput(
                player_id=str(mlbam_id),
                player_name=player_name,
                team=team,
                season=int(row_season),
                as_of_date=as_of_date,
                xera=xera,
                xwoba=xwoba,
                barrel_percent_allowed=barrel_percent_allowed,
                hard_hit_percent_allowed=hard_hit_percent_allowed,
                avg_exit_velocity_allowed=avg_exit_velocity_allowed,
                k_percent=k_percent,
                bb_percent=bb_percent,
                k_9=k_9,
                whiff_percent=whiff_percent,
                ip=ip,
                pitches=pitches,
                era=era,
                whip=whip,
            )
        )
    return pitchers


def backfill(season: int = 2026) -> dict:
    as_of_date = datetime.now(_ET).date()
    conn = psycopg2.connect(get_db_url())
    conn.autocommit = False
    cur = conn.cursor()
    try:
        pitchers = _fetch_pitchers(cur, season, as_of_date)
        scores = score_pitcher_population(pitchers)

        upserted = 0
        for score in scores:
            inputs = score.inputs
            cur.execute(
                """
                INSERT INTO savant_pitch_quality_scores (
                    player_id,
                    player_name,
                    team,
                    season,
                    as_of_date,
                    savant_pitch_quality,
                    arsenal_quality,
                    bat_missing_skill,
                    contact_suppression,
                    command_stability,
                    trend_adjustment,
                    sample_confidence,
                    signals,
                    inputs,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (player_id, season, as_of_date) DO UPDATE SET
                    player_name = EXCLUDED.player_name,
                    team = EXCLUDED.team,
                    savant_pitch_quality = EXCLUDED.savant_pitch_quality,
                    arsenal_quality = EXCLUDED.arsenal_quality,
                    bat_missing_skill = EXCLUDED.bat_missing_skill,
                    contact_suppression = EXCLUDED.contact_suppression,
                    command_stability = EXCLUDED.command_stability,
                    trend_adjustment = EXCLUDED.trend_adjustment,
                    sample_confidence = EXCLUDED.sample_confidence,
                    signals = EXCLUDED.signals,
                    inputs = EXCLUDED.inputs,
                    updated_at = NOW()
                """,
                (
                    score.player_id,
                    score.player_name,
                    inputs.get("team"),
                    score.season,
                    score.as_of_date,
                    score.savant_pitch_quality,
                    score.arsenal_quality,
                    score.bat_missing_skill,
                    score.contact_suppression,
                    score.command_stability,
                    score.trend_adjustment,
                    score.sample_confidence,
                    Json(score.signals),
                    Json(inputs),
                ),
            )
            upserted += cur.rowcount

        conn.commit()
        high_confidence = sum(1 for score in scores if score.sample_confidence >= 0.70)
        breakout_arms = sum(1 for score in scores if "BREAKOUT_ARM" in score.signals)
        watchlist = sum(1 for score in scores if "WATCHLIST" in score.signals)
        result = {
            "season": season,
            "as_of_date": as_of_date.isoformat(),
            "source_rows": len(pitchers),
            "scores_upserted": upserted,
            "high_confidence_scores": high_confidence,
            "breakout_arms": breakout_arms,
            "watchlist": watchlist,
        }
        logger.info("Savant Pitch Quality backfill complete: %s", result)
        return result
    except Exception as exc:
        conn.rollback()
        logger.error("Savant Pitch Quality backfill failed: %s", exc)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    backfill()
