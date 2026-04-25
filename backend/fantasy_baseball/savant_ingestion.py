"""
Savant Leaderboard Ingestion Pipeline (Phase 9)

Pulls aggregated Statcast metrics from Baseball Savant CSV leaderboards and
stores them in statcast_batter_metrics and statcast_pitcher_metrics tables.

This data powers the Statcast Proxy Engine for generating projections when
Steamer data is missing or outdated.

Usage:
    from backend.fantasy_baseball.savant_ingestion import SavantIngestionAgent

    agent = SavantIngestionAgent(db, season=2026)
    stats = agent.run_daily_ingestion()

Schedule:
    Runs daily at 6:00 AM ET via DailyIngestionOrchestrator.
    Advisory lock: 100_016 ("savant_daily_ingestion")
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any, Optional

import requests
from sqlalchemy import text
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from backend.models import StatcastBatterMetrics, StatcastPitcherMetrics

logger = logging.getLogger(__name__)

# Advisory lock ID for this job
SAVANT_LOCK_ID = 100_016


@dataclass
class SavantMetricsRow:
    """Parsed row from Savant CSV leaderboard."""
    mlbam_id: str
    player_name: str
    team: Optional[str]
    season: int
    metrics: dict[str, Any]


class SavantIngestionAgent:
    """
    Fetches Statcast leaderboards from Baseball Savant CSV endpoints.

    Savant leaderboards provide aggregated season stats including:
    - Batters: xwOBA, barrel%, hard hit%, exit velocity, whiff%, swing%
    - Pitchers: xERA, xwOBA against, barrel% allowed, K%, BB%, K/9

    Data is upserted into statcast_batter_metrics and statcast_pitcher_metrics.
    """

    BASE_URL = "https://baseballsavant.mlbbro.com/statcast_leaderboard"

    # Batter metrics columns from Savant leaderboard
    BATTER_METRICS = [
        "player_id", "last_name", "first_name", "team", "xwoba",
        "barrel_percent", "hard_hit_percent", "avg_exit_velocity",
        "max_exit_velocity", "whiff_percent", "swing_percent",
        "pa", "ab", "h", "hr", "r", "rbi", "sb", "avg", "slg", "ops"
    ]

    # Pitcher metrics columns from Savant leaderboard
    PITCHER_METRICS = [
        "player_id", "last_name", "first_name", "team", "xera",
        "xwoba", "barrel_percent", "hard_hit_percent", "avg_exit_velocity",
        "k_percent", "bb_percent", "k_9", "whiff_percent",
        "w", "l", "qs", "ip", "era", "whip", "sv", "h", "hr", "k"
    ]

    def __init__(self, db: Session, season: int = 2026):
        """
        Initialize the Savant ingestion agent.

        Args:
            db: SQLAlchemy database session
            season: MLB season year (default 2026)
        """
        self.db = db
        self.season = season

    def _fetch_csv(self, url: str) -> str:
        """
        Fetch CSV data from Savant URL.

        Args:
            url: Full Savant leaderboard URL with csv=true parameter

        Returns:
            CSV content as string

        Raises:
            requests.HTTPError: If the request fails
            requests.Timeout: If the request times out
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def _parse_batter_row(self, row: dict[str, str]) -> Optional[SavantMetricsRow]:
        """
        Parse a single batter CSV row into a SavantMetricsRow.

        Handles type conversions:
        - Strips '%' from percentage fields
        - Converts empty strings to None
        - Converts numeric fields to float/int

        Args:
            row: Dict from csv.DictReader

        Returns:
            SavantMetricsRow or None if row is invalid
        """
        try:
            mlbam_id = row.get("player_id", "").strip()
            if not mlbam_id:
                return None

            player_name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()

            metrics = {}
            for key, value in row.items():
                if not value or value.strip() == "":
                    metrics[key] = None
                elif key in ("barrel_percent", "hard_hit_percent", "whiff_percent",
                           "swing_percent", "k_percent", "bb_percent"):
                    # Strip % and convert to float
                    metrics[key] = float(value.rstrip("%"))
                elif key in ("pa", "ab", "h", "hr", "r", "rbi", "sb", "w", "l",
                           "qs", "h", "sv", "hr"):
                    metrics[key] = int(value) if value else None
                else:
                    # Float fields
                    metrics[key] = float(value) if value else None

            return SavantMetricsRow(
                mlbam_id=mlbam_id,
                player_name=player_name,
                team=row.get("team"),
                season=self.season,
                metrics=metrics
            )
        except (ValueError, KeyError) as e:
            logger.warning("Failed to parse batter row: %s — %s", row, e)
            return None

    def _parse_pitcher_row(self, row: dict[str, str]) -> Optional[SavantMetricsRow]:
        """Parse a single pitcher CSV row into a SavantMetricsRow."""
        try:
            mlbam_id = row.get("player_id", "").strip()
            if not mlbam_id:
                return None

            player_name = f"{row.get('last_name', '')}, {row.get('first_name', '')}".strip()

            metrics = {}
            for key, value in row.items():
                if not value or value.strip() == "":
                    metrics[key] = None
                elif key in ("barrel_percent", "hard_hit_percent", "k_percent",
                           "bb_percent", "whiff_percent"):
                    metrics[key] = float(value.rstrip("%")) if value else None
                elif key in ("w", "l", "qs", "h", "sv", "hr", "k"):
                    metrics[key] = int(value) if value else None
                else:
                    metrics[key] = float(value) if value else None

            return SavantMetricsRow(
                mlbam_id=mlbam_id,
                player_name=player_name,
                team=row.get("team"),
                season=self.season,
                metrics=metrics
            )
        except (ValueError, KeyError) as e:
            logger.warning("Failed to parse pitcher row: %s — %s", row, e)
            return None

    def _upsert_batters(self, rows: list[SavantMetricsRow]) -> dict[str, int]:
        """
        Upsert batter metrics into statcast_batter_metrics.

        Uses PostgreSQL ON CONFLICT for upsert semantics.

        Args:
            rows: Parsed batter rows

        Returns:
            Dict with 'inserted' and 'updated' counts
        """
        if not rows:
            return {"inserted": 0, "updated": 0}

        now = datetime.now(ZoneInfo("America/New_York"))

        inserted = 0
        updated = 0

        for row in rows:
            existing = self.db.query(StatcastBatterMetrics).filter_by(
                mlbam_id=row.mlbam_id
            ).first()

            m = row.metrics

            if existing:
                # Update existing row
                existing.player_name = row.player_name
                existing.team = row.team
                existing.xwoba = m.get("xwoba")
                existing.barrel_percent = m.get("barrel_percent")
                existing.hard_hit_percent = m.get("hard_hit_percent")
                existing.avg_exit_velocity = m.get("avg_exit_velocity")
                existing.max_exit_velocity = m.get("max_exit_velocity")
                existing.whiff_percent = m.get("whiff_percent")
                existing.swing_percent = m.get("swing_percent")
                existing.pa = m.get("pa")
                existing.ab = m.get("ab")
                existing.h = m.get("h")
                existing.hr = m.get("hr")
                existing.r = m.get("r")
                existing.rbi = m.get("rbi")
                existing.sb = m.get("sb")
                existing.avg = m.get("avg")
                existing.slg = m.get("slg")
                existing.ops = m.get("ops")
                existing.last_updated = now
                updated += 1
            else:
                # Insert new row
                batter = StatcastBatterMetrics(
                    mlbam_id=row.mlbam_id,
                    player_name=row.player_name,
                    team=row.team,
                    season=row.season,
                    xwoba=m.get("xwoba"),
                    barrel_percent=m.get("barrel_percent"),
                    hard_hit_percent=m.get("hard_hit_percent"),
                    avg_exit_velocity=m.get("avg_exit_velocity"),
                    max_exit_velocity=m.get("max_exit_velocity"),
                    whiff_percent=m.get("whiff_percent"),
                    swing_percent=m.get("swing_percent"),
                    pa=m.get("pa"),
                    ab=m.get("ab"),
                    h=m.get("h"),
                    hr=m.get("hr"),
                    r=m.get("r"),
                    rbi=m.get("rbi"),
                    sb=m.get("sb"),
                    avg=m.get("avg"),
                    slg=m.get("slg"),
                    ops=m.get("ops"),
                    last_updated=now
                )
                self.db.add(batter)
                inserted += 1

        self.db.commit()
        return {"inserted": inserted, "updated": updated}

    def _upsert_pitchers(self, rows: list[SavantMetricsRow]) -> dict[str, int]:
        """Upsert pitcher metrics into statcast_pitcher_metrics."""
        if not rows:
            return {"inserted": 0, "updated": 0}

        now = datetime.now(ZoneInfo("America/New_York"))

        inserted = 0
        updated = 0

        for row in rows:
            existing = self.db.query(StatcastPitcherMetrics).filter_by(
                mlbam_id=row.mlbam_id
            ).first()

            m = row.metrics

            if existing:
                existing.player_name = row.player_name
                existing.team = row.team
                existing.xera = m.get("xera")
                existing.xwoba = m.get("xwoba")
                existing.barrel_percent_allowed = m.get("barrel_percent")
                existing.hard_hit_percent_allowed = m.get("hard_hit_percent")
                existing.avg_exit_velocity_allowed = m.get("avg_exit_velocity")
                existing.k_percent = m.get("k_percent")
                existing.bb_percent = m.get("bb_percent")
                existing.k_9 = m.get("k_9")
                existing.whiff_percent = m.get("whiff_percent")
                existing.w = m.get("w")
                existing.l = m.get("l")
                existing.qs = m.get("qs")
                existing.ip = m.get("ip")
                existing.era = m.get("era")
                existing.whip = m.get("whip")
                existing.sv = m.get("sv")
                existing.h = m.get("h")
                existing.hr_pit = m.get("hr")
                existing.k_pit = m.get("k")
                existing.last_updated = now
                updated += 1
            else:
                pitcher = StatcastPitcherMetrics(
                    mlbam_id=row.mlbam_id,
                    player_name=row.player_name,
                    team=row.team,
                    season=row.season,
                    xera=m.get("xera"),
                    xwoba=m.get("xwoba"),
                    barrel_percent_allowed=m.get("barrel_percent"),
                    hard_hit_percent_allowed=m.get("hard_hit_percent"),
                    avg_exit_velocity_allowed=m.get("avg_exit_velocity"),
                    k_percent=m.get("k_percent"),
                    bb_percent=m.get("bb_percent"),
                    k_9=m.get("k_9"),
                    whiff_percent=m.get("whiff_percent"),
                    w=m.get("w"),
                    l=m.get("l"),
                    qs=m.get("qs"),
                    ip=m.get("ip"),
                    era=m.get("era"),
                    whip=m.get("whip"),
                    sv=m.get("sv"),
                    h=m.get("h"),
                    hr_pit=m.get("hr"),
                    k_pit=m.get("k"),
                    last_updated=now
                )
                self.db.add(pitcher)
                inserted += 1

        self.db.commit()
        return {"inserted": inserted, "updated": updated}

    def _acquire_advisory_lock(self) -> bool:
        """
        Acquire PostgreSQL advisory lock to prevent concurrent runs.

        Returns:
            True if lock acquired, False otherwise
        """
        result = self.db.execute(
            text(f"SELECT pg_try_advisory_lock({SAVANT_LOCK_ID})")
        ).scalar()
        return bool(result)

    def _release_advisory_lock(self) -> None:
        """Release the advisory lock."""
        self.db.execute(
            text(f"SELECT pg_advisory_unlock({SAVANT_LOCK_ID})")
        )

    def fetch_batter_leaderboard(self) -> list[SavantMetricsRow]:
        """
        Fetch and parse batter leaderboard from Savant.

        Returns:
            List of parsed SavantMetricsRow objects
        """
        metrics_str = ",".join([
            "xwoba", "barrel_percent", "hard_hit_percent", "avg_exit_velocity",
            "max_exit_velocity", "whiff_percent", "swing_percent",
            "pa", "ab", "h", "hr", "r", "rbi", "sb", "avg", "slg", "ops"
        ])

        url = (
            f"{self.BASE_URL}?year={self.season}&player_type=batter"
            f"&metrics={metrics_str}&csv=true"
        )

        csv_text = self._fetch_csv(url)
        rows = []

        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            parsed = self._parse_batter_row(row)
            if parsed:
                rows.append(parsed)

        logger.info("Fetched %d batter rows from Savant", len(rows))
        return rows

    def fetch_pitcher_leaderboard(self) -> list[SavantMetricsRow]:
        """
        Fetch and parse pitcher leaderboard from Savant.

        Returns:
            List of parsed SavantMetricsRow objects
        """
        metrics_str = ",".join([
            "xera", "xwoba", "barrel_percent", "hard_hit_percent",
            "avg_exit_velocity", "k_percent", "bb_percent", "k_9",
            "whiff_percent", "w", "l", "qs", "ip", "era", "whip", "sv",
            "h", "hr", "k"
        ])

        url = (
            f"{self.BASE_URL}?year={self.season}&player_type=pitcher"
            f"&metrics={metrics_str}&csv=true"
        )

        csv_text = self._fetch_csv(url)
        rows = []

        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            parsed = self._parse_pitcher_row(row)
            if parsed:
                rows.append(parsed)

        logger.info("Fetched %d pitcher rows from Savant", len(rows))
        return rows

    def run_daily_ingestion(self) -> dict[str, Any]:
        """
        Run the full Savant ingestion pipeline.

        Acquires advisory lock, fetches both leaderboards, upserts to database,
        and releases lock. Returns stats dictionary.

        Returns:
            Dict with keys: status, batters (inserted/updated), pitchers,
                          error (if failed)
        """
        if not self._acquire_advisory_lock():
            logger.warning("Savant ingestion already running, skipping")
            return {
                "status": "skipped",
                "reason": "advisory_lock_held",
                "batters": {"inserted": 0, "updated": 0},
                "pitchers": {"inserted": 0, "updated": 0}
            }

        try:
            logger.info("Starting Savant ingestion for season %d", self.season)

            # Fetch batters
            batter_rows = self.fetch_batter_leaderboard()
            batter_stats = self._upsert_batters(batter_rows)

            # Fetch pitchers
            pitcher_rows = self.fetch_pitcher_leaderboard()
            pitcher_stats = self._upsert_pitchers(pitcher_rows)

            logger.info(
                "Savant ingestion complete: %d batters (+%d new), %d pitchers (+%d new)",
                batter_stats["inserted"] + batter_stats["updated"],
                batter_stats["inserted"],
                pitcher_stats["inserted"] + pitcher_stats["updated"],
                pitcher_stats["inserted"]
            )

            return {
                "status": "success",
                "batters": batter_stats,
                "pitchers": pitcher_stats
            }

        except requests.Timeout as e:
            logger.error("Savant ingestion timeout: %s", e)
            self.db.rollback()
            return {
                "status": "error",
                "error": "timeout",
                "message": str(e)
            }
        except requests.HTTPError as e:
            logger.error("Savant ingestion HTTP error: %s", e)
            self.db.rollback()
            return {
                "status": "error",
                "error": "http_error",
                "message": str(e)
            }
        except Exception as e:
            logger.exception("Savant ingestion failed")
            self.db.rollback()
            return {
                "status": "error",
                "error": "unknown",
                "message": str(e)
            }
        finally:
            self._release_advisory_lock()


def run_savant_ingestion(db: Session, season: int = 2026) -> dict[str, Any]:
    """
    Convenience function to run Savant ingestion.

    Args:
        db: SQLAlchemy database session
        season: MLB season year

    Returns:
        Stats dictionary from run_daily_ingestion()
    """
    agent = SavantIngestionAgent(db, season)
    return agent.run_daily_ingestion()
