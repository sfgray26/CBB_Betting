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

    BASE_URL = "https://baseballsavant.mlb.com/leaderboard/custom"

    # Custom Leaderboard selections (URL-encoded comma-separated)
    # These are the actual column names returned by /leaderboard/custom
    BATTER_SELECTIONS = (
        "pa,xwoba,barrel_batted_rate,hard_hit_percent,exit_velocity_avg,"
        "whiff_percent,swing_percent,ab,h,hr,r,rbi,sb,"
        "batting_avg,slg_percent,on_base_plus_slg"
    )

    PITCHER_SELECTIONS = (
        "pa,xwoba,xera,barrel_batted_rate,hard_hit_percent,exit_velocity_avg,"
        "k_percent,bb_percent,k_9,whiff_percent,"
        "w,l,qs,ip,era,whip,sv,h,hr,k"
    )

    def __init__(self, db: Session, season: int = 2026):
        """
        Initialize the Savant ingestion agent.

        Args:
            db: SQLAlchemy database session
            season: MLB season year (default 2026)
        """
        self.db = db
        self.season = season

    @staticmethod
    def _savant_float(value: str) -> Optional[float]:
        """
        Parse a Baseball Savant numeric string to float.

        Handles:
        - Leading dots: '.000' → 0.0, '.352' → 0.352
        - Empty strings: '' → None
        - Percent signs: stripped by caller
        """
        if not value or not value.strip():
            return None
        s = value.strip().rstrip("%")
        if s.startswith("."):
            s = "0" + s
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _savant_int(value: str) -> Optional[int]:
        """Parse a Baseball Savant numeric string to int."""
        if not value or not value.strip():
            return None
        s = value.strip()
        try:
            return int(float(s))  # Handles '5.0' → 5
        except (ValueError, TypeError):
            return None

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
        # Strip UTF-8 BOM if present — Savant CSVs sometimes include it,
        # which breaks csv.DictReader's quoted-field parsing.
        return response.text.lstrip("\ufeff")

    def _parse_batter_row(self, row: dict[str, str]) -> Optional[SavantMetricsRow]:
        """
        Parse a single batter CSV row from Custom Leaderboard.

        Handles the combined name column and extracts MLBAM ID.
        Maps Custom Leaderboard column names to schema keys.
        """
        try:
            mlbam_id = row.get("player_id", "").strip()
            if not mlbam_id:
                return None

            # Extract player name from combined column
            player_name = ""
            name_col = row.get("last_name, first_name") or row.get("name")
            if name_col:
                if "," in name_col:
                    parts = name_col.split(",")
                    player_name = f"{parts[1].strip()} {parts[0].strip()}"
                else:
                    player_name = name_col.strip()

            if not player_name:
                return None

            # Map Custom Leaderboard columns to schema keys
            mapped_metrics = {
                "xwoba": self._savant_float(row.get("xwoba", "")),
                "barrel_percent": self._savant_float(row.get("barrel_batted_rate", "")),
                "hard_hit_percent": self._savant_float(row.get("hard_hit_percent", "")),
                "avg_exit_velocity": self._savant_float(row.get("exit_velocity_avg", "")),
                "whiff_percent": self._savant_float(row.get("whiff_percent", "")),
                "swing_percent": self._savant_float(row.get("swing_percent", "")),
                "pa": self._savant_int(row.get("pa", "")),
                "ab": self._savant_int(row.get("ab", "")),
                "h": self._savant_int(row.get("h", "")),
                "hr": self._savant_int(row.get("hr", "")),
                "r": self._savant_int(row.get("r", "")),
                "rbi": self._savant_int(row.get("rbi", "")),
                "sb": self._savant_int(row.get("sb", "")),
                "avg": self._savant_float(row.get("batting_avg", "")),
                "slg": self._savant_float(row.get("slg_percent", "")),
                "ops": self._savant_float(row.get("on_base_plus_slg", "")),
            }

            return SavantMetricsRow(
                mlbam_id=mlbam_id,
                player_name=player_name,
                team=row.get("team") or None,
                season=self.season,
                metrics=mapped_metrics
            )
        except Exception as e:
            logger.warning("Unexpected error parsing batter row: %s", e)
            return None
    def _parse_pitcher_row(self, row: dict[str, str]) -> Optional[SavantMetricsRow]:
        """Parse a single pitcher CSV row from Custom Leaderboard."""
        try:
            mlbam_id = row.get("player_id", "").strip()
            if not mlbam_id:
                return None

            player_name = ""
            name_col = row.get("last_name, first_name") or row.get("name")
            if name_col:
                if "," in name_col:
                    parts = name_col.split(",")
                    player_name = f"{parts[1].strip()} {parts[0].strip()}"
                else:
                    player_name = name_col.strip()

            # Map Custom Leaderboard columns to schema keys
            mapped_metrics = {
                "xwoba": self._savant_float(row.get("xwoba", "")),
                "xera": self._savant_float(row.get("xera", "")),
                "barrel_percent": self._savant_float(row.get("barrel_batted_rate", "")),
                "hard_hit_percent": self._savant_float(row.get("hard_hit_percent", "")),
                "avg_exit_velocity": self._savant_float(row.get("exit_velocity_avg", "")),
                "k_percent": self._savant_float(row.get("k_percent", "")),
                "bb_percent": self._savant_float(row.get("bb_percent", "")),
                "k_9": self._savant_float(row.get("k_9", "")),
                "whiff_percent": self._savant_float(row.get("whiff_percent", "")),
                "w": self._savant_int(row.get("w", "")),
                "l": self._savant_int(row.get("l", "")),
                "qs": self._savant_int(row.get("qs", "")),
                "ip": self._savant_float(row.get("ip", "")),
                "era": self._savant_float(row.get("era", "")),
                "whip": self._savant_float(row.get("whip", "")),
                "sv": self._savant_int(row.get("sv", "")),
                "h": self._savant_int(row.get("h", "")),
                "hr": self._savant_int(row.get("hr", "")),
                "k": self._savant_int(row.get("k", "")),
            }

            return SavantMetricsRow(
                mlbam_id=mlbam_id,
                player_name=player_name,
                team=row.get("team") or None,
                season=self.season,
                metrics=mapped_metrics
            )
        except Exception as e:
            logger.warning("Unexpected error parsing pitcher row: %s", e)
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
        Fetch and parse batter Custom Leaderboard from Savant.

        Uses the /leaderboard/custom endpoint with selections parameter.

        Returns:
            List of parsed SavantMetricsRow objects
        """
        url = (
            f"{self.BASE_URL}?year={self.season}&type=batter&filter=&min=0"
            f"&selections={self.BATTER_SELECTIONS}"
            f"&chart=false&x=pa&y=pa&r=no&chartType=beeswarm"
            f"&sort=xwoba&sortDir=desc&csv=true"
        )

        csv_text = self._fetch_csv(url)
        rows = []

        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            parsed = self._parse_batter_row(row)
            if parsed:
                rows.append(parsed)

        logger.info("Fetched %d batter rows from Savant Custom Leaderboard", len(rows))
        return rows

    def fetch_pitcher_leaderboard(self) -> list[SavantMetricsRow]:
        """
        Fetch and parse pitcher Custom Leaderboard from Savant.

        Uses the /leaderboard/custom endpoint with selections parameter.

        Returns:
            List of parsed SavantMetricsRow objects
        """
        url = (
            f"{self.BASE_URL}?year={self.season}&type=pitcher&filter=&min=0"
            f"&selections={self.PITCHER_SELECTIONS}"
            f"&chart=false&x=pa&y=pa&r=no&chartType=beeswarm"
            f"&sort=xwoba&sortDir=asc&csv=true"
        )

        csv_text = self._fetch_csv(url)
        rows = []

        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            parsed = self._parse_pitcher_row(row)
            if parsed:
                rows.append(parsed)

        logger.info("Fetched %d pitcher rows from Savant Custom Leaderboard", len(rows))
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
            import traceback as _tb
            logger.exception("Savant ingestion failed: %s\n%s", e, _tb.format_exc())
            self.db.rollback()
            return {
                "status": "error",
                "error": type(e).__name__,
                "message": str(e),
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
