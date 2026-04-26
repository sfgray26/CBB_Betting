#!/usr/bin/env python3
"""
Backfill Script: Orphan numeric player names in player_projections.

Phase 1 follow-up: the original `backfill_numeric_player_names.py`
resolved 260/353 numeric `player_name` rows by joining `player_id_mapping`
on `mlbam_id::text = pp.player_name`. 93 rows remain because the MLBAM ID
has no `player_id_mapping` row at all.

Strategy:
    1. Find all orphan rows: player_name ~ '^[0-9]+$' AND no matching
       row in player_id_mapping.
    2. For each orphan, treat the numeric name as an MLBAM ID and call
       MLB Stats API /api/v1/people/{id} to fetch full_name, position,
       team abbreviation.
    3. Optionally enrich with BDL ID via search_mlb_players(full_name).
    4. INSERT into player_id_mapping (mlbam_id, bdl_id, full_name,
       normalized_name, source='manual') ON CONFLICT DO UPDATE.
    5. UPDATE player_projections with resolved player_name + team +
       positions (only when those fields are NULL/empty).

Run:
    venv/Scripts/python scripts/backfill_orphan_numeric_names.py
    railway run python scripts/backfill_orphan_numeric_names.py
"""
import logging
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.models import PlayerIDMapping, PlayerProjection, SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MLB_STATS_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people/{id}"
REQUEST_TIMEOUT_SEC = 15


def _normalize_name(name: str) -> str:
    """Lowercase + strip accents — mirrors PlayerIDMapping.normalized_name semantics."""
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_accents.lower().strip()


def _fetch_from_mlb_stats(mlbam_id: int) -> Optional[dict]:
    """
    Fetch a player from MLB Stats API. Returns dict with:
        full_name, position, team_abbr — or None on failure.
    """
    try:
        resp = requests.get(
            MLB_STATS_PEOPLE_URL.format(id=mlbam_id),
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        people = data.get("people") or []
        if not people:
            return None
        p = people[0]
        full_name = p.get("fullName") or p.get("nameFirstLast") or ""
        if not full_name:
            return None
        position = (p.get("primaryPosition") or {}).get("abbreviation") or ""
        team_abbr = ((p.get("currentTeam") or {}).get("abbreviation")) or ""
        return {
            "full_name": full_name,
            "position": position,
            "team_abbr": team_abbr,
        }
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.info("MLB Stats: 404 for mlbam_id=%s", mlbam_id)
        else:
            logger.warning("MLB Stats: HTTP error for mlbam_id=%s: %s", mlbam_id, exc)
        return None
    except Exception as exc:
        logger.warning("MLB Stats: lookup failed for mlbam_id=%s: %s", mlbam_id, exc)
        return None


def _try_lookup_bdl_id(full_name: str) -> Optional[int]:
    """
    Best-effort BDL ID enrichment via search_mlb_players(name).
    Returns None if BDL is unreachable or no exact match.
    """
    try:
        from backend.services.balldontlie import BallDontLieClient
        bdl = BallDontLieClient()
    except Exception as exc:
        logger.info("BDL client unavailable, skipping bdl_id enrichment: %s", exc)
        return None

    try:
        results = bdl.search_mlb_players(full_name) or []
    except Exception as exc:
        logger.warning("BDL search failed for %r: %s", full_name, exc)
        return None

    norm_target = _normalize_name(full_name)
    for hit in results:
        if _normalize_name(hit.full_name) == norm_target:
            return hit.id
    return None


def _upsert_mapping(
    db,
    mlbam_id: int,
    bdl_id: Optional[int],
    full_name: str,
) -> None:
    """ON CONFLICT DO UPDATE — works if mlbam_id row exists, OR via bdl_id partial uniq."""
    normalized = _normalize_name(full_name)
    now = datetime.now(ZoneInfo("America/New_York"))
    stmt = pg_insert(PlayerIDMapping.__table__).values(
        mlbam_id=mlbam_id,
        bdl_id=bdl_id,
        full_name=full_name,
        normalized_name=normalized,
        source="manual",
        created_at=now,
        updated_at=now,
    )
    if bdl_id is not None:
        stmt = stmt.on_conflict_do_update(
            constraint="_pim_bdl_id_uc",
            set_={
                "mlbam_id": stmt.excluded.mlbam_id,
                "full_name": stmt.excluded.full_name,
                "normalized_name": stmt.excluded.normalized_name,
                "updated_at": stmt.excluded.updated_at,
            },
        )
    else:
        stmt = stmt.on_conflict_do_nothing()
    db.execute(stmt)


def backfill_orphans() -> dict:
    """Resolve numeric `player_name` orphans via MLB Stats API + optional BDL."""
    t0 = time.monotonic()
    db = SessionLocal()
    resolved = 0
    skipped = 0
    failed = 0
    try:
        rows = db.execute(text("""
            SELECT pp.player_id, pp.player_name, pp.team, pp.positions
            FROM player_projections pp
            LEFT JOIN player_id_mapping pid
              ON pid.mlbam_id::text = pp.player_name
            WHERE pp.player_name ~ '^[0-9]+$'
              AND pid.id IS NULL
            ORDER BY pp.player_id
        """)).fetchall()
        logger.info("Found %d orphan numeric-name rows", len(rows))

        for row in rows:
            numeric_name = row.player_name
            try:
                mlbam_id = int(numeric_name)
            except (TypeError, ValueError):
                logger.warning("Skipping non-integer player_name: %r", numeric_name)
                skipped += 1
                continue

            info = _fetch_from_mlb_stats(mlbam_id)
            if info is None:
                logger.warning("Could not resolve mlbam_id=%s", mlbam_id)
                failed += 1
                continue

            full_name = info["full_name"]
            team_abbr = info["team_abbr"] or None
            position = info["position"] or None
            bdl_id = _try_lookup_bdl_id(full_name)

            try:
                _upsert_mapping(db, mlbam_id, bdl_id, full_name)
            except Exception as map_err:
                db.rollback()
                logger.warning("Mapping upsert failed for mlbam_id=%s: %s", mlbam_id, map_err)
                failed += 1
                continue

            update_kwargs = {"player_name": full_name}
            if team_abbr and not row.team:
                update_kwargs["team"] = team_abbr
            if position and not row.positions:
                update_kwargs["positions"] = [position]

            try:
                proj = db.query(PlayerProjection).filter_by(player_id=row.player_id).first()
                if proj:
                    for key, value in update_kwargs.items():
                        setattr(proj, key, value)
                    db.commit()
                    resolved += 1
                    logger.info(
                        "Resolved mlbam_id=%s -> %r (team=%s, pos=%s, bdl=%s)",
                        mlbam_id, full_name, team_abbr, position, bdl_id,
                    )
                else:
                    skipped += 1
            except Exception as upd_err:
                db.rollback()
                logger.warning("Projection update failed for player_id=%s: %s", row.player_id, upd_err)
                failed += 1

        remaining = db.execute(text("""
            SELECT COUNT(*) FROM player_projections WHERE player_name ~ '^[0-9]+$'
        """)).scalar() or 0

        elapsed = time.monotonic() - t0
        return {
            "status": "success",
            "orphans_examined": len(rows),
            "resolved": resolved,
            "skipped": skipped,
            "failed": failed,
            "remaining_numeric_names": remaining,
            "elapsed_sec": round(elapsed, 1),
        }
    except Exception as exc:
        db.rollback()
        logger.error("Backfill failed: %s", exc)
        return {"status": "error", "error": str(exc)}
    finally:
        db.close()


if __name__ == "__main__":
    result = backfill_orphans()
    print(result)
    sys.exit(0 if result.get("status") == "success" else 1)
