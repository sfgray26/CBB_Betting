"""
Projection coverage reconciliation -- roster players vs player_scores.

Shared by:
  - GET /api/fantasy/projection-coverage (backend/routers/fantasy.py)
  - projection_coverage daily job (backend/services/daily_ingestion.py, lock 100_044)

Coverage classes per active roster player:
  covered            -- direct player_scores hit on the mapped bdl_id (14d window)
  covered_workaround -- score found via find_alternative_player_score fallback
  missing_mapping    -- no bdl_id resolvable from player_id_mapping
  missing_scores     -- bdl_id resolves but no player_scores row exists
  stale              -- newest score is older than STALE_AFTER_DAYS

Status thresholds (user spec 2026-07-10):
  green: 100% covered | yellow: 90-99% | red: <90%
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func

logger = logging.getLogger(__name__)

STALE_AFTER_DAYS = 2  # scores older than this are flagged (daily pipeline = 1 day lag max)


def compute_roster_projection_coverage(db, raw_players: list, target_date: str | None = None) -> dict:
    """
    Classify every non-IL roster player's projection coverage.

    Args:
        db: SQLAlchemy session
        raw_players: Yahoo roster dicts (player_key, name, status, ...)
        target_date: YYYY-MM-DD; defaults to today (America/New_York)

    Returns dict:
        {
          "target_date": str, "total": int, "covered": int,
          "coverage_pct": float, "status": "green"|"yellow"|"red",
          "players": [{player_key, name, coverage, bdl_id, as_of_date, score}],
          "missing": [subset of players with coverage.startswith("missing")],
        }
    """
    # Lazy imports: reuse the optimizer's exact resolution logic without
    # creating a router<->service import cycle at module load.
    from backend.models import PlayerScore
    from backend.routers.fantasy import _resolve_roster_player_bdl_ids
    from backend.services.player_id_resolver import find_alternative_player_score

    if target_date is None:
        target_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    players = [p for p in raw_players if isinstance(p, dict) and p.get("player_key")]

    # IL players accrue no rolling-window stats, so stale/missing scores are
    # expected -- exclude them from the denominator (mirrors the optimizer,
    # which never places IL players in active slots). Fail open: if overlay
    # loading breaks, treat everyone as active rather than hiding gaps.
    il_players = []
    try:
        from backend.routers.fantasy import _is_il_designated
        from backend.services.injury_overlay import load_injury_overlays_for_yahoo_players

        overlays = load_injury_overlays_for_yahoo_players(db, players)
        active = []
        for p in players:
            if _is_il_designated(p, overlays.get(p["player_key"])):
                il_players.append({
                    "player_key": p["player_key"],
                    "name": p.get("name", "Unknown"),
                    "coverage": "il_excluded",
                })
            else:
                active.append(p)
        players = active
    except Exception as exc:
        logger.debug("projection_coverage: IL overlay check skipped: %s", exc)
    player_key_to_ids = _resolve_roster_player_bdl_ids(db, players)

    bdl_ids = [
        ids["bdl_id"] for ids in player_key_to_ids.values()
        if ids.get("bdl_id") is not None
    ]
    latest_by_bdl: dict[int, tuple] = {}
    if bdl_ids:
        rows = (
            db.query(
                PlayerScore.bdl_player_id,
                func.max(PlayerScore.as_of_date).label("max_date"),
            )
            .filter(
                PlayerScore.bdl_player_id.in_(bdl_ids),
                PlayerScore.window_days == 14,
                PlayerScore.as_of_date <= target_date,
            )
            .group_by(PlayerScore.bdl_player_id)
            .all()
        )
        latest_by_bdl = {r.bdl_player_id: r.max_date for r in rows}

    stale_cutoff = (
        datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=STALE_AFTER_DAYS)
    ).date()

    results = []
    for p in players:
        player_key = p["player_key"]
        name = p.get("name", "Unknown")
        ids = player_key_to_ids.get(player_key) or {}
        bdl_id = ids.get("bdl_id")

        entry = {
            "player_key": player_key,
            "name": name,
            "bdl_id": bdl_id,
            "as_of_date": None,
            "coverage": "missing_mapping",
        }

        if bdl_id is not None:
            as_of = latest_by_bdl.get(bdl_id)
            if as_of is not None:
                entry["as_of_date"] = str(as_of)
                entry["coverage"] = "stale" if as_of < stale_cutoff else "covered"
            else:
                alt_score, alt_source = find_alternative_player_score(
                    db=db,
                    player_key=player_key,
                    bdl_id=bdl_id,
                    mlbam_id=ids.get("mlbam_id"),
                    full_name=name,
                    target_date=target_date,
                )
                if alt_score is not None:
                    entry["coverage"] = "covered_workaround"
                else:
                    entry["coverage"] = "missing_scores"
        results.append(entry)

    total = len(results)
    covered = sum(1 for r in results if r["coverage"] in ("covered", "covered_workaround"))
    coverage_pct = round(covered / total * 100.0, 1) if total else 100.0
    status = "green" if coverage_pct >= 100.0 else ("yellow" if coverage_pct >= 90.0 else "red")
    missing = [r for r in results if r["coverage"].startswith("missing") or r["coverage"] == "stale"]

    if missing:
        logger.warning(
            "projection_coverage: %d/%d roster players lack fresh projections (%s): %s",
            len(missing), total, target_date,
            ", ".join(f"{m['name']}[{m['coverage']}]" for m in missing),
        )

    return {
        "target_date": target_date,
        "total": total,
        "covered": covered,
        "coverage_pct": coverage_pct,
        "status": status,
        "players": results,
        "missing": missing,
        "il_excluded": il_players,
    }
