"""
Category scores builder — z-score computation for fantasy baseball projections.

Extracted from backend/routers/data_quality.py to enable testing and reuse.
Provides pure functions for:
  - Player classification (batter vs pitcher)
  - Category z-score computation (mirrors player_board._compute_zscores)
  - Full backfill pipeline (load → classify → score → team-lookup → write)

Usage:
    from backend.services.cat_scores_builder import run_backfill
    result = run_backfill(db)  # returns dict with stats
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session
from sqlalchemy.sql import text


# ---------------------------------------------------------------------------
# Constants — category weights and position sets
# ---------------------------------------------------------------------------

BATTER_WEIGHTS = {
    "r": 1.0, "h": 0.8, "hr": 1.3, "rbi": 1.2,
    "k_bat": -0.7, "tb": 0.9, "avg": 1.1, "ops": 1.4, "nsb": 1.0,
}

PITCHER_WEIGHTS = {
    "w": 1.0, "l": -0.8, "hr_pit": -1.0, "k_pit": 1.2,
    "era": -1.3, "whip": -1.3, "k9": 0.9, "qs": 1.0, "nsv": 1.1,
}

BATTER_POS = {"C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH", "UTIL"}
PITCHER_POS = {"SP", "RP", "P"}


# ---------------------------------------------------------------------------
# Z-score computation (mirrors player_board._zscore / _compute_zscores)
# ---------------------------------------------------------------------------

def _zscore(value: float, values: list[float], direction: float = 1.0) -> float:
    """Compute z-score with direction multiplier for weighted aggregation."""
    if len(values) < 2:
        return 0.0
    try:
        mu = statistics.mean(values)
        sd = statistics.stdev(values)
        if sd < 1e-9:
            return 0.0
        return ((value - mu) / sd) * direction
    except Exception:
        return 0.0


def compute_cat_scores(players: list[dict], weights: dict) -> None:
    """Compute per-category z-scores in-place. Mutates players[i]['cat_scores'].

    Args:
        players: List of dicts with 'proj' key containing category stats
        weights: Dict mapping category names to scalar weights
    """
    if not players:
        return

    cats = list(weights.keys())
    pool: dict[str, list[float]] = {c: [] for c in cats}

    for p in players:
        proj = p["proj"]
        for c in cats:
            pool[c].append(float(proj.get(c, 0) or 0))

    for p in players:
        proj = p["proj"]
        cat_scores: dict[str, float] = {}
        total = 0.0
        for c in cats:
            w = weights[c]
            direction = 1.0 if w >= 0 else -1.0
            z = _zscore(float(proj.get(c, 0) or 0), pool[c], direction=direction)
            weighted = z * abs(w)
            cat_scores[c] = round(z, 4)
            total += weighted
        p["cat_scores"] = cat_scores
        p["z_score"] = round(total, 4)


# ---------------------------------------------------------------------------
# Player classification
# ---------------------------------------------------------------------------

def classify_player(
    positions: list | str | None,
    era: float | None,
    hr: int | None,
    r_val: int | None,
    player_type: str | None = None,
) -> str | None:
    """Classify a player as 'batter' or 'pitcher'.

    Primary signal: player_type column (set by M34 migration and projection jobs).
    Fallback: position codes, then stats heuristic.

    Args:
        positions: List or comma-separated string of position codes
        era: Pitcher ERA (NULL for hitters post-M34)
        hr: Home run count (NULL for pitchers post-M34)
        r_val: Run count (NULL for pitchers post-M34)
        player_type: Explicit type from player_projections.player_type column

    Returns:
        'batter', 'pitcher', or None if truly ambiguous
    """
    # M34: trust the DB column first — it's set by the migration and projection jobs
    if player_type in ("hitter", "batter"):
        return "batter"
    if player_type == "pitcher":
        return "pitcher"

    # Fallback: position codes
    if isinstance(positions, str):
        positions = [p.strip() for p in positions.split(",") if p.strip()]
    pos_set = {str(p).upper() for p in (positions or [])}

    has_bat = bool(pos_set & BATTER_POS)
    has_pit = bool(pos_set & PITCHER_POS)

    if has_bat and not has_pit:
        return "batter"
    if has_pit and not has_bat:
        return "pitcher"
    if has_bat and has_pit:
        # Mixed — non-null ERA is a pitcher signal (post-M34 batters have NULL ERA)
        if era is not None:
            return "pitcher"
        return "batter"

    # Last resort: stats heuristic (pre-M34 rows only)
    if era is not None:
        return "pitcher"
    if (hr or 0) > 5 or (r_val or 0) > 30:
        return "batter"
    return None  # ambiguous


# ---------------------------------------------------------------------------
# Full backfill pipeline
# ---------------------------------------------------------------------------

def run_backfill(db: Session, force: bool = False) -> dict[str, Any]:
    """Compute and write cat_scores z-scores for all PlayerProjection rows.

    This function:
      1. Loads ALL PlayerProjection rows
      2. Classifies each as batter/pitcher/ambiguous
      3. Computes 9-category z-scores
      4. Resolves null team fields from statcast_performances
      5. Writes back cat_scores and team (idempotent — skips already-filled rows)
      6. Returns stats including verification count

    Args:
        db: SQLAlchemy Session (can be in-memory SQLite for testing)

    Returns:
        Dict with keys: status, cat_scores_updated, team_updated,
                      skipped_already_filled, ambiguous_rows,
                      verify_remaining_empty, target_met, generated_at
    """
    now = datetime.now(ZoneInfo("America/New_York"))

    # ------------------------------------------------------------------
    # 1. Load all rows
    # ------------------------------------------------------------------
    rows = db.execute(
        text(
            "SELECT player_id, team, positions, hr, r, rbi, sb, "
            "       avg, slg, ops, era, whip, k_per_nine, "
            "       w, l, qs, hr_pit, k_pit, nsv, "
            "       cat_scores, player_type "
            "FROM player_projections"
        )
    ).mappings().fetchall()

    # Helper to safely parse cat_scores from various storage formats
    def _parse_cat_scores(value) -> dict | None:
        """Parse cat_scores from dict, JSON string, or return None for empty/null."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value if value else None
        if isinstance(value, str):
            if value in ("{}", "[]", "", "null"):
                return None
            try:
                parsed = json.loads(value)
                return parsed if parsed else None
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    # ------------------------------------------------------------------
    # 2. Classify rows
    # ------------------------------------------------------------------
    batters, pitchers, ambiguous = [], [], []
    pa, ab = 550.0, 550.0 * 0.87

    for row in rows:
        avg = float(row["avg"] or 0.250)
        slg = float(row["slg"] or 0.400)
        ptype = classify_player(
            row["positions"], row["era"], row["hr"], row["r"],
            player_type=row.get("player_type"),
        )

        if ptype == "batter":
            proj = {
                "r": float(row["r"] or 65), "h": round(avg * ab),
                "hr": float(row["hr"] or 15), "rbi": float(row["rbi"] or 65),
                "k_bat": 0.0, "tb": round(slg * ab),
                "avg": avg, "ops": float(row["ops"] or 0.720),
                "nsb": float(row["sb"] or 5),
            }
            batters.append({
                "player_id": row["player_id"], "team": row["team"],
                "proj": proj, "cat_scores": {},
                "needs_cat": force or _parse_cat_scores(row["cat_scores"]) is None,
                "needs_team": not (row["team"] or "").strip(),
            })
        elif ptype == "pitcher":
            proj = {
                "w":      float(row["w"]      or 0),
                "l":      float(row["l"]      or 0),
                "hr_pit": float(row["hr_pit"] or 0),
                "k_pit":  float(row["k_pit"]  or 0),
                "era":    float(row["era"]    or 4.00),
                "whip":   float(row["whip"]   or 1.30),
                "k9":     float(row["k_per_nine"] or 8.5),
                "qs":     float(row["qs"]     or 0),
                "nsv":    float(row["nsv"]    or 0),
            }
            pitchers.append({
                "player_id": row["player_id"], "team": row["team"],
                "proj": proj, "cat_scores": {},
                "needs_cat": force or _parse_cat_scores(row["cat_scores"]) is None,
                "needs_team": not (row["team"] or "").strip(),
            })
        else:
            # Ambiguous — still queue with default pitcher proj so they get cat_scores
            proj = {
                "w":      float(row["w"]      or 0),
                "l":      float(row["l"]      or 0),
                "hr_pit": float(row["hr_pit"] or 0),
                "k_pit":  float(row["k_pit"]  or 0),
                "era":    float(row["era"]    or 4.00),
                "whip":   float(row["whip"]   or 1.30),
                "k9":     float(row["k_per_nine"] or 8.5),
                "qs":     float(row["qs"]     or 0),
                "nsv":    float(row["nsv"]    or 0),
            }
            ambiguous.append({
                "player_id": row["player_id"], "team": row["team"],
                "proj": proj, "cat_scores": {},
                "needs_cat": force or _parse_cat_scores(row["cat_scores"]) is None,
                "needs_team": not (row["team"] or "").strip(),
            })

    # ------------------------------------------------------------------
    # 3. Compute z-scores
    # ------------------------------------------------------------------
    compute_cat_scores(batters, BATTER_WEIGHTS)
    compute_cat_scores(pitchers, PITCHER_WEIGHTS)
    compute_cat_scores(ambiguous, PITCHER_WEIGHTS)

    # ------------------------------------------------------------------
    # 4. Team lookup from statcast_performances
    # ------------------------------------------------------------------
    all_players = batters + pitchers + ambiguous
    null_ids = {p["player_id"] for p in all_players if p["needs_team"]}
    team_map: dict = {}

    # Detect database dialect for SQL compatibility
    dialect_name = db.bind.dialect.name if hasattr(db.bind, 'dialect') else 'postgresql'

    if null_ids:
        id_list = ", ".join(f"'{pid}'" for pid in null_ids)

        # Use dialect-specific SQL for DISTINCT ON (PostgreSQL) vs GROUP BY (SQLite/others)
        if dialect_name == 'postgresql':
            team_query = text(
                f"SELECT DISTINCT ON (player_id) player_id, team "
                f"FROM statcast_performances "
                f"WHERE player_id IN ({id_list}) AND team IS NOT NULL "
                f"ORDER BY player_id, game_date DESC"
            )
        else:
            # SQLite-compatible: use GROUP BY and MAX(game_date) to get latest team
            team_query = text(
                f"SELECT sp.player_id, sp.team "
                f"FROM statcast_performances sp "
                f"INNER JOIN ("
                f"  SELECT player_id, MAX(game_date) as max_date "
                f"  FROM statcast_performances "
                f"  WHERE player_id IN ({id_list}) AND team IS NOT NULL "
                f"  GROUP BY player_id"
                f") latest ON sp.player_id = latest.player_id AND sp.game_date = latest.max_date"
            )

        team_rows = db.execute(team_query).fetchall()
        for tr in team_rows:
            if tr.team:
                team_map[str(tr.player_id)] = tr.team.upper().strip()

    # ------------------------------------------------------------------
    # 5. Write back to database
    # ------------------------------------------------------------------
    cat_updated = team_updated = skipped = 0

    for p in all_players:
        pid = p["player_id"]
        write_cat = p["needs_cat"]
        resolved_team = (p.get("team") or "").strip() or team_map.get(pid, "")
        write_team = p["needs_team"] and bool(resolved_team)

        if not write_cat and not write_team:
            skipped += 1
            continue

        parts = ["updated_at = :ts"]
        params: dict = {"ts": now.replace(tzinfo=None), "pid": pid}

        if write_cat:
            # PostgreSQL requires explicit JSONB cast; SQLite handles JSON natively
            if dialect_name == 'postgresql':
                parts.append("cat_scores = CAST(:cs AS JSONB)")
            else:
                parts.append("cat_scores = :cs")
            params["cs"] = json.dumps(p["cat_scores"])
            cat_updated += 1

        if write_team:
            parts.append("team = :team")
            params["team"] = resolved_team
            team_updated += 1

        db.execute(
            text("UPDATE player_projections SET " + ", ".join(parts) + " WHERE player_id = :pid"),
            params,
        )

    db.commit()

    # ------------------------------------------------------------------
    # 6. Verification
    # ------------------------------------------------------------------
    # Use dialect-specific SQL for JSON text comparison
    if dialect_name == 'postgresql':
        verify_query = text(
            "SELECT COUNT(*) FROM player_projections WHERE CAST(cat_scores AS TEXT) = '{}'"
        )
    else:
        # SQLite: check for empty JSON object
        verify_query = text(
            "SELECT COUNT(*) FROM player_projections WHERE cat_scores = '{}' OR cat_scores = '[]'"
        )

    remaining = db.execute(verify_query).scalar() or 0

    return {
        "status": "success",
        "cat_scores_updated": cat_updated,
        "team_updated": team_updated,
        "skipped_already_filled": skipped,
        "ambiguous_rows": len(ambiguous),
        "verify_remaining_empty": remaining,
        "target_met": remaining == 0,
        "generated_at": now.isoformat(),
    }
