#!/usr/bin/env python
"""
backfill_cat_scores.py — One-time backfill for PlayerProjection rows with
empty cat_scores ({}) or null team fields.

Usage:
    # Local (if DATABASE_URL is set in environment)
    venv/Scripts/python scripts/backfill_cat_scores.py

    # Railway (production DB)
    railway run python scripts/backfill_cat_scores.py

What it does:
    1. Queries ALL PlayerProjection rows.
    2. Classifies each as batter or pitcher from positions/stats.
    3. Computes 9-category z-scores using the same formula as player_board._compute_zscores.
    4. Writes back cat_scores to rows that have empty/null cat_scores.
    5. Fixes null team fields by querying StatcastPerformance (same MLBAM IDs).

Cat-score categories:
    Batters:  r, h, hr, rbi, k_bat, tb, avg, ops, nsb
    Pitchers: w, l, hr_pit, k_pit, era, whip, k9, qs, nsv

Notes:
    - h and tb are estimated from avg/slg × assumed 550 PA * 0.87 AB/PA.
    - k_bat, w, l, hr_pit, k_pit, qs remain 0 (no data in PlayerProjection columns).
    - z-scores computed from the pool of all rows in the same position group,
      matching the live _compute_zscores behaviour.
    - Rows with no clear batter/pitcher classification are skipped and reported.
"""

from __future__ import annotations

import os
import sys
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo

# Allow running from repo root: `railway run python scripts/backfill_cat_scores.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    sys.exit("ERROR: DATABASE_URL environment variable not set")

# --- DB setup ------------------------------------------------------------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2)
Session = sessionmaker(bind=engine)

# ---------------------------------------------------------------------------
# Z-score helpers (mirrors player_board._zscore / _compute_zscores exactly)
# ---------------------------------------------------------------------------

_BATTER_WEIGHTS = {
    "r": 1.0, "h": 0.8, "hr": 1.3, "rbi": 1.2,
    "k_bat": -0.7, "tb": 0.9, "avg": 1.1, "ops": 1.4, "nsb": 1.0,
}
_PITCHER_WEIGHTS = {
    "w": 1.0, "l": -0.8, "hr_pit": -1.0, "k_pit": 1.2,
    "era": -1.3, "whip": -1.3, "k9": 0.9, "qs": 1.0, "nsv": 1.1,
}


def _zscore(value: float, values: list[float], direction: float = 1.0) -> float:
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


def _compute_cat_scores(players: list[dict], player_type: str) -> None:
    """Compute per-category z-scores in-place.  Mutates players[i]['cat_scores']."""
    if not players:
        return

    if player_type == "batter":
        cats = list(_BATTER_WEIGHTS.keys())
        weights = _BATTER_WEIGHTS
    else:
        cats = list(_PITCHER_WEIGHTS.keys())
        weights = _PITCHER_WEIGHTS

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
# Player classification helpers
# ---------------------------------------------------------------------------

_BATTER_POS = {"C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH", "UTIL"}
_PITCHER_POS = {"SP", "RP", "P"}


def _classify(row) -> str | None:
    """Return 'batter', 'pitcher', or None if ambiguous."""
    positions = row.positions or []
    if isinstance(positions, str):
        positions = [positions]
    pos_set = {str(p).upper().strip() for p in positions}

    has_bat = bool(pos_set & _BATTER_POS)
    has_pit = bool(pos_set & _PITCHER_POS)

    if has_bat and not has_pit:
        return "batter"
    if has_pit and not has_bat:
        return "pitcher"
    if has_bat and has_pit:
        # Mixed — heuristic: ERA != default or k_per_nine != default → pitcher
        if row.era != 4.00 and row.era is not None:
            return "pitcher"
        return "batter"

    # No position info — fall back to stats
    era_is_default = row.era is None or abs(float(row.era or 4.00) - 4.00) < 0.01
    hr_is_high = (row.hr or 0) > 5
    r_is_high = (row.r or 0) > 30

    if not era_is_default:
        return "pitcher"
    if hr_is_high or r_is_high:
        return "batter"
    return None  # ambiguous


# ---------------------------------------------------------------------------
# Main backfill routine
# ---------------------------------------------------------------------------

def main() -> None:
    now = datetime.now(ZoneInfo("America/New_York"))
    print(f"[backfill_cat_scores] Starting at {now.isoformat()}")

    db = Session()
    try:
        # ------------------------------------------------------------------
        # 1. Load all PlayerProjection rows
        # ------------------------------------------------------------------
        rows = db.execute(
            text(
                "SELECT player_id, player_name, team, positions, "
                "       hr, r, rbi, sb, avg, obp, slg, ops, "
                "       era, whip, k_per_nine, "
                "       cat_scores "
                "FROM player_projections"
            )
        ).fetchall()

        print(f"[backfill_cat_scores] Total rows: {len(rows)}")

        # ------------------------------------------------------------------
        # 2. Classify rows
        # ------------------------------------------------------------------
        batters: list[dict] = []
        pitchers: list[dict] = []
        ambiguous: list[str] = []

        for row in rows:
            ptype = _classify(row)
            if ptype is None:
                ambiguous.append(row.player_id)
                continue

            pa_est = 550.0
            ab_est = pa_est * 0.87
            avg = float(row.avg or 0.250)
            slg = float(row.slg or 0.400)

            if ptype == "batter":
                proj = {
                    "r":     float(row.r or 65),
                    "h":     round(avg * ab_est),
                    "hr":    float(row.hr or 15),
                    "rbi":   float(row.rbi or 65),
                    "k_bat": 0.0,  # not stored in PlayerProjection
                    "tb":    round(slg * ab_est),
                    "avg":   avg,
                    "ops":   float(row.ops or 0.720),
                    "nsb":   float(row.sb or 5),
                }
                batters.append({
                    "player_id": row.player_id,
                    "team":      row.team,
                    "type":      "batter",
                    "proj":      proj,
                    "cat_scores": row.cat_scores or {},
                })
            else:
                era = float(row.era or 4.00)
                whip = float(row.whip or 1.30)
                k9 = float(row.k_per_nine or 8.5)
                proj = {
                    "w":      0.0,  # not stored
                    "l":      0.0,
                    "hr_pit": 0.0,
                    "k_pit":  0.0,  # k_per_nine is a rate — no raw K stored
                    "era":    era,
                    "whip":   whip,
                    "k9":     k9,
                    "qs":     0.0,
                    "nsv":    0.0,
                }
                pitchers.append({
                    "player_id": row.player_id,
                    "team":      row.team,
                    "type":      "pitcher",
                    "proj":      proj,
                    "cat_scores": row.cat_scores or {},
                })

        print(
            f"[backfill_cat_scores] Classified: {len(batters)} batters, "
            f"{len(pitchers)} pitchers, {len(ambiguous)} ambiguous (will skip)"
        )
        if ambiguous:
            print(f"[backfill_cat_scores] Ambiguous player_ids: {ambiguous[:20]}")

        # ------------------------------------------------------------------
        # 3. Compute z-scores for all groups
        # ------------------------------------------------------------------
        _compute_cat_scores(batters, "batter")
        _compute_cat_scores(pitchers, "pitcher")

        # ------------------------------------------------------------------
        # 4. Build team lookup from StatcastPerformance for null-team rows
        # ------------------------------------------------------------------
        null_team_ids = {
            p["player_id"]
            for p in batters + pitchers
            if not (p.get("team") or "").strip()
        }
        team_lookup: dict[str, str] = {}
        if null_team_ids:
            placeholders = ", ".join(f"'{pid}'" for pid in null_team_ids)
            team_rows = db.execute(
                text(
                    f"SELECT DISTINCT ON (player_id) player_id, team "
                    f"FROM statcast_performance "
                    f"WHERE player_id IN ({placeholders}) "
                    f"  AND team IS NOT NULL "
                    f"ORDER BY player_id, game_date DESC"
                )
            ).fetchall()
            for tr in team_rows:
                if tr.team:
                    team_lookup[str(tr.player_id)] = tr.team.upper().strip()
            print(
                f"[backfill_cat_scores] Team lookup resolved "
                f"{len(team_lookup)}/{len(null_team_ids)} null-team rows"
            )

        # ------------------------------------------------------------------
        # 5. Write back to database
        # ------------------------------------------------------------------
        cat_updated = 0
        team_updated = 0
        skipped_already_filled = 0

        for p in batters + pitchers:
            new_cat = p["cat_scores"]
            player_id = p["player_id"]

            # Resolve team
            current_team = (p.get("team") or "").strip()
            resolved_team = current_team or team_lookup.get(player_id, "")

            # Decide what to write
            write_cat = bool(new_cat) and not (
                isinstance(p.get("cat_scores"), dict)
                and len(p.get("cat_scores", {})) > 1
            )
            # More precisely: write if the *incoming* cat_scores (from row query) was empty
            original_empty = not p.get("cat_scores") or len(p.get("cat_scores", {})) <= 1

            write_team = resolved_team and not current_team

            if not original_empty and not write_team:
                skipped_already_filled += 1
                continue

            updates: dict = {"updated_at": now}
            if original_empty:
                updates["cat_scores"] = new_cat
                cat_updated += 1
            if write_team:
                updates["team"] = resolved_team
                team_updated += 1

            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            updates["player_id_bind"] = player_id

            db.execute(
                text(
                    f"UPDATE player_projections SET {set_clause} "
                    f"WHERE player_id = :player_id_bind"
                ),
                updates,
            )

        db.commit()

        print(
            f"\n[backfill_cat_scores] Complete:\n"
            f"  cat_scores updated:  {cat_updated}\n"
            f"  team field updated:  {team_updated}\n"
            f"  rows already filled: {skipped_already_filled}\n"
            f"  ambiguous / skipped: {len(ambiguous)}\n"
        )

    except Exception as exc:
        db.rollback()
        print(f"[backfill_cat_scores] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
