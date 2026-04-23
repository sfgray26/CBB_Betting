"""Data quality monitoring dashboard for fantasy baseball platform."""
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from sqlalchemy import func, and_, or_, text
from sqlalchemy.orm import Session

from backend.models import (
    get_db,
    PlayerProjection,
    MLBGameLog,
    MLBPlayerStats,
    StatcastPerformance,
    DataIngestionLog,
)

router = APIRouter(prefix="/api/admin/data-quality", tags=["admin"])


@router.get("/summary")
def get_data_quality_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Aggregate health metrics across all data sources.
    
    Returns 6 key metrics with red/yellow/green status indicators:
    1. Matchup detection rate (games today)
    2. Statcast coverage (% active players with xwOBA)
    3. Null field counts (players with missing data)
    4. Pipeline staleness (hours since last projection update)
    5. Data ingestion failure rate (last 7 days)
    6. Projection coverage (% players with cat_scores)
    """
    now = datetime.now(ZoneInfo("UTC"))
    
    # Metric 1: Matchup detection rate
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    games_today = db.query(func.count(MLBGameLog.id)).filter(
        MLBGameLog.game_date == today_et
    ).scalar() or 0
    
    # Metric 2: Statcast coverage
    active_players = db.query(func.count(PlayerProjection.id)).filter(
        PlayerProjection.updated_at > now - timedelta(days=7)
    ).scalar() or 0
    
    statcast_covered = db.query(
        func.count(func.distinct(StatcastPerformance.player_id))
    ).filter(
        StatcastPerformance.game_date > today_et - timedelta(days=7)
    ).scalar() or 0
    
    statcast_coverage_pct = (
        (statcast_covered / active_players * 100) if active_players > 0 else 0.0
    )
    
    # Metric 3: Null field counts
    null_teams = db.query(func.count(PlayerProjection.id)).filter(
        PlayerProjection.team.is_(None)
    ).scalar() or 0
    
    # Metric 4: Pipeline staleness
    last_projection_update = db.query(func.max(PlayerProjection.updated_at)).scalar()
    staleness_hours = (
        (now - last_projection_update).total_seconds() / 3600
        if last_projection_update
        else 999
    )
    
    # Metric 5: Data ingestion failure rate (last 7 days)
    total_jobs = db.query(func.count(DataIngestionLog.id)).filter(
        DataIngestionLog.run_at > now - timedelta(days=7)
    ).scalar() or 1
    
    failed_jobs = db.query(func.count(DataIngestionLog.id)).filter(
        DataIngestionLog.run_at > now - timedelta(days=7),
        DataIngestionLog.status == "failed",
    ).scalar() or 0
    
    failure_rate_pct = (failed_jobs / total_jobs * 100) if total_jobs > 0 else 0.0
    
    # Metric 6: Projection coverage
    total_projections = db.query(func.count(PlayerProjection.id)).scalar() or 0

    # Count projections with empty cat_scores using standard null/empty check
    empty_cat_scores_query = text("""
        SELECT COUNT(*) FROM player_projections
        WHERE cat_scores IS NULL OR cat_scores::text = '{}'
    """)
    empty_cat_scores = db.execute(empty_cat_scores_query).scalar() or 0
    
    projection_coverage_pct = (
        ((total_projections - empty_cat_scores) / total_projections * 100)
        if total_projections > 0
        else 0.0
    )
    
    return {
        "matchup_detection": {
            "games_today": games_today,
            "status": (
                "green" if games_today > 10
                else "yellow" if games_today > 0
                else "red"
            ),
        },
        "statcast_coverage": {
            "pct": round(statcast_coverage_pct, 1),
            "covered": statcast_covered,
            "total": active_players,
            "status": (
                "green" if statcast_coverage_pct > 85
                else "yellow" if statcast_coverage_pct > 60
                else "red"
            ),
        },
        "null_fields": {
            "null_teams": null_teams,
            "status": (
                "green" if null_teams < 10
                else "yellow" if null_teams < 50
                else "red"
            ),
        },
        "pipeline_staleness": {
            "hours": round(staleness_hours, 1),
            "last_update": (
                last_projection_update.isoformat() if last_projection_update else None
            ),
            "status": (
                "green" if staleness_hours < 12
                else "yellow" if staleness_hours < 36
                else "red"
            ),
        },
        "ingestion_failure_rate": {
            "pct": round(failure_rate_pct, 1),
            "failed": failed_jobs,
            "total": total_jobs,
            "status": (
                "green" if failure_rate_pct < 5
                else "yellow" if failure_rate_pct < 15
                else "red"
            ),
        },
        "projection_coverage": {
            "pct": round(projection_coverage_pct, 1),
            "with_projections": total_projections - empty_cat_scores,
            "total": total_projections,
            "status": (
                "green" if projection_coverage_pct > 90
                else "yellow" if projection_coverage_pct > 80
                else "red"
            ),
        },
        "generated_at": now.isoformat(),
    }


@router.get("/audit")
def run_data_quality_audit(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Run comprehensive data quality audit and return detailed issue list.
    
    This endpoint runs the same logic as scripts/audit_data_quality.py but
    returns JSON instead of CSV. Runs inside Railway network so DB access works.
    
    Returns:
        - issues: List of all data quality issues found
        - summary: Counts by priority/category
        - generated_at: Timestamp
    """
    all_issues = []
    
    # Priority levels
    P0, P1, P2, P3 = "P0", "P1", "P2", "P3"
    # Root cause categories
    A, B, C, D, E = "A", "B", "C", "D", "E"
    
    # === Audit 1: Null team fields ===
    null_team_players = db.query(PlayerProjection).filter(
        or_(
            PlayerProjection.team.is_(None),
            PlayerProjection.team == ""
        )
    ).limit(100).all()
    
    for player in null_team_players:
        all_issues.append({
            "issue_type": "null_team_field",
            "player_name": player.player_name,
            "player_id": player.player_id,
            "team": None,
            "pa": None,
            "expected_value": "<team_abbrev>",
            "actual_value": player.team or "NULL",
            "priority": P2,
            "category": C,
            "impact_score": 3.0,
            "description": "Player has null/empty team field"
        })
    
    # === Audit 2: Empty projections for active players ===
    week_ago = datetime.now(ZoneInfo("UTC")) - timedelta(days=7)

    # Use standard null/empty check for cat_scores
    empty_cats_query = text("""
        SELECT id FROM player_projections
        WHERE updated_at > :week_ago
        AND (cat_scores IS NULL OR cat_scores::text = '{}')
        LIMIT 100
    """)
    empty_cat_ids = [row[0] for row in db.execute(empty_cats_query, {"week_ago": week_ago}).fetchall()]
    players_with_empty_cats = db.query(PlayerProjection).filter(
        PlayerProjection.id.in_(empty_cat_ids)
    ).all()
    
    for player in players_with_empty_cats:
        pa = player.sample_size or 0
        
        if pa > 50:  # Active player
            all_issues.append({
                "issue_type": "empty_projection_active_player",
                "player_name": player.player_name,
                "player_id": player.player_id,
                "team": player.team,
                "pa": pa,
                "expected_value": "{category: z_score, ...}",
                "actual_value": "{}",
                "priority": P1,
                "category": B,
                "impact_score": 7.0 + (1.0 if pa > 200 else 0.0),
                "description": f"Active player (PA={pa}) has empty cat_scores"
            })
        elif pa > 10:  # Bench/streamer
            all_issues.append({
                "issue_type": "empty_projection_bench_player",
                "player_name": player.player_name,
                "player_id": player.player_id,
                "team": player.team,
                "pa": pa,
                "expected_value": "{category: z_score, ...}",
                "actual_value": "{}",
                "priority": P2,
                "category": B,
                "impact_score": 4.0,
                "description": f"Bench player (PA={pa}) has empty cat_scores"
            })
    
    # === Audit 3: Missing Statcast data for active players ===
    today_date = date.today()
    week_ago_date = today_date - timedelta(days=7)
    
    # Get players with recent stats
    active_players = db.query(
        MLBPlayerStats.player_name,
        MLBPlayerStats.bdl_player_id,
        MLBPlayerStats.team,
        func.sum(MLBPlayerStats.pa).label("total_pa")
    ).filter(
        MLBPlayerStats.game_date > week_ago_date
    ).group_by(
        MLBPlayerStats.player_name,
        MLBPlayerStats.bdl_player_id,
        MLBPlayerStats.team
    ).having(
        func.sum(MLBPlayerStats.pa) > 50
    ).limit(200).all()
    
    for player in active_players:
        # Skip if bdl_player_id is None
        if player.bdl_player_id is None:
            continue

        has_statcast = db.query(StatcastPerformance).filter(
            StatcastPerformance.player_id == str(player.bdl_player_id),
            StatcastPerformance.game_date > week_ago_date
        ).first()

        if not has_statcast:
            all_issues.append({
                "issue_type": "missing_statcast_data",
                "player_name": player.player_name,
                "player_id": str(player.bdl_player_id),
                "team": player.team,
                "pa": player.total_pa,
                "expected_value": "xwOBA, barrel%, exit_velo",
                "actual_value": "NULL",
                "priority": P2,
                "category": B,
                "impact_score": 5.0,
                "description": f"Active player (PA={player.total_pa}) missing Statcast"
            })
    
    # === Audit 4: Matchup detection issues ===
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    games_today = db.query(MLBGameLog).filter(
        MLBGameLog.game_date == today_et
    ).limit(50).all()
    
    if len(games_today) >= 5:  # Only audit if games scheduled
        for game in games_today:
            home_team = game.home_team
            
            # Check if recent player data exists
            home_players = db.query(MLBPlayerStats).filter(
                MLBPlayerStats.team == home_team,
                MLBPlayerStats.game_date > today_et - timedelta(days=3)
            ).limit(5).all()
            
            if not home_players:
                all_issues.append({
                    "issue_type": "matchup_detection_false_negative",
                    "player_name": f"{home_team} players",
                    "player_id": "N/A",
                    "team": home_team,
                    "pa": None,
                    "expected_value": f"has_game=True (vs {game.away_team})",
                    "actual_value": "no recent stats found",
                    "priority": P0,
                    "category": A,
                    "impact_score": 9.0,
                    "description": f"Game scheduled today but no recent player data"
                })
    
    # Sort by priority then impact score
    priority_order = {P0: 0, P1: 1, P2: 2, P3: 3}
    all_issues.sort(
        key=lambda x: (priority_order.get(x["priority"], 4), -x["impact_score"])
    )
    
    # Calculate summary stats
    priority_counts = {P0: 0, P1: 0, P2: 0, P3: 0}
    category_counts = {A: 0, B: 0, C: 0, D: 0, E: 0}
    
    for issue in all_issues:
        priority_counts[issue["priority"]] = priority_counts.get(issue["priority"], 0) + 1
        category_counts[issue["category"]] = category_counts.get(issue["category"], 0) + 1
    
    return {
        "issues": all_issues,
        "summary": {
            "total_issues": len(all_issues),
            "by_priority": {k: v for k, v in priority_counts.items() if v > 0},
            "by_category": {k: v for k, v in category_counts.items() if v > 0},
            "top_5": all_issues[:5] if all_issues else []
        },
        "generated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
    }


# ---------------------------------------------------------------------------
# One-shot backfill endpoint — runs inside Railway network (has internal DB access)
# ---------------------------------------------------------------------------

@router.post("/backfill-cat-scores")
def backfill_cat_scores(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Compute and write cat_scores z-scores for all PlayerProjection rows that
    still have empty cat_scores ({}) or null team fields.

    This endpoint runs on the production server inside Railway's network, giving
    it direct access to the internal Postgres instance.  Safe to call multiple
    times — rows already populated are skipped.

    Verification query is included in the response:
        remaining_empty = SELECT COUNT(*) WHERE cat_scores::text = '{}'
    """
    import json
    import statistics
    from zoneinfo import ZoneInfo as _ZI

    # --- z-score helpers (mirrors player_board._compute_zscores) ----------------
    _BAT_W = {
        "r": 1.0, "h": 0.8, "hr": 1.3, "rbi": 1.2,
        "k_bat": -0.7, "tb": 0.9, "avg": 1.1, "ops": 1.4, "nsb": 1.0,
    }
    _PIT_W = {
        "w": 1.0, "l": -0.8, "hr_pit": -1.0, "k_pit": 1.2,
        "era": -1.3, "whip": -1.3, "k9": 0.9, "qs": 1.0, "nsv": 1.1,
    }

    def _z(value: float, pool: list, direction: float = 1.0) -> float:
        if len(pool) < 2:
            return 0.0
        try:
            mu = statistics.mean(pool)
            sd = statistics.stdev(pool)
            return 0.0 if sd < 1e-9 else ((value - mu) / sd) * direction
        except Exception:
            return 0.0

    def _score(players: list, weights: dict) -> None:
        cats = list(weights.keys())
        pool = {c: [float(p["proj"].get(c, 0) or 0) for p in players] for c in cats}
        for p in players:
            scores, total = {}, 0.0
            for c in cats:
                w = weights[c]
                z = _z(float(p["proj"].get(c, 0) or 0), pool[c], 1.0 if w >= 0 else -1.0)
                scores[c] = round(z, 4)
                total += z * abs(w)
            p["cat_scores"] = scores
            p["z_score"] = round(total, 4)

    # --- classify ---------------------------------------------------------------
    BAT_POS = {"C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH", "UTIL"}
    PIT_POS = {"SP", "RP", "P"}

    def _classify(positions, era, hr, r_val):
        pos_set = {str(p).upper().strip() for p in (positions or [])}
        has_bat = bool(pos_set & BAT_POS)
        has_pit = bool(pos_set & PIT_POS)
        if has_bat and not has_pit:
            return "batter"
        if has_pit and not has_bat:
            return "pitcher"
        if has_bat and has_pit:
            return "pitcher" if (era is not None and abs(float(era) - 4.00) > 0.01) else "batter"
        # no position — stats heuristic
        if era is not None and abs(float(era) - 4.00) > 0.01:
            return "pitcher"
        if (hr or 0) > 5 or (r_val or 0) > 30:
            return "batter"
        return None  # ambiguous — will still receive default cat_scores

    # --- load all rows ----------------------------------------------------------
    rows = db.execute(
        text(
            "SELECT player_id, team, positions, hr, r, rbi, sb, "
            "       avg, slg, ops, era, whip, k_per_nine, cat_scores "
            "FROM player_projections"
        )
    ).mappings().fetchall()

    now = datetime.now(_ZI("America/New_York"))
    batters, pitchers, ambiguous = [], [], []

    for row in rows:
        pa, ab = 550.0, 550.0 * 0.87
        avg  = float(row["avg"]  or 0.250)
        slg  = float(row["slg"]  or 0.400)
        ptype = _classify(row["positions"], row["era"], row["hr"], row["r"])

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
                "needs_cat": not row["cat_scores"] or len(row["cat_scores"]) < 2,
                "needs_team": not (row["team"] or "").strip(),
            })
        elif ptype == "pitcher":
            proj = {
                "w": 0.0, "l": 0.0, "hr_pit": 0.0, "k_pit": 0.0,
                "era": float(row["era"] or 4.00), "whip": float(row["whip"] or 1.30),
                "k9": float(row["k_per_nine"] or 8.5), "qs": 0.0, "nsv": 0.0,
            }
            pitchers.append({
                "player_id": row["player_id"], "team": row["team"],
                "proj": proj, "cat_scores": {},
                "needs_cat": not row["cat_scores"] or len(row["cat_scores"]) < 2,
                "needs_team": not (row["team"] or "").strip(),
            })
        else:
            # ambiguous — still queue with default pitcher proj so they get some cat_scores
            proj = {
                "w": 0.0, "l": 0.0, "hr_pit": 0.0, "k_pit": 0.0,
                "era": float(row["era"] or 4.00), "whip": float(row["whip"] or 1.30),
                "k9": float(row["k_per_nine"] or 8.5), "qs": 0.0, "nsv": 0.0,
            }
            ambiguous.append({
                "player_id": row["player_id"], "team": row["team"],
                "proj": proj, "cat_scores": {},
                "needs_cat": not row["cat_scores"] or len(row["cat_scores"]) < 2,
                "needs_team": not (row["team"] or "").strip(),
            })

    # --- compute ----------------------------------------------------------------
    _score(batters, _BAT_W)
    _score(pitchers, _PIT_W)
    _score(ambiguous, _PIT_W)    # ambiguous get pitcher scores as safe default

    # --- team lookup from statcast ----------------------------------------------
    all_players = batters + pitchers + ambiguous
    null_ids = {p["player_id"] for p in all_players if p["needs_team"]}
    team_map: dict = {}
    if null_ids:
        id_list = ", ".join(f"'{pid}'" for pid in null_ids)
        team_rows = db.execute(
            text(
                f"SELECT DISTINCT ON (player_id) player_id, team "
                f"FROM statcast_performances "
                f"WHERE player_id IN ({id_list}) AND team IS NOT NULL "
                f"ORDER BY player_id, game_date DESC"
            )
        ).fetchall()
        for tr in team_rows:
            if tr.team:
                team_map[str(tr.player_id)] = tr.team.upper().strip()

    # --- write ------------------------------------------------------------------
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
            parts.append("cat_scores = :cs::jsonb")
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

    # --- verify -----------------------------------------------------------------
    remaining = db.execute(
        text("SELECT COUNT(*) FROM player_projections WHERE cat_scores::text = '{}'")
    ).scalar() or 0

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
