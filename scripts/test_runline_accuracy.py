#!/usr/bin/env python3
"""
MLB runline projection accuracy test (sprint commitment: Monday May 11).

Backtests yesterday's projections against actual game outcomes from mlb_game_log.
Run daily after games complete (evening) to measure model performance.

Usage:
    venv\Scripts\python scripts/test_runline_accuracy.py

Outputs:
    - Console report with MAE, RMSE, directional accuracy
    - Writes to reports/YYYY-MM-DD-runline-accuracy.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure backend is importable
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from backend.models import SessionLocal, MLBProjection, MLBGameLog
from sqlalchemy import func

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_runline_accuracy(target_date: date | None = None) -> dict:
    """
    Compare yesterday's runline projections against actual game results.

    Metrics:
      - Total projection MAE (mean absolute error on projected_total)
      - Home run MAE (projected_home_runs vs actual)
      - Away run MAE (projected_away_runs vs actual)
      - Directional accuracy (% of games where projected winner = actual winner)
      - Runline cover accuracy (% where projected runline margin sign matches actual)
    """
    target_date = target_date or (date.today() - timedelta(days=1))
    db = SessionLocal()
    try:
        projections = (
            db.query(MLBProjection)
            .filter(MLBProjection.projection_date == target_date)
            .all()
        )
        if not projections:
            logger.warning("No projections found for %s", target_date)
            return {"status": "no_data", "date": str(target_date)}

        results = []
        total_mae_sum = 0.0
        home_mae_sum = 0.0
        away_mae_sum = 0.0
        winner_correct = 0
        runline_correct = 0
        games_with_actual = 0

        for proj in projections:
            # Look up actual result in mlb_game_log by game_id
            game = (
                db.query(MLBGameLog)
                .filter(MLBGameLog.game_id == int(proj.game_id))
                .first()
            )
            if not game or game.home_runs is None or game.away_runs is None:
                continue

            games_with_actual += 1
            actual_total = game.home_runs + game.away_runs
            actual_margin = game.home_runs - game.away_runs
            proj_total = proj.projected_total
            proj_margin = proj.projected_runline_margin

            total_mae = abs(proj_total - actual_total)
            home_mae = abs(proj.projected_home_runs - game.home_runs)
            away_mae = abs(proj.projected_away_runs - game.away_runs)

            total_mae_sum += total_mae
            home_mae_sum += home_mae
            away_mae_sum += away_mae

            # Directional: did we predict the right winner?
            if (proj_margin > 0 and actual_margin > 0) or (proj_margin < 0 and actual_margin < 0) or (abs(proj_margin) < 0.1 and actual_margin == 0):
                winner_correct += 1

            # Runline cover: sign of margin matches
            if (proj_margin >= 0 and actual_margin >= 0) or (proj_margin < 0 and actual_margin < 0):
                runline_correct += 1

            results.append({
                "game_id": proj.game_id,
                "home_team": proj.home_team,
                "away_team": proj.away_team,
                "proj_total": proj_total,
                "actual_total": actual_total,
                "total_mae": round(total_mae, 2),
                "proj_margin": round(proj_margin, 2),
                "actual_margin": actual_margin,
            })

        if games_with_actual == 0:
            return {"status": "no_games_completed", "date": str(target_date)}

        report = {
            "status": "ok",
            "date": str(target_date),
            "projections_count": len(projections),
            "games_with_actual": games_with_actual,
            "total_mae": round(total_mae_sum / games_with_actual, 2),
            "home_mae": round(home_mae_sum / games_with_actual, 2),
            "away_mae": round(away_mae_sum / games_with_actual, 2),
            "directional_accuracy_pct": round(winner_correct / games_with_actual * 100, 1),
            "runline_cover_accuracy_pct": round(runline_correct / games_with_actual * 100, 1),
            "details": results,
        }

        # Write report
        reports_dir = _repo_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        out_path = reports_dir / f"{target_date}-runline-accuracy.json"
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)

        logger.info(
            "Runline accuracy for %s: %d games, total MAE=%.2f, dir_acc=%.1f%%, runline_acc=%.1f%%",
            target_date,
            games_with_actual,
            report["total_mae"],
            report["directional_accuracy_pct"],
            report["runline_cover_accuracy_pct"],
        )
        return report

    except Exception as exc:
        logger.error("test_runline_accuracy failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}
    finally:
        db.close()


if __name__ == "__main__":
    target = None
    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])
    result = test_runline_accuracy(target)
    print(json.dumps(result, indent=2))
