"""
Performance Sentinel — Monitors model health and system integrity.

K-7 thresholds (env-configurable):
  MAE_WARNING_THRESHOLD=9.5   -> status "WARNING"
  MAE_ELEVATED_THRESHOLD=12.0 -> status "ELEVATED"
  MAE_CRITICAL_THRESHOLD=15.0 -> status "CRITICAL"

  DRAWDOWN_YELLOW_PCT=8.0  -> "YELLOW"
  DRAWDOWN_ORANGE_PCT=12.0 -> "ORANGE"
  DRAWDOWN_RED_PCT=15.0    -> "RED"

During tournament period (March 18 - April 7):
  MAE thresholds relax by +1.5 pts (WARNING/ELEVATED) and +2.0 pts (CRITICAL).
  Drawdown YELLOW relaxes from 8% to 10%.
"""

import logging
import subprocess
import json
import sys
from datetime import datetime, date
from typing import Dict, Optional

from backend.models import SessionLocal
from backend.services.performance import calculate_model_accuracy, calculate_summary_stats
from backend.services.discord_notifier import send_health_briefing
from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-configurable thresholds (K-7 spec)
# ---------------------------------------------------------------------------

def _get_thresholds() -> Dict[str, float]:
    """Return the active threshold set, adjusted for tournament period."""
    mae_warning   = get_float_env("MAE_WARNING_THRESHOLD",   "9.5")
    mae_elevated  = get_float_env("MAE_ELEVATED_THRESHOLD",  "12.0")
    mae_critical  = get_float_env("MAE_CRITICAL_THRESHOLD",  "15.0")
    dd_yellow_pct = get_float_env("DRAWDOWN_YELLOW_PCT",     "8.0")
    dd_orange_pct = get_float_env("DRAWDOWN_ORANGE_PCT",     "12.0")
    dd_red_pct    = get_float_env("DRAWDOWN_RED_PCT",        "15.0")
    min_preds     = get_float_env("MIN_PREDICTIONS_FOR_MAE", "10")

    if _is_tournament_period():
        mae_warning  += 1.5
        mae_elevated += 1.5
        mae_critical += 2.0
        # Drawdown YELLOW relaxes from 8% to 10% during tournament
        if dd_yellow_pct < 10.0:
            dd_yellow_pct = 10.0

    return {
        "mae_warning":   mae_warning,
        "mae_elevated":  mae_elevated,
        "mae_critical":  mae_critical,
        "dd_yellow":     dd_yellow_pct / 100.0,
        "dd_orange":     dd_orange_pct / 100.0,
        "dd_red":        dd_red_pct    / 100.0,
        "min_preds":     int(min_preds),
    }


def _is_tournament_period() -> bool:
    """Return True if today falls in the NCAA Tournament window (March 18 - April 7)."""
    today = date.today()
    # Tournament window: March 18 through April 7 inclusive
    tournament_start = date(today.year, 3, 18)
    tournament_end   = date(today.year, 4, 7)
    return tournament_start <= today <= tournament_end


# ---------------------------------------------------------------------------
# MAE status classification
# ---------------------------------------------------------------------------

def _classify_mae(mae: Optional[float], count: int, thresholds: Dict[str, float]) -> str:
    """Classify MAE into a status string per K-7 spec."""
    if count < thresholds["min_preds"]:
        return "INSUFFICIENT_DATA"
    if mae is None:
        return "INSUFFICIENT_DATA"
    if mae > thresholds["mae_critical"]:
        return "CRITICAL"
    if mae > thresholds["mae_elevated"]:
        return "ELEVATED"
    if mae > thresholds["mae_warning"]:
        return "WARNING"
    return "HEALTHY"


# ---------------------------------------------------------------------------
# Drawdown status classification
# ---------------------------------------------------------------------------

def _classify_drawdown(drawdown: float, thresholds: Dict[str, float]) -> str:
    """Classify drawdown fraction into a status string per K-7 spec."""
    if drawdown >= thresholds["dd_red"]:
        return "RED"
    if drawdown >= thresholds["dd_orange"]:
        return "ORANGE"
    if drawdown >= thresholds["dd_yellow"]:
        return "YELLOW"
    return "GREEN"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_nightly_health_check() -> Dict:
    """
    Executes the master health check protocol:
    1. Performance (MAE) check
    2. Portfolio (Drawdown) check
    3. System (Pytest) check
    """
    logger.info("Sentinel: Starting nightly health check...")
    db = SessionLocal()

    thresholds = _get_thresholds()
    is_tournament = _is_tournament_period()

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "tournament_period": is_tournament,
        "performance": {},
        "portfolio": {},
        "system": {},
    }

    try:
        # 1. Performance Check (30 days)
        accuracy = calculate_model_accuracy(db, days=30)
        mae   = accuracy.get("mean_mae")
        count = accuracy.get("count", 0)
        perf_status = _classify_mae(mae, count, thresholds)
        summary["performance"] = {
            "mean_mae": mae,
            "count": count,
            "status": perf_status,
            "thresholds_used": {
                "warning":  thresholds["mae_warning"],
                "elevated": thresholds["mae_elevated"],
                "critical": thresholds["mae_critical"],
            },
        }
        logger.info(
            "Sentinel: MAE=%.2f (n=%d) -> %s%s",
            mae if mae is not None else 0.0,
            count,
            perf_status,
            " [tournament relaxed]" if is_tournament else "",
        )

        # 2. Portfolio Check
        # current_drawdown is nested under stats["overall"] — not at top level.
        stats    = calculate_summary_stats(db)
        drawdown = stats.get("overall", {}).get("current_drawdown", 0.0)
        drawdown_status = _classify_drawdown(drawdown, thresholds)
        summary["portfolio"] = {
            "current_drawdown_pct": drawdown,
            "status": drawdown_status,
            "thresholds_used": {
                "yellow_pct": thresholds["dd_yellow"] * 100.0,
                "orange_pct": thresholds["dd_orange"] * 100.0,
                "red_pct":    thresholds["dd_red"]    * 100.0,
            },
        }
        logger.info(
            "Sentinel: Drawdown=%.2f%% -> %s",
            drawdown * 100.0,
            drawdown_status,
        )

        # 3. System Check (Pytest)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
                capture_output=True,
                text=True,
            )
            last_line = result.stdout.splitlines()[-1] if result.stdout else "No output"
            summary["system"] = {
                "passed": result.returncode == 0,
                "output_summary": last_line,
                "status": "GREEN" if result.returncode == 0 else "RED",
            }
            logger.info("Sentinel: Pytest -> %s (%s)", summary["system"]["status"], last_line)
        except Exception as e:
            logger.error("Sentinel: Pytest execution failed: %s", e)
            summary["system"] = {"status": "RED", "error": str(e)}

        # 4. Dispatch Briefing
        logger.info("Sentinel: Health check complete. Dispatching briefing...")
        send_health_briefing(summary)

        return summary

    except Exception as e:
        logger.error("Sentinel: Fatal error during health check: %s", e)
        return {"error": str(e)}
    finally:
        db.close()


if __name__ == "__main__":
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_nightly_health_check(), indent=2))
