"""
Performance Sentinel — Monitors model health and system integrity.
"""

import logging
import subprocess
import json
import os
import sys
from datetime import datetime
from typing import Dict

from backend.models import SessionLocal
from backend.services.performance import calculate_model_accuracy, calculate_summary_stats
from backend.services.discord_notifier import send_health_briefing

logger = logging.getLogger(__name__)

def run_nightly_health_check():
    """
    Executes the master health check protocol:
    1. Performance (MAE) check
    2. Portfolio (Drawdown) check
    3. System (Pytest) check
    """
    logger.info("Sentinel: Starting nightly health check...")
    db = SessionLocal()
    
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "performance": {},
        "portfolio": {},
        "system": {}
    }

    try:
        # 1. Performance Check (30 days)
        accuracy = calculate_model_accuracy(db, days=30)
        mae = accuracy.get("mean_mae")
        count = accuracy.get("count", 0)
        if count < 5:
            perf_status = "INSUFFICIENT_DATA"
        elif mae is None:
            perf_status = "INSUFFICIENT_DATA"
        elif mae <= 3.0:
            perf_status = "GREEN"
        elif mae <= 5.0:
            perf_status = "YELLOW"
        else:
            perf_status = "RED"
        summary["performance"] = {
            "mean_mae": mae,
            "count": count,
            "status": perf_status,
        }

        # 2. Portfolio Check
        stats = calculate_summary_stats(db)
        drawdown = stats.get("current_drawdown", 0.0)
        
        drawdown_status = "GREEN"
        if drawdown > 0.15:
            drawdown_status = "RED"
        elif drawdown > 0.10:
            drawdown_status = "YELLOW"
            
        summary["portfolio"] = {
            "current_drawdown_pct": drawdown,
            "status": drawdown_status
        }

        # 3. System Check (Pytest)
        # Using -q for quiet and --tb=line for concise output
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
                capture_output=True,
                text=True,
            )
            summary["system"] = {
                "passed": result.returncode == 0,
                "output_summary": result.stdout.splitlines()[-1] if result.stdout else "No output",
                "status": "GREEN" if result.returncode == 0 else "RED"
            }
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
