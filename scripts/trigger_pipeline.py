"""Manually trigger downstream pipeline stages in order."""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.daily_ingestion import DailyIngestionOrchestrator


ALL_STAGES = [
    "rolling_windows",
    "player_scores",
    "player_momentum",
    "ros_simulation",
    "decision_optimization",
    "backtesting",
    "explainability",
    "snapshot",
]

# Override: pass stage names as CLI args to run a subset
import sys as _sys
STAGES = _sys.argv[1:] if len(_sys.argv) > 1 else ALL_STAGES

METHODS = {
    "rolling_windows": "_compute_rolling_windows",
    "player_scores": "_compute_player_scores",
    "player_momentum": "_compute_player_momentum",
    "ros_simulation": "_run_ros_simulation",
    "decision_optimization": "_run_decision_optimization",
    "backtesting": "_run_backtesting",
    "explainability": "_run_explainability",
    "snapshot": "_run_snapshot",
}


async def main():
    os.environ.setdefault("ENABLE_INGESTION_ORCHESTRATOR", "true")
    orchestrator = DailyIngestionOrchestrator()

    results = {}
    for stage in STAGES:
        method_name = METHODS[stage]
        method = getattr(orchestrator, method_name)
        print(f"\n{'='*60}")
        print(f"  STAGE: {stage}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            result = await method()
            elapsed = time.time() - t0
            status = result.get("status", "unknown") if isinstance(result, dict) else str(result)
            results[stage] = {"status": status, "elapsed": f"{elapsed:.1f}s", "detail": result}
            print(f"  Result: {result}")
            print(f"  Elapsed: {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results[stage] = {"status": "ERROR", "elapsed": f"{elapsed:.1f}s", "detail": str(e)}
            print(f"  ERROR: {e}")
            print(f"  Elapsed: {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for stage, info in results.items():
        print(f"  {stage:30s} {info['status']:15s} ({info['elapsed']})")


if __name__ == "__main__":
    asyncio.run(main())
