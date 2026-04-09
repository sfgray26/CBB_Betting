"""
Master Backfill Orchestration Script

Runs all backfill scripts in the correct order with validation between each step.
This ensures data dependencies are satisfied before proceeding.

Execution Order:
1. player_id_mapping — FOUNDATION (all cross-system integration depends on this)
2. position_eligibility — Depends on player_id_mapping (to map Yahoo→BDL IDs)
3. probable_pitchers — Depends on player_id_mapping (to map pitcher names→BDL IDs)
4. statcast_performances — Independent (uses MLBAM IDs from Baseball Savant)
5. mlb_player_stats — Depends on game logs (BDL data)
6. mlb_game_log — Independent (BDL schedule data)

Usage:
    python scripts/backfill_all_data.py

Expected Total Runtime: 20-30 minutes
"""
import logging
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backfill(script_name: str, description: str) -> dict:
    """
    Run a single backfill script and capture results.

    Args:
        script_name: Path to backfill script
        description: Human-readable description

    Returns:
        dict with success status and output
    """
    logger.info("=" * 60)
    logger.info(f"Running: {description}")
    logger.info(f"Script: {script_name}")
    logger.info("=" * 60)

    t0 = datetime.now(ZoneInfo("America/New_York"))

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout per script
        )

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds())

        if result.returncode == 0:
            logger.info(f"✅ SUCCESS: {description}")
            logger.info(f"   Output: {result.stdout[-500:]}")  # Last 500 chars
            return {
                'success': True,
                'description': description,
                'elapsed_seconds': elapsed
            }
        else:
            logger.error(f"❌ FAILED: {description}")
            logger.error(f"   Error: {result.stderr[-500:]}")  # Last 500 chars
            return {
                'success': False,
                'description': description,
                'error': result.stderr[-1000:],
                'elapsed_seconds': elapsed
            }

    except subprocess.TimeoutExpired:
        logger.error(f"❌ TIMEOUT: {description} (exceeded 30 minutes)")
        return {
            'success': False,
            'description': description,
            'error': 'Script execution timeout',
            'elapsed_seconds': 1800
        }
    except Exception as e:
        logger.exception(f"❌ ERROR running {description}: {e}")
        return {
            'success': False,
            'description': description,
            'error': str(e),
            'elapsed_seconds': int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds())
        }


def main():
    """Run all backfill scripts in dependency order."""
    t0 = datetime.now(ZoneInfo("America/New_York"))

    logger.info("=" * 80)
    logger.info("MASTER BACKFILL ORCHESTRATION")
    logger.info("=" * 80)
    logger.info(f"Started at: {t0.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info("")

    # Define backfill scripts in execution order
    backfills = [
        {
            'script': 'scripts/backfill_player_id_mapping.py',
            'description': 'Player ID Mapping (BDL ↔ MLB ↔ Yahoo)',
            'critical': True,
        },
        {
            'script': 'scripts/backfill_positions.py',
            'description': 'Position Eligibility (Yahoo Fantasy)',
            'critical': True,
        },
        {
            'script': 'scripts/backfill_probable_pitchers.py',
            'description': 'Probable Pitchers (March 20 - April 8, 2026)',
            'critical': True,
        },
        {
            'script': 'scripts/backfill_statcast.py',
            'description': 'Statcast Performances (March 20 - April 8, 2026)',
            'critical': True,
        },
        # TODO: Add remaining backfills when scripts are created
        # {
        #     'script': 'scripts/backfill_bdl_stats.py',
        #     'description': 'MLB Player Stats (March 20 - April 8, 2026)',
        #     'critical': False,
        # },
        # {
        #     'script': 'scripts/backfill_game_log.py',
        #     'description': 'MLB Game Log (March 20 - April 8, 2026)',
        #     'critical': False,
        # },
    ]

    results = []
    critical_failures = []

    for i, backfill in enumerate(backfills, 1):
        logger.info(f"\n[{i}/{len(backfills)}] Starting backfill step...")

        result = run_backfill(backfill['script'], backfill['description'])
        results.append(result)

        # Check for critical failures
        if not result['success'] and backfill['critical']:
            logger.error("")
            logger.error("!" * 60)
            logger.error(f"CRITICAL FAILURE: {backfill['description']}")
            logger.error("This blocks all subsequent backfills due to dependencies")
            logger.error("!" * 60)
            logger.error("")
            critical_failures.append(backfill['description'])

            # Stop execution on critical failure
            logger.error("Stopping backfill process. Fix the error and re-run.")
            break

    # Print summary
    total_elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds())

    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 80)

    for i, result in enumerate(results, 1):
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        logger.info(f"{i}. {status}: {result['description']}")
        if not result['success']:
            logger.info(f"   Error: {result.get('error', 'Unknown')[:100]}")

    logger.info("")
    logger.info(f"Total elapsed time: {total_elapsed}s ({total_elapsed // 60}m {total_elapsed % 60}s)")

    success_count = sum(1 for r in results if r['success'])

    if critical_failures:
        logger.error("")
        logger.error("CRITICAL FAILURES DETECTED:")
        for failure in critical_failures:
            logger.error(f"  - {failure}")
        logger.error("")
        logger.error("Backfill process FAILED. Please fix errors and re-run.")
        return 1

    if success_count == len(results):
        logger.info("")
        logger.info("🎉 ALL BACKFILLS COMPLETED SUCCESSFULLY!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Verify table counts via /admin/audit-tables endpoint")
        logger.info("2. Validate data quality via manual spot-checks")
        logger.info("3. Proceed with Railway deployment (G-30, G-29)")
        return 0
    else:
        logger.warning("")
        logger.warning(f"{len(results) - success_count} backfills failed")
        logger.warning("Review errors above and re-run failed scripts")
        return 1


if __name__ == "__main__":
    sys.exit(main())
