#!/usr/bin/env python3
"""
OpenClaw Scheduler Integration

Handles scheduled execution of OpenClaw intelligence tasks:
- Morning brief: Daily at 7:00 AM ET
- Telemetry check: Every 30 minutes (quiet mode)
- Live monitor: Every 2 hours on game days (Phase 2)

Usage:
    python scripts/openclaw_scheduler.py --morning-brief
    python scripts/openclaw_scheduler.py --telemetry-check
    python scripts/openclaw_scheduler.py --live-monitor

Cron setup:
    # Morning brief — daily at 7:00 AM ET
    0 7 * * * cd /app && python scripts/openclaw_scheduler.py --morning-brief
    
    # Telemetry check — every 30 minutes
    */30 * * * * cd /app && python scripts/openclaw_scheduler.py --telemetry-check

Author: Kimi CLI / Claude Code
Document: OPCL-001
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.openclaw_briefs_improved import generate_and_send_morning_brief
from backend.services.openclaw_telemetry import check_system_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("openclaw_scheduler")


async def run_morning_brief():
    """Generate and send the morning brief."""
    logger.info("=" * 50)
    logger.info("Starting morning brief generation")
    logger.info("=" * 50)
    
    try:
        success = await generate_and_send_morning_brief()
        
        if success:
            logger.info("✅ Morning brief sent successfully")
        else:
            logger.warning("⚠️ Morning brief generation completed but Discord send may have failed")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Failed to generate morning brief: {e}")
        return 1


async def run_telemetry_check(force_summary: bool = False):
    """Run telemetry check (only alerts if anomalies detected)."""
    logger.info("Running telemetry check...")
    
    try:
        sent = await check_system_health(force_summary=force_summary)
        
        if sent:
            logger.info("✅ Telemetry alert/summary sent")
        else:
            logger.info("ℹ️ No issues detected — no alert sent (quiet mode)")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Telemetry check failed: {e}")
        return 1


async def run_live_monitor():
    """
    Run live game monitor (Phase 2 — implemented in future iteration).
    
    For now, this is a placeholder that logs readiness.
    """
    logger.info("Live monitor check (Phase 2 — placeholder)")
    logger.info("Full implementation scheduled for March 19-25")
    
    # TODO: Implement live monitoring in Phase 2
    # This would:
    # 1. Query games within 4 hours of tip-off
    # 2. Run DDGS searches for each
    # 3. Quick integrity checks
    # 4. Send alerts if verdict changed
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw Scheduler — Scheduled intelligence tasks"
    )
    
    parser.add_argument(
        "--morning-brief",
        action="store_true",
        help="Generate and send daily morning brief (7 AM ET)"
    )
    
    parser.add_argument(
        "--telemetry-check",
        action="store_true",
        help="Run telemetry check (alerts only if anomalies)"
    )
    
    parser.add_argument(
        "--force-summary",
        action="store_true",
        help="Force daily summary in telemetry check even if no issues"
    )
    
    parser.add_argument(
        "--live-monitor",
        action="store_true",
        help="Run live game monitor (Phase 2)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: don't actually send to Discord"
    )
    
    args = parser.parse_args()
    
    # Validate args
    if sum([args.morning_brief, args.telemetry_check, args.live_monitor]) == 0:
        parser.print_help()
        print("\nError: Must specify one of --morning-brief, --telemetry-check, or --live-monitor")
        return 1
    
    if sum([args.morning_brief, args.telemetry_check, args.live_monitor]) > 1:
        print("Error: Can only run one task at a time")
        return 1
    
    # Log startup
    logger.info(f"OpenClaw Scheduler starting at {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"Task: morning-brief={args.morning_brief}, telemetry={args.telemetry_check}, live={args.live_monitor}")
    
    # Run appropriate task
    if args.morning_brief:
        exit_code = asyncio.run(run_morning_brief())
    elif args.telemetry_check:
        exit_code = asyncio.run(run_telemetry_check(force_summary=args.force_summary))
    elif args.live_monitor:
        exit_code = asyncio.run(run_live_monitor())
    else:
        exit_code = 1
    
    logger.info(f"Scheduler exiting with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
