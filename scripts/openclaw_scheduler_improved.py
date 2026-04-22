#!/usr/bin/env python3
"""
OpenClaw Scheduler — IMPROVED VERSION

Actually functional scheduled tasks for Discord notifications.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.openclaw_briefs_improved import generate_and_send_morning_brief_improved
from backend.services.discord_notifier import send_to_channel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("openclaw_scheduler")


# Color codes for embeds
_COLOR_GREEN = 0x2ECC71
_COLOR_BLUE = 0x3498DB


async def run_morning_brief():
    """Generate and send the morning brief."""
    logger.info("=" * 50)
    logger.info("Generating morning brief...")
    logger.info("=" * 50)
    
    try:
        success = generate_and_send_morning_brief_improved()
        
        if success:
            logger.info("✅ Morning brief sent")
            return 0
        else:
            logger.warning("⚠️ Morning brief failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Morning brief error: {e}")
        return 1


async def run_daily_picks():
    """
    Send daily picks summary.
    
    This runs after the nightly analysis and sends all bets.
    """
    logger.info("Sending daily picks...")
    
    try:
        from backend.services.analysis import nightly_analysis
        from backend.services.discord_notifier import send_todays_bets
        
        # Run analysis
        results = nightly_analysis()
        
        # Send to Discord
        send_todays_bets(results.get("bet_details"), results)
        
        logger.info("✅ Daily picks sent")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Daily picks error: {e}")
        return 1


async def run_end_of_day_results():
    """Send end-of-day results summary."""
    logger.info("Sending end-of-day results...")
    
    try:
        from backend.models import BetLog, Game, SessionLocal
        from backend.services.discord_bet_embeds import create_daily_results_embed
        from datetime import date, timedelta
        
        db = SessionLocal()
        
        # Get yesterday's results
        yesterday = date.today() - timedelta(days=1)
        bets = db.query(BetLog).join(Game).filter(
            Game.game_date >= yesterday,
            Game.game_date < date.today(),
            BetLog.outcome.isnot(None)
        ).all()
        
        results = []
        for bet in bets:
            game = db.query(Game).filter(Game.id == bet.game_id).first()
            if game:
                results.append({
                    'team': bet.pick.split()[0] if bet.pick else 'Unknown',
                    'outcome': bet.outcome,
                    'profit_loss_units': bet.profit_loss_units or 0
                })
        
        db.close()
        
        # Send results
        embed = create_daily_results_embed(results, yesterday.strftime("%b %d, %Y"))
        success = send_to_channel("cbb-bets", embed=embed)
        
        if success:
            logger.info("✅ End-of-day results sent")
            return 0
        else:
            logger.warning("⚠️ Failed to send results")
            return 1
            
    except Exception as e:
        logger.error(f"❌ End-of-day error: {e}")
        return 1


async def run_line_monitor_check():
    """
    Check for line movements and send BET NOW alerts.
    """
    logger.info("Checking line movements...")
    
    try:
        from backend.services.line_monitor import check_line_movements
        
        summary = check_line_movements()
        
        moves = summary.get('movements', [])
        logger.info(f"Found {len(moves)} significant line movements")
        
        # Send alert for favorable moves
        favorable = [m for m in moves if m.get('delta', 0) >= 1.5 and m.get('edge', 0) >= 0.025]
        
        if favorable:
            from backend.services.discord_bet_embeds import create_bet_now_alert_embed
            
            for move in favorable[:3]:  # Max 3 alerts
                embed = create_bet_now_alert_embed(
                    game_data={
                        'home_team': move.get('home_team'),
                        'away_team': move.get('away_team')
                    },
                    line_movement=move
                )
                send_to_channel("cbb-alerts", embed=embed)
            
            logger.info(f"✅ Sent {len(favorable)} BET NOW alerts")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Line monitor error: {e}")
        return 1


async def run_test_notification():
    """Send a test notification to verify Discord is working."""
    logger.info("Sending test notification...")
    
    embed = {
        "title": "🧪 OpenClaw Test",
        "description": "Discord notifications are working correctly!",
        "color": _COLOR_GREEN,
        "fields": [
            {"name": "Status", "value": "✅ Connected", "inline": True},
            {"name": "Time", "value": datetime.now(timezone.utc).strftime("%H:%M UTC"), "inline": True},
        ],
        "footer": {"text": "CBB Edge v9"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Send to all channels
    channels = ["cbb-bets", "cbb-morning-brief", "openclaw-briefs", "openclaw-health"]
    
    for channel in channels:
        try:
            success = send_to_channel(channel, embed=embed)
            if success:
                logger.info(f"✅ Test sent to #{channel}")
            else:
                logger.warning(f"⚠️ Failed to send to #{channel}")
        except Exception as e:
            logger.error(f"❌ Error sending to #{channel}: {e}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw Scheduler — Discord notifications"
    )
    
    parser.add_argument(
        "--morning-brief",
        action="store_true",
        help="Send morning brief (7 AM ET)"
    )
    
    parser.add_argument(
        "--daily-picks",
        action="store_true",
        help="Send daily picks (after nightly analysis)"
    )
    
    parser.add_argument(
        "--end-of-day",
        action="store_true",
        help="Send end-of-day results (11 PM ET)"
    )
    
    parser.add_argument(
        "--line-monitor",
        action="store_true",
        help="Check line movements and send alerts"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send test notification to all channels"
    )
    
    args = parser.parse_args()
    
    # Validate args
    tasks = [args.morning_brief, args.daily_picks, args.end_of_day, args.line_monitor, args.test]
    if sum(tasks) == 0:
        parser.print_help()
        print("\nError: Must specify a task")
        return 1
    
    if sum(tasks) > 1:
        print("Error: Can only run one task at a time")
        return 1
    
    # PAUSED (2026-04-21): OpenClaw is on hold until the baseball module is
    # fully implemented. All scheduled tasks are disabled.
    print("OpenClaw Scheduler is paused. Re-enable when baseball module is complete.")
    logger.info("OpenClaw Scheduler paused — baseball module not yet complete")
    return 0
    
    logger.info(f"Scheduler exiting with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
