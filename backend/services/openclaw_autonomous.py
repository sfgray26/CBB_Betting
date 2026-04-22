"""
OpenClawAutonomousLoop: daily fantasy intelligence workflow.
Morning routine (8:30 AM APScheduler):
  1. Morning brief
  2. Top waiver moves via WaiverEdgeDetector
  3. High-impact (win_prob_gain >= 0.05) -> immediate Discord priority-2 alert
  4. Rest -> batch digest
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

_HIGH_IMPACT_THRESHOLD = 0.05


class OpenClawAutonomousLoop:
    def __init__(self, waiver_edge_detector, discord_router, my_roster=None, opponent_roster=None):
        self.detector = waiver_edge_detector
        self.router = discord_router
        self.my_roster = my_roster or []
        self.opponent_roster = opponent_roster or []

    def run_morning_workflow(self) -> dict:
        """PAUSED (2026-04-21): OpenClaw is on hold until baseball module is complete.

        All Discord notifications and report generation are disabled.
        """
        logger.info("OpenClaw morning workflow skipped — paused until baseball module is complete")
        return {
            "brief_sent": False,
            "moves_evaluated": 0,
            "high_impact": 0,
            "digest_sent": False,
            "status": "paused",
        }

    def update_rosters(self, my_roster, opponent_roster):
        self.my_roster = my_roster
        self.opponent_roster = opponent_roster
