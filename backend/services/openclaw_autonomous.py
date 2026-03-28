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
        summary = {
            "brief_sent": False,
            "moves_evaluated": 0,
            "high_impact": 0,
            "digest_sent": False,
        }

        try:
            from backend.services.openclaw_briefs_improved import generate_and_send_morning_brief
            asyncio.run(generate_and_send_morning_brief())
            summary["brief_sent"] = True
        except Exception as e:
            logger.warning("Morning brief failed: %s", e)

        moves = []
        try:
            moves = self.detector.get_top_moves(
                self.my_roster, self.opponent_roster, n_candidates=10
            )
            summary["moves_evaluated"] = len(moves)
        except Exception as e:
            logger.warning("get_top_moves failed: %s", e)

        digest_items = []
        for move in moves:
            gain = move.get("win_prob_gain", 0.0)
            add_name = (move.get("add_player") or {}).get("name", "Unknown")
            drop_name = move.get("drop_player_name", "Unknown")
            label = f"ADD {add_name} / DROP {drop_name} - win% gain: {gain:+.1%}"
            if gain >= _HIGH_IMPACT_THRESHOLD:
                from backend.services.discord_router import IntelPackage
                self.router.route(IntelPackage(
                    channel="waiver-alerts",
                    message=f"HIGH IMPACT: {label}",
                    priority=2,
                    mention_admin=True,
                ))
                summary["high_impact"] += 1
            else:
                digest_items.append(label)

        if digest_items:
            from backend.services.discord_notifier import send_batch_digest
            try:
                summary["digest_sent"] = send_batch_digest("waiver-digest", digest_items)
            except Exception as e:
                logger.warning("Batch digest failed: %s", e)

        if self.router.should_flush():
            self.router.flush_batch()

        logger.info("OpenClaw morning workflow: %s", summary)
        return summary

    def update_rosters(self, my_roster, opponent_roster):
        self.my_roster = my_roster
        self.opponent_roster = opponent_roster
