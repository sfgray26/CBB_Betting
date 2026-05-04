"""
OpenClawAutonomousLoop: daily fantasy intelligence workflow.
Morning routine (8:30 AM APScheduler):
  1. Morning brief
  2. Top waiver moves via WaiverEdgeDetector
  3. High-impact (win_prob_gain >= 0.05) -> immediate Discord priority-2 alert
  4. Rest -> batch digest
"""
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
        """Run the daily morning workflow: brief, waiver moves, alerts, digest."""
        # Lazy imports so tests can patch them at the module level.
        from backend.services.openclaw_briefs_improved import generate_and_send_morning_brief
        from backend.services.discord_notifier import send_batch_digest
        from backend.services.discord_router import IntelPackage

        # 1. Morning brief
        brief_sent = False
        try:
            brief_sent = generate_and_send_morning_brief()
        except Exception as e:
            logger.warning("Morning brief failed: %s", e)

        # 2. Waiver moves
        moves = []
        try:
            moves = self.detector.get_top_moves(
                self.my_roster, self.opponent_roster, n_candidates=10
            )
        except Exception as e:
            logger.warning("Waiver move detection failed: %s", e)

        # 3. Evaluate moves
        high_impact = 0
        digest_items = []
        for move in moves:
            win_prob_gain = move.get("win_prob_gain", 0.0)
            add_player = move.get("add_player") or {}
            drop_name = move.get("drop_player_name", "")
            player_name = add_player.get("name", "Unknown")
            if win_prob_gain >= _HIGH_IMPACT_THRESHOLD:
                high_impact += 1
                msg = (
                    f"🚨 HIGH IMPACT WAIVER MOVE\n"
                    f"Add: {player_name}\n"
                    f"Drop: {drop_name}\n"
                    f"Win probability gain: +{win_prob_gain:.1%}"
                )
                pkg = IntelPackage(
                    channel="waiver-alerts",
                    message=msg,
                    priority=2,
                    mention_admin=True,
                )
                try:
                    self.router.route(pkg)
                except Exception as e:
                    logger.warning("Failed to route high-impact alert: %s", e)
            else:
                digest_items.append(
                    f"• {player_name} (drop {drop_name}) — win gain {win_prob_gain:+.1%}"
                )

        # 4. Batch digest
        digest_sent = False
        if digest_items:
            try:
                digest_sent = send_batch_digest("waiver-digest", digest_items)
            except Exception as e:
                logger.warning("Batch digest failed: %s", e)

        # 5. Router flush
        try:
            if self.router.should_flush():
                self.router.flush_batch()
        except Exception as e:
            logger.warning("Router flush failed: %s", e)

        return {
            "brief_sent": bool(brief_sent),
            "moves_evaluated": len(moves),
            "high_impact": high_impact,
            "digest_sent": digest_sent,
        }

    def update_rosters(self, my_roster, opponent_roster):
        self.my_roster = my_roster
        self.opponent_roster = opponent_roster
