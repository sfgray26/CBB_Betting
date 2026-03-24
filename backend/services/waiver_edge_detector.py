"""
WaiverEdgeDetector: scores FAs against current roster deficits,
enriches with MCMC win-probability. Rate-limit cache: FA list cached 10 min.
Returns [] (not raise) on YahooAuthError.
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_FA_CACHE: dict = {}
_FA_CACHE_TTL = 600
_INJURED_2B_Z_THRESHOLD = -1.0


class WaiverEdgeDetector:
    def __init__(self, mcmc_simulator=None, fa_cache_ttl=_FA_CACHE_TTL):
        self.mcmc = mcmc_simulator
        self.fa_cache_ttl = fa_cache_ttl

    def get_top_moves(self, my_roster, opponent_roster, n_candidates=10, force_refresh=False):
        free_agents = self._fetch_fas(force_refresh)
        if not free_agents:
            return []
        deficits = self._compute_deficits(my_roster, opponent_roster)
        drop_candidate = self._weakest_droppable(my_roster)
        moves = []
        for fa in free_agents[:40]:
            score = self._score_fa_against_deficits(fa, deficits)
            move = {
                "add_player": fa,
                "drop_player_name": drop_candidate.get("name", ""),
                "need_score": score,
                "win_prob_before": 0.5,
                "win_prob_after": 0.5,
                "win_prob_gain": 0.0,
                "mcmc_enabled": False,
                "category_win_probs": {},
            }
            if self._has_dead_2b(my_roster) and "2B" in (fa.get("positions") or []):
                move["need_score"] *= 1.25
            if self.mcmc and fa.get("cat_scores"):
                try:
                    r = self.mcmc.simulate_roster_move(
                        my_roster, opponent_roster,
                        add_player=fa,
                        drop_player_name=drop_candidate.get("name", ""),
                    )
                    move.update({
                        "win_prob_before": r.win_prob_before,
                        "win_prob_after": r.win_prob_after,
                        "win_prob_gain": r.win_prob_gain,
                        "mcmc_enabled": r.mcmc_enabled,
                        "category_win_probs": r.category_win_probs_after,
                    })
                except Exception as e:
                    logger.warning("MCMC enrichment failed for %s: %s", fa.get("name"), e)
            moves.append(move)
        moves.sort(key=lambda m: (m["win_prob_gain"], m["need_score"]), reverse=True)
        return moves[:n_candidates]

    def _fetch_fas(self, force_refresh):
        now = time.monotonic()
        if (
            not force_refresh
            and _FA_CACHE.get("fas")
            and (now - _FA_CACHE.get("ts", 0)) < self.fa_cache_ttl
        ):
            return _FA_CACHE["fas"]
        try:
            from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
            fas = YahooFantasyClient().get_free_agents()
            _FA_CACHE["fas"] = fas
            _FA_CACHE["ts"] = now
            return fas
        except Exception as e:
            logger.warning("FA fetch failed (%s) - returning cached", e)
            return _FA_CACHE.get("fas") or []

    def _compute_deficits(self, my_roster, opponent_roster):
        all_cats = set()
        for p in my_roster + opponent_roster:
            all_cats.update((p.get("cat_scores") or {}).keys())
        return {
            cat: (
                sum(p.get("cat_scores", {}).get(cat, 0.0) for p in opponent_roster)
                - sum(p.get("cat_scores", {}).get(cat, 0.0) for p in my_roster)
            )
            for cat in all_cats
        }

    def _weakest_droppable(self, roster):
        droppable = [p for p in roster if not p.get("is_undroppable", False)]
        if not droppable:
            return {}
        return min(droppable, key=lambda p: sum((p.get("cat_scores") or {}).values()))

    def _score_fa_against_deficits(self, fa, deficits):
        cat_scores = fa.get("cat_scores") or {}
        return sum(
            cat_scores.get(cat, 0.0) * max(0.0, deficit)
            for cat, deficit in deficits.items()
        )

    def _has_dead_2b(self, roster):
        for p in roster:
            if "2B" in (p.get("positions") or []):
                if sum((p.get("cat_scores") or {}).values()) < _INJURED_2B_Z_THRESHOLD:
                    return True
        return False
