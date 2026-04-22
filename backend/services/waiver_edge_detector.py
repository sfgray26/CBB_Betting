"""
WaiverEdgeDetector: scores FAs against current roster deficits,
enriches with MCMC win-probability. Rate-limit cache: FA list cached 10 min.
Returns [] (not raise) on YahooAuthError.
"""
import logging
import os
import time
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

_FA_CACHE: dict = {}
_FA_CACHE_TTL = 600
_INJURED_2B_Z_THRESHOLD = -1.0

# Statuses that indicate player is on IL (doesn't count against active roster)
_INACTIVE_STATUSES = frozenset({"IL", "IL10", "IL60", "NA", "OUT"})

# Yahoo IL slot position labels (selected_position values for IL-slotted players)
_IL_SLOT_POSITIONS = frozenset({"IL", "IL10", "IL60", "IL+"})
_DEFAULT_IL_SLOTS = int(os.getenv("YAHOO_IL_SLOTS", "2"))
_ELITE_Z_THRESHOLD = 4.0
_TIER_HOLD_FLOORS = {
    1: 4.5,
    2: 3.5,
    3: 2.75,
    4: 2.0,
    5: 1.25,
}


def _coerce_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default=999) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _player_lookup_id(player: dict) -> str:
    raw = (
        player.get("id")
        or player.get("player_id")
        or player.get("player_key")
        or player.get("name")
        or ""
    )
    normalized = unicodedata.normalize("NFKD", str(raw))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower().replace(" ", "_").replace(".", "").replace("'", "")


def long_term_hold_floor(player: dict) -> float:
    """Estimate a conservative rest-of-season hold value for drop decisions."""
    score = _coerce_float(player.get("z_score"), 0.0)
    tier = _coerce_int(player.get("tier"), 999)
    adp = _coerce_float(player.get("adp"), 9999.0)
    owned_pct = _coerce_float(
        player.get("percent_owned", player.get("owned_pct", 0.0)),
        0.0,
    )

    tier_floor = _TIER_HOLD_FLOORS.get(tier)
    if tier_floor is not None:
        score = max(score, tier_floor)

    if adp <= 30:
        score = max(score, 4.5)
    elif adp <= 75:
        score = max(score, 3.25)
    elif adp <= 130:
        score = max(score, 2.0)
    elif adp <= 180:
        score = max(score, 1.0)

    if owned_pct >= 95:
        score = max(score, 4.0)
    elif owned_pct >= 90:
        score = max(score, 3.25)
    elif owned_pct >= 80:
        score = max(score, 2.25)

    if player.get("is_keeper"):
        score = max(score, 4.0)

    try:
        from backend.fantasy_baseball.ballpark_factors import get_risk_profile

        risk_profile = get_risk_profile(_player_lookup_id(player))
    except Exception:
        risk_profile = None

    if (
        risk_profile
        and risk_profile.role_certainty in {"locked", "likely"}
        and adp <= 130
    ):
        score = max(score, 2.25)

    return score


def drop_candidate_value(player: dict) -> tuple:
    """Return a comparable tuple for ranking drop candidates (lower = more droppable).

    Tuple order: (primary_score, neg_tier, adp, neg_owned_pct, name_hash)
    - primary_score: cat_score sum or z_score (lower = worse)
    - neg_tier: negative tier (tier 5 → -5) so lower tiers drop first
    - adp: higher ADP means worse player (drafted later)
    - neg_owned_pct: negative ownership (less owned = more droppable)
    - name_hash: deterministic final tiebreaker

    Regression for April 21 Issue 3: Universal drop bug where 24/24 waiver
    decisions dropped the same player (Seiya Suzuki). The previous single-value
    return lacked discriminative power when cat_scores were empty and z_scores
    were similar across candidates.
    """
    cat_scores = player.get("cat_scores") or {}
    base_value = float(sum(cat_scores.values())) if cat_scores else _coerce_float(player.get("z_score"), 0.0)
    primary = max(base_value, long_term_hold_floor(player))

    tier = _coerce_int(player.get("tier"), 999)
    adp = _coerce_float(player.get("adp"), 9999.0)
    owned_pct = _coerce_float(
        player.get("percent_owned", player.get("owned_pct", 0.0)),
        0.0,
    )
    name = player.get("name", "")

    return (primary, -tier, adp, -owned_pct, hash(name))


def is_protected_drop_candidate(player: dict) -> bool:
    """Return True for players that should not be recommended as routine drops."""
    if player.get("is_undroppable") or player.get("is_keeper"):
        return True

    z_score = _coerce_float(player.get("z_score"), 0.0)
    tier = _coerce_int(player.get("tier"), 999)
    adp = _coerce_float(player.get("adp"), 9999.0)
    owned_pct = _coerce_float(
        player.get("percent_owned", player.get("owned_pct", 0.0)),
        0.0,
    )

    if z_score >= _ELITE_Z_THRESHOLD:
        return True
    if tier <= 2 or adp <= 30:
        return True
    if owned_pct >= 92 and z_score >= 1.5:
        return True
    if tier <= 4 and adp <= 130 and owned_pct >= 65:
        return True

    return long_term_hold_floor(player) >= 2.25 and tier <= 4 and adp <= 130


def count_il_slots_used(roster: list[dict]) -> int:
    """Count players currently occupying a Yahoo IL slot."""
    return sum(1 for p in roster if p.get("selected_position") in _IL_SLOT_POSITIONS)


def il_capacity_info(roster: list[dict]) -> dict:
    """Return {used, total, available} for IL slots."""
    used = count_il_slots_used(roster)
    total = _DEFAULT_IL_SLOTS
    return {"used": used, "total": total, "available": max(0, total - used)}

# Maps FA position → roster position group eligible for drop pairing.
# OF/LF/CF/RF all compete for the same outfield slots.
_POS_GROUP: dict[str, list[str]] = {
    "SP": ["SP", "RP", "P"],
    "RP": ["SP", "RP", "P"],
    "P":  ["SP", "RP", "P"],
    "C":  ["C"],
    "1B": ["1B"],
    "2B": ["2B"],
    "3B": ["3B"],
    "SS": ["SS"],
    "OF": ["OF", "LF", "CF", "RF"],
    "LF": ["OF", "LF", "CF", "RF"],
    "CF": ["OF", "LF", "CF", "RF"],
    "RF": ["OF", "LF", "CF", "RF"],
    "DH": ["DH", "1B", "OF"],
}


class WaiverEdgeDetector:
    def __init__(self, mcmc_simulator=None, fa_cache_ttl=_FA_CACHE_TTL):
        self.mcmc = mcmc_simulator
        self.fa_cache_ttl = fa_cache_ttl

    def get_top_moves(self, my_roster, opponent_roster, n_candidates=10, force_refresh=False):
        free_agents = self._enrich_players(self._fetch_fas(force_refresh))
        if not free_agents:
            return []
        my_roster = self._enrich_players(my_roster)
        opponent_roster = self._enrich_players(opponent_roster)
        deficits = self._compute_deficits(my_roster, opponent_roster)
        has_deficit_signal = any(abs(v) > 0 for v in deficits.values())
        moves = []
        for fa in free_agents[:40]:
            score = self._score_fa_against_deficits(fa, deficits)
            if score <= 0 and not has_deficit_signal:
                score = float(fa.get("z_score") or 0.0)
            fa_positions = fa.get("positions") or []
            drop_candidate = self._weakest_droppable_at(my_roster, fa_positions)
            move = {
                "add_player": fa,
                "drop_player_name": drop_candidate.get("name", "") if drop_candidate else "",
                "need_score": score,
                "win_prob_before": 0.5,
                "win_prob_after": 0.5,
                "win_prob_gain": 0.0,
                "mcmc_enabled": False,
                "category_win_probs": {},
            }
            if self._has_dead_2b(my_roster) and "2B" in fa_positions:
                move["need_score"] *= 1.25
            if self.mcmc and fa.get("cat_scores") and drop_candidate:
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

    def _enrich_players(self, players: list[dict]) -> list[dict]:
        return [self._enrich_player(player) for player in players]

    def _enrich_player(self, player: Optional[dict]) -> dict:
        if not player:
            return {}

        try:
            from backend.fantasy_baseball.player_board import get_or_create_projection
            projection = get_or_create_projection(player)
        except Exception as exc:
            logger.debug("Projection enrichment failed for %s: %s", player.get("name"), exc)
            projection = {}

        enriched = dict(player)

        if projection:
            enriched.setdefault("id", projection.get("id") or player.get("id") or "")
            enriched.setdefault("team", projection.get("team") or player.get("team") or "")
            enriched.setdefault("positions", projection.get("positions") or player.get("positions") or [])
            enriched["tier"] = projection.get("tier", enriched.get("tier"))
            enriched["adp"] = projection.get("adp", enriched.get("adp"))
            enriched["z_score"] = projection.get("z_score", enriched.get("z_score", 0.0))
            enriched["cat_scores"] = projection.get("cat_scores") or enriched.get("cat_scores") or {}
            enriched["proj"] = projection.get("proj") or enriched.get("proj") or {}
            enriched["is_keeper"] = bool(projection.get("is_keeper", enriched.get("is_keeper", False)))
            enriched["is_proxy"] = bool(projection.get("is_proxy", enriched.get("is_proxy", False)))

        return enriched

    def _fetch_fas(self, force_refresh):
        now = time.monotonic()
        if (
            not force_refresh
            and _FA_CACHE.get("fas")
            and (now - _FA_CACHE.get("ts", 0)) < self.fa_cache_ttl
        ):
            return _FA_CACHE["fas"]
        try:
            from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
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

    def _fa_position_group(self, fa_positions: list[str]) -> list[str]:
        """Return the roster position group that matches the FA's primary position."""
        for pos in fa_positions:
            group = _POS_GROUP.get(pos)
            if group:
                return group
        return fa_positions  # Unknown position — exact match only

    def _count_position_coverage(self, roster: list[dict], pos_group: list[str]) -> int:
        """Count non-undroppable, non-IL roster players that cover any position in pos_group.
        
        Players on IL (IL, IL10, IL60, NA, OUT) don't count against active roster spots
        and should not be considered as coverage for a position.
        """
        return sum(
            1 for p in roster
            if not p.get("is_undroppable", False)
            and p.get("selected_position") not in _INACTIVE_STATUSES
            and p.get("status") not in _INACTIVE_STATUSES
            and any(pos in (p.get("positions") or []) for pos in pos_group)
        )

    def _weakest_droppable_at(self, roster: list[dict], fa_positions: list[str]) -> Optional[dict]:
        """
        Return the weakest droppable roster player at the FA's position group.

        - 0 roster players at position: fall back to weakest overall
        - 1 roster player at position: protected (return None)
        - 2+ roster players at position: return weakest of those
        
        Note: IL players are excluded from consideration (they don't count against
        active roster spots and should not be suggested as drops).
        """
        pos_group = self._fa_position_group(fa_positions)
        coverage = self._count_position_coverage(roster, pos_group)

        if coverage == 0:
            # No coverage at this position — safe to drop anyone
            return self._weakest_droppable(roster)
        elif coverage == 1:
            # Only one cover at this position — protect them
            return None
        else:
            # 2+ players at position — drop weakest among them
            candidates = [
                p for p in roster
                if not p.get("is_undroppable", False)
                and p.get("selected_position") not in _INACTIVE_STATUSES
                and p.get("status") not in _INACTIVE_STATUSES
                and any(pos in (p.get("positions") or []) for pos in pos_group)
                and not is_protected_drop_candidate(p)
            ]
            if not candidates:
                return None
            return min(candidates, key=drop_candidate_value)

    def _weakest_droppable(self, roster):
        """Return weakest droppable player, excluding IL players."""
        droppable = [
            p for p in roster 
            if not p.get("is_undroppable", False)
            and p.get("selected_position") not in _INACTIVE_STATUSES
            and p.get("status") not in _INACTIVE_STATUSES
            and not is_protected_drop_candidate(p)
        ]
        if not droppable:
            return None
        return min(droppable, key=drop_candidate_value)

    @staticmethod
    def _player_value(player: dict) -> tuple:
        return drop_candidate_value(player)

    def _score_fa_against_deficits(self, fa, deficits):
        from backend.fantasy_baseball.category_aware_scorer import (
            CategoryNeedVector,
            PlayerCategoryImpactVector,
            score_fa_against_needs,
        )
        cat_scores = fa.get("cat_scores") or {}
        return score_fa_against_needs(
            PlayerCategoryImpactVector(impacts={k: float(v) for k, v in cat_scores.items()}),
            CategoryNeedVector(needs={k: float(v) for k, v in deficits.items()}),
        )

    def _has_dead_2b(self, roster):
        for p in roster:
            if "2B" in (p.get("positions") or []):
                if sum((p.get("cat_scores") or {}).values()) < _INJURED_2B_Z_THRESHOLD:
                    return True
        return False
