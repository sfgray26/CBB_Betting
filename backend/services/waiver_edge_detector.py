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
    # Use player id (Yahoo key) as tiebreaker — stable, unique, avoids name-hash bias
    # that caused the "always Seiya Suzuki" universal-drop bug (April 21).
    player_id = str(player.get("id") or player.get("player_key") or player.get("name") or "")

    return (primary, -tier, adp, -owned_pct, player_id)


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

    @staticmethod
    def _load_scarcity_lookup(player_keys: list[str]) -> dict[str, int]:
        """Bulk-load scarcity_rank from position_eligibility for the given yahoo_player_keys.
        Returns {yahoo_player_key: scarcity_rank}. Empty dict on any DB error (non-fatal)."""
        if not player_keys:
            return {}
        try:
            from backend.models import SessionLocal, PositionEligibility
            db = SessionLocal()
            try:
                rows = (
                    db.query(PositionEligibility.yahoo_player_key, PositionEligibility.scarcity_rank)
                    .filter(
                        PositionEligibility.yahoo_player_key.in_(player_keys),
                        PositionEligibility.scarcity_rank.isnot(None),
                    )
                    .all()
                )
                return {r.yahoo_player_key: r.scarcity_rank for r in rows}
            finally:
                db.close()
        except Exception as exc:
            logger.debug("_load_scarcity_lookup: DB query failed (%s) -- using _POS_GROUP fallback", exc)
            return {}

    def _load_market_scores(self, bdl_ids: list[int]) -> dict[int, float]:
        """PR 4.5: Bulk-load market_score from player_market_signals for the given BDL IDs.
        Returns {bdl_id: market_score}. Defaults to 50.0 (neutral) when not found."""
        if not bdl_ids:
            return {}
        try:
            from datetime import date
            from backend.models import SessionLocal
            from sqlalchemy import text

            today = date.today()
            db = SessionLocal()
            try:
                rows = db.execute(
                    text(
                        """
                        SELECT bdl_player_id, market_score
                        FROM player_market_signals
                        WHERE bdl_player_id = ANY(:bdl_ids)
                          AND as_of_date = :today
                        """
                    ),
                    {"bdl_ids": bdl_ids, "today": today},
                ).fetchall()
                # Default to 50.0 (neutral) for missing players
                return {r[0]: r[1] for r in rows} if rows else {}
            finally:
                db.close()
        except Exception as exc:
            logger.warning("_load_market_scores: DB query failed (%s)", exc)
            return {}

    def _load_matchup_scores(self, bdl_ids: list) -> dict:
        """Sprint 4: Bulk-load today's matchup context for given BDL player IDs.

        Returns {bdl_id: {"matchup_z": float, "matchup_score": float, "matchup_confidence": float}}.
        Empty dict on any failure -- never blocks waiver recommendations.
        """
        if not bdl_ids:
            return {}
        try:
            from datetime import date as _date
            from backend.models import SessionLocal
            from sqlalchemy import text

            today = _date.today()
            db = SessionLocal()
            try:
                rows = db.execute(
                    text(
                        "SELECT bdl_player_id, matchup_z, matchup_score, matchup_confidence "
                        "FROM matchup_context "
                        "WHERE bdl_player_id = ANY(:ids) AND game_date = :today"
                    ),
                    {"ids": list(bdl_ids), "today": today},
                ).fetchall()
                return {
                    r[0]: {
                        "matchup_z": float(r[1]) if r[1] is not None else 0.0,
                        "matchup_score": float(r[2]) if r[2] is not None else 50.0,
                        "matchup_confidence": float(r[3]) if r[3] is not None else 0.0,
                    }
                    for r in rows
                }
            finally:
                db.close()
        except Exception as exc:
            logger.debug("_load_matchup_scores failed (%s)", exc)
            return {}

    @staticmethod
    def _yahoo_id_to_bdl_id(yahoo_id: int) -> Optional[int]:
        """Map Yahoo player_id to BDL ID via player_id_mapping.
        Returns None if not found."""
        try:
            from backend.models import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                row = db.execute(
                    text("SELECT bdl_id FROM player_id_mapping WHERE yahoo_id = :yahoo_id"),
                    {"yahoo_id": yahoo_id},
                ).fetchone()
                return int(row[0]) if row else None
            finally:
                db.close()
        except Exception:
            return None

    @staticmethod
    def _bulk_yahoo_to_mlbam(yahoo_ids: list) -> dict:
        """Bulk-map Yahoo player IDs to MLBAM IDs via player_id_mapping.
        Returns {yahoo_id: mlbam_id}. Empty dict on any DB error."""
        if not yahoo_ids:
            return {}
        try:
            from backend.models import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                rows = db.execute(
                    text(
                        "SELECT yahoo_id, mlbam_id FROM player_id_mapping "
                        "WHERE yahoo_id = ANY(:ids) AND mlbam_id IS NOT NULL"
                    ),
                    {"ids": list(yahoo_ids)},
                ).fetchall()
                return {int(r[0]): int(r[1]) for r in rows}
            finally:
                db.close()
        except Exception as exc:
            logger.debug("_bulk_yahoo_to_mlbam failed (%s)", exc)
            return {}

    @staticmethod
    def _build_canonical_team_context(roster: list) -> Optional[object]:
        """Build TeamContext from roster Yahoo IDs using CanonicalProjection data.

        Returns TeamContext if CANONICAL_PROJECTION_V1 is enabled and roster has
        resolvable players; returns None on any failure (never raises).
        """
        try:
            from backend.models import SessionLocal
            from backend.fantasy_baseball.waiver_valuation_service import WaiverValuationService
            from backend.fantasy_baseball.id_resolution_service import get_quarantined_identity_ids

            roster_yahoo_ids = []
            for p in roster:
                yid = p.get("player_id") or p.get("id")
                if yid:
                    try:
                        roster_yahoo_ids.append(int(yid))
                    except (ValueError, TypeError):
                        pass

            if not roster_yahoo_ids:
                return None

            yahoo_to_mlbam = WaiverEdgeDetector._bulk_yahoo_to_mlbam(roster_yahoo_ids)
            roster_mlbam_ids = list(yahoo_to_mlbam.values())
            if not roster_mlbam_ids:
                return None

            db = SessionLocal()
            try:
                quarantined_ids = get_quarantined_identity_ids(db)
                svc = WaiverValuationService(db)
                ctx = svc.build_team_context(roster_mlbam_ids, quarantined_ids=quarantined_ids)
                logger.info(
                    "Canonical TeamContext: pa_denom=%.0f ip_denom=%.0f quarantined=%d",
                    ctx.rate_pa_denominator,
                    ctx.rate_ip_denominator,
                    len(ctx.quarantined_player_ids),
                )
                return ctx
            finally:
                db.close()
        except Exception as exc:
            logger.warning("_build_canonical_team_context failed (%s) -- no depth scaling", exc)
            return None

    def get_top_moves(self, my_roster, opponent_roster, n_candidates=10, force_refresh=False):
        free_agents = self._enrich_players(self._fetch_fas(force_refresh))
        if not free_agents:
            return []
        my_roster = self._enrich_players(my_roster)
        opponent_roster = self._enrich_players(opponent_roster)
        deficits = self._compute_deficits(my_roster, opponent_roster)
        has_deficit_signal = any(abs(v) > 0 for v in deficits.values())

        # Bulk-load scarcity_rank for all FA candidates in a single query.
        fa_keys = [fa.get("player_key") or fa.get("id") or "" for fa in free_agents[:40]]
        scarcity_lookup = self._load_scarcity_lookup([k for k in fa_keys if k])

        from backend.services.config_service import is_flag_enabled

        # Build single BDL ID map for all FAs -- reused by market signals + matchup context.
        fa_bdl_map: dict = {}  # yahoo_id -> bdl_id
        for fa in free_agents[:40]:
            yahoo_id = fa.get("player_id")
            if yahoo_id:
                bdl_id = self._yahoo_id_to_bdl_id(yahoo_id)
                if bdl_id:
                    fa_bdl_map[yahoo_id] = bdl_id
        bdl_id_list = list(fa_bdl_map.values())

        # PR 4.5: Load market_score as tertiary tiebreaker
        use_market_signals = is_flag_enabled("market_signals_enabled")
        market_lookup: dict = {}
        if use_market_signals and bdl_id_list:
            market_lookup = self._load_market_scores(bdl_id_list)

        # Sprint 4: Load today's matchup context for all FA batters
        use_matchup_context = is_flag_enabled("feature_matchup_enabled")
        matchup_lookup: dict = {}
        if use_matchup_context and bdl_id_list:
            matchup_lookup = self._load_matchup_scores(bdl_id_list)

        # CANONICAL_PROJECTION_V1: build TeamContext for roster-depth-aware scoring.
        # depth_factor adjusts the matchup need_score: shallow rosters (injured players,
        # IL-heavy) → factor > 1.0 (each FA adds more marginal value); deep rosters → < 1.0.
        # Falls back to depth_factor=1.0 on any failure — never blocks waiver recommendations.
        use_canonical = is_flag_enabled("CANONICAL_PROJECTION_V1")
        team_context = None
        batter_depth_factor = 1.0
        pitcher_depth_factor = 1.0
        if use_canonical:
            team_context = self._build_canonical_team_context(my_roster)
            if team_context is not None:
                pa_denom = float(getattr(team_context, "rate_pa_denominator", 0.0))
                ip_denom = float(getattr(team_context, "rate_ip_denominator", 0.0))
                if pa_denom > 0:
                    batter_depth_factor = max(0.8, min(1.2, 3600.0 / pa_denom))
                if ip_denom > 0:
                    pitcher_depth_factor = max(0.8, min(1.2, 900.0 / ip_denom))

        _PITCHER_POSITIONS = frozenset({"SP", "RP", "P"})

        moves = []
        for fa in free_agents[:40]:
            score = self._score_fa_against_deficits(fa, deficits)

            # Scarcity multiplier: boosts need_score for players at scarcer positions.
            # Formula: 1.0 + (13 - rank) * 0.05; clamped to ≥1.0 (no penalty for common positions).
            # Falls back to _POS_GROUP grouping when position_eligibility has no row.
            fa_key = fa.get("player_key") or fa.get("id") or ""
            scarcity_rank = scarcity_lookup.get(fa_key)
            if scarcity_rank is None:
                # _POS_GROUP fallback: use first position to estimate group scarcity
                fa_pos = (fa.get("positions") or ["OF"])[0].upper()
                _FALLBACK_RANK = {
                    "C": 1, "SS": 2, "2B": 3, "3B": 4, "CF": 5,
                    "SP": 6, "RP": 7, "LF": 8, "RF": 9, "1B": 10, "DH": 11, "OF": 12,
                }
                scarcity_rank = _FALLBACK_RANK.get(fa_pos, 13)
            scarcity_multiplier = max(1.0, 1.0 + (13 - scarcity_rank) * 0.05)
            score *= scarcity_multiplier

            # Resolve BDL ID for this FA (pre-built above -- single query per call, not N+1)
            fa_yahoo_id = fa.get("player_id")
            fa_bdl_id = fa_bdl_map.get(fa_yahoo_id) if fa_yahoo_id else None

            # PR 4.5: Market score as tertiary tiebreaker
            market_score = market_lookup.get(fa_bdl_id, 50.0) if fa_bdl_id else 50.0

            if score <= 0 and not has_deficit_signal:
                score = float(fa.get("z_score") or 0.0)

            # Sprint 4: Apply matchup context addend for batters only.
            # Hitter matchup context (opponent ERA/WHIP/park/splits) is only meaningful
            # for batting categories. Skip pitchers -- their matchup is opponent lineup quality,
            # which is a separate model (not yet implemented).
            fa_is_pitcher = bool(_PITCHER_POSITIONS.intersection(fa.get("positions") or []))
            matchup_row = matchup_lookup.get(fa_bdl_id, {}) if fa_bdl_id else {}
            matchup_z = matchup_row.get("matchup_z", 0.0) or 0.0
            matchup_score_val = matchup_row.get("matchup_score", 50.0) or 50.0
            matchup_confidence = matchup_row.get("matchup_confidence", 0.0) or 0.0
            if (
                use_matchup_context
                and not fa_is_pitcher
                and matchup_confidence >= 0.3
            ):
                # Confidence-weighted addend: at z=2.0, conf=1.0 -> +1.0 to score.
                # At z=-2.0, conf=1.0 -> -1.0 (bad matchup dampens recommendation).
                score += matchup_z * matchup_confidence * 0.5

            # CANONICAL_PROJECTION_V1: apply roster-depth factor to the matchup component.
            # Amplifies FA value when team is IL-heavy/shallow; dampens when roster is deep.
            applied_depth_factor = 1.0
            if team_context is not None:
                applied_depth_factor = pitcher_depth_factor if fa_is_pitcher else batter_depth_factor
                score *= applied_depth_factor

            fa_positions = fa.get("positions") or []
            drop_candidate = self._weakest_droppable_at(my_roster, fa_positions)
            move = {
                "add_player": fa,
                "drop_player_name": drop_candidate.get("name", "") if drop_candidate else "",
                "need_score": score,
                "market_score": market_score,
                "canonical_depth_factor": applied_depth_factor,
                "matchup_z": matchup_z if not fa_is_pitcher else 0.0,
                "matchup_score": matchup_score_val if not fa_is_pitcher else 50.0,
                "matchup_confidence": matchup_confidence if not fa_is_pitcher else 0.0,
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
        # PR 4.5: Sort with market_score as tertiary tiebreaker (higher = better buy signal)
        moves.sort(key=lambda m: (m["win_prob_gain"], m["need_score"], m["market_score"]), reverse=True)
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
