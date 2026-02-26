"""
Matchup-specific edge engine — play-by-play model infrastructure.

Moves beyond top-level efficiency metrics by modelling *how* two teams
interact.  Instead of ``Team A AdjEM - Team B AdjEM``, this engine
computes matchup-specific geometry adjustments:

    - Pace mismatch penalty (one team wants to run, other wants to grind)
    - Shot-type distribution clashes (3PA-heavy offence vs drop coverage)
    - Transition defence gaps
    - Rebounding differential impact on variance

Data sources (in priority order):
    1. Play-by-play feeds (via stats.ncaa.org / bigdataball)
    2. BartTorvik team profile pages (public)
    3. Manual configuration

This module is the *infrastructure* — it defines the data model and
adjustment calculations.  Populating the actual team profiles requires
the data ingestion pipeline (see ``scripts/ingest_pbp.py``).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team profile
# ---------------------------------------------------------------------------

@dataclass
class TeamPlayStyle:
    """
    Team-level play-style profile derived from play-by-play data.

    All rates are per-100-possessions or as fractions (0-1).
    """

    team: str

    # Pace & tempo
    pace: float = 68.0                # Possessions per 40 min (D1 avg ~68)
    avg_possession_length: float = 16.5  # Seconds

    # Shot distribution (fractions summing to ~1)
    rim_rate: float = 0.30            # % of FGA at the rim
    mid_range_rate: float = 0.15      # % of FGA from mid-range
    three_par: float = 0.36           # 3PA / FGA
    ft_rate: float = 0.30             # FTA / FGA (free-throw rate)

    # Shooting efficiency
    rim_fg_pct: float = 0.62
    mid_fg_pct: float = 0.38
    three_fg_pct: float = 0.34

    # Transition
    transition_freq: float = 0.15     # % of possessions in transition
    transition_ppp: float = 1.10      # Points per possession in transition

    # Rebounding
    orb_pct: float = 0.28             # Offensive rebound rate
    drb_pct: float = 0.72             # Defensive rebound rate

    # Defence scheme indicators
    drop_coverage_pct: float = 0.0    # % of PnR defended with drop
    blitz_pct: float = 0.0            # % of PnR defended with blitz/hedge
    zone_pct: float = 0.0             # % of possessions in zone defence

    # Turnover tendencies
    to_pct: float = 0.17              # Turnover rate
    steal_pct: float = 0.09           # Steal rate (defensive)

    # Defensive four factors (opponent stats allowed) — D1 averages as defaults.
    # Populated from BartTorvik EFGD%/TORD/FTRD/3PAD% columns.
    def_efg_pct: float = 0.505        # Opponent eFG% allowed
    def_to_pct: float = 0.175         # Opponent TO rate forced
    def_ft_rate: float = 0.280        # Opponent FT rate allowed
    def_three_par: float = 0.360      # Opponent 3PA rate allowed


# ---------------------------------------------------------------------------
# Matchup adjustment
# ---------------------------------------------------------------------------

@dataclass
class MatchupAdjustment:
    """Output of the matchup engine — margin and variance adjustments."""

    margin_adj: float = 0.0           # Points added to the raw margin
    sd_adj: float = 0.0              # Points added to the SD
    factors: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class MatchupEngine:
    """
    Computes matchup-specific adjustments to the model's margin and SD.

    These adjustments are *additive* to the base ratings-derived margin
    and capture second-order effects that top-level metrics miss.

    **Diminishing returns (two layers):**

    1. **Per-category RSS dampening**: Factors are grouped into broad
       correlation categories reflecting shared causal mechanisms:

       - *Possession Generating* (rebounding, turnover_battle, transition_gap):
         These factors all generate extra possessions through the same
         zero-sum resource pool.  A team cannot simultaneously dominate
         the glass, force turnovers, AND run in transition at full
         magnitude because each requires controlling the same possessions.

       - *Shot Quality* (3PT vs drop, zone vs 3PT, pace-driven effects):
         These factors modify per-possession efficiency through shot
         selection and defensive scheme interactions.

       Within each category, factors are split by sign and each side is
       compressed via L2 norm (RSS): ``rss_pos - rss_neg``.  This prevents
       L2-norm amplification on mixed-sign inputs while penalising the
       stacking of correlated same-sign advantages.  The L2 norm is the
       natural measure of combined magnitude for correlated signals —
       it equals the raw value for a single factor but grows sub-linearly
       as more same-sign factors pile up.

    2. **Global tanh activation**: After per-category RSS, the category
       totals are summed and passed through ``cap * tanh(raw / cap)``.
       This imposes a hard ceiling of ``MAX_TOTAL_ADJ`` points (default
       4.0) and provides smooth saturation at the extremes.

    **Mathematical justification:**
    Real basketball advantages are correlated and subject to diminishing
    returns.  Category-level RSS prevents "category stuffing" (exploiting
    many correlated factors within one causal family), while the global
    ``tanh`` prevents the sum of independent categories from growing
    without bound.
    """

    # Default hard cap on total margin adjustment (points).
    # At the D1-average game total of ~145, this is ≈ 2.75% of total — a
    # realistic upper bound on play-style-driven edge.
    MAX_TOTAL_ADJ: float = 4.0

    # Default per-category RSS cap (points).
    CATEGORY_CAP: float = 3.0

    # Reference game total used to normalise dynamic caps.
    # At D1 average pace (~68 poss/40 min) and eFG ≈ 0.50, expected total ≈ 145.
    _REFERENCE_TOTAL: float = 145.0

    # Factor → category mapping.  Factors not listed here are placed in
    # an "uncategorised" bucket and pass through without RSS dampening.
    FACTOR_CATEGORIES: Dict[str, str] = {
        # Possession Generating — extra possessions from boards, steals, fastbreaks
        "rebounding": "possession_generating",
        "turnover_battle": "possession_generating",
        "transition_gap": "possession_generating",
        # Shot Quality — per-possession efficiency from scheme/matchup geometry
        "home_3_vs_drop": "shot_quality",
        "away_3_vs_drop": "shot_quality",
        "home_zone_vs_away_3": "shot_quality",
        "away_zone_vs_home_3": "shot_quality",
    }

    def _effective_caps(
        self,
        game_total: Optional[float] = None,
        base_sd: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute dynamic (MAX_TOTAL_ADJ, CATEGORY_CAP) scaled to the game's
        expected scoring environment.

        Rationale
        ---------
        A rigid 4.0 / 3.0 cap treats a 120-point slow grind and a 170-point
        run-and-gun game identically.  But matchup factors (pace mismatch,
        3PA vs drop) represent a larger absolute edge in high-total games
        because more possessions amplify every per-possession advantage.

        Scaling rule::

            scale = sqrt(effective_total / _REFERENCE_TOTAL)

        where ``effective_total`` is derived (in priority order) from:
          1. ``game_total`` — sharp consensus total (most direct).
          2. ``base_sd``    — back-solve from SD ≈ sqrt(total) × 0.85, so
                             total ≈ (base_sd / 0.85)².
          3. Class defaults — if neither is available.

        ``sqrt`` ensures the caps grow sub-linearly: doubling the total adds
        only ~41% to the caps rather than doubling them, reflecting that
        individual play-style interactions do not scale 1-for-1 with pace.

        The caps are clamped to [0.5×default, 2.0×default] to prevent
        degenerate values on extreme inputs.

        Args:
            game_total: Expected combined score (over/under line).
            base_sd:    Effective game standard deviation (used if total absent).

        Returns:
            (effective_max_total_adj, effective_category_cap)
        """
        # Derive effective total from the best available source.
        if game_total is not None and game_total > 0:
            effective_total = game_total
        elif base_sd is not None and base_sd > 0:
            # Invert dynamic-SD formula: total ≈ (sd / 0.85)²
            effective_total = (base_sd / 0.85) ** 2
        else:
            return self.MAX_TOTAL_ADJ, self.CATEGORY_CAP

        scale = float(np.sqrt(effective_total / self._REFERENCE_TOTAL))

        # Clamp to [50%, 200%] of the class defaults.
        scale = max(0.5, min(scale, 2.0))

        return (
            round(self.MAX_TOTAL_ADJ * scale, 3),
            round(self.CATEGORY_CAP * scale, 3),
        )

    def analyze_matchup(
        self,
        home: TeamPlayStyle,
        away: TeamPlayStyle,
        game_total: Optional[float] = None,
        base_sd: Optional[float] = None,
    ) -> MatchupAdjustment:
        """
        Run the full matchup analysis between two team profiles.

        Returns ``MatchupAdjustment`` with margin and SD adjustments.

        The raw factor vector is aggregated via ``_apply_diminishing_returns``
        which applies:
          1. Per-category RSS dampening (``FACTOR_CATEGORIES``).
          2. Global ``tanh`` activation with a dynamic ceiling derived from
             ``game_total`` or ``base_sd`` (see ``_effective_caps()``).

        Args:
            home:       Home team play-style profile.
            away:       Away team play-style profile.
            game_total: Expected combined score (sharp consensus over/under).
                        When provided, scales MAX_TOTAL_ADJ and CATEGORY_CAP
                        relative to the D1-average reference total of 145.
            base_sd:    Effective game SD used to back-solve for a total when
                        ``game_total`` is unavailable.
        """
        adj = MatchupAdjustment()

        self._pace_mismatch(home, away, adj)
        self._three_point_vs_drop(home, away, adj)
        self._transition_gap(home, away, adj)
        self._rebounding_impact(home, away, adj)
        self._turnover_battle(home, away, adj)
        self._zone_vs_three(home, away, adj)

        max_adj, cat_cap = self._effective_caps(game_total=game_total, base_sd=base_sd)
        adj.margin_adj = round(
            self._apply_diminishing_returns(adj.factors, max_total_adj=max_adj, category_cap=cat_cap),
            3,
        )
        return adj

    def _apply_diminishing_returns(
        self,
        factors: Dict[str, float],
        max_total_adj: Optional[float] = None,
        category_cap: Optional[float] = None,
    ) -> float:
        """
        Aggregate raw matchup factors with categorical RSS + global tanh.

        **Layer 1 — Per-category split-sign RSS dampening:**
        Factors are grouped by ``FACTOR_CATEGORIES`` into correlation
        buckets.  Within each bucket, *positive* and *negative* factors
        are separated and their magnitudes are compressed independently
        via L2 norm (root-sum-square):

            rss_pos = sqrt(sum(f_i² for f_i > 0))
            rss_neg = sqrt(sum(f_i² for f_i < 0))
            category_adj = min(rss_pos, CAP) - min(rss_neg, CAP)

        This prevents a known amplification bug where mixed-sign factors
        within the same category inflate the result.  For example,
        ``[+2.0, -1.0]`` previously produced ``sign(1)*sqrt(5) = 2.24``
        instead of the correct net of ``+2.0 - 1.0 = +1.0``.  With the
        split-sign approach: ``min(2.0, CAP) - min(1.0, CAP) = 1.0``.

        The L2 norm is the natural aggregation for correlated signals:
        - For a single factor it equals the raw value (no distortion).
        - For N equal factors of magnitude m it equals m*sqrt(N), which
          grows sub-linearly vs the raw sum N*m.
        - ``category_cap`` prevents any single category from dominating.

        Factors not assigned to a category pass through at full value.

        **Layer 2 — Global tanh activation:**
        The sum of all category contributions is passed through:

            adjusted = max_total_adj * tanh(category_sum / max_total_adj)

        This ensures:
            - Small adjustments are approximately linear (tanh ≈ x for |x| << cap).
            - Large adjustments saturate toward ±max_total_adj.
            - The output never exceeds max_total_adj in absolute value.

        Args:
            factors:       Dict mapping factor name → point adjustment.
            max_total_adj: Override the global tanh ceiling (points).  If None,
                           uses ``self.MAX_TOTAL_ADJ``.  Callers that have already
                           computed dynamic caps via ``_effective_caps()`` pass the
                           result in here; tests can also pass explicit values.
            category_cap:  Override the per-category RSS ceiling (points).  If None,
                           uses ``self.CATEGORY_CAP``.

        Returns:
            Total margin adjustment after diminishing returns (float).
        """
        if not factors:
            return 0.0

        _max_adj = max_total_adj if max_total_adj is not None else self.MAX_TOTAL_ADJ
        _cat_cap = category_cap if category_cap is not None else self.CATEGORY_CAP

        # Layer 1: group factors into categories and apply per-category RSS
        buckets: Dict[str, List[float]] = {}
        uncategorised_sum = 0.0

        for name, value in factors.items():
            cat = self.FACTOR_CATEGORIES.get(name)
            if cat is not None:
                buckets.setdefault(cat, []).append(value)
            else:
                uncategorised_sum += value

        category_sum = uncategorised_sum
        for _cat, values in buckets.items():
            # Split positive and negative factors to avoid L2-norm
            # amplification on mixed-sign inputs.
            pos = [v for v in values if v > 0]
            neg = [v for v in values if v < 0]

            rss_pos = float(np.sqrt(sum(v ** 2 for v in pos))) if pos else 0.0
            rss_neg = float(np.sqrt(sum(v ** 2 for v in neg))) if neg else 0.0

            category_sum += min(rss_pos, _cat_cap) - min(rss_neg, _cat_cap)

        # Layer 2: global tanh activation with dynamic cap
        adjusted = _max_adj * float(np.tanh(category_sum / _max_adj))
        return adjusted

    # ------------------------------------------------------------------
    # Individual matchup factors
    # ------------------------------------------------------------------

    def _pace_mismatch(
        self, home: TeamPlayStyle, away: TeamPlayStyle, adj: MatchupAdjustment
    ) -> None:
        """
        Large pace mismatches (one team runs, the other grinds) create
        variance because the game tempo is uncertain.  The fast team gets
        a small margin boost if they also have a transition advantage.
        """
        pace_diff = abs(home.pace - away.pace)
        if pace_diff > 6:
            # Pace mismatch is a variance (SD) factor, not a margin factor.
            # Track it only via adj.sd_adj so it is not summed into margin_adj.
            adj.sd_adj += 0.5 * (pace_diff - 6) / 4.0
            adj.notes.append(
                f"Pace mismatch: {home.pace:.0f} vs {away.pace:.0f} (+SD)"
            )

    def _three_point_vs_drop(
        self, home: TeamPlayStyle, away: TeamPlayStyle, adj: MatchupAdjustment
    ) -> None:
        """
        A team with a high 3PA rate attacking a defence that plays heavy
        drop coverage gets more open looks from 3.  This creates a
        positive margin adjustment for the shooting team AND increases
        variance (3s are high-variance shots).
        """
        # Home offence vs away defence
        if home.three_par > 0.40 and away.drop_coverage_pct > 0.30:
            margin_boost = (home.three_par - 0.36) * (away.drop_coverage_pct) * 3.0
            adj.factors['home_3_vs_drop'] = margin_boost
            adj.sd_adj += 0.3
            adj.notes.append(
                f"Home 3PAr {home.three_par:.0%} vs Away drop {away.drop_coverage_pct:.0%}: "
                f"+{margin_boost:.1f} margin, +0.3 SD"
            )

        # Away offence vs home defence
        if away.three_par > 0.40 and home.drop_coverage_pct > 0.30:
            margin_penalty = -((away.three_par - 0.36) * home.drop_coverage_pct * 3.0)
            adj.factors['away_3_vs_drop'] = margin_penalty
            adj.sd_adj += 0.3
            adj.notes.append(
                f"Away 3PAr {away.three_par:.0%} vs Home drop {home.drop_coverage_pct:.0%}: "
                f"{margin_penalty:.1f} margin, +0.3 SD"
            )

    def _transition_gap(
        self, home: TeamPlayStyle, away: TeamPlayStyle, adj: MatchupAdjustment
    ) -> None:
        """
        A team that generates lots of transition possessions against
        a team that doesn't get back in transition creates easy points.
        """
        home_trans_edge = home.transition_freq * home.transition_ppp
        away_trans_edge = away.transition_freq * away.transition_ppp

        net = home_trans_edge - away_trans_edge
        if abs(net) > 0.02:
            # Scale: a 0.05 net edge ≈ ~1 point margin adjustment
            margin_adj = net * 20.0
            adj.factors['transition_gap'] = round(margin_adj, 2)
            adj.notes.append(
                f"Transition edge: home {home_trans_edge:.3f} vs away {away_trans_edge:.3f}"
            )

    def _rebounding_impact(
        self, home: TeamPlayStyle, away: TeamPlayStyle, adj: MatchupAdjustment
    ) -> None:
        """
        Offensive rebounding creates second-chance points and increases
        game variance (more possessions = more variance).
        """
        home_orb_edge = home.orb_pct - (1 - away.drb_pct)
        away_orb_edge = away.orb_pct - (1 - home.drb_pct)

        net_orb = home_orb_edge - away_orb_edge
        if abs(net_orb) > 0.05:
            # Each 5% ORB edge ≈ 0.8 point margin shift
            adj.factors['rebounding'] = round(net_orb * 16.0, 2)
            adj.sd_adj += abs(net_orb) * 2.0  # More boards = more variance

    def _turnover_battle(
        self, home: TeamPlayStyle, away: TeamPlayStyle, adj: MatchupAdjustment
    ) -> None:
        """
        A team that forces lots of turnovers against a turnover-prone
        team creates fast break opportunities and variance.
        """
        # Home defence stealing from turnover-prone away offence
        home_to_edge = away.to_pct * home.steal_pct
        away_to_edge = home.to_pct * away.steal_pct
        net = home_to_edge - away_to_edge

        if abs(net) > 0.005:
            adj.factors['turnover_battle'] = round(net * 50.0, 2)

    def _zone_vs_three(
        self, home: TeamPlayStyle, away: TeamPlayStyle, adj: MatchupAdjustment
    ) -> None:
        """
        Zone defence is vulnerable to good 3-point shooting teams.
        """
        # Home defence zones vs away 3-point shooting
        if home.zone_pct > 0.30 and away.three_fg_pct > 0.36:
            zone_penalty = -0.5 * home.zone_pct * (away.three_fg_pct - 0.34) * 10
            adj.factors['home_zone_vs_away_3'] = round(zone_penalty, 2)
            adj.notes.append(
                f"Home zone {home.zone_pct:.0%} vs Away 3FG% {away.three_fg_pct:.0%}"
            )

        # Away defence zones vs home 3-point shooting
        if away.zone_pct > 0.30 and home.three_fg_pct > 0.36:
            zone_bonus = 0.5 * away.zone_pct * (home.three_fg_pct - 0.34) * 10
            adj.factors['away_zone_vs_home_3'] = round(zone_bonus, 2)


# ---------------------------------------------------------------------------
# Team profile cache / loader
# ---------------------------------------------------------------------------

class TeamProfileCache:
    """
    In-memory cache of team play-style profiles.

    Profiles can be loaded from:
        - Database (if play-by-play data has been ingested)
        - JSON config file (manual seeding)
        - Scraped from BartTorvik (four-factors page)
    """

    def __init__(self):
        self._profiles: Dict[str, TeamPlayStyle] = {}

    def get(self, team: str) -> Optional[TeamPlayStyle]:
        """Return profile for a team, or None if not available."""
        return self._profiles.get(team)

    def set(self, team: str, profile: TeamPlayStyle) -> None:
        """Set or update a team profile."""
        self._profiles[team] = profile

    def load_from_barttorvik(self) -> int:
        """
        Load team four-factors (including defensive stats) from BartTorvik.

        Delegates to RatingsService.get_barttorvik_full_stats() which uses
        robust CSV header detection and the _safe_pct() scale normalizer.
        This replaces the previous hardcoded positional parser which mapped
        parts[4] to 'Rec' (win-loss record) instead of pace, causing all
        BartTorvik profiles to silently load with D1-average defaults and
        the profile cache to be functionally identical to the heuristic path.

        Defensive stats (def_efg_pct, def_to_pct, def_ft_rate, def_three_par)
        are now populated from the EFGD%/TORD/FTRD/3PAD% columns so the Markov
        simulator uses real per-team defensive data instead of D1 averages.

        Returns the number of teams loaded.
        """
        from backend.services.ratings import get_ratings_service

        service = get_ratings_service()
        full_stats = service.get_barttorvik_full_stats()
        if not full_stats:
            logger.warning("BartTorvik profile scrape returned no data")
            return 0

        count = 0
        for team_name, stat in full_stats.items():
            def _f(key: str, default: float) -> float:
                v = stat.get(key)
                return v if v is not None else default

            profile = TeamPlayStyle(
                team=team_name,
                pace=_f("pace", 68.0),
                three_par=_f("three_par", 0.36),
                ft_rate=_f("ft_rate", 0.30),
                to_pct=_f("to_pct", 0.175),
                # Offensive four-factors
                # (rim_rate / mid_range_rate / shooting pcts not in BartTorvik CSV;
                #  leave at dataclass defaults so the matchup engine uses sensible values)
                # Defensive four factors — the core fix
                def_efg_pct=_f("def_efg_pct", 0.505),
                def_to_pct=_f("def_to_pct", 0.175),
                def_ft_rate=_f("def_ft_rate", 0.280),
                def_three_par=_f("def_three_par", 0.360),
            )
            self._profiles[team_name] = profile
            count += 1

        n_def = sum(
            1 for t in full_stats.values() if t.get("def_efg_pct") is not None
        )
        logger.info(
            "Loaded %d team profiles from BartTorvik (%d with real def_efg_pct)",
            count, n_def,
        )
        return count

    def teams(self) -> List[str]:
        """Return the list of team names currently in the cache."""
        return list(self._profiles.keys())

    def has_profiles(self) -> bool:
        return len(self._profiles) > 0

    def __len__(self) -> int:
        return len(self._profiles)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_matchup_engine: Optional[MatchupEngine] = None
_profile_cache: Optional[TeamProfileCache] = None


def get_matchup_engine() -> MatchupEngine:
    global _matchup_engine
    if _matchup_engine is None:
        _matchup_engine = MatchupEngine()
    return _matchup_engine


def get_profile_cache() -> TeamProfileCache:
    global _profile_cache
    if _profile_cache is None:
        _profile_cache = TeamProfileCache()
    return _profile_cache
