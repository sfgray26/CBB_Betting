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
    """

    def analyze_matchup(
        self,
        home: TeamPlayStyle,
        away: TeamPlayStyle,
    ) -> MatchupAdjustment:
        """
        Run the full matchup analysis between two team profiles.

        Returns ``MatchupAdjustment`` with margin and SD adjustments.
        """
        adj = MatchupAdjustment()

        self._pace_mismatch(home, away, adj)
        self._three_point_vs_drop(home, away, adj)
        self._transition_gap(home, away, adj)
        self._rebounding_impact(home, away, adj)
        self._turnover_battle(home, away, adj)
        self._zone_vs_three(home, away, adj)

        adj.margin_adj = round(sum(adj.factors.values()), 3)
        return adj

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
            adj.sd_adj += 0.5 * (pace_diff - 6) / 4.0
            adj.factors['pace_mismatch_sd'] = 0.5 * (pace_diff - 6) / 4.0
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
        Scrape team four-factors from BartTorvik and build profiles.

        Returns the number of teams loaded.
        """
        import requests

        url = "https://barttorvik.com/2026_team_results.csv"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("BartTorvik profile scrape failed: %s", exc)
            return 0

        count = 0
        lines = resp.text.strip().split('\n')
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) < 20:
                continue
            try:
                team_name = parts[1].strip()
                profile = TeamPlayStyle(
                    team=team_name,
                    pace=float(parts[4]) if len(parts) > 4 and parts[4] else 68.0,
                    three_par=float(parts[10]) / 100.0 if len(parts) > 10 and parts[10] else 0.36,
                    ft_rate=float(parts[12]) / 100.0 if len(parts) > 12 and parts[12] else 0.30,
                    to_pct=float(parts[14]) / 100.0 if len(parts) > 14 and parts[14] else 0.17,
                    orb_pct=float(parts[16]) / 100.0 if len(parts) > 16 and parts[16] else 0.28,
                )
                self._profiles[team_name] = profile
                count += 1
            except (ValueError, IndexError):
                continue

        logger.info("Loaded %d team profiles from BartTorvik", count)
        return count

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
