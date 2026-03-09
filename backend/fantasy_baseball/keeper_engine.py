"""
Keeper Evaluation Engine — Treemendous League (Yahoo ID 72586)

Calculates "keeper surplus value": how much a player is worth relative to
where they'd be drafted fresh. Positive surplus = keep. Negative = let go.

League scoring (18 categories, H2H One Win):
  Batting (9):  R, H, HR, RBI, K(negative), TB, AVG, OPS, NSB
  Pitching (9): W, L(negative), HR(negative), K, ERA(negative),
                WHIP(negative), K/9, QS, NSV

Usage:
    from backend.fantasy_baseball.keeper_engine import KeeperEngine
    engine = KeeperEngine()
    report = engine.evaluate_roster(roster_players)
    engine.print_report(report)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring category configuration for this league
# ---------------------------------------------------------------------------

# For z-score direction: +1 means higher is better, -1 means lower is better
BATTING_CATEGORIES: dict[str, int] = {
    "R": 1,    # Runs
    "H": 1,    # Hits
    "HR": 1,   # Home Runs
    "RBI": 1,  # Runs Batted In
    "K": -1,   # Batter Strikeouts (negative!)
    "TB": 1,   # Total Bases
    "AVG": 1,  # Batting Average
    "OPS": 1,  # On-base + Slugging
    "NSB": 1,  # Net Stolen Bases (SB - CS)
}

PITCHING_CATEGORIES: dict[str, int] = {
    "W": 1,     # Wins
    "L": -1,    # Losses (negative!)
    "HR": -1,   # Home Runs Allowed (negative!)
    "K": 1,     # Strikeouts
    "ERA": -1,  # ERA (negative: lower is better)
    "WHIP": -1, # WHIP (negative: lower is better)
    "K9": 1,    # K/9
    "QS": 1,    # Quality Starts
    "NSV": 1,   # Net Saves (SV - BS)
}

# Category weights — based on scarcity and H2H impact
# (All categories equal weight in H2H One Win, but scarcity increases value)
CATEGORY_WEIGHTS: dict[str, float] = {
    # Batting
    "R": 1.0, "H": 0.9, "HR": 1.1, "RBI": 1.1, "K": 0.8,
    "TB": 1.0, "AVG": 1.1, "OPS": 1.2, "NSB": 1.4,  # NSB scarce
    # Pitching
    "W": 1.1, "L": 0.8, "HR_P": 0.7, "K_P": 1.1,
    "ERA": 1.1, "WHIP": 1.1, "K9": 1.0, "QS": 1.0, "NSV": 1.5,  # NSV very scarce
}

# Round cost bands — 12-team snake draft, 23+ rounds
ROUND_ADP_BANDS = [
    (1, 1.5),    # Round 1: picks 1-12
    (2, 2.5),    # Round 2: picks 13-24
    (3, 3.5),    # Round 3-4
    (5, 5.5),    # Round 5-6
    (7, 7.5),    # Round 7-9
    (10, 10.5),  # Round 10-13
    (14, 14.5),  # Round 14-17
    (18, 18.5),  # Round 18+
    (23, 23.5),  # End of draft
]


@dataclass
class PlayerProjection:
    """2026 projected stats for a single player."""
    name: str
    yahoo_player_key: str
    team: str
    positions: list[str]
    player_type: str  # 'batter' or 'pitcher'
    age: int = 0
    # Batting projections
    pa: int = 0
    r: float = 0.0
    h: float = 0.0
    hr: float = 0.0
    rbi: float = 0.0
    k_bat: float = 0.0     # Batter Ks (negative in this league)
    tb: float = 0.0
    avg: float = 0.0
    obp: float = 0.0
    slg: float = 0.0
    ops: float = 0.0
    sb: float = 0.0
    cs: float = 0.0
    nsb: float = 0.0       # sb - cs
    # Pitching projections
    ip: float = 0.0
    w: float = 0.0
    l: float = 0.0
    sv: float = 0.0
    bs: float = 0.0
    qs: float = 0.0
    k_pit: float = 0.0     # Pitcher Ks
    era: float = 0.0
    whip: float = 0.0
    k9: float = 0.0
    hr_pit: float = 0.0    # HR allowed
    nsv: float = 0.0       # sv - bs
    # Computed
    z_score: float = 0.0
    category_contributions: dict = field(default_factory=dict)
    # Draft context
    keeper_round_cost: int = 0   # What round this keeper costs
    adp_round: float = 0.0       # Where they'd be drafted in a fresh draft
    keeper_surplus: float = 0.0  # z_score surplus vs. round-equivalent replacement
    notes: list[str] = field(default_factory=list)


@dataclass
class KeeperReport:
    players: list[PlayerProjection]
    deadline: str = "Fri Mar 20 3:00am EDT"
    recommendation: list[str] = field(default_factory=list)


class CategoryValueEngine:
    """
    Computes z-scores for all 18 categories across a player pool.
    Handles batter K (negative) and pitcher L/HR/ERA/WHIP (negative) correctly.
    """

    def __init__(self, player_pool: list[PlayerProjection]):
        self.batters = [p for p in player_pool if p.player_type == "batter"]
        self.pitchers = [p for p in player_pool if p.player_type == "pitcher"]
        self._bat_stats: dict[str, list[float]] = {}
        self._pit_stats: dict[str, list[float]] = {}
        self._compute_pool_stats()

    def _compute_pool_stats(self):
        """Pre-compute mean/std for each category across the player pool."""
        if self.batters:
            self._bat_stats = {
                "R": [p.r for p in self.batters],
                "H": [p.h for p in self.batters],
                "HR": [p.hr for p in self.batters],
                "RBI": [p.rbi for p in self.batters],
                "K": [p.k_bat for p in self.batters],    # higher = worse
                "TB": [p.tb for p in self.batters],
                "AVG": [p.avg for p in self.batters],
                "OPS": [p.ops for p in self.batters],
                "NSB": [p.nsb for p in self.batters],
            }
        if self.pitchers:
            self._pit_stats = {
                "W": [p.w for p in self.pitchers],
                "L": [p.l for p in self.pitchers],        # higher = worse
                "HR_P": [p.hr_pit for p in self.pitchers], # higher = worse
                "K_P": [p.k_pit for p in self.pitchers],
                "ERA": [p.era for p in self.pitchers],     # higher = worse
                "WHIP": [p.whip for p in self.pitchers],   # higher = worse
                "K9": [p.k9 for p in self.pitchers],
                "QS": [p.qs for p in self.pitchers],
                "NSV": [p.nsv for p in self.pitchers],
            }

    @staticmethod
    def _z(value: float, values: list[float], direction: int = 1) -> float:
        """Compute z-score. direction=-1 flips sign (lower is better)."""
        import statistics
        if len(values) < 2:
            return 0.0
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        if std == 0:
            return 0.0
        raw_z = (value - mean) / std
        return raw_z * direction

    def score_batter(self, player: PlayerProjection) -> float:
        """Return total z-score for a batter across all 9 batting categories."""
        if not self._bat_stats:
            return 0.0
        contributions = {}
        total = 0.0

        category_map = [
            ("R", player.r, 1),
            ("H", player.h, 1),
            ("HR", player.hr, 1),
            ("RBI", player.rbi, 1),
            ("K", player.k_bat, -1),   # negative direction
            ("TB", player.tb, 1),
            ("AVG", player.avg, 1),
            ("OPS", player.ops, 1),
            ("NSB", player.nsb, 1),
        ]
        for cat, val, direction in category_map:
            z = self._z(val, self._bat_stats[cat], direction)
            weight = CATEGORY_WEIGHTS.get(cat, 1.0)
            weighted_z = z * weight
            contributions[cat] = round(weighted_z, 3)
            total += weighted_z

        player.category_contributions = contributions
        return round(total, 3)

    def score_pitcher(self, player: PlayerProjection) -> float:
        """Return total z-score for a pitcher across all 9 pitching categories."""
        if not self._pit_stats:
            return 0.0
        contributions = {}
        total = 0.0

        category_map = [
            ("W", player.w, 1, "W"),
            ("L", player.l, -1, "L"),           # negative direction
            ("HR_P", player.hr_pit, -1, "HR_P"), # negative direction
            ("K_P", player.k_pit, 1, "K_P"),
            ("ERA", player.era, -1, "ERA"),       # negative direction
            ("WHIP", player.whip, -1, "WHIP"),    # negative direction
            ("K9", player.k9, 1, "K9"),
            ("QS", player.qs, 1, "QS"),
            ("NSV", player.nsv, 1, "NSV"),
        ]
        for pool_key, val, direction, cat_label in category_map:
            z = self._z(val, self._pit_stats[pool_key], direction)
            weight = CATEGORY_WEIGHTS.get(pool_key, 1.0)
            weighted_z = z * weight
            contributions[cat_label] = round(weighted_z, 3)
            total += weighted_z

        player.category_contributions = contributions
        return round(total, 3)

    def score_all(self) -> list[PlayerProjection]:
        """Score every player in the pool and set their z_score."""
        for p in self.batters:
            p.z_score = self.score_batter(p)
        for p in self.pitchers:
            p.z_score = self.score_pitcher(p)
        return self.batters + self.pitchers


class KeeperEngine:
    """
    Evaluates keeper decisions by comparing projected player value
    against the cost of drafting them in a specific round.

    Key question: "Is this player worth more than the best available
    player I could draft at that round pick?"
    """

    def __init__(self, player_pool: Optional[list[PlayerProjection]] = None):
        self.pool = player_pool or []
        self._value_engine: Optional[CategoryValueEngine] = None
        if self.pool:
            self._value_engine = CategoryValueEngine(self.pool)
            self._value_engine.score_all()

    def set_pool(self, pool: list[PlayerProjection]) -> None:
        self.pool = pool
        self._value_engine = CategoryValueEngine(pool)
        self._value_engine.score_all()

    def _replacement_value_at_round(self, round_num: int, player_type: str) -> float:
        """
        Estimate the average z-score of a player drafted at a given round.
        Based on the player pool ranked by z_score.
        """
        relevant = [p for p in self.pool if p.player_type == player_type]
        relevant_sorted = sorted(relevant, key=lambda p: p.z_score, reverse=True)

        # In a 12-team snake draft, round N = pick positions (N-1)*12 to N*12
        pick_start = (round_num - 1) * 12
        pick_end = round_num * 12
        players_at_round = relevant_sorted[pick_start:pick_end]

        if not players_at_round:
            # Beyond end of draft — value approaches 0
            if relevant_sorted:
                return relevant_sorted[-1].z_score
            return 0.0

        return sum(p.z_score for p in players_at_round) / len(players_at_round)

    def compute_surplus(self, player: PlayerProjection) -> float:
        """
        Surplus value = player z_score minus the expected z_score of
        a player you'd draft at the same round (the opportunity cost).
        """
        if self._value_engine is None:
            return 0.0
        replacement = self._replacement_value_at_round(
            player.keeper_round_cost, player.player_type
        )
        return round(player.z_score - replacement, 3)

    def evaluate_roster(self, keepers: list[PlayerProjection]) -> KeeperReport:
        """
        Evaluate a list of keeper candidates against the full player pool.
        Sets keeper_surplus on each player and ranks recommendations.
        """
        if self._value_engine is None and self.pool:
            self._value_engine = CategoryValueEngine(self.pool)
            self._value_engine.score_all()

        # Score keepers if not already scored
        if self._value_engine:
            for p in keepers:
                if p.z_score == 0.0:
                    if p.player_type == "batter":
                        p.z_score = self._value_engine.score_batter(p)
                    else:
                        p.z_score = self._value_engine.score_pitcher(p)

        for p in keepers:
            p.keeper_surplus = self.compute_surplus(p)
            self._add_notes(p)

        keepers_sorted = sorted(keepers, key=lambda p: p.keeper_surplus, reverse=True)

        recommendations = self._build_recommendations(keepers_sorted)
        return KeeperReport(players=keepers_sorted, recommendation=recommendations)

    @staticmethod
    def _add_notes(player: PlayerProjection) -> None:
        """Flag strategic considerations for each player."""
        notes = []
        if player.player_type == "batter":
            if player.k_bat / max(player.pa, 1) > 0.28:
                notes.append(f"High K% ({player.k_bat/max(player.pa,1):.1%}) — K penalty hurts this player")
            if player.nsb > 25:
                notes.append(f"Elite NSB ({player.nsb:.0f}) — scarce category, strong keeper")
            if player.ops > 0.900:
                notes.append(f"Elite OPS ({player.ops:.3f})")
            if player.hr > 35:
                notes.append(f"Power upside ({player.hr:.0f} HR projected)")
        else:
            if player.nsv > 25:
                notes.append(f"Elite closer ({player.nsv:.0f} NSV) — scarce, strong keeper")
            if player.k9 > 10.0:
                notes.append(f"Elite K/9 ({player.k9:.1f}) — double value (K + K/9)")
            if player.era < 2.80:
                notes.append(f"Ace-tier ERA ({player.era:.2f})")
            if player.l > 10:
                notes.append(f"High L risk ({player.l:.0f} projected losses) — L category penalty")
            if player.qs > 18:
                notes.append(f"Workhorse SP ({player.qs:.0f} QS) — QS category leader")
        player.notes = notes

    @staticmethod
    def _build_recommendations(players: list[PlayerProjection]) -> list[str]:
        recs = ["=== KEEPER RECOMMENDATIONS (ranked by surplus value) ===\n"]
        for i, p in enumerate(players, 1):
            surplus_str = f"+{p.keeper_surplus:.2f}" if p.keeper_surplus >= 0 else f"{p.keeper_surplus:.2f}"
            verdict = "KEEP ✓" if p.keeper_surplus > 0.5 else ("BORDERLINE ~" if p.keeper_surplus > 0 else "RELEASE ✗")
            recs.append(
                f"  {i:2}. [{verdict}] {p.name:<22} | Round {p.keeper_round_cost:2} cost | "
                f"Z-Score: {p.z_score:+.2f} | Surplus: {surplus_str}"
            )
            for note in p.notes:
                recs.append(f"       → {note}")
        recs.append("\nKEEPER DEADLINE: Fri Mar 20 3:00am EDT")
        return recs

    def print_report(self, report: KeeperReport) -> None:
        print("\n".join(report.recommendation))


# ---------------------------------------------------------------------------
# Seed data: Juan Soto projection (2026 Steamer/ZiPS consensus)
# ---------------------------------------------------------------------------

def soto_2026_projection() -> PlayerProjection:
    """
    Juan Soto — 2026 consensus projection (Steamer/ZiPS blend).
    Adjust keeper_round_cost to your actual keeper round cost.
    """
    p = PlayerProjection(
        name="Juan Soto",
        yahoo_player_key="mlb.p.10234",  # verify actual key via API
        team="NYY",
        positions=["RF", "LF", "OF"],
        player_type="batter",
        age=27,
        pa=680,
        r=105,
        h=172,
        hr=38,
        rbi=100,
        k_bat=128,      # ~18.8% K rate — low for power hitter
        tb=330,
        avg=0.288,
        obp=0.407,
        slg=0.570,
        ops=0.977,
        sb=10,
        cs=3,
        nsb=7,          # Not a base stealer
        keeper_round_cost=1,  # ADJUST: what round does keeping Soto cost you?
    )
    return p
