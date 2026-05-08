"""
TeamContext -- ephemeral per-API-call struct for rate-category denominator math.

Built fresh per waiver/lineup call. Never persisted.
Quarantined players (IdentityQuarantine PENDING_REVIEW) contribute 0 PA/IP
and are excluded from all denominator calculations.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TeamContext:
    """
    Snapshot of a fantasy roster's playing-time context for a single API call.

    Attributes:
        roster_player_ids: Canonical MLBAM IDs of resolved (non-quarantined) roster players.
        projected_pa_by_player: {mlbam_id -> remaining-season projected PA} for batters.
        projected_ip_by_player: {mlbam_id -> remaining-season projected IP} for pitchers.
        rate_pa_denominator: Sum of projected_pa_by_player.values(). Used as AVG/OBP/OPS denominator.
        rate_ip_denominator: Sum of projected_ip_by_player.values(). Used as ERA/WHIP/K9 denominator.
        quarantined_player_ids: PlayerIdentity.id values in PENDING_REVIEW -- excluded from all math.
    """
    roster_player_ids: list[int] = field(default_factory=list)
    projected_pa_by_player: dict[int, float] = field(default_factory=dict)
    projected_ip_by_player: dict[int, float] = field(default_factory=dict)
    rate_pa_denominator: float = 0.0
    rate_ip_denominator: float = 0.0
    quarantined_player_ids: set[int] = field(default_factory=set)
    team_rate_numerators: dict[str, float] = field(default_factory=dict)
    team_rate_denominators: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        roster_player_ids: list[int],
        projected_pa_by_player: dict[int, float],
        projected_ip_by_player: dict[int, float],
        quarantined_player_ids: set[int] | None = None,
        team_rate_numerators: dict[str, float] | None = None,
        team_rate_denominators: dict[str, float] | None = None,
    ) -> "TeamContext":
        """
        Construct a TeamContext, computing denominators automatically.

        Args:
            roster_player_ids: Resolved roster MLBAM IDs (quarantined excluded by caller).
            projected_pa_by_player: {mlbam_id -> projected PA} for each batter.
            projected_ip_by_player: {mlbam_id -> projected IP} for each pitcher.
            quarantined_player_ids: Optional set of quarantined PlayerIdentity.id values.
            team_rate_numerators: Optional {category -> total numerator} from CategoryImpact.
            team_rate_denominators: Optional {category -> total denominator} from CategoryImpact.

        Returns:
            TeamContext with denominators pre-computed.
        """
        return cls(
            roster_player_ids=list(roster_player_ids),
            projected_pa_by_player=dict(projected_pa_by_player),
            projected_ip_by_player=dict(projected_ip_by_player),
            rate_pa_denominator=sum(projected_pa_by_player.values()),
            rate_ip_denominator=sum(projected_ip_by_player.values()),
            quarantined_player_ids=set(quarantined_player_ids or []),
            team_rate_numerators=team_rate_numerators or {},
            team_rate_denominators=team_rate_denominators or {},
        )

    def is_quarantined(self, player_id: int) -> bool:
        """Return True if player_id is in the quarantine set."""
        return player_id in self.quarantined_player_ids

    def batter_pa_share(self, player_id: int) -> float:
        """Return this player's fraction of team PA (0.0 if absent or denominator is zero)."""
        if self.rate_pa_denominator <= 0:
            return 0.0
        return self.projected_pa_by_player.get(player_id, 0.0) / self.rate_pa_denominator

    def pitcher_ip_share(self, player_id: int) -> float:
        """Return this player's fraction of team IP (0.0 if absent or denominator is zero)."""
        if self.rate_ip_denominator <= 0:
            return 0.0
        return self.projected_ip_by_player.get(player_id, 0.0) / self.rate_ip_denominator
