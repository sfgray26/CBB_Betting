"""
Constraint and data gap helpers — pure functions for UI contract population.

These 7 functions close L2 data gaps identified in the UI specification audit.
Each produces a value needed by ConstraintBudget, CanonicalPlayerRow, or
MatchupScoreboardRow contracts.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backend.contracts import IPPaceFlag


@dataclass(frozen=True)
class OpposingSPInfo:
    """Opposing starting pitcher context for a hitter."""
    sp_name: Optional[str]
    sp_handedness: Optional[str]  # "L" or "R"
    opponent_team: str
    home_away: str  # "home" or "away"


def count_weekly_acquisitions(
    transactions: list[dict],
    my_team_key: str,
    week_start: datetime,
    week_end: datetime,
) -> int:
    """Count add transactions by my team within the matchup week window.

    Args:
        transactions: Raw Yahoo transactions list from get_transactions(t_type="add")
        my_team_key: My Yahoo team key (e.g. "mlb.l.12345.t.3")
        week_start: Matchup week start date (inclusive)
        week_end: Matchup week end date (inclusive)

    Returns:
        Number of "add" type transactions by my team in the window.
    """
    week_start_ts = week_start.timestamp()
    week_end_ts = week_end.timestamp()

    count = 0
    for txn in transactions:
        # Filter to add transactions only
        if txn.get("type") not in ("add", "add/drop"):
            continue

        # Coerce timestamp — Yahoo returns timestamps as strings, not ints
        raw_ts = txn.get("timestamp")
        if raw_ts is None:
            continue
        try:
            ts_float = float(raw_ts)
        except (TypeError, ValueError):
            continue

        if not (week_start_ts <= ts_float <= week_end_ts):
            continue

        # Check destination team — try multiple structural paths Yahoo uses
        dest_team = (
            txn.get("destination_team_key")
            or txn.get("destination_team", {}).get("team_key")
        )

        # Fallback: walk transaction_data list for player-level destination
        if not dest_team:
            txn_data = txn.get("transaction_data") or txn.get("players") or []
            if isinstance(txn_data, list):
                for item in txn_data:
                    if not isinstance(item, dict):
                        continue
                    for _v in item.values():
                        if isinstance(_v, dict):
                            dest_team = (
                                _v.get("destination_team_key")
                                or _v.get("destination_team", {}).get("team_key")
                            )
                            if dest_team:
                                break
                    if dest_team:
                        break

        if dest_team == my_team_key:
            count += 1

    return count


def extract_ip_from_scoreboard(
    matchup_stats: dict[str, float],
) -> float:
    """Extract innings pitched value from a parsed matchup stats dict.

    The matchup stats dict is keyed by canonical codes (after disambiguation).
    IP is a display-only stat that Yahoo includes in scoreboard responses.

    Returns:
        Innings pitched as a float (e.g. 15.2 means 15 innings + 2 outs).
    """
    return matchup_stats.get("IP", 0.0)


def classify_ip_pace(
    ip_accumulated: float,
    ip_minimum: float,
    days_elapsed: int,
    days_total: int,
) -> IPPaceFlag:
    """Classify weekly IP pace relative to league minimum.

    Args:
        ip_accumulated: IP so far this matchup week (e.g. 12.1)
        ip_minimum: League IP minimum (18.0)
        days_elapsed: Days completed in matchup week (1-7)
        days_total: Total days in matchup week (7)

    Returns:
        IPPaceFlag.BEHIND if projected to miss minimum
        IPPaceFlag.ON_TRACK if projected to hit ±10% of minimum
        IPPaceFlag.AHEAD if projected to exceed minimum comfortably
    """
    if days_elapsed == 0:
        return IPPaceFlag.BEHIND

    daily_rate = ip_accumulated / days_elapsed
    projected_total = daily_rate * days_total

    lower_bound = ip_minimum * 0.9
    upper_bound = ip_minimum * 1.1

    if projected_total < lower_bound:
        return IPPaceFlag.BEHIND
    elif projected_total > upper_bound:
        return IPPaceFlag.AHEAD
    else:
        return IPPaceFlag.ON_TRACK


def count_games_remaining(
    team_abbr: str,
    schedule: dict[str, list[datetime]],
    today: datetime,
    week_end: datetime,
) -> int:
    """Count remaining games for a team between today (exclusive) and week_end (inclusive).

    Args:
        team_abbr: MLB team abbreviation (e.g. "NYY", "LAD")
        schedule: Dict mapping team abbreviation to list of game datetimes
        today: Current date (games today are INCLUDED if not yet started)
        week_end: Matchup week end date (inclusive)

    Returns:
        Number of games remaining in the matchup week.
    """
    team_games = schedule.get(team_abbr, [])
    today_date = today.date()
    week_end_date = week_end.date()

    count = 0
    for game_dt in team_games:
        game_date = game_dt.date()
        # Count games from today through week_end (inclusive)
        if today_date <= game_date <= week_end_date:
            count += 1

    return count


def extract_team_record(
    standings: list[dict],
    my_team_key: str,
) -> tuple[int, int, int]:
    """Extract W-L-T record from Yahoo standings response.

    Args:
        standings: Raw Yahoo standings list from get_standings()
        my_team_key: My Yahoo team key

    Returns:
        (wins, losses, ties) tuple
    """
    for team_entry in standings:
        # Navigate Yahoo standings structure
        team = team_entry.get("team") if isinstance(team_entry, dict) else None
        if not team:
            continue

        if team.get("team_key") == my_team_key:
            outcome_totals = team.get("outcome_totals", {})
            if isinstance(outcome_totals, dict):
                wins = int(outcome_totals.get("wins", 0))
                losses = int(outcome_totals.get("losses", 0))
                ties = int(outcome_totals.get("ties", 0))
                return (wins, losses, ties)

    return (0, 0, 0)


def lookup_opposing_sp(
    player_team: str,
    schedule_entry: dict,
    probable_pitchers: list[dict],
) -> Optional[OpposingSPInfo]:
    """Look up the opposing starting pitcher for a hitter's game today.

    Args:
        player_team: Hitter's team abbreviation
        schedule_entry: Today's game entry with home_team, away_team
        probable_pitchers: List of probable pitcher records from DB

    Returns:
        OpposingSPInfo or None if no game today.
    """
    if not schedule_entry:
        return None

    home_team = schedule_entry.get("home_team", "")
    away_team = schedule_entry.get("away_team", "")

    # Determine if player's team is home or away
    if player_team == home_team:
        opposing_team = away_team
        home_away = "home"
    elif player_team == away_team:
        opposing_team = home_team
        home_away = "away"
    else:
        return None

    # Find the opposing team's probable pitcher
    for pp in probable_pitchers:
        if pp.get("team") == opposing_team:
            return OpposingSPInfo(
                sp_name=pp.get("name"),
                sp_handedness=pp.get("handedness"),  # "L" or "R"
                opponent_team=opposing_team,
                home_away=home_away,
            )

    # Game found but no probable pitcher data
    return OpposingSPInfo(
        sp_name=None,
        sp_handedness=None,
        opponent_team=opposing_team,
        home_away=home_away,
    )


def resolve_playing_status(
    roster_entry: dict,
    team_schedule_today: bool,
) -> str:
    """Resolve a player's playing-today status.

    Args:
        roster_entry: Yahoo roster player dict with status fields
        team_schedule_today: Whether the player's team has a game today

    Returns:
        One of: "playing", "not_playing", "probable", "IL", "minors"
    """
    status = roster_entry.get("status", "")

    # Check IL status variants
    if status and status.upper().startswith("IL"):
        return "IL"

    # Check minors status
    if status in ("NA", "minors", "MINORS"):
        return "minors"

    # No game today
    if not team_schedule_today:
        return "not_playing"

    # TODO: Probable pitcher check would go here if probable_pitchers is passed
    # For Phase 1, we assume non-IL players with games are "playing"
    return "playing"
