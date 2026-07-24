"""Per-pitcher rotation projection for probable-pitcher inference.

Replaces the exact-modulo-5 fallback (`probable_pitcher_fallback.infer_probable_pitcher_for_team`)
which matched only ~31% of real rotation turns and produced zero rows post-All-Star.

Design (spec: reports/2026-07-22-streaming-rotation-projection-spec.md):
  * Rotation set = the <=6 most recent distinct starters per team (IP >= 4),
    excluding anyone idle > 14 days (IL / demotion / post-break uncertainty).
  * Personal cadence = median gap of the pitcher's last <=5 starts, clamped
    [4, 8], default 6 when < 2 prior gaps. 5-7 day gaps cover ~83% of real turns.
  * Walk each team's scheduled game dates chronologically, assigning the rotation
    pitcher whose projected next start is nearest the game date (within +/-2 days),
    then advancing that pitcher's next start. Advancing after each assignment is
    what surfaces a pitcher's *second* start inside the window (2-start detection).
  * Dates with an official probable are hard-anchored: they are not projected, and
    a matching rotation pitcher is re-anchored to that date so errors self-correct
    across the 3x/day resyncs.

`project_team_window` is a pure function (no DB) and is the unit-tested core.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from backend.models import MLBPlayerStats, PlayerIDMapping
from backend.services.probable_pitcher_fallback import (
    parse_innings_pitched,
    starter_team_name as _starter_team_name,
)

DEFAULT_CADENCE = 6
CADENCE_MIN = 4
CADENCE_MAX = 8
MAX_IDLE_DAYS = 14
MAX_ROTATION = 6
DEFAULT_LOOKBACK_DAYS = 45
DEFAULT_TOLERANCE = 2


@dataclass
class PitcherState:
    """A rotation pitcher with the history needed to project future starts.

    `start_dates` is ascending. `next_start` is mutated during a window walk.
    """

    team: str
    bdl_player_id: Optional[int]
    mlbam_id: Optional[int]
    pitcher_name: str
    start_dates: list[date]
    typical_ip: float = 5.0
    cadence: int = DEFAULT_CADENCE
    next_start: Optional[date] = field(default=None)

    @property
    def last_start(self) -> date:
        return self.start_dates[-1]

    def key(self) -> Optional[object]:
        """Identity for official-probable reconciliation (mlbam preferred)."""
        return self.mlbam_id if self.mlbam_id is not None else self.bdl_player_id


@dataclass
class ProjectedAssignment:
    game_date: date
    pitcher: PitcherState


def median_cadence(start_dates: list[date]) -> int:
    """Median gap (days) of the last <=5 starts, clamped [4, 8].

    Returns DEFAULT_CADENCE (6) when fewer than 2 gaps are available.
    """
    recent = sorted(start_dates)[-5:]
    gaps = [(recent[i + 1] - recent[i]).days for i in range(len(recent) - 1)]
    gaps = [g for g in gaps if g > 0]
    if len(gaps) < 2:
        return DEFAULT_CADENCE
    med = statistics.median(gaps)
    return int(max(CADENCE_MIN, min(CADENCE_MAX, round(med))))


def project_team_window(
    pitchers: list[PitcherState],
    game_dates: list[date],
    official_dates: Optional[dict[date, object]] = None,
    tolerance: int = DEFAULT_TOLERANCE,
) -> list[ProjectedAssignment]:
    """Project starters onto a single team's scheduled game dates.

    Pure function — the unit-tested core.

    Args:
        pitchers: the team's rotation set. Each pitcher's `cadence` must be set;
            `next_start` is (re)initialized here from `last_start + cadence`.
        game_dates: the team's scheduled game dates in the window (any order).
        official_dates: {date: pitcher_key} for dates already covered by an
            official probable. These dates are NOT projected; a matching rotation
            pitcher is re-anchored to that date.
        tolerance: max |next_start - game_date| in days to accept an assignment.

    Returns:
        Projected assignments (excludes official dates), in chronological order.
    """
    official_dates = official_dates or {}
    for p in pitchers:
        p.next_start = p.last_start + timedelta(days=p.cadence)

    assignments: list[ProjectedAssignment] = []
    for d in sorted(game_dates):
        if d in official_dates:
            official_key = official_dates[d]
            if official_key is not None:
                for p in pitchers:
                    if p.key() == official_key:
                        p.next_start = d + timedelta(days=p.cadence)
                        break
            continue

        if not pitchers:
            continue
        best = min(
            pitchers,
            key=lambda p: (
                abs((p.next_start - d).days),
                p.next_start.toordinal(),
                -p.typical_ip,
            ),
        )
        if abs((best.next_start - d).days) <= tolerance:
            assignments.append(ProjectedAssignment(game_date=d, pitcher=best))
            best.next_start = d + timedelta(days=best.cadence)
    return assignments


# ---------------------------------------------------------------------------
# DB builders
# ---------------------------------------------------------------------------

def build_rotation_sets(
    db: Session,
    today: date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_starter_ip: float = 4.0,
    max_idle_days: int = MAX_IDLE_DAYS,
    max_rotation: int = MAX_ROTATION,
) -> dict[str, list[PitcherState]]:
    """Build each team's rotation set with full recent start history.

    Unlike `build_recent_starter_candidates` (latest start only), this keeps every
    start date per pitcher so cadence can be estimated. Filters to the <=N most
    recent distinct starters per team, excluding anyone idle > max_idle_days.
    """
    window_start = today - timedelta(days=lookback_days)

    id_rows = (
        db.query(PlayerIDMapping.bdl_id, PlayerIDMapping.mlbam_id, PlayerIDMapping.full_name)
        .filter(PlayerIDMapping.bdl_id.isnot(None))
        .all()
    )
    id_map = {
        r.bdl_id: {"mlbam_id": r.mlbam_id, "full_name": r.full_name}
        for r in id_rows
        if r.bdl_id is not None
    }

    stat_rows = (
        db.query(MLBPlayerStats)
        .filter(
            MLBPlayerStats.game_date >= window_start,
            MLBPlayerStats.game_date < today,
            MLBPlayerStats.innings_pitched.isnot(None),
        )
        .all()
    )

    # (team, bdl_id) -> {"starts": [dates], "ips": [ip], "name": str, "mlbam": id}
    agg: dict[tuple[str, Optional[int]], dict] = {}
    for row in stat_rows:
        ip = parse_innings_pitched(row.innings_pitched)
        if ip is None or ip < min_starter_ip:
            continue
        team, name = _starter_team_name(row.raw_payload)
        if not team:
            continue
        bdl_id = row.bdl_player_id
        mapping = id_map.get(bdl_id, {})
        if not name:
            name = mapping.get("full_name") or ""
        if not name:
            continue
        key = (team, bdl_id)
        entry = agg.setdefault(
            key,
            {"starts": [], "ips": [], "name": name, "mlbam": mapping.get("mlbam_id"), "team": team, "bdl": bdl_id},
        )
        entry["starts"].append(row.game_date)
        entry["ips"].append(ip)

    by_team: dict[str, list[PitcherState]] = {}
    for entry in agg.values():
        starts = sorted(entry["starts"])
        if not starts:
            continue
        if (today - starts[-1]).days > max_idle_days:
            continue  # idle too long — defer to official probables
        state = PitcherState(
            team=entry["team"],
            bdl_player_id=entry["bdl"],
            mlbam_id=entry["mlbam"],
            pitcher_name=entry["name"],
            start_dates=starts,
            typical_ip=round(sum(entry["ips"]) / len(entry["ips"]), 2),
            cadence=median_cadence(starts),
        )
        by_team.setdefault(entry["team"], []).append(state)

    # Keep only the <=N most recent distinct starters per team.
    for team, pitchers in by_team.items():
        pitchers.sort(key=lambda p: (p.last_start.toordinal(), p.typical_ip), reverse=True)
        by_team[team] = pitchers[:max_rotation]

    return by_team


def project_probable_starters(
    db: Session,
    today: date,
    team_game_dates: dict[str, list[date]],
    official_by_team_date: Optional[dict[tuple[str, date], object]] = None,
    tolerance: int = DEFAULT_TOLERANCE,
) -> dict[tuple[str, date], PitcherState]:
    """Project starters for every (team, date) that lacks an official probable.

    Args:
        team_game_dates: {team: [scheduled game dates in the window]}.
        official_by_team_date: {(team, date): official pitcher key} — dates that
            already have an official probable (not projected; used for re-anchor).

    Returns:
        {(team, date): PitcherState} for projected (non-official) slots only.
    """
    official_by_team_date = official_by_team_date or {}
    rotation = build_rotation_sets(db, today)

    result: dict[tuple[str, date], PitcherState] = {}
    for team, dates in team_game_dates.items():
        pitchers = rotation.get(team, [])
        if not pitchers:
            continue
        official_dates = {
            d: key
            for (t, d), key in official_by_team_date.items()
            if t == team
        }
        assignments = project_team_window(pitchers, dates, official_dates, tolerance)
        for a in assignments:
            result[(team, a.game_date)] = a.pitcher
    return result


# ---------------------------------------------------------------------------
# Backtest harness (validation gate — run before enabling projected rows)
# ---------------------------------------------------------------------------

def _actual_starts_by_team_date(
    db: Session,
    start: date,
    end: date,
    min_starter_ip: float = 4.0,
) -> dict[tuple[str, date], set[str]]:
    """Actual starters (IP >= 4) per (team, date) in [start, end], names lowercased."""
    rows = (
        db.query(MLBPlayerStats)
        .filter(
            MLBPlayerStats.game_date >= start,
            MLBPlayerStats.game_date <= end,
            MLBPlayerStats.innings_pitched.isnot(None),
        )
        .all()
    )
    out: dict[tuple[str, date], set[str]] = {}
    for row in rows:
        ip = parse_innings_pitched(row.innings_pitched)
        if ip is None or ip < min_starter_ip:
            continue
        team, name = _starter_team_name(row.raw_payload)
        if not team or not name:
            continue
        out.setdefault((team, row.game_date), set()).add(name.strip().lower())
    return out


def backtest_rotation_projection(
    db: Session,
    days: int = 30,
    horizon: int = 7,
    tolerance: int = DEFAULT_TOLERANCE,
    today: Optional[date] = None,
) -> dict:
    """Replay the projection over the last `days` and report accuracy by offset.

    For each replay date D, build the rotation using only data < D, project each
    team's actual game dates in [D, D+horizon], and compare projected starters vs
    the real starters (IP >= 4) that pitched on those dates.

    Targets (spec §7): >= 70% exact-name hit on D+2..D+5, >= 85% within +/-1 day.
    """
    today = today or max(
        (r[0] for r in db.query(MLBPlayerStats.game_date).order_by(MLBPlayerStats.game_date.desc()).limit(1)),
        default=date.today(),
    )
    # per horizon-offset tallies
    by_offset: dict[int, dict[str, int]] = {
        off: {"total": 0, "exact": 0, "within1": 0} for off in range(horizon + 1)
    }

    for back in range(1, days + 1):
        replay = today - timedelta(days=back + horizon)  # ensure actuals exist through horizon
        win_start = replay
        win_end = replay + timedelta(days=horizon)
        actuals = _actual_starts_by_team_date(db, win_start, win_end)
        if not actuals:
            continue

        team_game_dates: dict[str, list[date]] = {}
        for (team, d) in actuals.keys():
            team_game_dates.setdefault(team, []).append(d)

        rotation = build_rotation_sets(db, replay)
        for team, dates in team_game_dates.items():
            pitchers = rotation.get(team, [])
            if not pitchers:
                continue
            assignments = project_team_window(pitchers, sorted(set(dates)), {}, tolerance)
            proj_by_date = {a.game_date: a.pitcher.pitcher_name.strip().lower() for a in assignments}
            for d in sorted(set(dates)):
                offset = (d - replay).days
                if offset < 0 or offset > horizon:
                    continue
                actual_names = actuals.get((team, d), set())
                if not actual_names:
                    continue
                by_offset[offset]["total"] += 1
                projected = proj_by_date.get(d)
                if projected and projected in actual_names:
                    by_offset[offset]["exact"] += 1
                    by_offset[offset]["within1"] += 1
                else:
                    # within +/-1 day: projected this pitcher on an adjacent date
                    hit1 = False
                    for adj in (d - timedelta(days=1), d + timedelta(days=1)):
                        if proj_by_date.get(adj) in actual_names and proj_by_date.get(adj):
                            hit1 = True
                            break
                    if hit1:
                        by_offset[offset]["within1"] += 1

    def _rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    summary = {
        str(off): {
            "total": t["total"],
            "exact_hit_rate": _rate(t["exact"], t["total"]),
            "within1_hit_rate": _rate(t["within1"], t["total"]),
        }
        for off, t in by_offset.items()
    }
    mid = [by_offset[o] for o in range(2, 6)]
    mid_total = sum(t["total"] for t in mid)
    mid_exact = sum(t["exact"] for t in mid)
    mid_within1 = sum(t["within1"] for t in mid)
    return {
        "replay_days": days,
        "horizon": horizon,
        "anchor_date": today.isoformat(),
        "by_offset": summary,
        "d2_d5_exact_hit_rate": _rate(mid_exact, mid_total),
        "d2_d5_within1_hit_rate": _rate(mid_within1, mid_total),
        "d2_d5_total": mid_total,
        "target_exact": 0.70,
        "target_within1": 0.85,
        "passes_gate": _rate(mid_exact, mid_total) >= 0.70 and _rate(mid_within1, mid_total) >= 0.85,
    }
