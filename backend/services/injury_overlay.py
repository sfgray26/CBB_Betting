"""
Shared MLB injury overlay helpers for fantasy roster, lineup, waiver, and dashboard paths.

BDL injury ingestion already persists rich injury rows into ingested_injuries.
These helpers translate those rows into lightweight UI/decision overlays with:
  - structured status (IL / DTD / 60-Day-IL)
  - short note text
  - return-date + freshness text for player cards
  - a modest waiver penalty for fresh active injuries
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.models import IngestedInjury, PlayerIDMapping

_ET = ZoneInfo("America/New_York")
_DEFAULT_FRESHNESS_MINUTES = 180
_IL_PENALTY = 0.75
_DTD_PENALTY = 0.25
_LONG_IL_PENALTY = 1.25

# Minimum IL stints by type (days from retroactive date until eligible to return).
_IL_DURATIONS: dict[str, int] = {
    "60": 60,
    "15": 15,
    "10": 10,
}


def _il_duration_days(status: str) -> Optional[int]:
    """Return the minimum IL duration in days for the given status string, or None."""
    normalized = status.upper().replace(" ", "").replace("-", "")
    if "IL60" in normalized or "60DAYIL" in normalized:
        return 60
    if "IL15" in normalized or "15DAYIL" in normalized:
        return 15
    if "IL10" in normalized or "10DAYIL" in normalized:
        return 10
    if "IL" in normalized:
        return 10  # Generic IL — use minimum
    return None


@dataclass(frozen=True)
class InjuryOverlay:
    status: str
    note: Optional[str]
    return_timeline: Optional[str]
    ingested_at: datetime
    is_stale: bool


def _coerce_et(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)


def _format_calendar_date(dt: datetime) -> str:
    dt_et = _coerce_et(dt)
    assert dt_et is not None
    return f"{dt_et.strftime('%b')} {dt_et.day}"


def _format_age(age: timedelta) -> str:
    minutes = max(0, int(round(age.total_seconds() / 60)))
    if minutes < 120:
        return f"updated {minutes}m ago"
    hours = max(1, round(minutes / 60))
    if hours < 48:
        return f"updated {hours}h ago"
    days = max(1, round(hours / 24))
    return f"updated {days}d ago"


def build_injury_overlay(
    *,
    status: str,
    note: Optional[str],
    return_date: Optional[datetime],
    ingested_at: datetime,
    injury_date: Optional[datetime] = None,
    now_et: Optional[datetime] = None,
    freshness_minutes: int = _DEFAULT_FRESHNESS_MINUTES,
) -> InjuryOverlay:
    """Build a lightweight overlay with a human-readable return/freshness string.

    ETA is computed from ``injury_date`` (retroactive start) plus the IL-type
    minimum duration when the status contains a recognised IL designator.  When
    the BDL-supplied ``return_date`` implies a duration more than 1.2× the IL
    minimum, the estimate is flagged as uncertain and shown as
    "ETA: TBD (eligible [earliest_date])" to avoid displaying a fabricated date.

    When no IL type can be inferred AND no ``return_date`` is available for an
    IL player, the timeline shows "ETA: Unknown" rather than a fabricated date.
    """
    now_et = _coerce_et(now_et) or datetime.now(_ET)
    ingested_et = _coerce_et(ingested_at) or now_et

    # Compute IL-type-aware ETA -------------------------------------------
    il_days = _il_duration_days(status)
    return_et: Optional[datetime] = None

    if il_days is not None:
        # Use injury_date (retroactive IL start) as the anchor; fall back to
        # ingested_at only when injury_date is missing.
        ref_date = _coerce_et(injury_date) or ingested_et
        computed_eta = ref_date + timedelta(days=il_days)  # earliest eligible date

        tbd_eligibility = False  # True → show "TBD (eligible ...)" not a hard date
        if return_date is None:
            # BDL provided no return_date — compute from IL type minimum.
            # Show as a specific date: the eligibility date is reliable.
            return_et = computed_eta
        else:
            bdl_et = _coerce_et(return_date)
            assert bdl_et is not None
            implied_days = (bdl_et - ref_date).days
            if implied_days > il_days * 1.2:
                # BDL return_date extends more than 1.2× the IL minimum beyond the
                # injury start — uncertain estimate (e.g. BDL applied the wrong
                # IL-type formula, or the doctor has given no confirmed timetable).
                # Show the eligibility date with a TBD qualifier instead of a
                # potentially fabricated specific date.
                return_et = computed_eta
                tbd_eligibility = True
            else:
                return_et = bdl_et
    else:
        # Non-IL or unrecognised status: trust BDL's return_date if present.
        return_et = _coerce_et(return_date)
        tbd_eligibility = False

    age = now_et - ingested_et
    is_stale = age > timedelta(minutes=freshness_minutes)
    freshness_label = _format_age(age)
    if is_stale:
        freshness_label = f"stale · {freshness_label}"

    timeline_parts: list[str] = []
    status_upper = status.upper().replace(" ", "").replace("-", "")
    if return_et is not None:
        if tbd_eligibility:
            timeline_parts.append(f"ETA: TBD (eligible {_format_calendar_date(return_et)})")
        else:
            timeline_parts.append(f"ETA {_format_calendar_date(return_et)}")
    elif "IL" in status_upper:
        # IL player but no date available — avoid fabricating a date.
        timeline_parts.append("ETA: Unknown")
    timeline_parts.append(freshness_label)

    return InjuryOverlay(
        status=status,
        note=note,
        return_timeline=" · ".join(part for part in timeline_parts if part) or None,
        ingested_at=ingested_et,
        is_stale=is_stale,
    )


def load_injury_overlays(
    db: Session,
    bdl_player_ids: Iterable[int],
    *,
    now_et: Optional[datetime] = None,
    freshness_minutes: int = _DEFAULT_FRESHNESS_MINUTES,
) -> dict[int, InjuryOverlay]:
    """Return the latest active injury overlay per BDL player ID."""
    player_ids = sorted({int(pid) for pid in bdl_player_ids if pid is not None})
    if not player_ids:
        return {}

    rows = (
        db.query(IngestedInjury)
        .filter(IngestedInjury.bdl_player_id.in_(player_ids))
        .order_by(IngestedInjury.bdl_player_id.asc(), IngestedInjury.ingested_at.desc())
        .all()
    )

    overlays: dict[int, InjuryOverlay] = {}
    for row in rows:
        if row.bdl_player_id in overlays:
            continue
        overlays[row.bdl_player_id] = build_injury_overlay(
            status=row.injury_status,
            note=row.short_comment or row.long_comment,
            return_date=row.return_date,
            injury_date=row.injury_date,
            ingested_at=row.ingested_at,
            now_et=now_et,
            freshness_minutes=freshness_minutes,
        )
    return overlays


def load_injury_overlays_for_yahoo_players(
    db: Session,
    raw_players: list[dict],
    *,
    now_et: Optional[datetime] = None,
    freshness_minutes: int = _DEFAULT_FRESHNESS_MINUTES,
) -> dict[str, InjuryOverlay]:
    """Resolve Yahoo player keys to BDL IDs, then fetch fresh injury overlays."""
    if not raw_players:
        return {}

    player_key_to_bdl: dict[str, int] = {}
    query_keys: set[str] = set()
    yahoo_ids: set[str] = set()

    for player in raw_players:
        player_key = str(player.get("player_key") or "").strip()
        if not player_key:
            continue

        direct_bdl_id = player.get("bdl_player_id")
        if direct_bdl_id is not None:
            player_key_to_bdl[player_key] = int(direct_bdl_id)
            continue

        query_keys.add(player_key)
        if ".p." in player_key:
            yahoo_ids.add(player_key.split(".p.", 1)[-1])
        elif player.get("player_id"):
            yahoo_ids.add(str(player.get("player_id")))

    if query_keys or yahoo_ids:
        predicates = []
        if query_keys:
            predicates.append(PlayerIDMapping.yahoo_key.in_(list(query_keys)))
        if yahoo_ids:
            predicates.append(PlayerIDMapping.yahoo_id.in_(list(yahoo_ids)))

        rows = (
            db.query(
                PlayerIDMapping.yahoo_key,
                PlayerIDMapping.yahoo_id,
                PlayerIDMapping.bdl_id,
            )
            .filter(
                PlayerIDMapping.bdl_id.isnot(None),
                or_(*predicates),
            )
            .all()
        )

        by_key = {row.yahoo_key: row.bdl_id for row in rows if row.yahoo_key and row.bdl_id is not None}
        by_yahoo_id = {row.yahoo_id: row.bdl_id for row in rows if row.yahoo_id and row.bdl_id is not None}

        for player in raw_players:
            player_key = str(player.get("player_key") or "").strip()
            if not player_key or player_key in player_key_to_bdl:
                continue

            bdl_id = by_key.get(player_key)
            if bdl_id is None:
                yahoo_id = player_key.split(".p.", 1)[-1] if ".p." in player_key else str(player.get("player_id") or "")
                bdl_id = by_yahoo_id.get(yahoo_id)
            if bdl_id is not None:
                player_key_to_bdl[player_key] = int(bdl_id)

    overlay_by_bdl = load_injury_overlays(
        db,
        player_key_to_bdl.values(),
        now_et=now_et,
        freshness_minutes=freshness_minutes,
    )

    return {
        player_key: overlay_by_bdl[bdl_id]
        for player_key, bdl_id in player_key_to_bdl.items()
        if bdl_id in overlay_by_bdl
    }


def apply_injury_penalty(score: float, overlay: Optional[InjuryOverlay]) -> tuple[float, Optional[str]]:
    """Apply a modest waiver penalty for fresh active injuries only."""
    if overlay is None or overlay.is_stale:
        return round(float(score), 3), None

    status = str(overlay.status or "").upper()
    penalty = 0.0
    if "60-DAY" in status or "IL60" in status:
        penalty = _LONG_IL_PENALTY
    elif "IL" in status:
        penalty = _IL_PENALTY
    elif "DTD" in status or "DAY-TO-DAY" in status:
        penalty = _DTD_PENALTY

    if penalty <= 0.0:
        return round(float(score), 3), None

    adjusted = max(0.0, float(score) - penalty)
    note = f"Injury penalty -{penalty:.2f} ({overlay.status})"
    return round(adjusted, 3), note
