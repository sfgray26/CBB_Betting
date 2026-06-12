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
    expired_eta: bool = False  # NEW: Track if ETA has passed


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
    expired_eta: bool = False,  # NEW: Pass through expired flag
) -> InjuryOverlay:
    """Build a lightweight overlay with a human-readable return/freshness string.

    ETA is computed from ``injury_date`` (retroactive start) plus the IL-type
    minimum duration when the status contains a recognised IL designator.  When
    the BDL-supplied ``return_date`` implies a duration more than 1.2× the IL
    minimum, the estimate is flagged as uncertain and shown as
    \"ETA: TBD (eligible [earliest_date])\" to avoid displaying a fabricated date.

    When no IL type can be inferred AND no ``return_date`` is available for an
    IL player, the timeline shows \"ETA: Unknown\" rather than a fabricated date.
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
    
    # Add expired ETA warning
    if expired_eta and return_et and return_et.date() < now_et.date():
        timeline_parts.append("⚠️ ETA EXPIRED — CHECK STATUS")
    
    if return_et is not None and not expired_eta:
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
        expired_eta=expired_eta,
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

    # Check for expired ETAs in real-time and mark them in DB
    now_et = _coerce_et(now_et) or datetime.now(_ET)
    today_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Auto-mark expired ETAs (real-time check, not just nightly cron)
    try:
        expired_injuries = (
            db.query(IngestedInjury)
            .filter(
                IngestedInjury.bdl_player_id.in_(player_ids),
                IngestedInjury.return_date.isnot(None),
                IngestedInjury.return_date < today_start,
                IngestedInjury.expired_eta.is_(False),
            )
            .all()
        )
        if expired_injuries:
            logger.warning(
                "injury_overlay: marking %d injuries as expired_eta (real-time check)",
                len(expired_injuries),
            )
            for injury in expired_injuries:
                injury.expired_eta = True
            db.commit()
    except Exception as e:
        logger.error("injury_overlay: failed to mark expired ETAs: %s", e)
        db.rollback()

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
        
        # Check if ETA has expired for display
        return_et = None
        if row.return_date:
            return_et = _coerce_et(row.return_date)
        
        expired_eta_display = False
        if return_et and return_et.date() < now_et.date() and row.expired_eta:
            expired_eta_display = True
        
        overlays[row.bdl_player_id] = build_injury_overlay(
            status=row.injury_status,
            note=row.short_comment or row.long_comment,
            return_date=row.return_date,
            injury_date=row.injury_date,
            ingested_at=row.ingested_at,
            now_et=now_et,
            freshness_minutes=freshness_minutes,
            expired_eta=expired_eta_display,
        )
    
    return overlays


def load_injury_overlays_for_yahoo_players(
    db: Session,
    yahoo_players: list[dict],
    *,
    now_et: Optional[datetime] = None,
    freshness_minutes: int = _DEFAULT_FRESHNESS_MINUTES,
) -> dict[str, InjuryOverlay]:
    """Load injury overlays for Yahoo player objects, keyed by yahoo_player_key.

    This helper resolves the crosswalk from ``yahoo_player_key`` → ``bdl_player_id``
    via PlayerIDMapping, then calls ``load_injury_overlays`` with the resolved BDL
    IDs.
    """
    if not yahoo_players:
        return {}

    # Resolve yahoo_player_key → bdl_player_id
    yahoo_keys = [p.get("player_key") for p in yahoo_players if p.get("player_key")]
    if not yahoo_keys:
        return {}

    mappings = (
        db.query(PlayerIDMapping)
        .filter(PlayerIDMapping.yahoo_player_key.in_(yahoo_keys))
        .all()
    )

    key_to_bdl_id: dict[str, int] = {
        m.yahoo_player_key: m.bdl_player_id
        for m in mappings
        if m.bdl_player_id is not None
    }

    bdl_ids = set(key_to_bdl_id.values())
    overlays_by_bdl_id = load_injury_overlays(
        db,
        bdl_ids,
        now_et=now_et,
        freshness_minutes=freshness_minutes,
    )

    # Map back to yahoo_player_key
    return {
        yahoo_key: overlays_by_bdl_id[bdl_id]
        for yahoo_key, bdl_id in key_to_bdl_id.items()
        if bdl_id in overlays_by_bdl_id
    }


def apply_injury_penalty(
    need_score: float,
    injury: Optional[InjuryOverlay],
) -> tuple[float, Optional[str]]:
    """Apply a modest waiver penalty for fresh active injuries.

    Args:
        need_score: The raw need score computed from category deficits.
        injury: Injury overlay if the player is injured, else None.

    Returns:
        A tuple of (adjusted_need_score, penalty_note).
    """
    if injury is None:
        return need_score, None

    penalty_note = None
    adjusted_score = need_score

    # Only apply penalty for fresh injuries (non-stale)
    if not injury.is_stale:
        status_upper = injury.status.upper().replace(" ", "").replace("-", "")

        if "IL60" in status_upper or "60DAYIL" in status_upper:
            penalty_note = "60-Day IL"
            adjusted_score *= _LONG_IL_PENALTY
        elif "IL15" in status_upper or "15DAYIL" in status_upper:
            penalty_note = "15-Day IL"
            adjusted_score *= _IL_PENALTY
        elif "IL10" in status_upper or "10DAYIL" in status_upper:
            penalty_note = "10-Day IL"
            adjusted_score *= _IL_PENALTY
        elif "IL" in status_upper:
            penalty_note = "IL"
            adjusted_score *= _IL_PENALTY
        elif "DTD" in status_upper:
            penalty_note = "DTD"
            adjusted_score *= _DTD_PENALTY

    return adjusted_score, penalty_note