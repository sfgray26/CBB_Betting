"""League transaction feed — within-league drop intelligence for waiver scoring.

Pulls recent drop transactions from Yahoo Fantasy API and exposes them
so the waiver endpoint can annotate candidates with "dropped_by" context.

Usage:
    from backend.services.league_transaction_feed import get_recent_league_drops, build_drop_lookup
    drops = get_recent_league_drops(client, days=7)
    lookup = build_drop_lookup(drops)
    drop_info = lookup["by_key"].get("469.p.10001")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LeagueDrop:
    """A single drop transaction from the league transaction log."""

    player_key: str         # Yahoo player key, e.g. "469.p.10001"
    player_name: str        # Full display name
    team: str               # MLB team abbreviation
    positions: List[str]    # Eligible positions, e.g. ["OF", "Util"]
    dropped_by_team: str    # Manager's Yahoo team key
    dropped_by_name: str    # Manager's team display name
    dropped_at: datetime    # UTC datetime of the drop
    days_ago: float         # Fractional days since drop (pre-computed)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _flatten_meta(raw) -> dict:
    """Flatten Yahoo's list-of-single-key-dicts player metadata to a plain dict."""
    if isinstance(raw, list):
        out: dict = {}
        for item in raw:
            if isinstance(item, dict):
                out.update(item)
        return out
    return raw if isinstance(raw, dict) else {}


def _parse_player_slot(player_raw) -> Optional[dict]:
    """Parse one player entry inside a transaction's players section.

    Yahoo structures each player as a 2-element list:
        [ metadata_list, transaction_data_dict ]

    Returns a flattened dict with all metadata fields plus a ``_txn_data`` key.
    Returns ``None`` if the input cannot be parsed.
    """
    if not isinstance(player_raw, list) or len(player_raw) < 2:
        return None

    meta = _flatten_meta(player_raw[0])
    txn_raw = player_raw[1]

    txn_data: dict = {}
    if isinstance(txn_raw, dict):
        # Yahoo sometimes wraps inside {"transaction_data": {...}}
        txn_data = txn_raw.get("transaction_data", txn_raw)

    return {**meta, "_txn_data": txn_data}


def _player_key(parsed: dict) -> str:
    return str(parsed.get("player_key", ""))


def _player_name(parsed: dict) -> str:
    name = parsed.get("name", {})
    if isinstance(name, dict):
        return name.get("full", "")
    return str(name) if name else ""


def _team_abbr(parsed: dict) -> str:
    return str(parsed.get("editorial_team_abbr", ""))


def _positions(parsed: dict) -> List[str]:
    ep = parsed.get("eligible_positions", {})
    if isinstance(ep, list):
        return [
            p.get("position", "")
            for p in ep
            if isinstance(p, dict) and p.get("position")
        ]
    if isinstance(ep, dict):
        pos = ep.get("position", [])
        if isinstance(pos, list):
            return [str(p) for p in pos if p]
        return [str(pos)] if pos else []
    return []


def _flatten_players_section(players_raw) -> dict:
    """Normalise Yahoo's players section to a plain dict {count, "0": {...}, ...}."""
    if isinstance(players_raw, list):
        flat: dict = {}
        for item in players_raw:
            if isinstance(item, dict):
                flat.update(item)
        return flat
    return players_raw if isinstance(players_raw, dict) else {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recent_league_drops(client, days: int = 7) -> List[LeagueDrop]:
    """Return drop transactions from the last *days* days, sorted newest-first.

    Args:
        client: A ``YahooFantasyClient`` (or compatible) instance.
        days:   Look-back window in days. Default 7.

    Returns:
        List of :class:`LeagueDrop` objects sorted newest-first.
        Returns an empty list on API failure — caller should handle gracefully.
    """
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=days)

    try:
        txns = client.get_transactions(t_type="drop")
    except Exception as exc:
        logger.warning("league_transaction_feed: get_transactions error: %s", exc)
        return []

    drops: List[LeagueDrop] = []

    for txn in txns:
        if not isinstance(txn, dict):
            continue

        # Accept "drop" and "add/drop" (which contain drop legs)
        txn_type = txn.get("type", "")
        if "drop" not in txn_type.lower():
            continue

        # Skip pending / rejected transactions
        status = txn.get("status", "successful")
        if status and status.lower() not in ("successful", ""):
            continue

        # Parse Unix timestamp
        ts_raw = txn.get("timestamp", "")
        try:
            dropped_at = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            continue

        if dropped_at < cutoff:
            continue

        days_ago = (now - dropped_at).total_seconds() / 86400.0

        players_section = _flatten_players_section(txn.get("players", {}))

        try:
            count = int(players_section.get("count", 0))
        except (ValueError, TypeError):
            count = 0

        for i in range(count):
            slot = players_section.get(str(i), {})
            if not isinstance(slot, dict):
                continue

            parsed = _parse_player_slot(slot.get("player", []))
            if parsed is None:
                continue

            txn_data = parsed.get("_txn_data", {})
            if not isinstance(txn_data, dict):
                continue

            # In add/drop transactions the same players section contains both
            # add and drop legs — only take the drop leg.
            if txn_data.get("type", "drop") != "drop":
                continue

            pkey = _player_key(parsed)
            pname = _player_name(parsed)
            if not pkey and not pname:
                continue

            drops.append(
                LeagueDrop(
                    player_key=pkey,
                    player_name=pname,
                    team=_team_abbr(parsed),
                    positions=_positions(parsed),
                    dropped_by_team=str(txn_data.get("source_team_key", "")),
                    dropped_by_name=str(txn_data.get("source_team_name", "")),
                    dropped_at=dropped_at,
                    days_ago=round(days_ago, 1),
                )
            )

    drops.sort(key=lambda d: d.dropped_at, reverse=True)
    return drops


def build_drop_lookup(drops: List[LeagueDrop]) -> Dict[str, Dict]:
    """Build key-based and name-based lookup dicts from a drops list.

    Returns:
        {
            "by_key":  {player_key  → LeagueDrop, ...},  # most recent per key
            "by_name": {lower_name  → LeagueDrop, ...},  # fallback for unkeyed matches
        }
    """
    by_key: Dict[str, LeagueDrop] = {}
    by_name: Dict[str, LeagueDrop] = {}

    for drop in drops:  # already sorted newest-first
        if drop.player_key and drop.player_key not in by_key:
            by_key[drop.player_key] = drop
        if drop.player_name:
            nk = drop.player_name.lower()
            if nk not in by_name:
                by_name[nk] = drop

    return {"by_key": by_key, "by_name": by_name}
