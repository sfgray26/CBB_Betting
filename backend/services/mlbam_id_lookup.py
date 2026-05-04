"""
MLBAM ID Lookup Service

Uses pybaseball.playerid_lookup() to cross-reference player names
to MLBAM IDs (Savant IDs) for Statcast integration.

Caching: Results are cached in-memory during backfill to avoid repeated
API calls for the same player.
"""
import logging
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import pybaseball

logger = logging.getLogger(__name__)


class MLBAMLookupError(Exception):
    """Raised when MLBAM ID lookup fails."""
    pass


def build_mlbam_cache(players: list) -> Dict[str, Dict[str, Optional[int]]]:
    """
    Build a cache of MLBAM IDs for all players using pybaseball.

    Args:
        players: List of player dicts with 'full_name' and optionally 'first_name', 'last_name'

    Returns:
        Dict mapping normalized_name -> {'mlbam_id': int or None, 'confidence': float}
        - normalized_name: lowercase full_name for lookup
        - mlbam_id: MLBAM ID (int) or None if not found
        - confidence: 1.0 for exact match, 0.5 for fuzzy match
    """
    cache: Dict[str, Dict[str, Optional[int]]] = {}

    # Group by last name to batch lookups
    by_last_name: Dict[str, list] = {}
    for p in players:
        full_name = p.get("full_name", "")
        if not full_name:
            continue

        # Parse name components
        parts = full_name.strip().split()
        if len(parts) < 2:
            # Single name (rare), use full as last
            last_name = full_name
            first_name = ""
        else:
            last_name = parts[-1]
            first_name = parts[0]

        norm_name = full_name.lower()
        by_last_name.setdefault(last_name.lower(), []).append({
            "full_name": full_name,
            "first_name": first_name,
            "last_name": last_name,
            "norm_name": norm_name
        })

    logger.info(f"Looking up MLBAM IDs for {len(players)} players across {len(by_last_name)} unique last names...")

    # Batch lookup by last name
    for last_norm, player_list in by_last_name.items():
        try:
            # Try exact last name match first
            df = pybaseball.playerid_lookup(last=last_norm, fuzzy=False)

            if df.empty:
                # Try fuzzy match as fallback
                df = pybaseball.playerid_lookup(last=last_norm, fuzzy=True)

            if df.empty:
                # No results - mark all as None
                for p in player_list:
                    cache[p["norm_name"]] = {"mlbam_id": None, "confidence": 0.0}
                continue

            # Match first names within results
            for p in player_list:
                first_norm = p["first_name"].lower()

                # Exact match on first name
                match = df[df["name_first"].str.lower() == first_norm]

                if match.empty:
                    # Try first initial match (e.g., "J" for "Juan")
                    first_initial = first_norm[0] if first_norm else ""
                    match = df[df["name_first"].str.lower().str.startswith(first_initial)]

                if not match.empty:
                    # Take first match
                    row = match.iloc[0]
                    mlbam_id = int(row["key_mlbam"]) if row["key_mlbam"] else None
                    cache[p["norm_name"]] = {"mlbam_id": mlbam_id, "confidence": 1.0}
                else:
                    cache[p["norm_name"]] = {"mlbam_id": None, "confidence": 0.0}

        except Exception as e:
            logger.warning(f"Failed to lookup MLBAM ID for last name '{last_norm}': {e}")
            for p in player_list:
                cache[p["norm_name"]] = {"mlbam_id": None, "confidence": 0.0}

    # Report statistics
    found = sum(1 for v in cache.values() if v["mlbam_id"] is not None)
    logger.info(f"MLBAM ID lookup complete: {found}/{len(players)} players found ({100*found/len(players):.1f}%)")

    return cache


def get_mlbam_id_for_player(
    full_name: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    cache: Optional[Dict[str, Dict[str, Optional[int]]]] = None
) -> Optional[int]:
    """
    Get MLBAM ID for a single player.

    Args:
        full_name: Player's full name
        first_name: Player's first name (optional, extracted from full_name if not provided)
        last_name: Player's last name (optional, extracted from full_name if not provided)
        cache: Optional cache dict from build_mlbam_cache()

    Returns:
        MLBAM ID as int, or None if not found
    """
    if cache:
        norm_name = full_name.lower()
        if norm_name in cache:
            return cache[norm_name]["mlbam_id"]

    # Direct lookup (slower, use only for single queries)
    try:
        if not last_name:
            parts = full_name.strip().split()
            last_name = parts[-1] if len(parts) > 1 else full_name
            first_name = parts[0] if len(parts) > 1 else None

        df = pybaseball.playerid_lookup(last=last_name, first=first_name, fuzzy=False)

        if df.empty and first_name:
            # Try fuzzy match
            df = pybaseball.playerid_lookup(last=last_name, first=first_name, fuzzy=True)

        if not df.empty:
            row = df.iloc[0]
            return int(row["key_mlbam"]) if row["key_mlbam"] else None

    except Exception as e:
        logger.warning(f"MLBAM lookup failed for {full_name}: {e}")

    return None
