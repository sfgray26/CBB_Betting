"""
PlayerIDResolver -- BDL player.id -> mlbam_id identity resolution service.

Resolution order:
  1. Cache hit: player_id_mapping WHERE bdl_id = bdl_player_id
     Manual overrides (source='manual') take precedence via ORDER BY.
  2. pybaseball fallback: playerid_lookup(last, first) -> mlbam_id
  3. Persist successful pybaseball lookups to cache with source='pybaseball',
     resolution_confidence=1.0.

Returns None if resolution fails -- callers decide whether to skip or flag.

This is a pure Python service: no async, no background jobs.
It does NOT call the BDL API -- it consumes BDL player.id from ingested data.

ADR-004: Never import betting_model or analysis from this file.
"""

import contextlib
import io
import logging
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """
    Unicode-normalize a player name for fuzzy matching.

    Steps:
      1. NFKD decomposition (splits combined chars into base + combining)
      2. Strip combining characters (category 'Mn' = Mark, Nonspacing)
      3. Lowercase
      4. Strip leading/trailing whitespace

    Example: "Adolis Garcia" -> "adolis garcia"
             "Yordan Alvarez" -> "yordan alvarez" (accent stripped)
    """
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return stripped.lower().strip()


class PlayerIDResolver:
    """
    Resolves BDL player IDs to canonical mlbam IDs.

    Resolution order:
    1. Cache hit: check player_id_mapping WHERE bdl_id = bdl_player_id
       Manual overrides (source='manual') are returned first.
    2. pybaseball fallback: playerid_lookup(last, first) -> mlbam_id
    3. Persist successful lookups to cache with source='pybaseball',
       resolution_confidence=1.0.

    Returns None if resolution fails (caller decides whether to skip or flag).
    """

    def __init__(self, db_session):
        self._db = db_session

    def resolve(self, bdl_player_id: int, full_name: str) -> Optional[int]:
        """
        Return mlbam_id for the given BDL player ID, or None if unresolvable.

        Args:
            bdl_player_id: BDL internal player.id (from stat or injury rows)
            full_name:     Player's full name (e.g. "Shohei Ohtani") for pybaseball fallback
        """
        # Step 1: cache lookup (manual overrides always win)
        cached = self._cache_lookup(bdl_player_id)
        if cached is not None:
            return cached

        # Step 2: pybaseball name lookup
        mlbam_id = self._pybaseball_lookup(full_name)
        if mlbam_id is None:
            logger.debug(
                "PlayerIDResolver: no mlbam_id found for bdl_player_id=%d name=%r",
                bdl_player_id, full_name,
            )
            return None

        # Step 3: persist to cache so future calls hit step 1
        self._persist_to_cache(bdl_player_id, full_name, mlbam_id)
        return mlbam_id

    def _cache_lookup(self, bdl_player_id: int) -> Optional[int]:
        """
        Query player_id_mapping by bdl_id.

        Manual overrides (source='manual') take precedence: we ORDER BY
        (source = 'manual') DESC so the manual row sorts first when multiple
        rows share the same bdl_id (should not happen in practice but guards
        against data entry errors).

        Returns mlbam_id or None if no row found.
        """
        try:
            from backend.models import PlayerIDMapping
            from sqlalchemy import case

            # Priority: manual > pybaseball > api (anything else)
            priority = case(
                (PlayerIDMapping.source == "manual", 0),
                else_=1,
            )
            row = (
                self._db.query(PlayerIDMapping)
                .filter(PlayerIDMapping.bdl_id == bdl_player_id)
                .order_by(priority)
                .first()
            )
            if row is None:
                return None
            return row.mlbam_id
        except Exception as exc:
            logger.warning(
                "PlayerIDResolver._cache_lookup(bdl_player_id=%d) failed: %s",
                bdl_player_id, exc,
            )
            return None

    def _pybaseball_lookup(self, full_name: str) -> Optional[int]:
        """
        Use pybaseball.playerid_lookup(last, first) to find mlbam_id.

        Handles:
          - Name parsing: split on first space; everything after is first name
          - No results: return None
          - Multiple results: prefer rows where key_mlbam is not null;
            if still multiple, take the last row (most recent player)
          - pd.NA / np.nan in the mlbam column

        Suppresses pybaseball's stdout progress output via redirect.

        Returns mlbam_id (key_mlbam) as int, or None.
        """
        # Parse name: "Last First" or "Last, First" both handled.
        # pybaseball convention: playerid_lookup(last, first)
        name = full_name.strip()
        if "," in name:
            # "Garcia, Adolis" format
            parts = [p.strip() for p in name.split(",", 1)]
            last_name = parts[0]
            first_name = parts[1] if len(parts) > 1 else ""
        elif " " in name:
            # "Adolis Garcia" -> first="Adolis", last="Garcia"
            # But pybaseball takes (last, first)
            parts = name.rsplit(" ", 1)
            first_name = parts[0]
            last_name = parts[1]
        else:
            # Single name token -- try as last name only
            last_name = name
            first_name = ""

        try:
            import pybaseball  # type: ignore[import]

            # Suppress pybaseball's "Gathering player lookup table." stdout
            with contextlib.redirect_stdout(io.StringIO()):
                if first_name:
                    df = pybaseball.playerid_lookup(last_name, first_name)
                else:
                    df = pybaseball.playerid_lookup(last_name)
        except ImportError:
            logger.error(
                "PlayerIDResolver: pybaseball not installed -- cannot resolve %r", full_name
            )
            return None
        except Exception as exc:
            logger.warning(
                "PlayerIDResolver._pybaseball_lookup(%r) failed: %s", full_name, exc
            )
            return None

        if df is None or df.empty:
            return None

        # Filter to rows where key_mlbam is present (not null / NaN)
        try:
            import pandas as pd
            valid = df[df["key_mlbam"].notna()]
            if valid.empty:
                return None
            # Multiple results: take the last row (most recent player for this name)
            last_row = valid.iloc[-1]
            raw_val = last_row["key_mlbam"]
            # Guard against pd.NA, np.nan, and other non-int values
            if pd.isna(raw_val):
                return None
            return int(raw_val)
        except Exception as exc:
            logger.warning(
                "PlayerIDResolver._pybaseball_lookup(%r) result parse failed: %s",
                full_name, exc,
            )
            return None

    def _persist_to_cache(
        self,
        bdl_player_id: int,
        full_name: str,
        mlbam_id: int,
    ) -> None:
        """
        Insert or update player_id_mapping with the resolved IDs.

        bdl_id has a UNIQUE constraint, so there is at most one row per BDL player.
        We still guard manual overrides as sacred.

        Sets:
          source               = 'pybaseball'
          resolution_confidence = 1.0
          normalized_name      = normalize(full_name)
          mlbam_id             = mlbam_id

        Does NOT overwrite rows with source='manual' -- manual rows are sacred.
        """
        from backend.models import PlayerIDMapping

        normalized = _normalize_name(full_name)

        try:
            # Check if a manual override row exists -- never overwrite it
            existing = (
                self._db.query(PlayerIDMapping)
                .filter(PlayerIDMapping.bdl_id == bdl_player_id)
                .filter(PlayerIDMapping.source == "manual")
                .first()
            )
            if existing is not None:
                logger.debug(
                    "PlayerIDResolver._persist_to_cache: skipping -- manual override "
                    "exists for bdl_player_id=%d",
                    bdl_player_id,
                )
                return

            # bdl_id is unique -- check for any existing row
            existing_row = (
                self._db.query(PlayerIDMapping)
                .filter(PlayerIDMapping.bdl_id == bdl_player_id)
                .first()
            )
            if existing_row is not None:
                # Update in-place -- mlbam may have been refined
                existing_row.mlbam_id = mlbam_id
                existing_row.full_name = full_name
                existing_row.normalized_name = normalized
                existing_row.source = "pybaseball"
                existing_row.resolution_confidence = 1.0
                self._db.commit()
            else:
                # New row -- no yahoo_key known yet (NULL)
                new_row = PlayerIDMapping(
                    yahoo_key=None,
                    yahoo_id=None,
                    mlbam_id=mlbam_id,
                    bdl_id=bdl_player_id,
                    full_name=full_name,
                    normalized_name=normalized,
                    source="pybaseball",
                    resolution_confidence=1.0,
                )
                self._db.add(new_row)
                self._db.commit()

            logger.info(
                "PlayerIDResolver: cached bdl_player_id=%d -> mlbam_id=%d (name=%r)",
                bdl_player_id, mlbam_id, full_name,
            )
        except Exception as exc:
            logger.warning(
                "PlayerIDResolver._persist_to_cache(bdl_player_id=%d) failed: %s",
                bdl_player_id, exc,
            )
            try:
                self._db.rollback()
            except Exception:
                pass
