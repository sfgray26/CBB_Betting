"""
IdentityResolutionService — Sprint 1a

Resolves a set of incoming provider identifiers (Yahoo GUID/ID, MLBAM ID,
FanGraphs ID, full name) to a canonical ``player_identities.id`` value.

Resolution order
----------------
1. yahoo_guid  (unique index — fastest path)
2. yahoo_id    (unique index)
3. mlbam_id    (unique index)
4. fangraphs_id (unique index)
5. normalized_name (exact match after NFC + lower + strip)

On failure the incoming record is routed to ``identity_quarantine`` with
status PENDING_REVIEW and the function returns None.  Callers must skip
or defer any downstream writes for None results.

No fuzzy matching is performed here.  Fuzzy candidate scoring in the
quarantine entry uses difflib.SequenceMatcher for display purposes only —
the scored candidates are stored as informational metadata for manual review.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.models import IdentityQuarantine, PlayerIdentity


def _normalize_name(name: str) -> str:
    """Return accent-stripped, lower-cased, stripped name.

    Uses NFKD decomposition to remove combining characters (accents) so that
    'José Ramírez' and 'Jose Ramirez' resolve to the same key ('jose ramirez').
    This matches FanGraphs/Steamer ASCII names against Yahoo-sourced accented names.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _top_candidates(
    session: Session,
    normalized: str,
    limit: int = 3,
) -> List[dict]:
    """
    Return the top ``limit`` PlayerIdentity rows ranked by
    SequenceMatcher ratio against *normalized*.  Used only for
    informational metadata in the quarantine entry.
    """
    rows = session.query(
        PlayerIdentity.id,
        PlayerIdentity.full_name,
        PlayerIdentity.normalized_name,
    ).all()

    scored = [
        {
            "player_id": r.id,
            "full_name": r.full_name,
            "score": round(SequenceMatcher(None, normalized, r.normalized_name).ratio(), 3),
        }
        for r in rows
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


class IdentityResolutionService:
    """
    Stateless service — all methods accept an explicit ``db`` Session so the
    service is compatible with both request-scoped FastAPI sessions and
    standalone ingestion scripts.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        db: Session,
        *,
        yahoo_guid: Optional[str] = None,
        yahoo_id: Optional[str] = None,
        mlbam_id: Optional[int] = None,
        fangraphs_id: Optional[str] = None,
        full_name: Optional[str] = None,
        provider: str = "MANUAL",
        raw_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Return ``player_identities.id`` or None (quarantined).

        Parameters
        ----------
        db : Session
            Active SQLAlchemy session (caller manages commit/rollback).
        yahoo_guid, yahoo_id, mlbam_id, fangraphs_id, full_name
            Any subset of identifiers may be provided.
        provider
            Originating data source label stored in the quarantine row
            (FANGRAPHS | STATCAST | YAHOO | BDL | MANUAL).
        raw_id
            The provider's raw record identifier, stored in quarantine for
            diagnostic purposes.
        """
        identity = (
            self._lookup_by_yahoo_guid(db, yahoo_guid)
            or self._lookup_by_yahoo_id(db, yahoo_id)
            or self._lookup_by_mlbam_id(db, mlbam_id)
            or self._lookup_by_fangraphs_id(db, fangraphs_id)
            or self._lookup_by_normalized_name(db, full_name)
        )

        if identity is not None:
            return identity.id

        # ── Nothing matched — quarantine ────────────────────────────────
        raw_name = full_name or ""
        normalized = _normalize_name(raw_name) if raw_name else ""
        candidates = _top_candidates(db, normalized) if normalized else []
        top_score = candidates[0]["score"] if candidates else 0.0

        # Deduplicate: skip if there is already an open PENDING_REVIEW row
        # for this exact (provider, raw_name, raw_id) combination.
        existing = (
            db.query(IdentityQuarantine)
            .filter(
                IdentityQuarantine.incoming_provider == provider,
                IdentityQuarantine.incoming_raw_name == raw_name,
                IdentityQuarantine.incoming_raw_id == raw_id,
                IdentityQuarantine.status == "PENDING_REVIEW",
            )
            .first()
        )
        if existing is None:
            quarantine_row = IdentityQuarantine(
                incoming_provider=provider,
                incoming_raw_name=raw_name,
                incoming_raw_id=raw_id,
                proposed_player_id=candidates[0]["player_id"] if candidates else None,
                match_score=top_score,
                match_candidates=candidates,
                status="PENDING_REVIEW",
            )
            db.add(quarantine_row)
            # Flush but do NOT commit — caller controls the transaction.
            db.flush()

        return None

    # ------------------------------------------------------------------
    # Private lookup helpers (return ORM row or None)
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_by_yahoo_guid(
        db: Session, yahoo_guid: Optional[str]
    ) -> Optional[PlayerIdentity]:
        if not yahoo_guid:
            return None
        return (
            db.query(PlayerIdentity)
            .filter(PlayerIdentity.yahoo_guid == yahoo_guid)
            .first()
        )

    @staticmethod
    def _lookup_by_yahoo_id(
        db: Session, yahoo_id: Optional[str]
    ) -> Optional[PlayerIdentity]:
        if not yahoo_id:
            return None
        return (
            db.query(PlayerIdentity)
            .filter(PlayerIdentity.yahoo_id == yahoo_id)
            .first()
        )

    @staticmethod
    def _lookup_by_mlbam_id(
        db: Session, mlbam_id: Optional[int]
    ) -> Optional[PlayerIdentity]:
        if mlbam_id is None:
            return None
        return (
            db.query(PlayerIdentity)
            .filter(PlayerIdentity.mlbam_id == mlbam_id)
            .first()
        )

    @staticmethod
    def _lookup_by_fangraphs_id(
        db: Session, fangraphs_id: Optional[str]
    ) -> Optional[PlayerIdentity]:
        if not fangraphs_id:
            return None
        return (
            db.query(PlayerIdentity)
            .filter(PlayerIdentity.fangraphs_id == fangraphs_id)
            .first()
        )

    @staticmethod
    def _lookup_by_normalized_name(
        db: Session, full_name: Optional[str]
    ) -> Optional[PlayerIdentity]:
        if not full_name:
            return None
        normalized = _normalize_name(full_name)
        return (
            db.query(PlayerIdentity)
            .filter(PlayerIdentity.normalized_name == normalized)
            .first()
        )


def get_quarantined_identity_ids(session: Session) -> set:
    """
    Return PlayerIdentity.id values that have at least one PENDING_REVIEW
    quarantine entry with a proposed match.

    TeamContext uses this to exclude proposed-but-unconfirmed identity matches
    from rate-category denominators.

    Args:
        session: SQLAlchemy Session.

    Returns:
        Set of PlayerIdentity.id integers. Empty set if none pending.
    """
    rows = (
        session.query(IdentityQuarantine.proposed_player_id)
        .filter(
            IdentityQuarantine.status == "PENDING_REVIEW",
            IdentityQuarantine.proposed_player_id.isnot(None),
        )
        .all()
    )
    return {r.proposed_player_id for r in rows}
