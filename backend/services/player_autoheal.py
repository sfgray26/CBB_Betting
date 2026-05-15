"""
PlayerAutoHealService — BDL search fallback for unmapped Yahoo players.

When yahoo_id_sync cannot match a Yahoo player to a BDL ID by normalized name
(new call-ups, name-format mismatches), this service searches BDL by player name
and inserts a provisional mapping so downstream enrichment can proceed.

Source tag: 'bdl_search' — never overwrites 'manual' rows.
TTL: AUTO_HEAL_TTL_DAYS (7). Fresh bdl_search rows are skipped on re-runs;
     stale ones are refreshed. Failed heals (no BDL match) are retried every run.

ADR-004: Never import betting_model from this file.
"""

import logging
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

AUTO_HEAL_TTL_DAYS = 7
CONFIDENCE_THRESHOLD = 0.85


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return stripped.lower().strip()


def _name_confidence(query_norm: str, candidate_norm: str) -> float:
    """Score name match quality between 0.0 and 1.0."""
    if query_norm == candidate_norm:
        return 1.0
    if query_norm in candidate_norm or candidate_norm in query_norm:
        return 0.9
    return 0.0


def _is_fresh(dt: datetime, ttl_days: int = AUTO_HEAL_TTL_DAYS) -> bool:
    """Return True if dt is within ttl_days of now."""
    now = datetime.now(ZoneInfo("America/New_York"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
    return (now - dt) < timedelta(days=ttl_days)


class PlayerAutoHealService:
    """
    BDL name-search fallback for Yahoo players with no player_id_mapping row.

    Intended to run as the final phase of yahoo_id_sync after the main
    name-match loop, processing players that had no BDL index hit.

    Usage:
        service = PlayerAutoHealService(db_session=db, bdl_client=bdl)
        summary = service.batch_heal(unmatched_list)
    """

    def __init__(self, db_session, bdl_client):
        self._db = db_session
        self._bdl = bdl_client

    def heal_player(
        self,
        yahoo_id: str,
        yahoo_key: str,
        name: str,
    ) -> bool:
        """
        Attempt to resolve a single unmapped Yahoo player via BDL name search.

        Returns True if a mapping was inserted/updated, False otherwise.

        Skip conditions:
          - source='manual' row exists (sacred — never overwrite)
          - source='bdl_search' row exists with bdl_id AND is fresh (< TTL)

        Retry conditions:
          - source='bdl_search' row exists with bdl_id=None (previous heal failed)
          - source='bdl_search' row exists but is stale (>= TTL)
        """
        from backend.models import PlayerIDMapping

        norm_name = _normalize(name)

        existing = (
            self._db.query(PlayerIDMapping)
            .filter(PlayerIDMapping.yahoo_key == yahoo_key)
            .first()
        )

        if existing is not None:
            if existing.source == "manual":
                logger.debug("auto_heal: skip %s — manual override exists", yahoo_key)
                return False
            if (
                existing.source == "bdl_search"
                and existing.bdl_id is not None
                and existing.updated_at is not None
                and _is_fresh(existing.updated_at)
            ):
                logger.debug("auto_heal: skip %s — fresh bdl_search row", yahoo_key)
                return False

        try:
            results = self._bdl.search_mlb_players(name)
        except Exception as exc:
            logger.warning("auto_heal: BDL search failed for %r: %s", name, exc)
            return False

        best_result = None
        best_confidence = 0.0
        for r in results:
            conf = _name_confidence(norm_name, _normalize(r.full_name))
            if conf > best_confidence:
                best_confidence = conf
                best_result = r

        if best_result is None or best_confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                "auto_heal: no confident BDL match for %r (best_conf=%.2f)",
                name, best_confidence,
            )
            return False

        try:
            now = datetime.now(ZoneInfo("America/New_York"))
            if existing is not None:
                existing.bdl_id = best_result.id
                existing.full_name = best_result.full_name
                existing.normalized_name = _normalize(best_result.full_name)
                existing.source = "bdl_search"
                existing.resolution_confidence = best_confidence
                existing.updated_at = now
            else:
                conflict = (
                    self._db.query(PlayerIDMapping)
                    .filter(PlayerIDMapping.bdl_id == best_result.id)
                    .first()
                )
                if conflict is not None:
                    logger.warning(
                        "auto_heal: bdl_id=%d already mapped (row id=%d), skipping %r",
                        best_result.id, conflict.id, name,
                    )
                    return False

                new_row = PlayerIDMapping(
                    yahoo_id=str(yahoo_id),
                    yahoo_key=yahoo_key,
                    bdl_id=best_result.id,
                    full_name=best_result.full_name,
                    normalized_name=_normalize(best_result.full_name),
                    source="bdl_search",
                    resolution_confidence=best_confidence,
                )
                self._db.add(new_row)

            self._db.commit()
            logger.info(
                "auto_heal: healed yahoo_key=%s -> bdl_id=%d name=%r (conf=%.2f)",
                yahoo_key, best_result.id, name, best_confidence,
            )
            return True

        except Exception as exc:
            logger.warning("auto_heal: DB write failed for %r: %s", name, exc)
            try:
                self._db.rollback()
            except Exception:
                pass
            return False

    def batch_heal(self, unmatched: List[Dict]) -> Dict:
        """
        Attempt to heal a batch of unmatched Yahoo players.

        Args:
            unmatched: List of dicts with keys 'name', 'yahoo_id', 'yahoo_key',
                       and optional 'reason' (bdl_name_collision, bdl_id_conflict
                       are skipped — those require manual resolution).

        Returns:
            {"healed": int, "skipped": int, "failed": int}
        """
        healed = 0
        skipped = 0
        failed = 0

        for player in unmatched:
            name = player.get("name", "")
            yahoo_id = str(player.get("yahoo_id", ""))
            yahoo_key = player.get("yahoo_key", "")

            if not name or not yahoo_id or not yahoo_key:
                skipped += 1
                continue

            reason = player.get("reason", "")
            if reason in ("bdl_name_collision", "bdl_id_conflict"):
                skipped += 1
                continue

            try:
                if self.heal_player(yahoo_id=yahoo_id, yahoo_key=yahoo_key, name=name):
                    healed += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.warning("auto_heal: batch_heal error for %r: %s", name, exc)
                failed += 1

        logger.info(
            "auto_heal: batch complete — healed=%d skipped=%d failed=%d total=%d",
            healed, skipped, failed, healed + skipped + failed,
        )
        return {"healed": healed, "skipped": skipped, "failed": failed}
