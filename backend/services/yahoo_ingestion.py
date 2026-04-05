"""
Validated Yahoo ingestion helpers -- Layer 2 adapter.

These functions wrap the existing YahooFantasyClient methods and parse
their output through Pydantic V2 contracts. The client is NOT modified.

Any ValidationError is logged at ERROR level with the item index and raw
value. Never suppressed. Returns an empty list on total failure so
callers always receive a typed list, never a raw dict or None.

Layer model:
    Layer 0: YahooPlayer / YahooRosterEntry / YahooWaiverCandidate (contracts)
    Layer 2: This module (adapter -- raw dict -> validated model)
    Layer 3: daily_ingestion.py (scheduler -- calls these helpers)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from backend.data_contracts import YahooPlayer, YahooRosterEntry, YahooWaiverCandidate

if TYPE_CHECKING:
    from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

logger = logging.getLogger(__name__)


def get_validated_roster(client: "YahooFantasyClient") -> list[YahooRosterEntry]:
    """Return validated roster entries.

    Wraps client.get_roster(). Any item that fails YahooRosterEntry
    validation is logged at ERROR and skipped. Never raises.
    """
    raw = client.get_roster()
    results: list[YahooRosterEntry] = []
    for i, item in enumerate(raw):
        try:
            results.append(YahooRosterEntry.model_validate(item))
        except ValidationError as exc:
            logger.error(
                "get_validated_roster item[%d] failed: %s | raw=%r",
                i,
                exc,
                item,
            )
    return results


def get_validated_free_agents(client: "YahooFantasyClient") -> list[YahooWaiverCandidate]:
    """Return validated waiver/FA candidates.

    Wraps client.get_free_agents(). Any item that fails
    YahooWaiverCandidate validation is logged at ERROR and skipped.
    Never raises.
    """
    raw = client.get_free_agents()
    results: list[YahooWaiverCandidate] = []
    for i, item in enumerate(raw):
        try:
            results.append(YahooWaiverCandidate.model_validate(item))
        except ValidationError as exc:
            logger.error(
                "get_validated_free_agents item[%d] failed: %s | raw=%r",
                i,
                exc,
                item,
            )
    return results


def get_validated_adp_feed(client: "YahooFantasyClient") -> list[YahooPlayer]:
    """Return validated ADP/injury feed players.

    Wraps client.get_adp_and_injury_feed(). Any item that fails
    YahooPlayer validation is logged at ERROR and skipped. Never raises.

    This is the embargo-critical method for job 100_013. 100% parse rate
    required before that job is re-enabled.
    """
    raw = client.get_adp_and_injury_feed()
    results: list[YahooPlayer] = []
    for i, item in enumerate(raw):
        try:
            results.append(YahooPlayer.model_validate(item))
        except ValidationError as exc:
            logger.error(
                "get_validated_adp_feed item[%d] failed: %s | raw=%r",
                i,
                exc,
                item,
            )
    return results
