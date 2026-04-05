"""
YahooWaiverCandidate -- free agent / waiver wire player with stats dict.

stats field only present on get_free_agents() responses.
Stat values are raw strings from Yahoo (not floats).

CRITICAL from live capture 2026-04-05:
    Stat ID 60 returns "H/AB" combined format e.g. "8/20" -- NOT the same as
    stat ID 8 (raw H count). These are semantically distinct fields.
    Stat IDs 28 vs 42: likely pitching K vs batting K respectively.
    Do not collapse to a single "K" mapping.
"""
from __future__ import annotations

from typing import Optional

from backend.data_contracts.yahoo_player import YahooPlayer


class YahooWaiverCandidate(YahooPlayer):
    stats: Optional[dict[str, str]] = None  # Stat ID -> value string. FA responses only.
