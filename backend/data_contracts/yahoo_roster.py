"""
YahooRosterEntry -- roster player with selected_position.

Only returned by get_roster(), not FA or ADP endpoints.
selected_position reflects the active roster slot assignment e.g. "C", "SP", "BN", "IL".
"""
from __future__ import annotations

from backend.data_contracts.yahoo_player import YahooPlayer


class YahooRosterEntry(YahooPlayer):
    selected_position: str  # Active slot: "C", "SP", "RP", "BN", "IL", "P", etc.
