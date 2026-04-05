"""
MLBBettingOdd — BDL /mlb/v1/odds data item contract.

Ground truth: reports/SCHEMA_DISCOVERY.md
CRITICAL: spread_home_value, spread_away_value, total_value are STRINGS
in the API response (e.g. "1.5", "-1.5", "3.5"). They are NOT floats.
Use the @property accessors for numeric comparisons.

Verified vendors from 6-record sample:
    fanduel, fanatics, betrivers, caesars, draftkings, betmgm
Vendor list is NOT exhaustive — typed as str, not Literal.

All odds are American format integers (e.g. -110, +150).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class MLBBettingOdd(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    game_id: int
    vendor: str                         # Book name e.g. "draftkings". Not a closed enum.
    spread_home_value: str              # STRING e.g. "1.5" — use spread_home_float property
    spread_home_odds: int               # American odds e.g. -110
    spread_away_value: str              # STRING e.g. "-1.5"
    spread_away_odds: int               # American odds
    moneyline_home_odds: int            # American odds
    moneyline_away_odds: int            # American odds
    total_value: str                    # STRING e.g. "8.5" — use total_float property
    total_over_odds: int                # American odds
    total_under_odds: int               # American odds
    updated_at: str                     # ISO 8601 UTC timestamp

    @field_validator("spread_home_value", "spread_away_value", "total_value", mode="before")
    @classmethod
    def must_be_numeric_string(cls, v: object) -> str:
        """Reject values that cannot be parsed to float — catches API regressions early."""
        if not isinstance(v, str):
            raise ValueError(f"spread/total value must be a string, got {type(v).__name__}: {v!r}")
        try:
            float(v)
        except ValueError:
            raise ValueError(f"spread/total value '{v}' cannot be parsed as float")
        return v

    @property
    def spread_home_float(self) -> float:
        return float(self.spread_home_value)

    @property
    def spread_away_float(self) -> float:
        return float(self.spread_away_value)

    @property
    def total_float(self) -> float:
        return float(self.total_value)
