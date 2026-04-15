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

Nullability: spread_* and total_* fields are Optional because BDL returns
None when a book has not set a line (moneyline-only event, pre-open window).
Moneyline odds remain required — if a book has no moneyline, it has no row.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class MLBBettingOdd(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    game_id: int
    vendor: str                                 # Book name e.g. "draftkings". Not a closed enum.
    spread_home_value: Optional[str] = None     # STRING e.g. "1.5" — None when book has no spread
    spread_home_odds: Optional[int] = None      # American odds e.g. -110 — None when no spread
    spread_away_value: Optional[str] = None     # STRING e.g. "-1.5"
    spread_away_odds: Optional[int] = None      # American odds
    moneyline_home_odds: int                    # Required — defines a row's existence
    moneyline_away_odds: int                    # Required
    total_value: Optional[str] = None           # STRING e.g. "8.5" — None when book has no total
    total_over_odds: Optional[int] = None       # American odds
    total_under_odds: Optional[int] = None      # American odds
    updated_at: str                             # ISO 8601 UTC timestamp

    @field_validator("spread_home_value", "spread_away_value", "total_value", mode="before")
    @classmethod
    def must_be_numeric_string_or_none(cls, v: object) -> Optional[str]:
        """Accept None (book has no line); otherwise enforce numeric string contract."""
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"spread/total value must be a string, got {type(v).__name__}: {v!r}")
        try:
            float(v)
        except ValueError:
            raise ValueError(f"spread/total value '{v}' cannot be parsed as float")
        return v

    @property
    def has_spread(self) -> bool:
        return (
            self.spread_home_value is not None
            and self.spread_away_value is not None
            and self.spread_home_odds is not None
            and self.spread_away_odds is not None
        )

    @property
    def has_total(self) -> bool:
        return (
            self.total_value is not None
            and self.total_over_odds is not None
            and self.total_under_odds is not None
        )

    @property
    def spread_home_float(self) -> Optional[float]:
        return float(self.spread_home_value) if self.spread_home_value is not None else None

    @property
    def spread_away_float(self) -> Optional[float]:
        return float(self.spread_away_value) if self.spread_away_value is not None else None

    @property
    def total_float(self) -> Optional[float]:
        return float(self.total_value) if self.total_value is not None else None
