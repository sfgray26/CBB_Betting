"""
BDL pagination envelope — generic wrapper for all BDL responses.

Ground truth: reports/SCHEMA_DISCOVERY.md
"""


from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class BDLMeta(BaseModel):
    model_config = ConfigDict(strict=True)

    per_page: int
    next_cursor: Optional[int] = None


class BDLResponse(BaseModel, Generic[T]):
    """
    Generic wrapper: BDLResponse[MLBGame], BDLResponse[MLBBettingOdd], etc.

    Usage:
        payload = BDLResponse[MLBGame].model_validate(raw_dict)
        games: list[MLBGame] = payload.data
        cursor: int | None = payload.meta.next_cursor
    """

    model_config = ConfigDict(strict=False)  # Generic models need strict=False

    data: list[T]
    meta: BDLMeta

# Rebuild models to handle forward references correctly
BDLResponse.model_rebuild()
BDLMeta.model_rebuild()
