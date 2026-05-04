"""Load and validate the fantasy stat contract at startup."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .schema import FantasyStatContract

_CONTRACT_PATH = Path(__file__).parent / "fantasy_stat_contract.json"


@lru_cache(maxsize=1)
def load_contract() -> FantasyStatContract:
    """Load and validate the stat contract. Cached — call freely."""
    if not _CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Stat contract not found at {_CONTRACT_PATH}. "
            "Run the contract generator or place the JSON manually."
        )
    raw = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    return FantasyStatContract.model_validate(raw)
