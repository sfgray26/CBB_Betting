import json
from pathlib import Path

from backend.utils.fantasy_stat_contract import (
    CATEGORY_NEED_STAT_MAP,
    LOWER_IS_BETTER,
    YAHOO_STAT_ID_FALLBACK,
    get_fantasy_stat_contract,
)


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "frontend" / "lib" / "fantasy-stat-contract.json"


def test_shared_stat_contract_loader_reads_canonical_json():
    contract = get_fantasy_stat_contract()
    raw = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["fallbackStatIds"] == raw["fallbackStatIds"]
    assert contract["statLabels"] == raw["statLabels"]


def test_shared_stat_contract_keeps_critical_ids_aligned():
    assert YAHOO_STAT_ID_FALLBACK["57"] == "K/9"
    assert YAHOO_STAT_ID_FALLBACK["60"] == "NSB"
    assert CATEGORY_NEED_STAT_MAP["60"] == "nsb"
    assert "29" not in LOWER_IS_BETTER
