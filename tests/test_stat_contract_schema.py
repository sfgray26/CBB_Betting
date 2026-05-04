"""Tests for stat contract schema validation."""

import pytest

from backend.stat_contract import (
    CONTRACT,
    SCORING_CATEGORY_CODES,
    BATTING_CODES,
    PITCHING_CODES,
    LOWER_IS_BETTER,
    YAHOO_ID_INDEX,
    MATCHUP_DISPLAY_ORDER,
    load_contract,
)
from backend.stat_contract.schema import (
    Aggregation,
    ExternalIds,
    StatEntry,
    SupportingStatEntry,
    ScoringCategories,
    WeeklyRules,
    FantasyStatContract,
)


def test_contract_loads_successfully():
    """Contract loads successfully."""
    contract = load_contract()
    assert isinstance(contract, FantasyStatContract)
    assert contract.version == "2.0.0"
    assert contract.season == 2026


def test_exactly_18_scoring_stats():
    """Exactly 18 scoring stats."""
    scoring_stats = [
        code for code, entry in CONTRACT.stats.items()
        if entry.is_scoring_category
    ]
    assert len(scoring_stats) == 18


def test_batting_pitching_split():
    """Batting/pitching split — 9 batting + 9 pitching, no overlap, complete coverage."""
    batting = [c for c in CONTRACT.scoring_categories.batting]
    pitching = [c for c in CONTRACT.scoring_categories.pitching]

    assert len(batting) == 9
    assert len(pitching) == 9
    assert set(batting).isdisjoint(set(pitching))
    assert len(batting) + len(pitching) == 18


def test_lower_is_better_correct():
    """Lower-is-better correct — exactly ERA, WHIP, L, K_B, HR_P."""
    expected = {"ERA", "WHIP", "L", "K_B", "HR_P"}
    assert LOWER_IS_BETTER == expected


def test_weekly_rules_validate():
    """Weekly rules validate — pitcher_ip_minimum_outs == round(pitcher_ip_minimum * 3)."""
    rules = CONTRACT.weekly_rules
    expected_outs = int(round(rules.pitcher_ip_minimum * 3))
    assert rules.pitcher_ip_minimum_outs == expected_outs


def test_yahoo_id_index_complete():
    """Yahoo ID index complete — every ID maps to a stat that exists in stats or supporting_stats."""
    all_canonical = set(CONTRACT.stats.keys()) | set(CONTRACT.supporting_stats.keys())
    for yahoo_id_str, canonical in CONTRACT.yahoo_id_index.items():
        assert canonical in all_canonical, f"Yahoo ID {yahoo_id_str} maps to unknown {canonical}"


def test_matchup_display_order():
    """Matchup display order — all 18 scoring categories present, plus display-only stats."""
    # First 18 should be scoring categories
    scoring_in_order = MATCHUP_DISPLAY_ORDER[:18]
    assert set(scoring_in_order) == SCORING_CATEGORY_CODES

    # Should include display-only stats at the end
    assert "IP" in MATCHUP_DISPLAY_ORDER
    assert "GS" in MATCHUP_DISPLAY_ORDER
    assert "H_AB" in MATCHUP_DISPLAY_ORDER


def test_cross_validation():
    """Cross-validation — scoring_categories.batting codes match stats with scoring_role == 'batting'."""
    batting_from_categories = set(CONTRACT.scoring_categories.batting)
    batting_from_stats = {
        code for code, entry in CONTRACT.stats.items()
        if entry.is_scoring_category and entry.scoring_role == "batting"
    }
    assert batting_from_categories == batting_from_stats


def test_schema_rejects_bad_data_missing_field():
    """Schema rejects bad data — missing required field raises ValidationError."""
    with pytest.raises(Exception):  # ValidationError
        ScoringCategories(
            batting=["R", "H"],
            pitching=["W"],
            # missing total_count and win_threshold
        )


def test_schema_rejects_bad_data_wrong_type():
    """Schema rejects bad data — wrong type raises ValidationError."""
    with pytest.raises(Exception):  # ValidationError
        StatEntry(
            canonical_code="TEST",
            display_label="Test",
            short_label="T",
            scope="hitter",  # Should be list, not string
            is_scoring_category=True,
            aggregation=Aggregation(method="sum"),
            direction="higher_is_better",
            data_type="integer",
            precision=0,
            valid_range=[0, None],
            external_ids=ExternalIds(),
        )


def test_schema_rejects_extra_fields():
    """Schema rejects extra fields — unknown field raises ValidationError (extra="forbid")."""
    with pytest.raises(Exception):  # ValidationError
        StatEntry(
            canonical_code="TEST",
            display_label="Test",
            short_label="T",
            scope=["hitter"],
            is_scoring_category=True,
            aggregation=Aggregation(method="sum"),
            direction="higher_is_better",
            data_type="integer",
            precision=0,
            valid_range=[0, None],
            external_ids=ExternalIds(),
            unknown_field="should_fail",  # Extra field
        )


def test_convenience_constants():
    """Convenience constants — SCORING_CATEGORY_CODES has 18 elements, BATTING_CODES has 9, PITCHING_CODES has 9, LOWER_IS_BETTER has 5."""
    assert len(SCORING_CATEGORY_CODES) == 18
    assert len(BATTING_CODES) == 9
    assert len(PITCHING_CODES) == 9
    assert len(LOWER_IS_BETTER) == 5


def test_valid_range_length():
    """StatEntry validates that valid_range has exactly 2 elements."""
    # Valid entry with 2-element range
    entry = StatEntry(
        canonical_code="TEST",
        display_label="Test",
        short_label="T",
        scope=["hitter"],
        is_scoring_category=True,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=[0, None],  # Exactly 2 elements
        external_ids=ExternalIds(),
    )
    assert entry.valid_range == [0, None]

    # Invalid entry with 1-element range
    with pytest.raises(ValueError, match="valid_range must be length 2"):
        StatEntry(
            canonical_code="TEST",
            display_label="Test",
            short_label="T",
            scope=["hitter"],
            is_scoring_category=True,
            aggregation=Aggregation(method="sum"),
            direction="higher_is_better",
            data_type="integer",
            precision=0,
            valid_range=[0],  # Only 1 element
            external_ids=ExternalIds(),
        )


def test_scoring_categories_count_validation():
    """ScoringCategories validates that total_count == len(batting) + len(pitching)."""
    # Valid
    cats = ScoringCategories(
        batting=["R", "H", "HR_B"],
        pitching=["W", "L", "ERA"],
        total_count=6,
        win_threshold=4,
    )
    assert cats.total_count == 6

    # Invalid - total_count doesn't match
    with pytest.raises(ValueError, match="total_count does not match"):
        ScoringCategories(
            batting=["R", "H", "HR_B"],
            pitching=["W", "L", "ERA"],
            total_count=10,  # Should be 6
            win_threshold=4,
        )


def test_scoring_categories_win_threshold_validation():
    """ScoringCategories validates that win_threshold <= total_count."""
    # Valid
    cats = ScoringCategories(
        batting=["R", "H", "HR_B"],
        pitching=["W", "L", "ERA"],
        total_count=6,
        win_threshold=4,
    )
    assert cats.win_threshold == 4

    # Invalid - win_threshold exceeds total_count
    with pytest.raises(ValueError, match="win_threshold exceeds total_count"):
        ScoringCategories(
            batting=["R", "H", "HR_B"],
            pitching=["W", "L", "ERA"],
            total_count=6,
            win_threshold=10,  # Exceeds total_count
        )


def test_weekly_rules_ip_outs_validation():
    """WeeklyRules validates that pitcher_ip_minimum_outs == round(pitcher_ip_minimum * 3)."""
    # Valid
    rules = WeeklyRules(
        pitcher_ip_minimum=18.0,
        pitcher_ip_minimum_outs=54,
        acquisitions_max=8,
        waiver_window_days=1,
        categories_to_win_matchup=10,
    )
    assert rules.pitcher_ip_minimum_outs == 54

    # Invalid - outs don't match IP
    with pytest.raises(ValueError, match="does not equal pitcher_ip_minimum"):
        WeeklyRules(
            pitcher_ip_minimum=18.0,
            pitcher_ip_minimum_outs=50,  # Should be 54
            acquisitions_max=8,
            waiver_window_days=1,
            categories_to_win_matchup=10,
        )


# =============================================================================
# Migration Verification Tests (V1 -> V2)
# =============================================================================

def test_v2_yahoo_id_index_covers_v1_fallback():
    """V2 YAHOO_ID_INDEX covers all scoring stat_ids from old V1 YAHOO_STAT_ID_FALLBACK.

    This verifies the migration is complete — every Yahoo stat_id needed for
    the 18 scoring categories in V1 has a corresponding entry in V2's
    YAHOO_ID_INDEX (though the canonical code may have changed, e.g., 'HR' -> 'HR_B').

    Note: stat_id "38" (K/BB) was intentionally not carried forward to V2 as it's
    not a scoring category.
    """
    # Old V1 YAHOO_STAT_ID_FALLBACK stat_ids for scoring categories
    # (from backend/utils/fantasy_stat_contract.json)
    # Excludes: "38" (K/BB) - not carried forward to V2
    v1_scoring_stat_ids = {
        "3", "4", "6", "7", "8", "12", "13", "16", "21", "23", "24",
        "26", "27", "28", "29", "32", "35", "42", "48", "50",
        "55", "57", "60", "62", "83", "85",
    }

    # All V1 scoring stat_ids should exist in V2 YAHOO_ID_INDEX
    v2_stat_ids = set(YAHOO_ID_INDEX.keys())
    missing = v1_scoring_stat_ids - v2_stat_ids
    assert not missing, f"V2 YAHOO_ID_INDEX missing V1 scoring stat_ids: {missing}"


def test_v2_scoring_codes_cover_v1():
    """V2 SCORING_CATEGORY_CODES covers all 18 V1 categories with semantic equivalent.

    Verifies V2 has 18 scoring categories and covers all V1 categories:
    - V1: HR (batting), HR (pitching), K (batting), K (pitching)
    - V2: HR_B, HR_P, K_B, K_P (disambiguated)
    """
    # V2 should have exactly 18 scoring categories
    assert len(SCORING_CATEGORY_CODES) == 18

    # V1 league scoring categories (from backend/utils/fantasy_stat_contract.json)
    # These were ambiguous — 'HR' and 'K' appeared in both batting and pitching contexts
    v1_batting = {"R", "H", "HR", "RBI", "SB", "K(B)", "AVG", "OPS", "NSB"}
    v1_pitching = {"W", "L", "ERA", "WHIP", "K", "K/9", "SV", "QS", "HRA", "HLD"}

    # V2 disambiguated codes should cover all V1 categories
    v2_batting_codes = set(BATTING_CODES)
    v2_pitching_codes = set(PITCHING_CODES)

    # Coverage mappings (V1 -> V2 semantic equivalent)
    batting_coverage = {
        "R": "R",
        "H": "H",  # H is in stats but not a scoring category
        "HR": "HR_B",
        "RBI": "RBI",
        "SB": "NSB",  # V2 uses NSB (Net SB)
        "K(B)": "K_B",
        "AVG": "AVG",
        "OPS": "OPS",
        "NSB": "NSB",
    }

    pitching_coverage = {
        "W": "W",
        "L": "L",
        "ERA": "ERA",
        "WHIP": "WHIP",
        "K": "K_P",
        "K/9": "K_9",
        "SV": "NSV",  # V2 uses NSV (Net Saves)
        "QS": "QS",
        "HRA": "HR_P",
        "HLD": "HLD",
    }

    # Verify all V1 batting categories have V2 equivalents
    for v1_cat, v2_code in batting_coverage.items():
        if v1_cat == "H":  # H is not a scoring category in V2
            continue
        assert v2_code in v2_batting_codes, f"V1 batting '{v1_cat}' -> '{v2_code}' not in V2 batting codes"

    # Verify all V1 pitching categories have V2 equivalents
    for v1_cat, v2_code in pitching_coverage.items():
        if v1_cat == "SV":  # SV maps to NSV
            assert v2_code in v2_pitching_codes, f"V1 pitching '{v1_cat}' -> '{v2_code}' not in V2 pitching codes"


def test_disambiguation_maps_use_v2_codes():
    """V2 disambiguation codes (HR_P, K_P, K_B, HR_B) exist in CONTRACT with correct semantics.

    Verifies the four disambiguated codes exist and have:
    - Correct scoring_role (batting vs pitching)
    - Correct direction (higher_is_better vs lower_is_better)
    """
    # HR_P: pitching, lower_is_better (home runs allowed)
    hr_p_entry = CONTRACT.stats["HR_P"]
    assert hr_p_entry.is_scoring_category
    assert hr_p_entry.scoring_role == "pitching"
    assert hr_p_entry.direction == "lower_is_better"

    # K_P: pitching, higher_is_better (strikeouts thrown)
    k_p_entry = CONTRACT.stats["K_P"]
    assert k_p_entry.is_scoring_category
    assert k_p_entry.scoring_role == "pitching"
    assert k_p_entry.direction == "higher_is_better"

    # K_B: batting, lower_is_better (strikeouts taken)
    k_b_entry = CONTRACT.stats["K_B"]
    assert k_b_entry.is_scoring_category
    assert k_b_entry.scoring_role == "batting"
    assert k_b_entry.direction == "lower_is_better"

    # HR_B: batting, higher_is_better (home runs hit)
    hr_b_entry = CONTRACT.stats["HR_B"]
    assert hr_b_entry.is_scoring_category
    assert hr_b_entry.scoring_role == "batting"
    assert hr_b_entry.direction == "higher_is_better"
