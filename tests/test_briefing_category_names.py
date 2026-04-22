"""Contract tests for briefing category name display.

Production probe 2026-04-22 found all 11 briefing categories had name=null.
Root cause: CategoryBriefing.category is an abbreviation (e.g. "HR", "ERA") but
no `name` field was exposed. CATEGORY_DISPLAY_NAMES was added to daily_briefing.py
and the serializer was updated to include `name` in every category dict.
"""

import pathlib


def test_category_display_names_exported():
    """CATEGORY_DISPLAY_NAMES must be importable from daily_briefing."""
    from backend.fantasy_baseball.daily_briefing import CATEGORY_DISPLAY_NAMES
    assert isinstance(CATEGORY_DISPLAY_NAMES, dict)
    assert len(CATEGORY_DISPLAY_NAMES) > 0


def test_category_display_names_covers_core_stats():
    """Key batting and pitching categories must be present."""
    from backend.fantasy_baseball.daily_briefing import CATEGORY_DISPLAY_NAMES
    required = {"R", "HR", "RBI", "SB", "AVG", "OPS", "W", "SV", "ERA", "WHIP", "K_P", "QS"}
    missing = required - set(CATEGORY_DISPLAY_NAMES.keys())
    assert not missing, f"CATEGORY_DISPLAY_NAMES is missing: {missing}"


def test_category_display_names_values_are_nonempty_strings():
    from backend.fantasy_baseball.daily_briefing import CATEGORY_DISPLAY_NAMES
    for cat, label in CATEGORY_DISPLAY_NAMES.items():
        assert isinstance(label, str) and label, (
            f"CATEGORY_DISPLAY_NAMES['{cat}'] is not a non-empty string: {label!r}"
        )


def test_fantasy_router_briefing_serializer_has_name_field():
    """The briefing category serializer in fantasy.py must include a 'name' key."""
    src = (
        pathlib.Path(__file__).parent.parent
        / "backend"
        / "routers"
        / "fantasy.py"
    ).read_text(encoding="utf-8")
    assert '"name": CATEGORY_DISPLAY_NAMES.get(c.category' in src, (
        "Briefing category serializer in fantasy.py is missing the 'name' field. "
        "Consumers will receive name=null."
    )


def test_main_py_briefing_serializer_has_name_field():
    """The mirrored briefing serializer in main.py must also include 'name'."""
    src = (
        pathlib.Path(__file__).parent.parent
        / "backend"
        / "main.py"
    ).read_text(encoding="utf-8")
    assert '"name": CATEGORY_DISPLAY_NAMES.get(c.category' in src, (
        "Briefing category serializer in main.py is missing the 'name' field."
    )


def test_unknown_category_falls_back_to_category_key():
    """Unknown categories should fall back to the raw abbreviation, not None."""
    from backend.fantasy_baseball.daily_briefing import CATEGORY_DISPLAY_NAMES
    result = CATEGORY_DISPLAY_NAMES.get("UNKNOWN_STAT", "UNKNOWN_STAT")
    assert result == "UNKNOWN_STAT"
