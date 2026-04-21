"""Regression tests for daily briefing — no-game semantics.

Production postman capture (postman_collections/responses/api_fantasy_
briefing_2026_04_20_20260420_181810.json) showed the briefing recommending
"START" for every player while simultaneously emitting "Starting but no
game today" alerts. Fix: route no-game cards to MONITOR and reword the
smart-selector warning so it does not contradict the routing.

Also verifies that briefing uses `opposing_pitcher.team` as the opponent
string instead of the hard-coded "TBD".
"""

from unittest.mock import MagicMock, patch

from backend.fantasy_baseball.daily_briefing import DailyBriefingGenerator
from backend.fantasy_baseball.smart_lineup_selector import (
    SmartBatterRanking,
    OpposingPitcher,
    Handedness,
)


def _ranking(name: str, team: str, has_game: bool, opp_team: str = "") -> SmartBatterRanking:
    r = SmartBatterRanking(
        name=name,
        player_id=name.lower().replace(" ", "_"),
        team=team,
        positions=["OF"],
        has_game=has_game,
        smart_score=2.5,
    )
    if opp_team:
        r.opposing_pitcher = OpposingPitcher(
            name=f"{opp_team} Starter",
            team=opp_team,
            handedness=Handedness.R,
        )
    return r


def test_briefing_routes_no_game_players_to_monitor_not_start():
    """When has_game is False, cards must NOT be labeled START.

    A START recommendation with a 'no game today' factor is self-
    contradictory and erodes trust. MONITOR is the accurate routing.
    """
    gen = DailyBriefingGenerator.__new__(DailyBriefingGenerator)
    gen.record_decisions = False
    gen.smart_selector = MagicMock()
    gen.category_tracker = MagicMock()
    gen.context_builder = MagicMock()
    gen.context_builder.build_context.return_value = None

    rankings = [
        _ranking("No-Game Bat", "CHC", has_game=False),
        _ranking("Another No-Game Bat", "NYM", has_game=False),
    ]
    gen.smart_selector.select_optimal_lineup.return_value = (rankings, [])
    gen.category_tracker.get_category_needs.return_value = []

    with patch.object(gen, "_record_all_decisions"):
        briefing = gen.generate(roster=[], projections=[], game_date="2026-04-20")

    assert len(briefing.start_recommendations) == 0, (
        "No-game players must not be recommended to START"
    )
    assert len(briefing.monitor_list) == 2, "No-game players should route to MONITOR"
    for card in briefing.monitor_list:
        assert card.recommendation == "MONITOR"


def test_briefing_opponent_uses_opposing_pitcher_team_not_tbd():
    """Opponent must come from opposing_pitcher.team when available."""
    gen = DailyBriefingGenerator.__new__(DailyBriefingGenerator)
    gen.record_decisions = False
    gen.smart_selector = MagicMock()
    gen.category_tracker = MagicMock()
    gen.context_builder = MagicMock()
    gen.context_builder.build_context.return_value = None

    rankings = [
        _ranking("Juan Soto", "NYM", has_game=True, opp_team="PHI"),
    ]
    gen.smart_selector.select_optimal_lineup.return_value = (rankings, [])
    gen.category_tracker.get_category_needs.return_value = []

    with patch.object(gen, "_record_all_decisions"):
        briefing = gen.generate(roster=[], projections=[], game_date="2026-04-20")

    all_cards = (
        briefing.start_recommendations
        + briefing.bench_recommendations
        + briefing.monitor_list
    )
    soto = next(c for c in all_cards if c.player_name == "Juan Soto")
    assert soto.opponent == "PHI", f"expected opponent PHI, got {soto.opponent!r}"


def test_smart_selector_no_game_warning_is_not_contradictory():
    """The upstream warning string must not claim 'Starting' for no-game players.

    Checks that the smart selector source no longer emits the literal phrase
    'Starting but no game' which pipes through briefing alerts as a
    contradiction to the MONITOR routing.
    """
    import pathlib

    src_path = pathlib.Path(__file__).parent.parent / "backend" / "fantasy_baseball" / "smart_lineup_selector.py"
    source = src_path.read_text(encoding="utf-8")
    assert "Starting but no game today" not in source, (
        "smart_lineup_selector must not emit 'Starting but no game today' — it "
        "contradicts the briefing MONITOR routing for no-game players"
    )
