from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.contracts import LineupOptimizationRequest
from backend.fantasy_baseball.league_contract import (
    ACTIVE_ROSTER_SLOTS,
    ACTIVE_SCORING_CATEGORIES,
    CANONICAL_SCORING_STAT_ID_MAP,
    build_scoring_stat_map_from_settings,
)
from backend.fantasy_baseball.position_normalizer import (
    Player,
    PositionNormalizer,
    RosterSlot,
    YahooRoster,
)


FAKE_ENV = {
    "YAHOO_CLIENT_ID": "test_id",
    "YAHOO_CLIENT_SECRET": "test_secret",
    "YAHOO_LEAGUE_ID": "72586",
    "YAHOO_REFRESH_TOKEN": "test_refresh",
    "YAHOO_ACCESS_TOKEN": "test_access",
}


def _scoring_stat(stat_id: str, display_name: str, position_type: str) -> dict:
    return {
        "stat": {
            "stat_id": stat_id,
            "display_name": display_name,
            "position_type": position_type,
            "is_scoring_category": 1,
            "is_only_display_stat": 0,
        }
    }


def _display_only_stat(stat_id: str, display_name: str, position_type: str) -> dict:
    return {
        "stat": {
            "stat_id": stat_id,
            "display_name": display_name,
            "position_type": position_type,
            "is_scoring_category": 0,
            "is_only_display_stat": 1,
        }
    }


def test_build_scoring_stat_map_from_settings_returns_exact_18_stats():
    settings_payload = {
        "settings": [
            {
                "stat_categories": {
                    "stats": [
                        _scoring_stat("7", "R", "B"),
                        _scoring_stat("8", "H", "B"),
                        _scoring_stat("12", "HR", "B"),
                        _scoring_stat("13", "RBI", "B"),
                        _scoring_stat("42", "K", "B"),
                        _scoring_stat("6", "TB", "B"),
                        _scoring_stat("3", "AVG", "B"),
                        _scoring_stat("55", "OPS", "B"),
                        _scoring_stat("60", "NSB", "B"),
                        _scoring_stat("23", "W", "P"),
                        _scoring_stat("24", "L", "P"),
                        _scoring_stat("35", "HR", "P"),
                        _scoring_stat("28", "K", "P"),
                        _scoring_stat("26", "ERA", "P"),
                        _scoring_stat("27", "WHIP", "P"),
                        _scoring_stat("57", "K/9", "P"),
                        _scoring_stat("29", "QS", "P"),
                        _scoring_stat("83", "NSV", "P"),
                        _display_only_stat("85", "OBP", "B"),
                        _display_only_stat("38", "K/BB", "P"),
                        _display_only_stat("21", "IP", "P"),
                    ]
                }
            }
        ]
    }

    stat_map = build_scoring_stat_map_from_settings(settings_payload)

    assert stat_map == CANONICAL_SCORING_STAT_ID_MAP
    assert len(stat_map) == 18
    assert list(stat_map.values()) == list(ACTIVE_SCORING_CATEGORIES)


def test_lineup_request_rejects_missing_ssot_fields():
    with pytest.raises(ValidationError) as excinfo:
        LineupOptimizationRequest.model_validate(
            {"target_date": "2026-04-03", "risk_tolerance": "balanced"}
        )

    message = str(excinfo.value)
    assert "league_key" in message
    assert "team_key" in message
    assert "scoring_categories" in message
    assert "roster_positions" in message
    assert "available_players" in message


def test_lineup_request_rejects_generic_of_slot():
    payload = {
        "league_key": "mlb.l.72586",
        "team_key": "469.l.72586.t.7",
        "scoring_categories": list(ACTIVE_SCORING_CATEGORIES),
        "roster_positions": [
            "C", "1B", "2B", "3B", "SS", "OF", "CF", "RF",
            "Util", "SP", "SP", "RP", "RP", "P", "P", "P",
        ],
        "available_players": [
            {
                "player_id": "469.p.123",
                "name": "Test Player",
                "eligible_positions": ["LF", "CF", "RF"],
            }
        ],
        "target_date": "2026-04-03",
        "risk_tolerance": "balanced",
    }

    with pytest.raises(ValidationError):
        LineupOptimizationRequest.model_validate(payload)


def test_position_normalizer_routes_outfielders_to_explicit_slots():
    yahoo_roster = YahooRoster(
        slots=[
            RosterSlot(id="slot_1_LF", position="LF"),
            RosterSlot(id="slot_2_CF", position="CF"),
            RosterSlot(id="slot_3_RF", position="RF"),
        ],
        players=[
            Player(id="469.p.1", name="OF One", positions=["OF"]),
            Player(id="469.p.2", name="OF Two", positions=["OF"]),
            Player(id="469.p.3", name="OF Three", positions=["OF"]),
        ],
    )

    assignments = PositionNormalizer.normalize_lineup(
        {"starters": [
            {"id": "469.p.1", "name": "OF One", "positions": ["OF"]},
            {"id": "469.p.2", "name": "OF Two", "positions": ["OF"]},
            {"id": "469.p.3", "name": "OF Three", "positions": ["OF"]},
        ]},
        yahoo_roster,
    )

    assert set(assignments.keys()) == {"slot_1_LF", "slot_2_CF", "slot_3_RF"}
    assert all("OF" not in slot_id for slot_id in assignments)


def test_get_roster_accepts_date_kwarg_without_error():
    with patch.dict("os.environ", FAKE_ENV):
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = YahooFantasyClient()
        client._get = MagicMock(return_value={
            "fantasy_content": {
                "team": {
                    "roster": {"0": {"players": {"count": 0}}}
                }
            }
        })

        roster = client.get_roster(team_key="469.l.72586.t.7", date="2026-04-03")

        assert roster == []
        client._get.assert_called_once_with(
            "team/469.l.72586.t.7/roster/players",
            params={"date": "2026-04-03"},
        )


def test_normalize_player_key_uses_numeric_game_prefix():
    with patch.dict("os.environ", FAKE_ENV):
        from backend.fantasy_baseball.yahoo_client_resilient import ResilientYahooClient

        client = ResilientYahooClient()
        assert client._normalize_player_key("mlb.p.12345", "469.l.72586.t.7") == "469.p.12345"
        assert client._normalize_player_key("469.p.12345", "469.l.72586.t.7") == "469.p.12345"