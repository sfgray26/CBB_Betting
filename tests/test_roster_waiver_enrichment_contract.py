"""Regression tests for roster enrichment and waiver matchup context.

UAT capture `tasks/uat_findings_post_deploy_v5.md` (2026-04-21) showed:

1. `GET /api/fantasy/roster` returned 200 but `players_with_stats = 0/23`.
   `season_stats`, `rolling_*`, `ros_projection`, `row_projection`,
   `game_context`, `bdl_player_id`, `mlbam_id` were null for every player.

   Root cause: the canonical roster handler in `backend/routers/fantasy.py`
   never called `get_players_stats_batch()` after `get_roster()`. The
   player_mapper reads `yahoo_player["stats"]` to populate season_stats,
   but Yahoo's roster/players subresource does not return stats inline,
   so the mapper always received an empty dict.

2. `GET /api/fantasy/waiver` returned 200 but `matchup_opponent = "TBD"`
   and `category_deficits = []`.

   Root cause: the waiver handler used a shallow two-level dict/list
   descent to pull team_key and team_stats out of the scoreboard response.
   Yahoo's actual payload nests these one level deeper than the parser
   handled, so the handler could not find the user's team in the matchup
   and the opponent stayed at the default "TBD" sentinel. That default
   also short-circuited category_deficits construction.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fantasy_client():
    """TestClient fixture that skips scheduler bootstrap and bypasses auth."""
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.auth import verify_api_key
            from backend.fantasy_app import app
            from fastapi.testclient import TestClient

            async def _no_auth() -> str:
                return "test_user"

            app.dependency_overrides[verify_api_key] = _no_auth
            try:
                with TestClient(app) as client:
                    yield client
            finally:
                app.dependency_overrides.pop(verify_api_key, None)


# ---------------------------------------------------------------------------
# Roster enrichment
# ---------------------------------------------------------------------------

def test_roster_populates_season_stats_from_batch(fantasy_client):
    """Roster endpoint must hydrate season_stats via get_players_stats_batch."""
    mock_roster = [
        {
            "player_key": "469.l.72586.p.10001",
            "name": "Test Batter",
            "team": "NYY",
            "positions": ["OF", "Util"],
            "selected_position": "OF",
        },
        {
            "player_key": "469.l.72586.p.10002",
            "name": "Test Pitcher",
            "team": "LAD",
            "positions": ["SP", "P"],
            "selected_position": "SP",
        },
    ]

    # Yahoo batch-stats response keyed by stat_id string.
    # 7=R, 8=H, 12=HR_B, 13=RBI, 42=K_B (per YAHOO_ID_INDEX)
    mock_stats = {
        "469.l.72586.p.10001": {
            "7": "42", "8": "78", "12": "11", "13": "38", "42": "52",
        },
        # 26=ERA, 27=WHIP, 28=K_P, 29=QS, 57=K_9
        "469.l.72586.p.10002": {
            "26": "2.85", "27": "1.05", "28": "64", "29": "4", "57": "9.2",
        },
    }

    mock_client = MagicMock()
    mock_client.get_roster.return_value = mock_roster
    mock_client.get_players_stats_batch.return_value = mock_stats

    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
         patch("backend.routers.fantasy.fetch_rolling_stats_for_players", return_value={}):
        response = fantasy_client.get("/api/fantasy/roster")

    assert response.status_code == 200
    payload = response.json()
    players = payload["players"]
    assert len(players) == 2

    # get_players_stats_batch must have been called with the roster keys
    mock_client.get_players_stats_batch.assert_called_once()
    call_args, call_kwargs = mock_client.get_players_stats_batch.call_args
    requested_keys = call_args[0] if call_args else call_kwargs.get("player_keys")
    assert set(requested_keys) == {
        "469.l.72586.p.10001", "469.l.72586.p.10002",
    }

    # season_stats must be populated (not null) and carry canonical codes
    batter = next(p for p in players if p["yahoo_player_key"] == "469.l.72586.p.10001")
    assert batter["season_stats"] is not None, (
        "season_stats must hydrate from get_players_stats_batch — roster UAT "
        "showed this field was null for all 23 players"
    )
    batter_values = batter["season_stats"]["values"]
    assert batter_values["R"] == 42.0
    assert batter_values["H"] == 78.0
    assert batter_values["HR_B"] == 11.0
    assert batter_values["RBI"] == 38.0
    assert batter_values["K_B"] == 52.0

    pitcher = next(p for p in players if p["yahoo_player_key"] == "469.l.72586.p.10002")
    assert pitcher["season_stats"] is not None
    pitcher_values = pitcher["season_stats"]["values"]
    assert pitcher_values["ERA"] == 2.85
    assert pitcher_values["WHIP"] == 1.05
    assert pitcher_values["K_P"] == 64.0
    assert pitcher_values["QS"] == 4.0
    assert pitcher_values["K_9"] == 9.2


def test_roster_survives_stats_batch_failure(fantasy_client):
    """Roster must still return 200 when get_players_stats_batch raises."""
    mock_roster = [{
        "player_key": "469.l.72586.p.10003",
        "name": "Survivor",
        "team": "CHC",
        "positions": ["3B", "Util"],
        "selected_position": "3B",
    }]

    mock_client = MagicMock()
    mock_client.get_roster.return_value = mock_roster
    mock_client.get_players_stats_batch.side_effect = RuntimeError("boom")

    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
         patch("backend.routers.fantasy.fetch_rolling_stats_for_players", return_value={}):
        response = fantasy_client.get("/api/fantasy/roster")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["players"]) == 1
    # Degrades gracefully: no season_stats but handler must not 5xx.
    assert payload["players"][0]["season_stats"] is None


# ---------------------------------------------------------------------------
# Waiver matchup context
# ---------------------------------------------------------------------------

def _nested_scoreboard(my_key: str, opp_key: str, my_stats: dict, opp_stats: dict) -> list:
    """Build a Yahoo-shaped scoreboard payload that nests team_key and
    team_stats one level deeper than a 2-level descent can reach.

    The outer list wraps each team metadata and stats in an inner list,
    which mirrors what Yahoo actually returns and broke the previous
    waiver parser.
    """
    def _team_block(team_key: str, name: str, stats_by_sid: dict) -> list:
        stats_list = [
            {"stat": {"stat_id": sid, "value": val}}
            for sid, val in stats_by_sid.items()
        ]
        return [
            [
                {"team_key": team_key},
                {"name": name},
                {"team_id": team_key.rsplit(".", 1)[-1]},
            ],
            {"team_stats": {"stats": stats_list}},
        ]

    return [
        {
            "week": 5,
            "teams": {
                "count": 2,
                "0": {"team": _team_block(my_key, "Lindor Truffles", my_stats)},
                "1": {"team": _team_block(opp_key, "Bartolo's Colon", opp_stats)},
            },
        }
    ]


def test_waiver_populates_opponent_and_category_deficits(fantasy_client, monkeypatch):
    """Waiver must resolve opponent name and produce category_deficits."""
    monkeypatch.setenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    my_key = "469.l.72586.t.7"
    opp_key = "469.l.72586.t.3"
    my_sids = {"7": "48", "26": "3.10"}   # R=48, ERA=3.10
    opp_sids = {"7": "41", "26": "3.55"}  # R=41 (I'm winning), ERA=3.55 (I'm winning — lower is better)

    mock_client = MagicMock()
    mock_client.get_my_team_key.return_value = my_key
    mock_client.get_roster.return_value = []
    mock_client.get_faab_balance.return_value = 100
    mock_client.get_free_agents.return_value = []
    mock_client.get_scoreboard.return_value = _nested_scoreboard(
        my_key, opp_key, my_sids, opp_sids,
    )
    # No league settings -> handler falls back to YAHOO_ID_INDEX for sid_map
    mock_client.get_league_settings.side_effect = RuntimeError("no settings in test")

    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
        response = fantasy_client.get("/api/fantasy/waiver")

    assert response.status_code == 200
    payload = response.json()

    assert payload["matchup_opponent"] == "Bartolo's Colon", (
        "waiver handler must parse the nested Yahoo scoreboard and "
        "resolve opponent name — not fall back to 'TBD'"
    )

    deficits = payload["category_deficits"]
    assert len(deficits) > 0, (
        "category_deficits must populate once opponent is resolved — "
        "UAT capture showed empty list"
    )

    # R (higher is better): my 48 vs opp 41 -> winning
    r_row = next((d for d in deficits if d["category"] == "R"), None)
    assert r_row is not None
    assert r_row["my_total"] == 48.0
    assert r_row["opponent_total"] == 41.0
    assert r_row["winning"] is True

    # ERA (lower is better): my 3.10 vs opp 3.55 -> winning
    era_row = next((d for d in deficits if d["category"] == "ERA"), None)
    assert era_row is not None
    assert era_row["my_total"] == 3.10
    assert era_row["opponent_total"] == 3.55
    assert era_row["winning"] is True


def test_waiver_scoreboard_calls_once(fantasy_client, monkeypatch):
    """Scoreboard must be fetched exactly once per waiver request.

    The previous implementation called get_scoreboard twice — once for
    opponent resolution and once for category_deficits. Consolidating
    to one call keeps Yahoo rate-limit pressure lower.
    """
    monkeypatch.setenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    my_key = "469.l.72586.t.7"
    opp_key = "469.l.72586.t.3"
    scoreboard = _nested_scoreboard(
        my_key, opp_key,
        {"7": "10"}, {"7": "9"},
    )

    mock_client = MagicMock()
    mock_client.get_my_team_key.return_value = my_key
    mock_client.get_roster.return_value = []
    mock_client.get_faab_balance.return_value = 100
    mock_client.get_free_agents.return_value = []
    mock_client.get_scoreboard.return_value = scoreboard
    mock_client.get_league_settings.side_effect = RuntimeError("no settings in test")

    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
        response = fantasy_client.get("/api/fantasy/waiver")

    assert response.status_code == 200
    assert mock_client.get_scoreboard.call_count == 1, (
        "waiver handler must call get_scoreboard at most once per request"
    )


# ---------------------------------------------------------------------------
# Pure helper — scoreboard parsing
# ---------------------------------------------------------------------------

def test_flatten_scoreboard_team_entry_recovers_nested_team_key():
    """The recursive parser must recover team_key nested deeper than 2 levels.

    The historical waiver parser did 2-level descent only; Yahoo returns
    metadata one level deeper, which is why matchup_opponent was stuck
    on the 'TBD' default.
    """
    from backend.routers.fantasy import _flatten_scoreboard_team_entry

    nested = [
        [
            {"team_key": "469.l.72586.t.7"},
            {"name": "Lindor Truffles"},
        ],
        {
            "team_stats": {
                "stats": [
                    {"stat": {"stat_id": "7", "value": "42"}},
                ]
            }
        },
    ]

    team_key, team_name, stats_raw = _flatten_scoreboard_team_entry(nested)
    assert team_key == "469.l.72586.t.7"
    assert team_name == "Lindor Truffles"
    assert len(stats_raw) == 1
    assert stats_raw[0]["stat"]["stat_id"] == "7"
