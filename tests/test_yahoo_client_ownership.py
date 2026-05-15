"""Tests for ownership % batch enrichment in get_free_agents."""
from unittest.mock import MagicMock


def _make_client():
    from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
    client = YahooFantasyClient.__new__(YahooFantasyClient)
    client.league_key = "mlb.l.99999"
    client._cache = {}
    client._circuit_open = False
    client._failure_count = 0
    client.logger = MagicMock()
    client.get_players_stats_batch = MagicMock(return_value={})
    return client


def _fake_players_response(player_key: str = "mlb.p.1"):
    return {
        "fantasy_content": {
            "league": [
                {},
                {"players": {
                    "0": {"player": [{"player_key": player_key, "player_id": "1",
                                      "full_name": "Test Player",
                                      "editorial_team_abbr": "NYY",
                                      "eligible_positions": [{"position": "OF"}]}]},
                    "count": 1,
                }}
            ]
        }
    }


def _fake_ownership_response(player_key: str = "mlb.p.1", pct: str = "34.5"):
    return {
        "fantasy_content": {
            "players": {
                "0": {"player": [
                    {"player_key": player_key},
                    {"ownership": {"percent_rostered": {"value": pct}}},
                ]},
                "count": 1,
            }
        }
    }


def test_ownership_merged_from_batch_call():
    """Ownership % from secondary batch call must be merged into player dicts."""
    client = _make_client()

    def fake_get(path, params=None):
        if "ownership" in path:
            return _fake_ownership_response()
        return _fake_players_response()

    client._get = fake_get

    players = client.get_free_agents()
    assert len(players) == 1
    assert players[0]["percent_owned"] == pytest.approx(34.5), (
        f"Expected 34.5 from ownership batch, got {players[0]['percent_owned']}"
    )


def test_ownership_batch_failure_is_non_fatal():
    """If ownership batch call raises, players still returned with 0.0 ownership."""
    client = _make_client()

    def fake_get(path, params=None):
        if "ownership" in path:
            raise Exception("Yahoo 400")
        return _fake_players_response()

    client._get = fake_get

    players = client.get_free_agents()
    assert len(players) == 1
    assert players[0]["percent_owned"] == 0.0


def test_ownership_not_overwritten_if_already_set():
    """If _parse_player already extracted a non-zero ownership, batch must not overwrite it."""
    client = _make_client()

    def fake_players_with_ownership(path, params=None):
        if "ownership" in path:
            # Batch returns 50% but player already has 25% from _parse_player
            return _fake_ownership_response(pct="50.0")
        # Include ownership block in primary response so _parse_player gets 25.0
        return {
            "fantasy_content": {
                "league": [
                    {},
                    {"players": {
                        "0": {"player": [
                            {"player_key": "mlb.p.1", "player_id": "1",
                             "full_name": "Test Player",
                             "editorial_team_abbr": "NYY",
                             "eligible_positions": [{"position": "OF"}],
                             "ownership": {"percent_rostered": {"value": "25.0"}}},
                        ]},
                        "count": 1,
                    }}
                ]
            }
        }

    client._get = fake_players_with_ownership

    players = client.get_free_agents()
    assert len(players) == 1
    # _parse_player already found 25.0, batch should NOT overwrite with 50.0
    assert players[0]["percent_owned"] == pytest.approx(25.0), (
        f"Expected 25.0 (from primary parse), got {players[0]['percent_owned']}"
    )


def test_adp_loader_accepts_current_yahoo_headers(tmp_path):
    """Current ADP CSV uses PLAYER NAME/AVG, not the legacy Name/ADP headers."""
    client = _make_client()
    adp_path = tmp_path / "adp_yahoo_2026.csv"
    adp_path.write_text(
        "PLAYER NAME,TEAM,POS,AVG,BEST,WORST,# TEAMS,STDEV\n"
        "Ranger Suarez,PHI,SP,156.4,124,193,6,25.0\n",
        encoding="utf-8",
    )
    client.adp_data_path = str(adp_path)

    adp_data = client._load_adp_data()

    assert adp_data["Ranger Suarez"] == pytest.approx(156.4)
    assert client._estimate_ownership_from_adp("Ranger Suarez", adp_data) > 0.0


import pytest
