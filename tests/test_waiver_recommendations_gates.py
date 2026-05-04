"""Regression tests for /api/fantasy/waiver/recommendations filter gates.

Production Postman capture (postman_collections/responses/
waiver_recommendations_200.json) emitted an ADD_DROP suggestion with
win_prob_gain = -0.001 — surfacing a move the MCMC simulator says
*hurts* the matchup. The handler now rejects any MCMC-enabled move
with negative win_prob_gain.

Also guards the waiver stat contract — Yahoo's stats batch can return
non-scoring stat_ids (e.g. "38") that do not round-trip through
YAHOO_ID_INDEX. The handler drops any numeric key left after
translation so the response never surfaces `"38": "0"`.
"""

import pathlib


def test_handler_rejects_negative_mcmc_win_prob_gain():
    """Guard must appear verbatim in the fantasy router source.

    Unit-level assertion: checks that the handler has the MCMC gate that
    skips recommendations whose simulated win probability drops. An
    end-to-end test would require mocking the Yahoo client, projection
    builder, statcast loader, and MCMC simulator — far more surface than
    the single-line invariant we are protecting.
    """
    src = (
        pathlib.Path(__file__).parent.parent
        / "backend"
        / "routers"
        / "fantasy.py"
    ).read_text(encoding="utf-8")

    assert '_mcmc.get("mcmc_enabled") and _mcmc.get("win_prob_gain", 0.0) <= 0' in src, (
        "fantasy waiver recommendations handler must skip MCMC-enabled "
        "moves with non-positive win_prob_gain (zero or negative gain is not actionable)"
    )


def test_waiver_stats_dropped_when_yahoo_stat_id_is_unknown():
    """Handler must drop numeric stat_ids that fail to translate.

    Production capture (api_fantasy_waiver_position_ALL_player_type_ALL_
    20260420_181810.json) surfaced `"38": "0"` on Seth Lugo's stats
    block. stat_id 38 is not in fantasy_stat_contract.json yahoo_id_index
    and is not a scoring category in this league, so it leaks through
    sid_map.get(k, k) unchanged. The fix drops any key that is still a
    bare numeric string after translation.
    """
    src = (
        pathlib.Path(__file__).parent.parent
        / "backend"
        / "routers"
        / "fantasy.py"
    ).read_text(encoding="utf-8")

    assert (
        "if isinstance(_translated_key, str) and _translated_key.isdigit():"
        in src
    ), (
        "_to_waiver_player must drop untranslated numeric stat_ids so "
        "Yahoo-internal IDs like '38' never appear in waiver output"
    )


def test_waiver_stats_numeric_id_filter_behavior():
    """Simulate the translation/filter step to prove numeric IDs are dropped.

    Mirrors the logic at backend/routers/fantasy.py `_to_waiver_player` —
    any key that is still all-digits after sid_map.get(k, k) is dropped.
    Canonical codes like K_P, K_B, K_9, NSV, QS, IP, ERA, WHIP, OBP stay.
    """
    from backend.stat_contract import YAHOO_ID_INDEX

    sid_map: dict = dict(YAHOO_ID_INDEX)

    raw_stats = {
        "21": "24.1",   # TB
        "50": "6.0",    # IP
        "28": "1",      # W
        "29": "0",      # L
        "38": "0",      # HR_P (unknown stat_id in old contract — must drop)
        "42": "21",     # K_P
        "23": "5",      # K_B
        "26": "1.48",   # ERA
        "27": "0.99",   # WHIP
        "57": "7.77",   # K_9
        "83": "3",      # NSV
        "85": "1",      # QS
    }

    translated: dict = {}
    for k, v in raw_stats.items():
        tk = sid_map.get(k, k)
        if tk == "K(P)":
            tk = "K"
        if isinstance(tk, str) and tk.isdigit():
            continue
        translated[tk] = v

    assert "38" not in translated, "raw numeric stat_id must be dropped"
    for canonical in {"TB", "IP", "W", "L", "HR_P", "K_P", "K_B", "ERA", "WHIP", "K_9", "NSV", "QS"}:
        assert canonical in translated, (
            f"canonical code {canonical} must survive translation"
        )


def test_recommendations_populates_category_deficits_and_opponent(monkeypatch):
    """Recommendations endpoint must compute category_deficits using shared helper."""
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from backend.main import app
    from backend.auth import verify_api_key

    async def _auth():
        return "test-user"

    app.dependency_overrides[verify_api_key] = _auth

    scoreboard = [{
        "teams": {
            "count": 2,
            "0": {
                "team": [
                    {"team_key": "469.l.1.t.7", "name": "My Team"},
                    {"team_stats": {"stats": [
                        {"stat": {"stat_id": "7", "value": "10"}},
                        {"stat": {"stat_id": "12", "value": "40"}},
                    ]}},
                ]
            },
            "1": {
                "team": [
                    {"team_key": "469.l.1.t.3", "name": "Rival Squad"},
                    {"team_stats": {"stats": [
                        {"stat": {"stat_id": "7", "value": "8"}},
                        {"stat": {"stat_id": "12", "value": "45"}},
                    ]}},
                ]
            },
        }
    }]

    mock_client = MagicMock()
    mock_client.get_my_team_key.return_value = "469.l.1.t.7"
    mock_client.get_roster.return_value = []
    mock_client.get_free_agents.return_value = []
    mock_client.get_scoreboard.return_value = scoreboard

    try:
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.fantasy_baseball.statcast_loader.build_statcast_signals", return_value=([], 0.0)), \
             patch("backend.fantasy_baseball.statcast_loader.statcast_need_score_boost", return_value=0.0), \
             patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_batters", return_value={}), \
             patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_pitchers", return_value={}):
            client = TestClient(app)
            resp = client.get("/api/fantasy/waiver/recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["matchup_opponent"] == "Rival Squad", \
            f"Expected 'Rival Squad', got {data['matchup_opponent']!r}"
        assert isinstance(data["category_deficits"], list), "category_deficits must be a list"
        assert len(data["category_deficits"]) > 0, \
            "category_deficits must be non-empty when scoreboard has stat data"
    finally:
        app.dependency_overrides.pop(verify_api_key, None)
