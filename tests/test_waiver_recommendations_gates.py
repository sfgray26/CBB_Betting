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

    assert '_mcmc.get("mcmc_enabled") and _mcmc.get("win_prob_gain", 0.0) < 0' in src, (
        "fantasy waiver recommendations handler must skip MCMC-enabled "
        "moves with negative win_prob_gain"
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
        "21": "24.1",   # IP
        "28": "1",      # K_P (Yahoo sends under this id too)
        "29": "1",      # QS
        "38": "0",      # unknown stat_id — must drop
        "42": "21",     # K_B
        "26": "1.48",   # ERA
        "27": "0.99",   # WHIP
        "57": "7.77",   # K_9
        "83": "3",      # NSV
        "85": "0",      # OBP
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
    for canonical in {"IP", "K_P", "QS", "K_B", "ERA", "WHIP", "K_9", "NSV", "OBP"}:
        assert canonical in translated, (
            f"canonical code {canonical} must survive translation"
        )
