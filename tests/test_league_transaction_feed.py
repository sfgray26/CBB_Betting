"""Tests for backend/services/league_transaction_feed.py."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from backend.services.league_transaction_feed import (
    LeagueDrop,
    _flatten_meta,
    _parse_player_slot,
    _positions,
    build_drop_lookup,
    get_recent_league_drops,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

NOW_UTC = datetime.now(tz=timezone.utc)
TS_1_DAY_AGO = str(int((NOW_UTC - timedelta(days=1)).timestamp()))
TS_3_DAYS_AGO = str(int((NOW_UTC - timedelta(days=3)).timestamp()))
TS_10_DAYS_AGO = str(int((NOW_UTC - timedelta(days=10)).timestamp()))


def _make_txn(
    player_key: str = "469.p.10001",
    player_name: str = "Jack Dreyer",
    team_abbr: str = "SD",
    positions: list = None,
    source_team_key: str = "469.l.72586.t.3",
    source_team_name: str = "Marte Partay",
    timestamp: str = TS_1_DAY_AGO,
    txn_type: str = "drop",
    status: str = "successful",
) -> dict:
    """Build a synthetic Yahoo transaction dict matching get_transactions() output."""
    positions = positions or [{"position": "SP"}]
    return {
        "transaction_key": f"469.l.72586.tr.{player_key}",
        "type": txn_type,
        "status": status,
        "timestamp": timestamp,
        "players": {
            "count": 1,
            "0": {
                "player": [
                    # metadata list (already flattened by get_transactions)
                    [
                        {"player_key": player_key},
                        {"player_id": player_key.split(".")[-1]},
                        {"name": {"full": player_name}},
                        {"editorial_team_abbr": team_abbr},
                        {"eligible_positions": positions},
                    ],
                    # transaction_data dict
                    {
                        "transaction_data": {
                            "type": "drop",
                            "source_type": "team",
                            "source_team_key": source_team_key,
                            "source_team_name": source_team_name,
                        }
                    },
                ]
            },
        },
    }


def _make_client(txns: list) -> MagicMock:
    client = MagicMock()
    client.get_transactions.return_value = txns
    return client


# ---------------------------------------------------------------------------
# Unit: _flatten_meta
# ---------------------------------------------------------------------------

class TestFlattenMeta:
    def test_list_of_dicts_flattened(self):
        raw = [{"player_key": "469.p.1"}, {"name": {"full": "A"}}]
        result = _flatten_meta(raw)
        assert result == {"player_key": "469.p.1", "name": {"full": "A"}}

    def test_plain_dict_passthrough(self):
        raw = {"player_key": "469.p.1"}
        assert _flatten_meta(raw) == raw

    def test_empty_returns_empty(self):
        assert _flatten_meta([]) == {}
        assert _flatten_meta({}) == {}


# ---------------------------------------------------------------------------
# Unit: _parse_player_slot
# ---------------------------------------------------------------------------

class TestParsePlayerSlot:
    def test_returns_none_for_non_list(self):
        assert _parse_player_slot({}) is None
        assert _parse_player_slot("bad") is None

    def test_returns_none_for_short_list(self):
        assert _parse_player_slot([{"player_key": "469.p.1"}]) is None

    def test_parses_well_formed_slot(self):
        slot = [
            [{"player_key": "469.p.1"}, {"name": {"full": "Josh"}}],
            {"transaction_data": {"type": "drop", "source_team_name": "Team A"}},
        ]
        result = _parse_player_slot(slot)
        assert result is not None
        assert result["player_key"] == "469.p.1"
        assert result["_txn_data"]["source_team_name"] == "Team A"

    def test_handles_unwrapped_txn_data(self):
        """transaction_data key absent — top-level dict treated as txn_data."""
        slot = [
            [{"player_key": "469.p.2"}],
            {"type": "drop", "source_team_name": "Team B"},
        ]
        result = _parse_player_slot(slot)
        assert result is not None
        assert result["_txn_data"]["source_team_name"] == "Team B"


# ---------------------------------------------------------------------------
# Unit: _positions
# ---------------------------------------------------------------------------

class TestPositions:
    def test_list_of_position_dicts(self):
        parsed = {"eligible_positions": [{"position": "SP"}, {"position": "P"}]}
        assert _positions(parsed) == ["SP", "P"]

    def test_dict_with_list_position(self):
        parsed = {"eligible_positions": {"position": ["OF", "Util"]}}
        assert _positions(parsed) == ["OF", "Util"]

    def test_dict_with_single_position(self):
        parsed = {"eligible_positions": {"position": "C"}}
        assert _positions(parsed) == ["C"]

    def test_missing_returns_empty(self):
        assert _positions({}) == []


# ---------------------------------------------------------------------------
# Unit: get_recent_league_drops
# ---------------------------------------------------------------------------

class TestGetRecentLeagueDrops:
    def test_returns_drop_within_window(self):
        client = _make_client([_make_txn(timestamp=TS_1_DAY_AGO)])
        drops = get_recent_league_drops(client, days=7)
        assert len(drops) == 1
        assert drops[0].player_key == "469.p.10001"
        assert drops[0].player_name == "Jack Dreyer"
        assert drops[0].dropped_by_name == "Marte Partay"

    def test_excludes_drop_outside_window(self):
        client = _make_client([_make_txn(timestamp=TS_10_DAYS_AGO)])
        drops = get_recent_league_drops(client, days=7)
        assert drops == []

    def test_excludes_pending_transactions(self):
        client = _make_client([_make_txn(timestamp=TS_1_DAY_AGO, status="pending")])
        drops = get_recent_league_drops(client, days=7)
        assert drops == []

    def test_excludes_non_drop_transactions(self):
        txn = _make_txn(timestamp=TS_1_DAY_AGO)
        txn["type"] = "add"
        client = _make_client([txn])
        drops = get_recent_league_drops(client, days=7)
        assert drops == []

    def test_accepts_add_drop_transaction_type(self):
        txn = _make_txn(timestamp=TS_1_DAY_AGO, txn_type="add/drop")
        client = _make_client([txn])
        drops = get_recent_league_drops(client, days=7)
        assert len(drops) == 1

    def test_returns_empty_on_api_error(self):
        client = MagicMock()
        client.get_transactions.side_effect = RuntimeError("API down")
        drops = get_recent_league_drops(client)
        assert drops == []

    def test_multiple_drops_sorted_newest_first(self):
        txns = [
            _make_txn(player_key="469.p.1", player_name="Older", timestamp=TS_3_DAYS_AGO),
            _make_txn(player_key="469.p.2", player_name="Newer", timestamp=TS_1_DAY_AGO),
        ]
        client = _make_client(txns)
        drops = get_recent_league_drops(client, days=7)
        assert len(drops) == 2
        assert drops[0].player_name == "Newer"
        assert drops[1].player_name == "Older"

    def test_days_ago_is_approximately_correct(self):
        client = _make_client([_make_txn(timestamp=TS_1_DAY_AGO)])
        drops = get_recent_league_drops(client, days=7)
        assert 0.9 <= drops[0].days_ago <= 1.2

    def test_positions_parsed_correctly(self):
        txn = _make_txn(positions=[{"position": "SP"}, {"position": "P"}])
        client = _make_client([txn])
        drops = get_recent_league_drops(client, days=7)
        assert drops[0].positions == ["SP", "P"]

    def test_invalid_timestamp_skipped(self):
        txn = _make_txn()
        txn["timestamp"] = "not-a-number"
        client = _make_client([txn])
        drops = get_recent_league_drops(client, days=7)
        assert drops == []

    def test_empty_players_section_skipped(self):
        txn = _make_txn()
        txn["players"] = {}
        client = _make_client([txn])
        drops = get_recent_league_drops(client, days=7)
        assert drops == []


# ---------------------------------------------------------------------------
# Unit: build_drop_lookup
# ---------------------------------------------------------------------------

class TestBuildDropLookup:
    def _drop(self, key: str, name: str, days_ago: float = 1.0) -> LeagueDrop:
        return LeagueDrop(
            player_key=key,
            player_name=name,
            team="NYY",
            positions=["OF"],
            dropped_by_team="469.l.72586.t.3",
            dropped_by_name="Marte Partay",
            dropped_at=NOW_UTC - timedelta(days=days_ago),
            days_ago=days_ago,
        )

    def test_by_key_lookup(self):
        drops = [self._drop("469.p.1", "Player A")]
        lk = build_drop_lookup(drops)
        assert "469.p.1" in lk["by_key"]

    def test_by_name_lookup_case_insensitive(self):
        drops = [self._drop("469.p.1", "Player A")]
        lk = build_drop_lookup(drops)
        assert "player a" in lk["by_name"]

    def test_most_recent_drop_wins(self):
        """When same player dropped twice, most recent (index 0) wins."""
        recent = self._drop("469.p.1", "Player A", days_ago=1.0)
        older = self._drop("469.p.1", "Player A", days_ago=3.0)
        lk = build_drop_lookup([recent, older])  # recent first (sorted)
        assert lk["by_key"]["469.p.1"].days_ago == 1.0

    def test_empty_input(self):
        lk = build_drop_lookup([])
        assert lk == {"by_key": {}, "by_name": {}}
