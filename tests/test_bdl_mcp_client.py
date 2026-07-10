# -*- coding: utf-8 -*-
"""Tests for BDLPlayerResolver (backend/services/bdl_mcp_client.py)."""
from unittest.mock import MagicMock

import pytest

from backend.data_contracts import MLBPlayer, MLBSeasonStats
from backend.services.bdl_mcp_client import BDLPlayerResolver, _normalize_name


def _player(pid: int, first: str, last: str, active: bool = True) -> MLBPlayer:
    return MLBPlayer(
        id=pid,
        first_name=first,
        last_name=last,
        full_name=f"{first} {last}",
        position="SP",
        active=active,
    )


SANCHEZ = _player(40, "Cristopher", "Sanchez")
OTHER_SANCHEZ = _player(999, "Sixto", "Sanchez")
NUNEZ = _player(164, "Nasim", "Nunez")


@pytest.fixture
def client():
    return MagicMock()


@pytest.fixture
def resolver(client):
    return BDLPlayerResolver(client=client)


class TestSearchPlayers:
    def test_strips_accents_from_query(self, resolver, client):
        client.search_mlb_players.return_value = [SANCHEZ]
        resolver.search_players("Sánchez")
        client.search_mlb_players.assert_called_once_with("Sanchez")

    def test_empty_query_returns_empty_without_api_call(self, resolver, client):
        assert resolver.search_players("") == []
        assert resolver.search_players("   ") == []
        client.search_mlb_players.assert_not_called()


class TestGetPlayerByName:
    def test_accented_yahoo_name_resolves_exact_match(self, resolver, client):
        client.search_mlb_players.return_value = [SANCHEZ, OTHER_SANCHEZ]
        result = resolver.get_player_by_name("Cristopher Sánchez")
        assert result is not None
        assert result.id == 40

    def test_ambiguous_names_return_none(self, resolver, client):
        twin_a = _player(1, "Luis", "Garcia")
        twin_b = _player(2, "Luis", "Garcia")
        client.search_mlb_players.return_value = [twin_a, twin_b]
        assert resolver.get_player_by_name("Luis Garcia") is None

    def test_no_match_returns_none(self, resolver, client):
        client.search_mlb_players.return_value = []
        assert resolver.get_player_by_name("Nonexistent Player") is None

    def test_loose_match_single_active_candidate(self, resolver, client):
        # "Chris Sanchez" (nickname drift) -> single active C. Sanchez
        client.search_mlb_players.return_value = [SANCHEZ, _player(5, "Ali", "Sanchez")]
        result = resolver.get_player_by_name("Cris Sanchez")
        assert result is not None
        assert result.id == 40

    def test_inactive_exact_dupe_prefers_active(self, resolver, client):
        retired = _player(7, "Cristopher", "Sanchez", active=False)
        client.search_mlb_players.return_value = [retired, SANCHEZ]
        result = resolver.get_player_by_name("Cristopher Sanchez")
        assert result is not None
        assert result.id == 40


class TestGetPlayerStats:
    def test_returns_matching_row(self, resolver, client):
        row = MLBSeasonStats(player=SANCHEZ, season=2026, pitching_gp=19, pitching_era=2.62)
        client.get_mlb_season_stats.return_value = [row]
        result = resolver.get_player_stats(40, season=2026)
        assert result is not None
        assert result.pitching_era == 2.62
        client.get_mlb_season_stats.assert_called_once_with(season=2026, player_ids=[40])

    def test_no_rows_returns_none(self, resolver, client):
        client.get_mlb_season_stats.return_value = []
        assert resolver.get_player_stats(40) is None


class TestGetProjections:
    def test_packages_season_stats_with_source_tag(self, resolver, client):
        row = MLBSeasonStats(player=SANCHEZ, season=2026, pitching_gp=19, pitching_k=137)
        client.get_mlb_season_stats.return_value = [row]
        proj = resolver.get_projections(40)
        assert proj is not None
        assert proj["source"] == "bdl"
        assert proj["bdl_id"] == 40
        assert proj["is_pitcher"] is True
        assert proj["stats"]["pitching_k"] == 137

    def test_missing_stats_returns_none(self, resolver, client):
        client.get_mlb_season_stats.return_value = []
        assert resolver.get_projections(40) is None


class TestNormalizeName:
    def test_accents_and_suffixes(self):
        assert _normalize_name("Cristopher Sánchez") == "cristopher sanchez"
        assert _normalize_name("Nasim Nuñez") == "nasim nunez"
        assert _normalize_name("Ronald Acuña Jr.") == "ronald acuna"
