from __future__ import annotations

import sys
from contextlib import nullcontext
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from backend.data_contracts.mlb_player import MLBPlayer
from backend.data_contracts.mlb_player_stats import MLBPlayerStats
from backend.services import daily_ingestion
from backend.services.daily_ingestion import (
    DailyIngestionOrchestrator,
    _parse_innings_pitched,
    _validate_mlb_stats,
)


def _make_player(player_id: int = 12345) -> MLBPlayer:
    return MLBPlayer(
        id=player_id,
        first_name="Test",
        last_name="Player",
        full_name="Test Player",
        position="P",
        active=True,
    )


async def _passthrough_lock(_lock_id, _job_name, inner):
    return await inner()


async def _no_sleep(*_args, **_kwargs):
    return None


def test_parse_innings_pitched_rejects_invalid_baseball_out_digits():
    assert _parse_innings_pitched("6.3") is None
    assert _parse_innings_pitched("6.9") is None
    assert _parse_innings_pitched("-1.1") is None


def test_validate_mlb_stats_rejects_invalid_counting_stat_combinations():
    stat = MLBPlayerStats(
        player=_make_player(),
        game_id=67890,
        ab=2,
        h=3,
        double=2,
        triple=1,
        hr=1,
        obp=1.2,
    )

    assert _validate_mlb_stats(stat) is False


@pytest.mark.asyncio
async def test_supplement_statsapi_counting_stats_patches_batter_strikeouts(  # noqa: E501
    monkeypatch,
):
    target_date = date(2026, 4, 13)
    row = SimpleNamespace(
        game_date=target_date,
        raw_payload={
            "player": {"full_name": "Jose Ramirez"},
        },
        ab=None,
        runs=None,
        hits=None,
        doubles=None,
        triples=None,
        home_runs=None,
        rbi=None,
        walks=2,
        strikeouts_bat=None,
        stolen_bases=None,
        caught_stealing=None,
    )
    db_query = SimpleNamespace(
        filter=lambda *_fargs, **_fkwargs: SimpleNamespace(all=lambda: [row]),
    )
    db = SimpleNamespace(
        query=lambda *_args, **_kwargs: db_query,
        commit=lambda: None,
        rollback=lambda: None,
        close=lambda: None,
    )

    statsapi_stub = SimpleNamespace(
        schedule=lambda **_kwargs: [{"game_id": 42, "status": "Final"}],
        boxscore_data=lambda _game_id: {
            "playerInfo": {"ID99": {"fullName": "José Ramírez"}},
            "awayBatters": [
                {
                    "personId": 99,
                    "ab": "4",
                    "r": "1",
                    "h": "2",
                    "doubles": "1",
                    "triples": "0",
                    "hr": "0",
                    "rbi": "2",
                    "bb": "3",
                    "strikeouts": "2",
                    "sb": "1",
                    "cs": "0",
                }
            ],
            "homeBatters": [],
            "awayPitchers": [],
            "homePitchers": [],
        },
    )

    monkeypatch.setattr(
        daily_ingestion,
        "_with_advisory_lock",
        _passthrough_lock,
    )
    monkeypatch.setattr(daily_ingestion, "today_et", lambda: target_date)
    monkeypatch.setattr(daily_ingestion, "SessionLocal", lambda: db)
    monkeypatch.setattr(daily_ingestion.asyncio, "sleep", _no_sleep)
    monkeypatch.setitem(sys.modules, "statsapi", statsapi_stub)

    orch = DailyIngestionOrchestrator()
    monkeypatch.setattr(
        orch,
        "_record_job_run",
        lambda *_args, **_kwargs: None,
    )

    result = await orch._supplement_statsapi_counting_stats()

    assert result["status"] == "success"
    assert row.ab == 4
    assert row.hits == 2
    assert row.strikeouts_bat == 2
    assert row.walks == 2


@pytest.mark.asyncio
async def test_poll_mlb_odds_normalizes_vendor_before_upsert(monkeypatch):
    target_date = date(2026, 4, 13)
    now = datetime(2026, 4, 13, 14, 17)
    captured = []
    db = SimpleNamespace(
        execute=lambda stmt: captured.append(stmt),
        commit=lambda: None,
        rollback=lambda: None,
        close=lambda: None,
    )

    game = SimpleNamespace(
        id=10,
        date="2026-04-13T23:05:00Z",
        season=2026,
        season_type="regular",
        status="STATUS_FINAL",
        venue="Test Park",
        attendance=25000,
        period=9,
        home_team=SimpleNamespace(
            id=1,
            abbreviation="NYY",
            name="Yankees",
            display_name="New York Yankees",
            short_display_name="Yankees",
            location="New York",
            slug="yankees",
            league="AL",
            division="East",
        ),
        away_team=SimpleNamespace(
            id=2,
            abbreviation="BOS",
            name="Red Sox",
            display_name="Boston Red Sox",
            short_display_name="Red Sox",
            location="Boston",
            slug="red-sox",
            league="AL",
            division="East",
        ),
        home_team_data=SimpleNamespace(runs=5, hits=9, errors=0),
        away_team_data=SimpleNamespace(runs=4, hits=8, errors=1),
        model_dump=lambda: {"game_id": 10},
    )
    odd = SimpleNamespace(
        id=99,
        game_id=10,
        vendor=" FanDuel ",
        spread_home_value="-1.5",
        spread_away_value="+1.5",
        spread_home_odds=-110,
        spread_away_odds=-110,
        moneyline_home_odds=-140,
        moneyline_away_odds=120,
        total_value="8.5",
        total_over_odds=-105,
        total_under_odds=-115,
        model_dump=lambda: {"vendor": " FanDuel "},
    )

    class _FakeBDLClient:
        def get_mlb_games(self, _date_str):
            return [game]

        def get_mlb_odds(self, _game_id):
            return [odd]

    monkeypatch.setattr(
        daily_ingestion,
        "_with_advisory_lock",
        _passthrough_lock,
    )
    monkeypatch.setattr(daily_ingestion, "today_et", lambda: target_date)
    monkeypatch.setattr(daily_ingestion, "now_et", lambda: now)
    monkeypatch.setattr(daily_ingestion, "SessionLocal", lambda: db)
    monkeypatch.setitem(
        sys.modules,
        "backend.services.balldontlie",
        SimpleNamespace(BallDontLieClient=_FakeBDLClient),
    )

    orch = DailyIngestionOrchestrator()
    monkeypatch.setattr(
        orch,
        "_record_job_run",
        lambda *_args, **_kwargs: None,
    )

    result = await orch._poll_mlb_odds()

    assert result["status"] == "success"
    odds_stmt = captured[-1]
    params = odds_stmt.compile(dialect=postgresql.dialect()).params
    assert params["vendor"] == "fanduel"
    assert params["raw_payload"]["vendor"] == "fanduel"


@pytest.mark.asyncio
async def test_mlb_box_stats_upsert_uses_unique_constraint(monkeypatch):  # noqa: E501
    target_date = date(2026, 4, 13)
    upsert_calls = []

    lookup_query = SimpleNamespace(
        filter=lambda *_fargs, **_fkwargs: SimpleNamespace(
            all=lambda: [(123,)],
        ),
    )
    lookup_db = SimpleNamespace(
        query=lambda *_args, **_kwargs: lookup_query,
        close=lambda: None,
    )
    write_query = SimpleNamespace(
        filter=lambda *_fargs, **_fkwargs: SimpleNamespace(
            all=lambda: [SimpleNamespace(game_id=123, game_date=target_date)],
        ),
    )
    write_db = SimpleNamespace(
        query=lambda *_args, **_kwargs: write_query,
        execute=lambda stmt: upsert_calls.append(stmt),
        begin_nested=lambda: nullcontext(),
        commit=lambda: None,
        rollback=lambda: None,
        close=lambda: None,
    )

    stat = MLBPlayerStats(
        id=555,
        player=_make_player(),
        game_id=123,
        season=2026,
        ab=4,
        h=2,
        double=1,
        r=1,
        rbi=2,
        bb=1,
        so=1,
        sb=0,
        cs=0,
        obp=0.400,
        slg=0.750,
        ip="6.0",
        h_allowed=5,
        r_allowed=2,
        er=2,
        bb_allowed=1,
        k=7,
        era=3.0,
    )

    class _FakeBDLClient:
        def get_mlb_stats(self, game_ids):
            assert game_ids == [123]
            return [stat]

    monkeypatch.setattr(
        daily_ingestion,
        "_with_advisory_lock",
        _passthrough_lock,
    )
    monkeypatch.setattr(daily_ingestion, "today_et", lambda: target_date)
    monkeypatch.setattr(
        daily_ingestion,
        "now_et",
        lambda: datetime(2026, 4, 13, 3, 0),
    )
    monkeypatch.setitem(
        sys.modules,
        "backend.services.balldontlie",
        SimpleNamespace(BallDontLieClient=_FakeBDLClient),
    )

    session_iter = iter([lookup_db, write_db])
    monkeypatch.setattr(
        daily_ingestion,
        "SessionLocal",
        lambda: next(session_iter),
    )

    orch = DailyIngestionOrchestrator()
    monkeypatch.setattr(
        orch,
        "_record_job_run",
        lambda *_args, **_kwargs: None,
    )

    result = await orch._ingest_mlb_box_stats()

    assert result["status"] == "success"
    compiled = str(upsert_calls[0].compile(dialect=postgresql.dialect()))
    assert (
        "ON CONFLICT ON CONSTRAINT _mps_player_game_uc DO UPDATE"
        in compiled
    )
