import pytest
from unittest.mock import MagicMock, patch

from backend.services.dashboard_service import DashboardService
from backend.services.injury_overlay import InjuryOverlay


@pytest.mark.asyncio
async def test_dashboard_waiver_targets_preserve_owned_pct_and_priority_score():
    service = DashboardService.__new__(DashboardService)

    mock_client = MagicMock()
    mock_client.get_scoreboard.return_value = []
    mock_client.get_my_team_key.return_value = "469.l.72586.t.7"
    # _get_waiver_targets calls get_free_agents directly (not waiver_detector.get_top_moves)
    mock_client.get_free_agents.return_value = [{
        "player_id": "9329",
        "name": "Michael Wacha",
        "team": "KC",
        "positions": ["SP", "P"],
        "percent_owned": 44.2,
    }]
    service._get_yahoo_client = MagicMock(return_value=mock_client)

    with patch("backend.fantasy_baseball.player_board.get_or_create_projection", return_value={}), \
         patch("backend.fantasy_baseball.category_aware_scorer.compute_need_score", return_value=1.25):
        targets = await service._get_waiver_targets(user_id="user-1", prefs=MagicMock())

    assert len(targets) == 1
    assert targets[0].percent_owned == pytest.approx(44.2)
    assert targets[0].priority_score == pytest.approx(1.25)
    assert targets[0].reason == "Need score: 1.25"


@pytest.mark.asyncio
async def test_dashboard_waiver_targets_apply_injury_penalty_and_eta():
    service = DashboardService.__new__(DashboardService)

    mock_client = MagicMock()
    mock_client.get_scoreboard.return_value = []
    mock_client.get_my_team_key.return_value = "469.l.72586.t.7"
    mock_client.get_free_agents.return_value = [{
        "player_key": "469.l.72586.p.9329",
        "player_id": "9329",
        "name": "Michael Wacha",
        "team": "KC",
        "positions": ["SP", "P"],
        "percent_owned": 44.2,
    }]
    service._get_yahoo_client = MagicMock(return_value=mock_client)

    overlay = InjuryOverlay(
        status="15-Day-IL",
        note="Hamstring strain",
        return_timeline="ETA May 24 · updated 30m ago",
        ingested_at=None,  # type: ignore[arg-type]
        is_stale=False,
    )

    with patch("backend.fantasy_baseball.player_board.get_or_create_projection", return_value={}), \
         patch("backend.fantasy_baseball.category_aware_scorer.compute_need_score", return_value=1.25), \
         patch("backend.services.dashboard_service.load_injury_overlays_for_yahoo_players", return_value={
             "469.l.72586.p.9329": overlay,
         }):
        targets = await service._get_waiver_targets(user_id="user-1", prefs=MagicMock())

    assert len(targets) == 1
    assert targets[0].priority_score < 1.25
    assert targets[0].injury_status == "15-Day-IL"
    assert targets[0].estimated_return == "ETA May 24 · updated 30m ago"
    assert "Injury penalty" in targets[0].reason
