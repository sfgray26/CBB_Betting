import pytest
from unittest.mock import MagicMock

from backend.services.dashboard_service import DashboardService


@pytest.mark.asyncio
async def test_dashboard_waiver_targets_preserve_owned_pct_and_priority_score():
    service = DashboardService.__new__(DashboardService)
    service.waiver_detector = MagicMock()

    mock_client = MagicMock()
    mock_client.get_roster.return_value = []
    mock_client.get_scoreboard.return_value = []
    mock_client.get_my_team_key.return_value = "469.l.72586.t.7"
    service._get_yahoo_client = MagicMock(return_value=mock_client)

    service.waiver_detector.get_top_moves.return_value = [{
        "add_player": {
            "player_key": "469.p.9329",
            "name": "Michael Wacha",
            "team": "KC",
            "positions": ["SP", "P"],
            "owned_pct": 44.2,
        },
        "need_score": 1.25,
        "win_prob_gain": 0.0,
        "drop_player_name": "Eury Perez",
    }]

    targets = await service._get_waiver_targets(user_id="user-1", prefs=MagicMock())

    assert len(targets) == 1
    assert targets[0].percent_owned == pytest.approx(44.2)
    assert targets[0].priority_score == pytest.approx(1.25)
    assert targets[0].reason == "Need score: 1.25 | Drop: Eury Perez"