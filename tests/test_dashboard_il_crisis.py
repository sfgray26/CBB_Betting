"""Tests for IL crisis detection in _get_lineup_gaps."""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict


def _make_roster_player(name: str, selected_position: str, injury_status: str | None = None) -> dict:
    return {
        "name": name,
        "selected_position": selected_position,
        "positions": [selected_position],
        "injury_status": injury_status,
    }


def test_lineup_gap_has_action_url_field():
    """LineupGap dataclass must have an action_url field."""
    from backend.services.dashboard_service import LineupGap
    gap = LineupGap(
        position="ROSTER",
        severity="critical",
        message="test",
        action_url="/war-room/roster",
    )
    assert gap.action_url == "/war-room/roster"
    d = asdict(gap)
    assert "action_url" in d


def test_lineup_gap_action_url_defaults_none():
    """action_url defaults to None for normal gaps."""
    from backend.services.dashboard_service import LineupGap
    gap = LineupGap(position="OF", severity="warning", message="No eligible player")
    assert gap.action_url is None


@pytest.mark.asyncio
async def test_il_crisis_appended_when_three_injured_active():
    """_get_lineup_gaps must append a ROSTER EMERGENCY gap when 3+ injured active players."""
    from backend.services.dashboard_service import DashboardService

    roster = [
        _make_roster_player("Player A", "BN", injury_status="il"),
        _make_roster_player("Player B", "BN", injury_status="il10"),
        _make_roster_player("Player C", "OF", injury_status="out"),
        _make_roster_player("Healthy D", "1B"),
        _make_roster_player("Healthy E", "SS"),
    ]

    service = DashboardService.__new__(DashboardService)
    service.reliability_engine = MagicMock()
    service.reliability_engine.validate_yahoo_roster = MagicMock(
        return_value=MagicMock(is_valid=True, errors=[])
    )

    mock_client = MagicMock()
    mock_client.get_roster = MagicMock(return_value=roster)
    service._get_yahoo_client = MagicMock(return_value=mock_client)

    with patch("backend.services.dashboard_service.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db   # SessionLocal() called directly, not as context manager
        service._detect_pitcher_swap_gaps = MagicMock(return_value=[])

        gaps, _, _ = await service._get_lineup_gaps("user1", None)

    crisis_gaps = [g for g in gaps if g.position == "ROSTER"]
    assert len(crisis_gaps) == 1, f"Expected 1 ROSTER crisis gap, got {len(crisis_gaps)}: {gaps}"
    assert crisis_gaps[0].severity == "critical"
    assert "ROSTER EMERGENCY" in crisis_gaps[0].message
    assert crisis_gaps[0].action_url == "/war-room/roster"


@pytest.mark.asyncio
async def test_no_il_crisis_when_fewer_than_three_injured():
    """No ROSTER EMERGENCY gap when fewer than 3 injured active players."""
    from backend.services.dashboard_service import DashboardService

    roster = [
        _make_roster_player("Player A", "BN", injury_status="il"),
        _make_roster_player("Healthy B", "1B"),
        _make_roster_player("Healthy C", "SS"),
    ]

    service = DashboardService.__new__(DashboardService)
    service.reliability_engine = MagicMock()
    service.reliability_engine.validate_yahoo_roster = MagicMock(
        return_value=MagicMock(is_valid=True, errors=[])
    )
    mock_client = MagicMock()
    mock_client.get_roster = MagicMock(return_value=roster)
    service._get_yahoo_client = MagicMock(return_value=mock_client)

    with patch("backend.services.dashboard_service.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db
        service._detect_pitcher_swap_gaps = MagicMock(return_value=[])

        gaps, _, _ = await service._get_lineup_gaps("user1", None)

    crisis_gaps = [g for g in gaps if g.position == "ROSTER"]
    assert len(crisis_gaps) == 0


@pytest.mark.asyncio
async def test_injury_counts_normalize_overlay_longhand():
    """UAT 2026-07-17: overlay longhand statuses ("15-Day-IL", "Day-to-Day")
    must be counted as INJURED, not healthy. Kyle Harrison (15-Day-IL) and
    Soto/Perdomo (Day-to-Day) were slipping into the healthy bucket because the
    overlay status string didn't literally match the canonical short codes.
    """
    from backend.services.dashboard_service import DashboardService

    # Roster players keyed the way _get_injury_flags reads them: "status" + "name"
    roster = [
        # Healthy player — no status, no overlay
        {"name": "Healthy Aaron", "player_key": "k1", "selected_position": "1B", "status": ""},
        # IL15 via Yahoo short code
        {"name": "Kyle Harrison", "player_key": "k2", "selected_position": "SP", "status": "IL15"},
        # DTD via Yahoo short code
        {"name": "Juan Soto", "player_key": "k3", "selected_position": "RF", "status": "DTD"},
        # OUT
        {"name": "Gabe Perdomo", "player_key": "k4", "selected_position": "SS", "status": "OUT"},
    ]

    # Overlays keyed by player_key — emit LONGHAND statuses the way the DB does
    overlays = {
        "k2": MagicMock(status="15-Day-IL"),   # longhand IL15
        "k3": MagicMock(status="Day-to-Day"),  # longhand DTD
        "k4": MagicMock(status="OUT"),
    }

    service = DashboardService.__new__(DashboardService)
    service.reliability_engine = MagicMock()
    service.reliability_engine.validate_yahoo_roster = MagicMock(
        return_value=MagicMock(is_valid=True, errors=[])
    )
    mock_client = MagicMock()
    mock_client.get_roster = MagicMock(return_value=roster)
    service._get_yahoo_client = MagicMock(return_value=mock_client)

    with patch(
        "backend.services.dashboard_service.load_injury_overlays_for_yahoo_players",
        return_value=overlays,
    ), patch("backend.services.dashboard_service.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db

        flags, healthy, injured = await service._get_injury_flags("user1")

    # Only the truly healthy player lands in the healthy bucket.
    assert healthy == 1, f"Expected 1 healthy, got {healthy} (longhand overlay miscounted)"
    # The three injured players (IL15, DTD, OUT) all land in the injured bucket.
    assert injured == 3, f"Expected 3 injured, got {injured} (longhand overlay miscounted)"
    # None of the injured players are in an IL slot, so all three produce flags.
    assert len(flags) == 3
