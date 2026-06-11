"""Tests for DashboardService._detect_pitcher_swap_gaps() — Phase 2 sub-optimal placement."""

from datetime import date
from unittest.mock import MagicMock, patch

from backend.services.dashboard_service import (
    DashboardService,
    LineupGap,
    SUBOPTIMAL_SCORE_THRESHOLD,
)


def _make_service():
    """Instantiate DashboardService with all heavy deps patched out."""
    with patch("backend.services.dashboard_service.DailyLineupOptimizer"), \
         patch("backend.services.dashboard_service.WaiverEdgeDetector"), \
         patch("backend.services.dashboard_service.get_reliability_engine"):
        return DashboardService()


def _pid_mock(mappings):
    """Mock for db.query(PlayerIDMapping).filter(...).all() chain."""
    m = MagicMock()
    m.filter.return_value.all.return_value = mappings
    return m


def _ps_mock(scores):
    """Mock for db.query(PlayerScore).filter(...).order_by(...).all() chain."""
    m = MagicMock()
    m.filter.return_value.order_by.return_value.all.return_value = scores
    return m


class FakeMapping:
    def __init__(self, normalized_name, bdl_id):
        self.normalized_name = normalized_name
        self.bdl_id = bdl_id


class FakeScore:
    def __init__(self, bdl_player_id, score_0_100):
        self.bdl_player_id = bdl_player_id
        self.score_0_100 = score_0_100
        self.as_of_date = date.today()


def test_no_bench_pitchers_returns_empty():
    """No bench pitchers on the roster → no optimization gaps."""
    service = _make_service()
    roster = [
        {"name": "Eury Pérez", "positions": ["SP"], "selected_position": "SP", "status": None},
    ]
    db = MagicMock()
    result = service._detect_pitcher_swap_gaps(roster, db)
    assert result == []


def test_emits_optimization_gap_when_bench_scorer_higher():
    """Harrison (score 90) benched while Pérez (score 20) starts SP → optimization gap."""
    service = _make_service()
    roster = [
        {"name": "Eury Pérez",    "positions": ["SP"],        "selected_position": "SP", "status": None},
        {"name": "Kyle Harrison", "positions": ["SP", "RP"],  "selected_position": "BN", "status": None},
    ]
    db = MagicMock()
    mappings = [FakeMapping("eury pérez", 101), FakeMapping("kyle harrison", 102)]
    scores   = [FakeScore(101, 20.0), FakeScore(102, 90.0)]
    db.query.side_effect = [_pid_mock(mappings), _ps_mock(scores)]

    result = service._detect_pitcher_swap_gaps(roster, db)

    assert len(result) == 1
    gap = result[0]
    assert gap.severity == "optimization"
    assert gap.position == "SP"
    assert "Kyle Harrison" in gap.message
    assert "Eury Pérez" in gap.message
    assert gap.suggested_add == "Kyle Harrison"


def test_no_gap_when_score_difference_at_or_below_threshold():
    """Score gap exactly at SUBOPTIMAL_SCORE_THRESHOLD → no gap (must be strictly greater)."""
    service = _make_service()
    roster = [
        {"name": "Starter A", "positions": ["SP"], "selected_position": "SP", "status": None},
        {"name": "Bench B",   "positions": ["SP"], "selected_position": "BN", "status": None},
    ]
    db = MagicMock()
    mappings = [FakeMapping("starter a", 201), FakeMapping("bench b", 202)]
    scores   = [FakeScore(201, 50.0), FakeScore(202, 50.0 + SUBOPTIMAL_SCORE_THRESHOLD)]
    db.query.side_effect = [_pid_mock(mappings), _ps_mock(scores)]

    result = service._detect_pitcher_swap_gaps(roster, db)
    assert result == []


def test_db_error_returns_empty_list_without_raising():
    """DB exception inside the method → empty list, no crash."""
    service = _make_service()
    roster = [
        {"name": "Eury Pérez",    "positions": ["SP"], "selected_position": "SP", "status": None},
        {"name": "Kyle Harrison", "positions": ["SP"], "selected_position": "BN", "status": None},
    ]
    db = MagicMock()
    db.query.side_effect = Exception("DB connection lost")

    result = service._detect_pitcher_swap_gaps(roster, db)
    assert result == []
