"""
Tests for TeamContext (team_context.py) and WaiverValuationService (waiver_valuation_service.py).

All DB interactions are mocked -- no real connections required.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from backend.fantasy_baseball.team_context import TeamContext
from backend.fantasy_baseball.waiver_valuation_service import WaiverValuationService


# ===========================================================================
# Helpers
# ===========================================================================

def _make_projection(**kwargs) -> MagicMock:
    """Return a MagicMock that behaves like a CanonicalProjection row."""
    defaults = {
        "player_type": "BATTER",
        "projected_pa": 500.0,
        "projected_ip": None,
        "projection_date": date(2026, 5, 1),
        "proj_r": 70,
        "proj_hr": 20,
        "proj_rbi": 70,
        "proj_sb": 12,
        "proj_avg": 0.270,
        "proj_ops": 0.780,
        "proj_w": None,
        "proj_k": None,
        "proj_sv": None,
        "proj_era": None,
        "proj_whip": None,
        "proj_k9": None,
    }
    defaults.update(kwargs)
    proj = MagicMock()
    for attr, val in defaults.items():
        setattr(proj, attr, val)
    return proj


def _make_pitcher_projection(**kwargs) -> MagicMock:
    defaults = {
        "player_type": "PITCHER",
        "projected_pa": None,
        "projected_ip": 180.0,
        "projection_date": date(2026, 5, 1),
        "proj_r": None,
        "proj_hr": None,
        "proj_rbi": None,
        "proj_sb": None,
        "proj_avg": None,
        "proj_ops": None,
        "proj_w": 12,
        "proj_k": 180,
        "proj_sv": 0,
        "proj_era": 3.50,
        "proj_whip": 1.20,
        "proj_k9": 9.0,
    }
    defaults.update(kwargs)
    proj = MagicMock()
    for attr, val in defaults.items():
        setattr(proj, attr, val)
    return proj


# ===========================================================================
# TestTeamContext
# ===========================================================================

class TestTeamContext:
    def test_build_computes_pa_denominator(self):
        ctx = TeamContext.build(
            roster_player_ids=[1, 2, 3],
            projected_pa_by_player={1: 400.0, 2: 500.0, 3: 600.0},
            projected_ip_by_player={},
        )
        assert ctx.rate_pa_denominator == pytest.approx(1500.0)

    def test_build_computes_ip_denominator(self):
        ctx = TeamContext.build(
            roster_player_ids=[10, 11],
            projected_pa_by_player={},
            projected_ip_by_player={10: 150.0, 11: 180.0},
        )
        assert ctx.rate_ip_denominator == pytest.approx(330.0)

    def test_batter_pa_share_correct(self):
        ctx = TeamContext.build(
            roster_player_ids=[1, 2, 3],
            projected_pa_by_player={1: 400.0, 2: 500.0, 3: 600.0},
            projected_ip_by_player={},
        )
        # player 2: 500 / 1500 = 0.333...
        assert ctx.batter_pa_share(2) == pytest.approx(500.0 / 1500.0, rel=1e-4)

    def test_pitcher_ip_share_correct(self):
        ctx = TeamContext.build(
            roster_player_ids=[10, 11],
            projected_pa_by_player={},
            projected_ip_by_player={10: 150.0, 11: 180.0},
        )
        assert ctx.pitcher_ip_share(11) == pytest.approx(180.0 / 330.0, rel=1e-4)

    def test_is_quarantined_true(self):
        ctx = TeamContext.build(
            roster_player_ids=[],
            projected_pa_by_player={},
            projected_ip_by_player={},
            quarantined_player_ids={42},
        )
        assert ctx.is_quarantined(42) is True

    def test_is_quarantined_false(self):
        ctx = TeamContext.build(
            roster_player_ids=[],
            projected_pa_by_player={},
            projected_ip_by_player={},
            quarantined_player_ids={42},
        )
        assert ctx.is_quarantined(99) is False

    def test_empty_roster_denominators_zero(self):
        ctx = TeamContext.build(
            roster_player_ids=[],
            projected_pa_by_player={},
            projected_ip_by_player={},
        )
        assert ctx.rate_pa_denominator == pytest.approx(0.0)
        assert ctx.rate_ip_denominator == pytest.approx(0.0)

    def test_pa_share_zero_when_denominator_zero(self):
        ctx = TeamContext.build(
            roster_player_ids=[],
            projected_pa_by_player={},
            projected_ip_by_player={},
        )
        assert ctx.batter_pa_share(999) == pytest.approx(0.0)


# ===========================================================================
# TestBuildTeamContext
# ===========================================================================

class TestBuildTeamContext:
    def _make_service(self, query_return_value):
        """Return a WaiverValuationService whose DB query chain returns the given value."""
        db = MagicMock()
        (
            db.query.return_value
            .filter.return_value
            .order_by.return_value
            .first.return_value
        ) = query_return_value
        return WaiverValuationService(db)

    def test_uses_projected_pa_from_canonical(self):
        proj = _make_projection(player_type="BATTER", projected_pa=550.0)
        svc = self._make_service(proj)
        ctx = svc.build_team_context([101])
        assert ctx.projected_pa_by_player.get(101) == pytest.approx(550.0)

    def test_uses_projected_ip_from_canonical(self):
        proj = _make_pitcher_projection(projected_ip=180.0)
        svc = self._make_service(proj)
        ctx = svc.build_team_context([202])
        assert ctx.projected_ip_by_player.get(202) == pytest.approx(180.0)

    def test_fallback_pa_when_no_canonical(self):
        svc = self._make_service(None)
        ctx = svc.build_team_context([303])
        # Unknown type -> universal PA fallback of 450
        assert ctx.projected_pa_by_player.get(303) == pytest.approx(450.0)

    def test_quarantined_player_excluded(self):
        proj = _make_projection(player_type="BATTER", projected_pa=500.0)
        svc = self._make_service(proj)
        ctx = svc.build_team_context([42], quarantined_ids={42})
        assert 42 not in ctx.roster_player_ids
        assert 42 not in ctx.projected_pa_by_player


# ===========================================================================
# TestAddDropSurplus
# ===========================================================================

class TestAddDropSurplus:
    def _make_service_two_projections(self, add_proj, drop_proj):
        """
        Return a WaiverValuationService whose _latest_projection method
        returns add_proj for the first call and drop_proj for the second.
        """
        db = MagicMock()
        svc = WaiverValuationService(db)
        svc._latest_projection = MagicMock(side_effect=[add_proj, drop_proj])
        return svc

    def _empty_context(self) -> TeamContext:
        return TeamContext.build(
            roster_player_ids=[],
            projected_pa_by_player={},
            projected_ip_by_player={},
        )

    def test_returns_no_data_when_add_missing(self):
        svc = self._make_service_two_projections(None, _make_projection())
        result = svc.add_drop_surplus(1, 2, self._empty_context())
        assert result == {"status": "no_data"}

    def test_returns_no_data_when_drop_missing(self):
        svc = self._make_service_two_projections(_make_projection(), None)
        result = svc.add_drop_surplus(1, 2, self._empty_context())
        assert result == {"status": "no_data"}

    def test_positive_surplus_when_add_better(self):
        add_proj = _make_projection(player_type="BATTER", proj_hr=30)
        drop_proj = _make_projection(player_type="BATTER", proj_hr=10)
        svc = self._make_service_two_projections(add_proj, drop_proj)
        ctx = TeamContext.build(
            roster_player_ids=[1],
            projected_pa_by_player={1: 500.0},
            projected_ip_by_player={},
        )
        result = svc.add_drop_surplus(1, 2, ctx)
        assert result["category_deltas"]["HR"] == pytest.approx(20.0)

    def test_surplus_score_present(self):
        add_proj = _make_projection(player_type="BATTER")
        drop_proj = _make_projection(player_type="BATTER")
        svc = self._make_service_two_projections(add_proj, drop_proj)
        result = svc.add_drop_surplus(1, 2, self._empty_context())
        assert "surplus_score" in result

    def test_data_source_canonical_when_both_found(self):
        add_proj = _make_projection(player_type="BATTER")
        drop_proj = _make_projection(player_type="BATTER")
        svc = self._make_service_two_projections(add_proj, drop_proj)
        result = svc.add_drop_surplus(1, 2, self._empty_context())
        assert result.get("data_source") == "canonical"
