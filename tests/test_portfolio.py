"""
Tests for portfolio-level Kelly sizing
Run with: pytest tests/test_portfolio.py -v
"""

import pytest
from backend.services.portfolio import PortfolioManager, PortfolioPosition


class TestPortfolioBasics:
    """Test basic portfolio state tracking"""

    def test_initial_state(self):
        pm = PortfolioManager(starting_bankroll=1000)

        assert pm.current_bankroll == 1000
        assert pm.drawdown_pct == 0.0
        assert not pm.is_halted
        assert pm.total_exposure_pct == 0.0

    def test_drawdown_calculation(self):
        pm = PortfolioManager(starting_bankroll=1000)
        pm.update_bankroll(850)

        assert pm.drawdown_pct == 15.0

    def test_drawdown_halt(self):
        pm = PortfolioManager(starting_bankroll=1000, max_drawdown_pct=15.0)
        pm.update_bankroll(849)

        assert pm.is_halted

    def test_no_halt_at_boundary(self):
        pm = PortfolioManager(starting_bankroll=1000, max_drawdown_pct=15.0)
        pm.update_bankroll(851)

        assert not pm.is_halted


class TestPortfolioExposure:
    """Test total exposure tracking"""

    def test_single_position(self):
        pm = PortfolioManager(starting_bankroll=1000)
        pm.add_position(PortfolioPosition(
            game_id=1, pick="Duke -4.5",
            kelly_fractional=0.05, recommended_units=1.5,
            edge_conservative=0.02,
        ))

        assert pm.total_exposure_pct == 1.5

    def test_multiple_positions(self):
        pm = PortfolioManager(starting_bankroll=1000)
        for i in range(5):
            pm.add_position(PortfolioPosition(
                game_id=i, pick=f"Team{i}",
                kelly_fractional=0.04, recommended_units=2.0,
                edge_conservative=0.02,
            ))

        assert pm.total_exposure_pct == 10.0

    def test_clear_settled(self):
        pm = PortfolioManager(starting_bankroll=1000)
        pm.add_position(PortfolioPosition(
            game_id=1, pick="Duke", kelly_fractional=0.05,
            recommended_units=1.5, edge_conservative=0.02,
        ))
        pm.add_position(PortfolioPosition(
            game_id=2, pick="UNC", kelly_fractional=0.05,
            recommended_units=1.5, edge_conservative=0.02,
        ))

        pm.clear_settled([1])
        assert pm.total_exposure_pct == 1.5


class TestPortfolioAdjustedKelly:
    """Test portfolio-adjusted Kelly sizing"""

    def test_no_adjustment_when_room(self):
        pm = PortfolioManager(
            starting_bankroll=1000,
            max_total_exposure_pct=15.0,
            max_single_bet_pct=3.0,
        )

        sizing = pm.adjust_kelly(raw_kelly_frac=0.05, raw_units=1.5)

        assert sizing.adjusted_units == 1.5
        assert sizing.scaling_factor == 1.0

    def test_single_bet_cap(self):
        pm = PortfolioManager(
            starting_bankroll=1000,
            max_single_bet_pct=2.0,
        )

        sizing = pm.adjust_kelly(raw_kelly_frac=0.10, raw_units=5.0)

        assert sizing.adjusted_units == 2.0

    def test_total_exposure_cap(self):
        pm = PortfolioManager(
            starting_bankroll=1000,
            max_total_exposure_pct=10.0,
            max_single_bet_pct=5.0,
        )

        # Fill 8% exposure
        for i in range(4):
            pm.add_position(PortfolioPosition(
                game_id=i, pick=f"Team{i}",
                kelly_fractional=0.04, recommended_units=2.0,
                edge_conservative=0.02,
            ))

        # Try to add 3% more — only 2% headroom
        sizing = pm.adjust_kelly(raw_kelly_frac=0.06, raw_units=3.0)

        assert sizing.adjusted_units == 2.0  # Capped at headroom

    def test_halt_returns_zero(self):
        pm = PortfolioManager(
            starting_bankroll=1000,
            max_drawdown_pct=10.0,
        )
        pm.update_bankroll(899)  # > 10% drawdown

        sizing = pm.adjust_kelly(raw_kelly_frac=0.05, raw_units=1.5)

        assert sizing.adjusted_units == 0.0
        assert "HALTED" in sizing.reason

    def test_conference_correlation_penalty(self):
        pm = PortfolioManager(
            starting_bankroll=1000,
            max_total_exposure_pct=20.0,
            conference_correlation=0.10,
        )

        # Add a Big Ten bet
        pm.add_position(PortfolioPosition(
            game_id=1, pick="Purdue -5",
            kelly_fractional=0.05, recommended_units=1.5,
            edge_conservative=0.02, conference="Big Ten",
        ))

        # Another Big Ten bet should get penalized
        sizing_corr = pm.adjust_kelly(
            raw_kelly_frac=0.05, raw_units=1.5, conference="Big Ten",
        )
        # An ACC bet should not
        sizing_diff = pm.adjust_kelly(
            raw_kelly_frac=0.05, raw_units=1.5, conference="ACC",
        )

        assert sizing_corr.adjusted_units < sizing_diff.adjusted_units

    def test_get_state(self):
        pm = PortfolioManager(starting_bankroll=1000)
        pm.add_position(PortfolioPosition(
            game_id=1, pick="Duke",
            kelly_fractional=0.05, recommended_units=1.5,
            edge_conservative=0.02,
        ))

        state = pm.get_state()

        assert state.current_bankroll == 1000
        assert len(state.positions) == 1
        assert state.total_exposure_pct == 1.5
        assert not state.is_halted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
