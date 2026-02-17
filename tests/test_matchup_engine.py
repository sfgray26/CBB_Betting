"""
Tests for matchup engine play-style analysis
Run with: pytest tests/test_matchup_engine.py -v
"""

import pytest
from backend.services.matchup_engine import (
    MatchupEngine,
    TeamPlayStyle,
    TeamProfileCache,
)


class TestMatchupAdjustments:
    """Test that matchup-specific adjustments are computed correctly"""

    def test_identical_teams_no_adjustment(self):
        engine = MatchupEngine()
        home = TeamPlayStyle(team="TeamA")
        away = TeamPlayStyle(team="TeamB")

        adj = engine.analyze_matchup(home, away)

        # Identical default profiles should produce near-zero adjustments
        assert abs(adj.margin_adj) < 0.5
        assert adj.sd_adj < 0.5

    def test_pace_mismatch_increases_sd(self):
        engine = MatchupEngine()
        fast = TeamPlayStyle(team="Fast", pace=78.0)
        slow = TeamPlayStyle(team="Slow", pace=60.0)

        adj = engine.analyze_matchup(fast, slow)

        assert adj.sd_adj > 0
        assert 'pace_mismatch_sd' in adj.factors

    def test_three_point_vs_drop_creates_margin(self):
        engine = MatchupEngine()
        shooter = TeamPlayStyle(team="Shooter", three_par=0.45, three_fg_pct=0.38)
        dropper = TeamPlayStyle(team="Dropper", drop_coverage_pct=0.50)

        # Home = shooter, away = dropper (drop defence)
        adj = engine.analyze_matchup(shooter, dropper)

        assert 'home_3_vs_drop' in adj.factors
        assert adj.factors['home_3_vs_drop'] > 0  # Shooter gets margin boost

    def test_zone_vs_three_penalty(self):
        engine = MatchupEngine()
        zoner = TeamPlayStyle(team="Zone", zone_pct=0.50)
        sniper = TeamPlayStyle(team="Sniper", three_fg_pct=0.40)

        # Home = zone, away = sniper
        adj = engine.analyze_matchup(zoner, sniper)

        assert 'home_zone_vs_away_3' in adj.factors
        assert adj.factors['home_zone_vs_away_3'] < 0  # Zone gets punished

    def test_transition_gap(self):
        engine = MatchupEngine()
        runner = TeamPlayStyle(team="Runner", transition_freq=0.22, transition_ppp=1.15)
        halfcourt = TeamPlayStyle(team="Halfcourt", transition_freq=0.10, transition_ppp=1.00)

        adj = engine.analyze_matchup(runner, halfcourt)

        assert 'transition_gap' in adj.factors
        assert adj.factors['transition_gap'] > 0  # Runner gets advantage


class TestTeamProfileCache:
    """Test the team profile cache"""

    def test_set_and_get(self):
        cache = TeamProfileCache()
        profile = TeamPlayStyle(team="Duke", pace=72.0, three_par=0.38)

        cache.set("Duke", profile)

        assert cache.get("Duke") is not None
        assert cache.get("Duke").pace == 72.0

    def test_missing_team_returns_none(self):
        cache = TeamProfileCache()

        assert cache.get("Nonexistent") is None

    def test_has_profiles(self):
        cache = TeamProfileCache()
        assert not cache.has_profiles()

        cache.set("Duke", TeamPlayStyle(team="Duke"))
        assert cache.has_profiles()

    def test_len(self):
        cache = TeamProfileCache()
        cache.set("Duke", TeamPlayStyle(team="Duke"))
        cache.set("UNC", TeamPlayStyle(team="UNC"))

        assert len(cache) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
