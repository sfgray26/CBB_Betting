"""
Tests for _heuristic_style_from_rating in analysis.py.

Run with: pytest tests/test_heuristic_style.py -v
"""

import pytest
from backend.services.analysis import _heuristic_style_from_rating


class TestHeuristicStyleFromRating:
    """Validate the Four-Factor heuristic fallback used when the BartTorvik
    profile cache misses."""

    def test_none_input_returns_none(self):
        assert _heuristic_style_from_rating(None) is None

    # ── Absolute-efficiency scale (BartTorvik AdjOE ≈ 90–130) ──────────────

    def test_d1_average_efg_at_105(self):
        """AdjOE = 105 (D1 avg) → eFG% should be exactly the D1 baseline."""
        style = _heuristic_style_from_rating(105.0)
        assert style is not None
        assert style["efg_pct"] == pytest.approx(0.505, abs=0.002)

    def test_above_average_offense_raises_efg(self):
        """AdjOE = 115 → eFG% > D1 average."""
        style = _heuristic_style_from_rating(115.0)
        assert style["efg_pct"] > 0.505

    def test_below_average_offense_lowers_efg(self):
        """AdjOE = 95 → eFG% < D1 average."""
        style = _heuristic_style_from_rating(95.0)
        assert style["efg_pct"] < 0.505

    def test_elite_offense_clamped_at_0650(self):
        """Extremely high AdjOE should not push eFG% above the cap."""
        style = _heuristic_style_from_rating(200.0)
        assert style["efg_pct"] <= 0.650

    def test_weak_offense_clamped_at_0350(self):
        """Floor dropped from 0.400 to 0.350 to allow extreme-weak offenses."""
        style = _heuristic_style_from_rating(60.0)
        assert style["efg_pct"] >= 0.350
        assert style["efg_pct"] < 0.400  # Far below D1 avg but floored

    # ── Margin scale (KenPom AdjEM ≈ −30 to +40) ───────────────────────────

    def test_kenpom_positive_margin_raises_efg(self):
        """KenPom AdjEM = +20 (strong team) → eFG% above average."""
        style = _heuristic_style_from_rating(20.0)  # abs < 50 → margin path
        assert style["efg_pct"] > 0.505

    def test_kenpom_zero_margin_near_average(self):
        """KenPom AdjEM = 0 → estimated AdjO ≈ 105 → eFG% ≈ 0.505."""
        style = _heuristic_style_from_rating(0.0)
        assert style is not None
        assert style["efg_pct"] == pytest.approx(0.505, abs=0.002)

    def test_kenpom_negative_margin_lowers_efg(self):
        """KenPom AdjEM = −15 → eFG% below average."""
        style = _heuristic_style_from_rating(-15.0)
        assert style["efg_pct"] < 0.505

    # ── Output structure ────────────────────────────────────────────────────

    def test_all_required_keys_present(self):
        style = _heuristic_style_from_rating(105.0)
        for key in ("pace", "efg_pct", "to_pct", "ft_rate", "three_par"):
            assert key in style, f"Missing key: {key}"

    def test_tempo_and_shot_selection_baselines(self):
        """Pace, ft_rate, three_par remain D1 baselines; to_pct is dynamic."""
        style = _heuristic_style_from_rating(110.0)
        assert style["pace"]      == pytest.approx(68.0, abs=0.1)
        assert style["ft_rate"]   == pytest.approx(0.32, abs=0.001)
        assert style["three_par"] == pytest.approx(0.36, abs=0.001)
        # to_pct is now dynamic: 0.175 * (105 / 110) ≈ 0.1670
        assert style["to_pct"] == pytest.approx(0.175 * (105.0 / 110.0), abs=0.001)
        assert style["to_pct"] < 0.175  # Above-average offense → fewer turnovers

    # ── Defensive rating path ───────────────────────────────────────────────

    def test_def_rating_provided_produces_def_efg(self):
        """When def_rating is supplied, def_efg_pct should be in the output."""
        style = _heuristic_style_from_rating(115.0, def_rating_raw=98.0)
        assert "def_efg_pct" in style

    def test_elite_defense_lowers_def_efg(self):
        """Strong defensive AdjDE (low number = better) → lower def_efg_pct."""
        weak_def  = _heuristic_style_from_rating(105.0, def_rating_raw=115.0)
        elite_def = _heuristic_style_from_rating(105.0, def_rating_raw=92.0)
        assert elite_def["def_efg_pct"] < weak_def["def_efg_pct"]


class TestBuildProfileFromStyle:
    """Test _build_profile_from_style fallback builder in analysis.py (P1)."""

    def test_returns_team_play_style_instance(self):
        """Output is a TeamPlayStyle with correct team name."""
        from backend.services.analysis import _build_profile_from_style
        from backend.services.matchup_engine import TeamPlayStyle
        profile = _build_profile_from_style({"pace": 70.0, "efg_pct": 0.520}, "Duke")
        assert isinstance(profile, TeamPlayStyle)
        assert profile.team == "Duke"

    def test_four_factor_fields_populated(self):
        """All supplied four-factor stats are reflected in the profile."""
        from backend.services.analysis import _build_profile_from_style
        style = {
            "pace": 72.5,
            "efg_pct": 0.530,
            "to_pct": 0.165,
            "def_efg_pct": 0.490,
            "def_to_pct": 0.190,
        }
        profile = _build_profile_from_style(style, "Team")
        assert profile.pace == pytest.approx(72.5)
        assert profile.efg_pct == pytest.approx(0.530)
        assert profile.to_pct == pytest.approx(0.165)
        assert profile.def_efg_pct == pytest.approx(0.490)
        assert profile.def_to_pct == pytest.approx(0.190)

    def test_missing_efg_pct_yields_none(self):
        """efg_pct absent from style dict → profile.efg_pct is None (heuristic path)."""
        from backend.services.analysis import _build_profile_from_style
        profile = _build_profile_from_style({"pace": 68.0, "to_pct": 0.175}, "Team")
        assert profile.efg_pct is None

    def test_pbp_fields_remain_at_defaults(self):
        """PBP-derived fields (drop_coverage_pct, zone_pct) stay at 0 defaults."""
        from backend.services.analysis import _build_profile_from_style
        profile = _build_profile_from_style({"pace": 68.0}, "Team")
        assert profile.drop_coverage_pct == pytest.approx(0.0, abs=0.001)
        assert profile.zone_pct == pytest.approx(0.0, abs=0.001)
