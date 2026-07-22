"""
P28 Scoring Engine — rate-floor imputation tests.

Verifies the fix for the reliever denominator-shrink distortion ("Latz bug"):
when a pitcher is below MIN_RATE_IP, their rate categories (ERA/WHIP/K9) are
imputed as neutral 0.0 rather than dropped to None. Dropping shrank the weighted-
mean denominator and collapsed an elite reliever's composite onto counting stats
only, causing straight inversions (Latz 1.57 ERA scoring below Williams 4.00 ERA).

All tests use plain SimpleNamespace stubs — zero DB or I/O dependencies.
"""

from datetime import date
from types import SimpleNamespace

import pytest

from backend.services import scoring_engine
from backend.services.scoring_engine import (
    MIN_RATE_IP,
    PlayerScoreResult,
    compute_league_zscores,
)


AS_OF = date(2026, 7, 22)
WINDOW = 14


# ---------------------------------------------------------------------------
# Stub builders
# ---------------------------------------------------------------------------

def _starter(
    pid: int,
    era: float = 4.00,
    whip: float = 1.30,
    k9: float = 8.0,
    k_pit: float = 20.0,
    qs: float = 2.0,
    ip: float = 12.0,  # well above MIN_RATE_IP=8.0
    games: int = 10,
) -> SimpleNamespace:
    """A starting pitcher with enough IP to clear the rate floor."""
    return SimpleNamespace(
        bdl_player_id=pid,
        as_of_date=AS_OF,
        window_days=WINDOW,
        games_in_window=games,
        w_ab=None,
        w_ip=ip,
        w_era=era,
        w_whip=whip,
        w_k_per_9=k9,
        w_strikeouts_pit=k_pit,
        w_qs=qs,
        # hitter fields absent
        w_home_runs=None, w_rbi=None, w_stolen_bases=None,
        w_net_stolen_bases=None, w_runs=None, w_hits=None, w_tb=None,
        w_strikeouts_bat=None, w_avg=None, w_obp=None, w_ops=None,
        w_doubles=None, w_triples=None, w_walks=None, w_caught_stealing=None,
    )


def _reliever(
    pid: int,
    era: float = 1.50,
    whip: float = 0.80,
    k9: float = 12.0,
    k_pit: float = 8.0,
    qs: float = 0.0,
    ip: float = 5.0,  # below MIN_RATE_IP=8.0
    games: int = 8,
) -> SimpleNamespace:
    """A short reliever below the IP floor — the Latz profile."""
    return _starter(pid, era=era, whip=whip, k9=k9, k_pit=k_pit, qs=qs, ip=ip, games=games)


def _varied_starters() -> list:
    """A cohort of starters with VARIED stats so rate categories are non-degenerate
    (spread > 0) and actually get computed. Identical values across the cohort
    would make every category degenerate and skip it for everyone."""
    return [
        _starter(pid=1, era=3.00, whip=1.05, k9=10.0, k_pit=25.0, qs=3.0),
        _starter(pid=2, era=3.80, whip=1.20, k9=8.5, k_pit=20.0, qs=2.0),
        _starter(pid=3, era=4.50, whip=1.35, k9=7.0, k_pit=15.0, qs=1.0),
        _starter(pid=4, era=5.20, whip=1.50, k9=6.0, k_pit=12.0, qs=0.0),
        _starter(pid=5, era=4.10, whip=1.28, k9=8.0, k_pit=18.0, qs=2.0),
    ]


def _result_for(results: list[PlayerScoreResult], pid: int) -> PlayerScoreResult:
    for r in results:
        if r.bdl_player_id == pid:
            return r
    raise KeyError(f"No result for pid={pid}")


# ===========================================================================
# 1. Sub-floor reliever: rate categories imputed 0.0, not None
# ===========================================================================

class TestRateFloorImputation:
    """The core P28 fix — impute neutral Z instead of dropping the category."""

    def test_sub_floor_reliever_gets_zero_not_none_for_rate_categories(self):
        """A reliever below MIN_RATE_IP should have z_era/z_whip/z_k_per_9 == 0.0,
        not None. These are 'league-average, unconfirmed' placeholders."""
        # Varied starters clear the floor (cohort computes the rate categories)
        starters = _varied_starters()
        # 1 elite reliever below the floor
        reliever = _reliever(pid=99, era=0.50, whip=0.50, k9=15.0)
        rows = starters + [reliever]

        results = compute_league_zscores(rows, AS_OF, WINDOW)
        r = _result_for(results, 99)

        # BEFORE the fix these were None; now they are 0.0 (imputed neutral)
        assert r.z_era == 0.0, f"Expected imputed z_era=0.0, got {r.z_era}"
        assert r.z_whip == 0.0, f"Expected imputed z_whip=0.0, got {r.z_whip}"
        assert r.z_k_per_9 == 0.0, f"Expected imputed z_k_per_9=0.0, got {r.z_k_per_9}"

    def test_imputed_categories_tracked_for_explainability(self):
        """The imputed category keys must be recorded for downstream explainability."""
        starters = _varied_starters()
        reliever = _reliever(pid=99)
        rows = starters + [reliever]

        results = compute_league_zscores(rows, AS_OF, WINDOW)
        r = _result_for(results, 99)

        assert set(r.imputed_categories) == {"z_era", "z_whip", "z_k_per_9"}, (
            f"Expected all 3 pitcher rate categories imputed, got {r.imputed_categories}"
        )

    def test_starters_above_floor_have_no_imputed_categories(self):
        """A starter clearing MIN_RATE_IP must be completely unaffected — real Z values."""
        starters = _varied_starters()
        results = compute_league_zscores(starters, AS_OF, WINDOW)
        for r in results:
            assert r.imputed_categories == [], (
                f"Starter pid={r.bdl_player_id} should have no imputed categories, "
                f"got {r.imputed_categories}"
            )
            # Real (non-imputed) values present
            assert r.z_era is not None, (
                f"Starter pid={r.bdl_player_id} should have a real z_era, got None"
            )

    def test_starters_keep_real_rate_z_values(self):
        """Starters above the floor get actual computed Z-scores, not imputed 0.0.
        The best-ERA starter should have a clearly positive z_era."""
        starters = _varied_starters()
        results = compute_league_zscores(starters, AS_OF, WINDOW)
        # pid=1 has the best ERA (3.00) in the cohort
        r = _result_for(results, 1)
        assert r.z_era > 0.0, f"Best-ERA starter should have positive z_era, got {r.z_era}"
        assert r.imputed_categories == []


# ===========================================================================
# 2. Composite stability — the denominator no longer shrinks
# ===========================================================================

class TestCompositeDenominatorStability:
    """The distortion the fix targets: dropping categories shrank the weighted-mean
    denominator. With imputation, an elite reliever is scored on the same category
    count as a starter."""

    def test_elite_reliever_outscores_mediocre_starter(self):
        """The Latz/Williams inversion: an elite RP (low ERA, low WHIP, high K9)
        should score at least as well as a mediocre SP, once rate categories are
        imputed neutral rather than dropped.

        Before the fix, the reliever lost all rate categories and could score
        BELOW the mediocre starter purely from denominator shrinkage."""
        # Varied starters to establish a non-degenerate cohort
        mediocre = _varied_starters()
        # 1 elite reliever (Latz profile): tiny IP but dominant rates
        elite_rp = _reliever(pid=50, era=0.50, whip=0.60, k9=14.0, k_pit=10.0)
        rows = mediocre + [elite_rp]

        results = compute_league_zscores(rows, AS_OF, WINDOW)
        rp_score = _result_for(results, 50).score_0_100
        sp_scores = [_result_for(results, i).score_0_100 for i in range(1, 6)]

        # The elite reliever should NOT score below the mediocre starters.
        # (Before the fix, the RP could land in the bottom of the cohort.)
        assert rp_score >= min(sp_scores), (
            f"Elite reliever (score={rp_score}) scored below the worst mediocre "
            f"starter (min={min(sp_scores)}) — denominator distortion not fixed"
        )


# ===========================================================================
# 3. Flag-off parity — disable flag reverts to legacy None-drop behavior
# ===========================================================================

class TestDisableFlagRevertsBehavior:
    """The scoring.disable_rate_imputation flag must fully revert to the pre-P28
    behavior (None-drop) so we can roll back without a redeploy."""

    def test_disable_flag_restores_none_drop(self, monkeypatch):
        """When scoring.disable_rate_imputation is True, sub-floor rate categories
        go back to None and imputed_categories is empty."""
        monkeypatch.setattr(
            scoring_engine, "is_flag_enabled",
            lambda flag: flag == "scoring.disable_rate_imputation",
        )

        starters = _varied_starters()
        reliever = _reliever(pid=99)
        rows = starters + [reliever]

        results = compute_league_zscores(rows, AS_OF, WINDOW)
        r = _result_for(results, 99)

        # Legacy behavior: rate categories are None (dropped)
        assert r.z_era is None, f"Flag-off should restore z_era=None, got {r.z_era}"
        assert r.z_whip is None
        assert r.z_k_per_9 is None
        assert r.imputed_categories == [], (
            f"Flag-off should produce no imputed categories, got {r.imputed_categories}"
        )

    def test_default_flag_state_is_imputation_on(self, monkeypatch):
        """With no flags set (production default), imputation must be ON."""
        monkeypatch.setattr(scoring_engine, "is_flag_enabled", lambda flag: False)

        starters = _varied_starters()
        reliever = _reliever(pid=99)
        rows = starters + [reliever]

        results = compute_league_zscores(rows, AS_OF, WINDOW)
        r = _result_for(results, 99)

        assert r.z_era == 0.0, "Default (no flag) should impute z_era=0.0"
        assert len(r.imputed_categories) == 3
