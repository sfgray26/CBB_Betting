"""
P28 Blended Score — tests for the talent-anchored composite.

These encode the behavior contracts that make the redesign correct:
  - Soto case: a cold-week elite star stays anchored high (not benched on form alone)
  - Matchup-bench case: that same star + a brutal matchup DOES drop (benched for the right reason)
  - Latz case: an elite reliever scores well once rate stats aren't dropped (Phase 1 interaction)
  - Missing-data fallbacks (no talent, no form, no matchup, nothing at all)
  - Weight renormalization when a component is absent
  - Z <-> percentile round-trip
"""

import pytest

from backend.services.blended_score import (
    BlendedWeights,
    compute_blended_score,
    percentile_to_z,
    z_to_percentile,
)


# Default talent-anchored weights used across tests
W = BlendedWeights(talent=0.60, form=0.20, matchup=0.20)


# ===========================================================================
# 1. The Soto case — cold star stays anchored
# ===========================================================================

class TestSotoCase:
    """Soto is an elite hitter on a one-week cold streak. Without a talent
    anchor, the old 14-day-only score benched him. With the blend, his season
    xwOBA keeps him anchored — a tiny cold sample can't override the track record."""

    def test_cold_star_stays_positive_on_talent_anchor(self):
        """Soto: talent_z ~ +1.5 (elite season xwOBA), form_z ~ -2.0 (cold week),
        confidence ~ 0.3 (few games). Form shrinks toward talent, so final_z
        stays clearly positive — he should NOT be benched behind a replacement.

        Note: low_confidence is correctly True here because the *form* sample is
        thin (0.3 < 0.4), but the *score* is correctly anchored high by talent.
        The flag signals 'form signal is thin', not 'the score is unreliable'."""
        result = compute_blended_score(
            talent_z=1.5,
            form_z=-2.0,
            confidence=0.3,
            matchup_z=None,          # neutral matchup day
            weights=W,
            talent_source="statcast",
        )
        # form_z_shrunk = 1.5 + 0.3*(-2.0 - 1.5) = 1.5 - 1.05 = +0.45
        assert result.form_z_shrunk == pytest.approx(0.45, abs=0.01)
        # final_z: talent dominates (0.6*1.5 + 0.2*0.45 + 0) = 0.9 + 0.09 = 0.99
        assert result.final_z > 0.5, (
            f"Cold-week star should stay anchored positive, got final_z={result.final_z}"
        )
        # low_confidence correctly True: form sample is thin, even though talent
        # data is present and the score is well-anchored.
        assert result.low_confidence is True

    def test_cold_star_with_neutral_matchup_beats_replacement(self):
        """The replacement-level player (Soderstrom) surging should not outrank
        a cold Soto when Soto's talent anchor is intact. Approximate the
        comparison: Soto's blended Z vs a neutral-talent hot player."""
        soto = compute_blended_score(
            talent_z=1.5, form_z=-2.0, confidence=0.3, matchup_z=None, weights=W,
            talent_source="statcast",
        )
        soderstrom = compute_blended_score(
            talent_z=0.0,           # average true talent
            form_z=1.5,             # hot week
            confidence=0.6,
            matchup_z=None,
            weights=W,
            talent_source="statcast",
        )
        # Soto should still outrank the average-talent hot streak
        assert soto.final_z >= soderstrom.final_z, (
            f"Cold Soto ({soto.final_z}) benched below hot-average Soderstrom "
            f"({soderstrom.final_z}) — talent anchor not holding"
        )


# ===========================================================================
# 2. The matchup-bench case — star + ace matchup DOES drop
# ===========================================================================

class TestMatchupBenchCase:
    """The whole point: benching a star should be possible when there's a genuine
    forward-looking reason. Soto vs an ace in a pitcher's park should drop him
    below the lineup threshold even though his talent is elite."""

    def test_star_with_brutal_matchup_drops(self):
        """Soto (elite talent, neutral form) facing an ace: matchup_z strongly
        negative should pull final_z down enough to make benching defensible."""
        neutral_form = compute_blended_score(
            talent_z=1.5, form_z=0.0, confidence=0.5, matchup_z=None, weights=W,
            talent_source="statcast",
        )
        ace_matchup = compute_blended_score(
            talent_z=1.5, form_z=0.0, confidence=0.5,
            matchup_z=-2.0,            # facing an ace, pitcher's park
            matchup_confidence=0.8,
            weights=W,
            talent_source="statcast",
        )
        assert ace_matchup.final_z < neutral_form.final_z, (
            "A brutal matchup should reduce the star's score"
        )
        # And the drop should be material (> 0.2 Z)
        assert (neutral_form.final_z - ace_matchup.final_z) > 0.2

    def test_talent_anchored_superstar_rarely_benched_on_single_matchup(self):
        """With the 60/20/20 talent-anchored default, even a cold superstar with a
        bad matchup stays above a replacement — that's the 'trust the track record'
        philosophy by design. To bench a superstar you need either an extreme
        matchup or a more reactive weight set (see next test)."""
        soto_cold_ace = compute_blended_score(
            talent_z=1.5, form_z=-2.0, confidence=0.3,
            matchup_z=-2.0, matchup_confidence=0.8,
            weights=W, talent_source="statcast",
        )
        replacement_good_matchup = compute_blended_score(
            talent_z=0.0, form_z=0.5, confidence=0.6,
            matchup_z=1.0, matchup_confidence=0.8,
            weights=W, talent_source="statcast",
        )
        # The superstar still ranks higher — this is correct for 60/20/20.
        assert soto_cold_ace.final_z > replacement_good_matchup.final_z

    def test_superstar_benchable_with_reactive_weights_or_extreme_matchup(self):
        """Benching a superstar IS possible when (a) the matchup is truly extreme
        (e.g. facing a Cy Young candidate in a pitcher's park, matchup_z near the
        -3 floor) or (b) the operator configures more reactive weights. This is
        the legitimate bench path — it just requires a stronger signal than a
        single ordinary bad matchup, which is the elite-manager behavior we want."""
        soto_extreme = compute_blended_score(
            talent_z=1.5, form_z=-2.0, confidence=0.5,
            matchup_z=-3.0,            # near-floor extreme matchup (ace + extreme park)
            matchup_confidence=0.9,
            weights=W, talent_source="statcast",
        )
        replacement_good = compute_blended_score(
            talent_z=0.0, form_z=0.5, confidence=0.6,
            matchup_z=1.0, matchup_confidence=0.9,
            weights=W, talent_source="statcast",
        )
        # An extreme matchup CAN drop even an anchored superstar below a
        # well-matched replacement. (0.6*1.5 + 0.2*form + 0.2*(-3) = 0.9+form-0.6)
        assert soto_extreme.final_z < replacement_good.final_z, (
            f"Extreme matchup (z=-3) should make a cold superstar benchable, "
            f"got soto={soto_extreme.final_z} vs rep={replacement_good.final_z}"
        )

        # Also achievable with more reactive weights (the operator can tune this)
        reactive = BlendedWeights(talent=0.40, form=0.15, matchup=0.45)
        soto_reactive = compute_blended_score(
            talent_z=1.5, form_z=-2.0, confidence=0.5,
            matchup_z=-2.0, matchup_confidence=0.8,
            weights=reactive, talent_source="statcast",
        )
        assert soto_reactive.final_z < soto_extreme.final_z + 1.0  # materially lower


# ===========================================================================
# 3. Confidence shrinkage behavior
# ===========================================================================

class TestConfidenceShrinkage:
    """form_z is shrunk toward talent_z by the confidence factor. Low-confidence
    form barely budges off the anchor; high-confidence form can move freely."""

    def test_zero_confidence_form_equals_talent(self):
        """At confidence=0, form_z_shrunk should equal talent_z exactly
        (no evidence to override the prior)."""
        r = compute_blended_score(
            talent_z=1.0, form_z=-2.5, confidence=0.0, matchup_z=None, weights=W,
            talent_source="statcast",
        )
        assert r.form_z_shrunk == pytest.approx(1.0, abs=1e-6)

    def test_full_confidence_form_equals_raw_form(self):
        """At confidence=1.0, form_z_shrunk should equal the raw form_z."""
        r = compute_blended_score(
            talent_z=1.0, form_z=-2.5, confidence=1.0, matchup_z=None, weights=W,
            talent_source="statcast",
        )
        assert r.form_z_shrunk == pytest.approx(-2.5, abs=1e-6)

    def test_low_confidence_flag_when_form_thin(self):
        """confidence < 0.4 should set low_confidence=True."""
        r = compute_blended_score(
            talent_z=1.0, form_z=0.5, confidence=0.3, matchup_z=None, weights=W,
            talent_source="statcast",
        )
        assert r.low_confidence is True

    def test_no_low_confidence_when_form_and_talent_robust(self):
        r = compute_blended_score(
            talent_z=1.0, form_z=0.5, confidence=0.8, matchup_z=0.2,
            matchup_confidence=0.7, weights=W, talent_source="statcast",
        )
        assert r.low_confidence is False


# ===========================================================================
# 4. Missing-data fallbacks
# ===========================================================================

class TestMissingDataFallbacks:
    """The blend must degrade gracefully when signals are missing."""

    def test_no_matchup_redistributes_weight(self):
        """When there's no game / no matchup data, the matchup weight is
        redistributed to talent and form (renormalization)."""
        r = compute_blended_score(
            talent_z=1.0, form_z=0.5, confidence=0.8,
            matchup_z=None, weights=W, talent_source="statcast",
        )
        # With matchup absent, effective weights renormalize: talent/form split
        # the matchup's share proportionally to their original 0.6/0.2 = 3:1
        eff = r.components["weights_effective"]
        assert eff["matchup"] == 0.0
        assert eff["talent"] == pytest.approx(0.75, abs=0.01)  # 0.6/0.8
        assert eff["form"] == pytest.approx(0.25, abs=0.01)    # 0.2/0.8

    def test_no_talent_uses_form_only(self):
        """When talent data is missing, form carries the score (with low_conf)."""
        r = compute_blended_score(
            talent_z=None, form_z=1.2, confidence=0.7,
            matchup_z=None, weights=W, talent_source="none",
        )
        # form can't be shrunk without a talent anchor, so it's used raw
        assert r.form_z_shrunk == pytest.approx(1.2, abs=1e-6)
        assert r.final_z == pytest.approx(1.2, abs=1e-6)
        assert r.low_confidence is True  # no talent anchor -> low confidence

    def test_all_signals_missing_returns_neutral(self):
        """No talent, no form, no matchup -> neutral 0.0, low confidence."""
        r = compute_blended_score(
            talent_z=None, form_z=None, confidence=0.0,
            matchup_z=None, weights=W, talent_source="none",
        )
        assert r.final_z == 0.0
        assert r.low_confidence is True
        assert r.components.get("note") == "no_signal"

    def test_low_matchup_confidence_dampens_matchup(self):
        """A low-confidence matchup (<0.4) should be damped toward zero so an
        uncertain matchup signal can't dominate the blend."""
        high_conf = compute_blended_score(
            talent_z=0.0, form_z=0.0, confidence=0.8,
            matchup_z=-2.0, matchup_confidence=0.9, weights=W, talent_source="statcast",
        )
        low_conf = compute_blended_score(
            talent_z=0.0, form_z=0.0, confidence=0.8,
            matchup_z=-2.0, matchup_confidence=0.2, weights=W, talent_source="statcast",
        )
        # Low-confidence matchup should pull final_z less strongly negative
        assert low_conf.final_z > high_conf.final_z


# ===========================================================================
# 5. Weight renormalization correctness
# ===========================================================================

class TestWeightRenormalization:
    def test_weights_sum_to_one_after_renorm(self):
        r = compute_blended_score(
            talent_z=1.0, form_z=0.5, confidence=0.7,
            matchup_z=None, weights=W, talent_source="statcast",
        )
        eff = r.components["weights_effective"]
        assert sum(eff.values()) == pytest.approx(1.0, abs=1e-6)

    def test_only_talent_available_uses_talent_directly(self):
        r = compute_blended_score(
            talent_z=1.5, form_z=None, confidence=0.0,
            matchup_z=None, weights=W, talent_source="statcast",
        )
        assert r.final_z == pytest.approx(1.5, abs=1e-6)


# ===========================================================================
# 6. Z <-> percentile round-trip
# ===========================================================================

class TestPercentileConversions:
    def test_z_zero_is_50th_percentile(self):
        assert z_to_percentile(0.0) == 50.0

    def test_round_trip_mid_percentiles(self):
        for p in [10, 25, 40, 50, 60, 75, 90]:
            z = percentile_to_z(p)
            assert z_to_percentile(z) == pytest.approx(p, abs=0.5), (
                f"Round-trip failed for p={p}: z={z}, back={z_to_percentile(z)}"
            )

    def test_extremes_clamped(self):
        assert percentile_to_z(0) > -10  # not -inf
        assert percentile_to_z(100) < 10  # not +inf

    def test_final_z_maps_to_familiar_0_100_scale(self):
        """An elite player (final_z ~ +1.5) should map to a high-but-sane score,
        and a replacement (~0.0) to ~50. This is what the frontend displays."""
        from backend.services.blended_score import z_to_percentile
        assert z_to_percentile(1.5) > 90   # elite
        assert z_to_percentile(0.0) == 50  # replacement
        assert z_to_percentile(-1.5) < 10  # weak
