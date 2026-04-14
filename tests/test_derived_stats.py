"""
Regression tests for backend.services.derived_stats.

These tests lock the NULL-propagation contract: any missing component produces
None, never a silent 0.0. They exist specifically to catch the anti-pattern
called out in the April 13 readiness assessment — `.get(key, 0.0)` defaults
polluting derived stats with fake zeros.
"""

import math

import pytest

from backend.services.derived_stats import (
    compute_avg,
    compute_era,
    compute_iso,
    compute_ops,
    compute_whip,
    parse_innings_pitched,
)


# ---------------------------------------------------------------------------
# parse_innings_pitched — BDL "6.2" convention
# ---------------------------------------------------------------------------

class TestParseInningsPitched:
    def test_none_returns_none(self):
        assert parse_innings_pitched(None) is None

    def test_empty_string_returns_none(self):
        assert parse_innings_pitched("") is None
        assert parse_innings_pitched("  ") is None

    def test_zero_innings_returns_none(self):
        """0 IP is mathematically undefined for WHIP/ERA — must be None."""
        assert parse_innings_pitched("0.0") is None
        assert parse_innings_pitched("0") is None
        assert parse_innings_pitched(0) is None
        assert parse_innings_pitched(0.0) is None

    def test_whole_innings(self):
        assert parse_innings_pitched("7.0") == pytest.approx(7.0)
        assert parse_innings_pitched("1.0") == pytest.approx(1.0)

    def test_one_out(self):
        # .1 = 1 out = 1/3 inning
        assert parse_innings_pitched("6.1") == pytest.approx(6 + 1 / 3.0)

    def test_two_outs(self):
        # .2 = 2 outs = 2/3 inning
        assert parse_innings_pitched("6.2") == pytest.approx(6 + 2 / 3.0)

    def test_invalid_outs_digit_returns_none(self):
        """BDL convention allows only .0/.1/.2 — '6.3' is garbage."""
        assert parse_innings_pitched("6.3") is None
        assert parse_innings_pitched("6.9") is None

    def test_non_numeric_returns_none(self):
        assert parse_innings_pitched("abc") is None
        assert parse_innings_pitched("N/A") is None

    def test_numeric_input_accepted(self):
        assert parse_innings_pitched(6.0) == pytest.approx(6.0)
        assert parse_innings_pitched(6) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# compute_ops
# ---------------------------------------------------------------------------

class TestComputeOps:
    def test_happy_path(self):
        assert compute_ops(0.350, 0.450) == pytest.approx(0.800)

    def test_none_obp_propagates(self):
        assert compute_ops(None, 0.450) is None

    def test_none_slg_propagates(self):
        assert compute_ops(0.350, None) is None

    def test_both_none_propagates(self):
        assert compute_ops(None, None) is None

    def test_zero_inputs_are_real_data(self):
        """OBP=0, SLG=0 → OPS=0 is a valid 'no hits' outcome, not NULL."""
        assert compute_ops(0.0, 0.0) == 0.0

    def test_int_inputs_accepted(self):
        assert compute_ops(0, 0) == 0.0


# ---------------------------------------------------------------------------
# compute_avg
# ---------------------------------------------------------------------------

class TestComputeAvg:
    def test_happy_path(self):
        assert compute_avg(30, 100) == pytest.approx(0.300)

    def test_zero_ab_returns_none(self):
        """Zero at-bats is undefined AVG, must be None (not 0.0)."""
        assert compute_avg(0, 0) is None
        assert compute_avg(1, 0) is None

    def test_none_hits_propagates(self):
        assert compute_avg(None, 100) is None

    def test_none_ab_propagates(self):
        assert compute_avg(30, None) is None

    def test_zero_hits_is_real_data(self):
        """0 hits in 10 AB → .000, not NULL."""
        assert compute_avg(0, 10) == 0.0


# ---------------------------------------------------------------------------
# compute_iso
# ---------------------------------------------------------------------------

class TestComputeIso:
    def test_happy_path(self):
        # ISO = SLG - AVG
        assert compute_iso(0.500, 0.300) == pytest.approx(0.200)

    def test_none_slg_propagates(self):
        assert compute_iso(None, 0.300) is None

    def test_none_avg_propagates(self):
        assert compute_iso(0.500, None) is None

    def test_zero_iso_is_real_data(self):
        """All-singles hitter → ISO=0 is valid, not NULL."""
        assert compute_iso(0.250, 0.250) == 0.0


# ---------------------------------------------------------------------------
# compute_whip
# ---------------------------------------------------------------------------

class TestComputeWhip:
    def test_happy_path_whole_innings(self):
        # (10 + 20) / 10.0 = 3.0
        assert compute_whip(10, 20, "10.0") == pytest.approx(3.0)

    def test_happy_path_partial_innings(self):
        # (1 + 5) / 6.667 = 0.9 approx
        ip = 6 + 2 / 3.0
        assert compute_whip(1, 5, "6.2") == pytest.approx(6.0 / ip)

    def test_zero_ip_returns_none(self):
        assert compute_whip(1, 2, "0.0") is None
        assert compute_whip(1, 2, 0) is None

    def test_none_walks_propagates(self):
        assert compute_whip(None, 5, "7.0") is None

    def test_none_hits_propagates(self):
        assert compute_whip(2, None, "7.0") is None

    def test_none_ip_propagates(self):
        assert compute_whip(2, 5, None) is None

    def test_invalid_ip_propagates(self):
        assert compute_whip(2, 5, "junk") is None

    def test_perfect_game(self):
        """0 walks + 0 hits over 9 IP → WHIP = 0.0 (valid, not NULL)."""
        assert compute_whip(0, 0, "9.0") == 0.0


# ---------------------------------------------------------------------------
# compute_era
# ---------------------------------------------------------------------------

class TestComputeEra:
    def test_happy_path(self):
        # 2 ER over 9 IP = 2.00 ERA
        assert compute_era(2, "9.0") == pytest.approx(2.00)

    def test_happy_path_partial_innings(self):
        # 3 ER over 6.2 IP = (3*9)/6.667 = 4.05
        expected = (3 * 9.0) / (6 + 2 / 3.0)
        assert compute_era(3, "6.2") == pytest.approx(expected)

    def test_zero_ip_returns_none(self):
        assert compute_era(1, "0.0") is None

    def test_zero_er_is_real_data(self):
        """0 ER over 9 IP → ERA = 0.00 (valid, not NULL)."""
        assert compute_era(0, "9.0") == 0.0

    def test_none_er_propagates(self):
        assert compute_era(None, "7.0") is None

    def test_none_ip_propagates(self):
        assert compute_era(2, None) is None


# ---------------------------------------------------------------------------
# Anti-regression: the contract is NULL-propagation, not silent 0.0
# ---------------------------------------------------------------------------

class TestNoSilentZeroPollution:
    """
    The April 13 readiness assessment called out that `.get(key, 0.0)` defaults
    were polluting downstream stats with fake zeros. This test class locks the
    rule: missing input → None, never 0.0.
    """

    @pytest.mark.parametrize("fn, args", [
        (compute_ops, (None, None)),
        (compute_ops, (None, 0.4)),
        (compute_avg, (None, 10)),
        (compute_avg, (5, None)),
        (compute_avg, (0, 0)),
        (compute_iso, (None, 0.3)),
        (compute_iso, (0.5, None)),
        (compute_whip, (None, 5, "7.0")),
        (compute_whip, (5, None, "7.0")),
        (compute_whip, (5, 5, None)),
        (compute_whip, (5, 5, "0.0")),
        (compute_era, (None, "7.0")),
        (compute_era, (5, None)),
        (compute_era, (5, "0.0")),
    ])
    def test_missing_component_returns_none_not_zero(self, fn, args):
        result = fn(*args)
        assert result is None, (
            f"{fn.__name__}{args} returned {result!r}; contract is None on missing input"
        )
