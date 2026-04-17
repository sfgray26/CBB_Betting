"""
End-to-end data pipeline validation tests.

Validates the complete computation chain (scoring -> simulation -> VORP)
using synthetic but realistic data. No external APIs, no database.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from backend.services.scoring_engine import compute_league_zscores
from backend.services.simulation_engine import simulate_player
from backend.services.vorp_engine import compute_vorp


# ---------------------------------------------------------------------------
# Helpers -- build synthetic PlayerRollingStats-like mock rows
# ---------------------------------------------------------------------------

def _make_hitter(
    pid: int,
    name: str,
    games: int = 10,
    hr: float = 0.15,
    rbi: float = 0.8,
    sb: float = 0.1,
    avg: float = 0.270,
    obp: float = 0.340,
    ab: float = 35.0,
    hits: float = 9.5,
    as_of: date = date(2026, 4, 10),
) -> MagicMock:
    row = MagicMock()
    row.bdl_player_id = pid
    row.player_name = name
    row.as_of_date = as_of
    row.window_days = 14
    row.games_in_window = games
    row.player_type = "hitter"
    row.w_home_runs = hr
    row.w_rbi = rbi
    row.w_stolen_bases = sb
    row.w_avg = avg
    row.w_obp = obp
    row.w_ab = ab
    row.w_hits = hits
    row.w_games = games
    # Pitcher fields must be None for hitter detection
    row.w_ip = None
    row.w_era = None
    row.w_whip = None
    row.w_k_per_9 = None
    row.w_strikeouts_pit = None
    row.w_earned_runs = None
    row.w_hits_allowed = None
    row.w_walks_allowed = None
    return row


def _make_pitcher(
    pid: int,
    name: str,
    games: int = 8,
    era: float = 3.50,
    whip: float = 1.20,
    k_per_9: float = 9.0,
    ip: float = 24.0,
    earned_runs: float = 8.0,
    hits_allowed: float = 18.0,
    walks_allowed: float = 8.4,
    strikeouts: float = 36.0,
    as_of: date = date(2026, 4, 10),
) -> MagicMock:
    row = MagicMock()
    row.bdl_player_id = pid
    row.player_name = name
    row.as_of_date = as_of
    row.window_days = 14
    row.games_in_window = games
    row.player_type = "pitcher"
    row.w_era = era
    row.w_whip = whip
    row.w_k_per_9 = k_per_9
    row.w_ip = ip
    row.w_earned_runs = earned_runs
    row.w_hits_allowed = hits_allowed
    row.w_walks_allowed = walks_allowed
    row.w_strikeouts_pit = strikeouts
    row.w_games = games
    # Hitter fields must be None for pitcher detection
    row.w_ab = None
    row.w_home_runs = None
    row.w_rbi = None
    row.w_stolen_bases = None
    row.w_avg = None
    row.w_obp = None
    row.w_hits = None
    return row


# ===================================================================
# Test Suite 1: Scoring Engine E2E
# ===================================================================

class TestScoringPipelineE2E:
    """Validate Z-score computation with synthetic player pools."""

    def test_zscore_output_range(self):
        """Z-scores for a 50-player pool should all fall in [-4, 4]."""
        import random as rng
        rng.seed(42)

        rows = []
        # 35 hitters
        for i in range(35):
            rows.append(_make_hitter(
                pid=1000 + i,
                name=f"Hitter_{i}",
                games=rng.randint(5, 14),
                hr=round(rng.uniform(0.0, 0.35), 3),
                rbi=round(rng.uniform(0.3, 1.5), 3),
                sb=round(rng.uniform(0.0, 0.4), 3),
                avg=round(rng.uniform(0.200, 0.340), 3),
                obp=round(rng.uniform(0.280, 0.420), 3),
                ab=round(rng.uniform(20, 50), 1),
                hits=round(rng.uniform(5, 16), 1),
            ))
        # 15 pitchers
        for i in range(15):
            rows.append(_make_pitcher(
                pid=2000 + i,
                name=f"Pitcher_{i}",
                games=rng.randint(5, 14),
                era=round(rng.uniform(2.0, 6.0), 2),
                whip=round(rng.uniform(0.90, 1.60), 2),
                k_per_9=round(rng.uniform(6.0, 13.0), 1),
                ip=round(rng.uniform(15, 40), 1),
                earned_runs=round(rng.uniform(3, 15), 1),
                hits_allowed=round(rng.uniform(10, 30), 1),
                walks_allowed=round(rng.uniform(4, 14), 1),
                strikeouts=round(rng.uniform(15, 55), 1),
            ))

        results = compute_league_zscores(rows, date(2026, 4, 10), 14)

        assert len(results) == 50

        z_fields = [
            "z_hr", "z_rbi", "z_sb", "z_avg", "z_obp",
            "z_era", "z_whip", "z_k_per_9",
        ]
        for res in results:
            for zf in z_fields:
                val = getattr(res, zf)
                if val is not None:
                    assert -4.0 <= val <= 4.0, (
                        f"Player {res.bdl_player_id} {zf}={val} out of range"
                    )

    def test_composite_z_monotonic_with_skill(self):
        """Players with linearly scaling stats should have monotonic composite_z."""
        rows = []
        for i in range(30):
            # Scale all hitter stats linearly with index
            t = i / 29.0  # 0.0 to 1.0
            rows.append(_make_hitter(
                pid=3000 + i,
                name=f"Scaled_{i}",
                games=10,
                hr=round(0.02 + t * 0.30, 3),
                rbi=round(0.30 + t * 1.20, 3),
                sb=round(0.01 + t * 0.35, 3),
                avg=round(0.200 + t * 0.140, 3),
                obp=round(0.280 + t * 0.140, 3),
                ab=35.0,
                hits=round(7.0 + t * 8.0, 1),
            ))

        results = compute_league_zscores(rows, date(2026, 4, 10), 14)
        by_id = {r.bdl_player_id: r for r in results}

        best = by_id[3029]  # highest index
        worst = by_id[3000]  # lowest index
        assert best.composite_z > worst.composite_z, (
            f"Best ({best.composite_z}) should exceed worst ({worst.composite_z})"
        )


# ===================================================================
# Test Suite 2: Simulation Engine E2E
# ===================================================================

class TestSimulationEngineE2E:
    """Validate Monte Carlo simulation outputs with synthetic rows."""

    def test_percentiles_ordered(self):
        """P10 <= P25 <= P50 <= P75 <= P90 for every hitter stat."""
        row = _make_hitter(
            pid=5001, name="OrderTest", games=10,
            hr=0.20, rbi=0.9, sb=0.15, avg=0.275, obp=0.350,
            ab=38.0, hits=10.5,
        )
        result = simulate_player(row, remaining_games=130, seed=42)

        for stat in ["hr", "rbi", "sb", "avg"]:
            p10 = getattr(result, f"proj_{stat}_p10")
            p25 = getattr(result, f"proj_{stat}_p25")
            p50 = getattr(result, f"proj_{stat}_p50")
            p75 = getattr(result, f"proj_{stat}_p75")
            p90 = getattr(result, f"proj_{stat}_p90")
            assert p10 <= p25 <= p50 <= p75 <= p90, (
                f"{stat} percentiles not ordered: "
                f"{p10}, {p25}, {p50}, {p75}, {p90}"
            )

    def test_reproducibility_with_seed(self):
        """Same seed=42 produces identical results across two runs."""
        row = _make_hitter(
            pid=5002, name="ReproTest", games=10,
            hr=0.18, rbi=0.85, sb=0.10, avg=0.260, obp=0.330,
            ab=36.0, hits=9.4,
        )
        r1 = simulate_player(row, remaining_games=130, seed=42)
        r2 = simulate_player(row, remaining_games=130, seed=42)

        assert r1.proj_hr_p50 == r2.proj_hr_p50
        assert r1.proj_rbi_p50 == r2.proj_rbi_p50
        assert r1.proj_sb_p50 == r2.proj_sb_p50
        assert r1.proj_avg_p50 == r2.proj_avg_p50

    def test_hr_projection_sanity(self):
        """A 0.3 HR/game hitter over 130 games: P50 HR in [30, 50]."""
        row = _make_hitter(
            pid=5003, name="HRSlugger", games=10,
            hr=3.0,  # 3.0 HR in 10 games = 0.3 HR/game rate
            rbi=1.0, sb=0.05, avg=0.260, obp=0.340,
            ab=40.0, hits=10.4,
        )
        result = simulate_player(row, remaining_games=130, seed=42)
        assert 30.0 <= result.proj_hr_p50 <= 50.0, (
            f"P50 HR = {result.proj_hr_p50}, expected [30, 50]"
        )

    def test_pitcher_projections(self):
        """Pitcher sim produces non-None ERA/WHIP/K percentiles."""
        row = _make_pitcher(
            pid=5004, name="AcePitcher", games=8,
            era=3.00, whip=1.10, k_per_9=10.5,
            ip=24.0, earned_runs=8.0,
            hits_allowed=18.0, walks_allowed=8.4,
            strikeouts=36.0,
        )
        result = simulate_player(row, remaining_games=130, seed=42)

        assert result.player_type == "pitcher"
        # ERA percentiles exist and are ordered
        assert result.proj_era_p10 is not None
        assert result.proj_era_p10 <= result.proj_era_p50 <= result.proj_era_p90
        # WHIP percentiles exist and are ordered
        assert result.proj_whip_p10 is not None
        assert result.proj_whip_p10 <= result.proj_whip_p50 <= result.proj_whip_p90
        # K percentiles exist and are ordered
        assert result.proj_k_p10 is not None
        assert result.proj_k_p10 <= result.proj_k_p50 <= result.proj_k_p90

    def test_pitcher_micro_sample_does_not_use_team_games_as_starts(self):
        """One strong outing should not extrapolate to 130 pitcher appearances."""
        row = _make_pitcher(
            pid=5005, name="OneStartWonder", games=1,
            era=2.00, whip=1.00, k_per_9=12.0,
            ip=6.0, earned_runs=1.0,
            hits_allowed=4.0, walks_allowed=2.0,
            strikeouts=8.0,
        )
        result = simulate_player(row, remaining_games=130, seed=42)

        assert result.proj_k_p50 is not None
        assert result.proj_k_p50 < 300.0, (
            f"P50 K = {result.proj_k_p50}, expected well below team-games extrapolation."
        )


# ===================================================================
# Test Suite 3: VORP Engine E2E
# ===================================================================

class TestVORPEngineE2E:
    """Validate VORP computation for different player profiles."""

    def test_positive_vorp_for_good_player(self):
        """composite_z=2.0 at SS (replacement=-4.0) -> VORP = 6.0."""
        vorp = compute_vorp(composite_z=2.0, positions=["SS"])
        assert vorp is not None
        assert vorp > 0, f"Expected positive VORP, got {vorp}"
        assert vorp == 6.0, f"Expected 6.0 (2.0 - (-4.0)), got {vorp}"

    def test_negative_vorp_for_replacement_player(self):
        """composite_z=-5.0 at OF (replacement=-2.5) -> VORP = -2.5."""
        vorp = compute_vorp(composite_z=-5.0, positions=["OF"])
        assert vorp is not None
        assert vorp < 0, f"Expected negative VORP, got {vorp}"
        assert vorp == -2.5, f"Expected -2.5 (-5.0 - (-2.5)), got {vorp}"

    def test_multi_position_uses_scarcest(self):
        """Multi-eligible player uses lowest replacement level (C=-5.5)."""
        vorp = compute_vorp(composite_z=1.0, positions=["C", "1B", "DH"])
        # Should use C at -5.5 (scarcest)
        assert vorp == round(1.0 - (-5.5), 4)

    def test_no_positions_returns_none(self):
        """No eligible positions -> VORP is None."""
        vorp = compute_vorp(composite_z=2.0, positions=[])
        assert vorp is None
