"""
Tests for the constraint-aware lineup solver (DailyLineupOptimizer.solve_lineup).

Core correctness: position slots filled by eligibility, scarcest first,
so multi-eligible flex players (Castro 2B/3B) cover uncovered positions
rather than being benched behind score-ranked duplicates.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from backend.fantasy_baseball.daily_lineup_optimizer import (
    DailyLineupOptimizer,
    BatterRanking,
    LineupSlotResult,
    _DEFAULT_BATTER_SLOTS,
    _INACTIVE_STATUSES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_opt() -> DailyLineupOptimizer:
    opt = DailyLineupOptimizer()
    opt._api_key = "fake"
    return opt


def _make_batter(name, team, positions, score=3.0, status=None) -> BatterRanking:
    return BatterRanking(
        name=name,
        team=team,
        positions=positions,
        implied_team_runs=5.0,
        park_factor=1.0,
        lineup_score=score,
        reason="test",
        status=status,
    )


def _patch_ranked(opt, ranked):
    """Patch rank_batters to return a fixed sorted list (no Odds API needed)."""
    opt.rank_batters = lambda *a, **kw: sorted(ranked, key=lambda b: b.lineup_score, reverse=True)


def _patch_no_odds(opt):
    """Patch fetch_mlb_odds to return empty list (no off-day filtering applied)."""
    opt.fetch_mlb_odds = lambda *a, **kw: []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSolveLineupConstraints:

    def test_castro_fills_3b_not_benched(self):
        """
        Roster: Semien (2B only), Castro (2B/3B), Chapman (3B only), Soto (OF), Alonso (1B),
                Diaz (C), Perdomo (SS), Frelick (OF), Nimmo (OF), Buxton (OF).

        Expected: Semien → 2B, Chapman → 3B, Castro → Util (not bench).
        Castro must NOT be benched while a pure-position player fills his slot.
        """
        opt = _make_opt()
        _patch_no_odds(opt)
        roster = [
            _make_batter("Semien",  "TEX", ["2B"],       score=3.5),
            _make_batter("Chapman", "TOR", ["3B"],        score=2.8),
            _make_batter("Castro",  "NYY", ["2B", "3B"], score=2.5),
            _make_batter("Soto",    "NYM", ["OF"],        score=4.5),
            _make_batter("Alonso",  "NYM", ["1B"],        score=4.0),
            _make_batter("Diaz",    "NYM", ["C"],         score=3.0),
            _make_batter("Perdomo", "ARI", ["SS"],        score=2.2),
            _make_batter("Frelick", "MIL", ["OF"],        score=2.6),
            _make_batter("Nimmo",   "NYM", ["OF"],        score=2.4),
            _make_batter("Buxton",  "MIN", ["OF"],        score=3.2),
        ]
        _patch_ranked(opt, roster)

        results, warnings = opt.solve_lineup(roster=roster, projections=[], game_date="2026-04-01")

        slot_map = {r.slot: r.player_name for r in results if r.slot != "BN"}
        bench = {r.player_name for r in results if r.slot == "BN"}

        assert slot_map.get("2B") == "Semien",  f"Expected Semien at 2B, got {slot_map}"
        assert slot_map.get("3B") == "Chapman", f"Expected Chapman at 3B, got {slot_map}"
        # Castro should fill Util (not be benched)
        assert slot_map.get("Util") == "Castro", f"Expected Castro at Util, got {slot_map}"
        assert "Castro" not in bench, "Castro should not be benched"

    def test_scarce_position_filled_first(self):
        """C slot must be filled before OF slots, even when the catcher has a lower score."""
        opt = _make_opt()
        _patch_no_odds(opt)
        roster = [
            _make_batter("Diaz",    "NYM", ["C"],  score=2.0),  # low score
            _make_batter("Soto",    "NYM", ["OF"], score=5.0),
            _make_batter("Nimmo",   "NYM", ["OF"], score=4.0),
            _make_batter("Buxton",  "MIN", ["OF"], score=3.5),
            _make_batter("Alonso",  "NYM", ["1B"], score=3.0),
            _make_batter("Semien",  "TEX", ["2B"], score=2.5),
            _make_batter("Chapman", "TOR", ["3B"], score=2.2),
            _make_batter("Perdomo", "ARI", ["SS"], score=2.1),
            _make_batter("Castro",  "NYY", ["2B", "3B"], score=1.8),
        ]
        _patch_ranked(opt, roster)

        results, _ = opt.solve_lineup(roster=roster, projections=[], game_date="2026-04-01")
        slot_map = {r.slot: r.player_name for r in results if r.slot != "BN"}

        assert slot_map.get("C") == "Diaz", f"Diaz should fill C slot, got {slot_map}"
        # OF slots should go to the three highest-scoring OFs
        of_names = {v for k, v in slot_map.items() if k == "OF"}
        assert "Soto" in {r.player_name for r in results if r.slot == "OF"}

    def test_il_player_excluded_from_active_slots(self):
        """IL player must never appear in a START slot."""
        opt = _make_opt()
        _patch_no_odds(opt)
        roster = [
            _make_batter("Westburg",  "BAL", ["2B"], score=3.0, status="IL"),
            _make_batter("Semien",    "TEX", ["2B"], score=2.5),
            _make_batter("Diaz",      "NYM", ["C"],  score=2.0),
            _make_batter("Perdomo",   "ARI", ["SS"], score=1.8),
            _make_batter("Chapman",   "TOR", ["3B"], score=2.2),
            _make_batter("Alonso",    "NYM", ["1B"], score=3.5),
            _make_batter("Soto",      "NYM", ["OF"], score=4.5),
            _make_batter("Nimmo",     "NYM", ["OF"], score=2.4),
            _make_batter("Buxton",    "MIN", ["OF"], score=3.2),
            _make_batter("Castro",    "NYY", ["2B", "3B"], score=2.0),
        ]
        _patch_ranked(opt, roster)

        results, _ = opt.solve_lineup(roster=roster, projections=[], game_date="2026-04-01")
        active_names = {r.player_name for r in results if r.slot != "BN"}

        assert "Westburg" not in active_names, "IL player must not appear in active slots"
        assert slot_map_from(results, "2B") == "Semien", (
            f"Semien should fill 2B after Westburg excluded; got {slot_map_from(results, '2B')}"
        )

    def test_off_day_player_sits_when_full_slate(self):
        """
        When odds API covers 10+ teams (full slate), a player whose team has no game
        should be deprioritised and replaced by an in-game player at the same position.
        """
        opt = _make_opt()

        # Simulate a full slate: NYM, TEX, ARI, BAL, MIN + 5 more pairs (10 abbrevs total)
        fake_team_odds = {t: {"implied_runs": 4.5, "is_home": True, "opponent": "X", "park_factor": 1.0}
                         for t in ["NYM", "TEX", "ARI", "BAL", "MIN", "BOS", "LAD", "ATL", "HOU", "CLE"]}
        opt._build_team_odds_map = lambda games: fake_team_odds
        opt.fetch_mlb_odds = lambda *a, **kw: ["stub"] * 5  # len=5 → at least 1 game marker

        roster = [
            _make_batter("Diaz",    "NYM", ["C"],  score=2.0),   # has game
            _make_batter("Alonso",  "NYM", ["1B"], score=3.5),   # has game
            _make_batter("Semien",  "TEX", ["2B"], score=2.5),   # has game
            _make_batter("Chapman", "CWS", ["3B"], score=4.0),   # OFF DAY — not in fake_team_odds
            _make_batter("Nimmo",   "BAL", ["3B"], score=2.0),   # has game, lower score
            _make_batter("Perdomo", "ARI", ["SS"], score=1.8),   # has game
            _make_batter("Soto",    "NYM", ["OF"], score=4.5),   # has game
            _make_batter("Buxton",  "MIN", ["OF"], score=3.2),   # has game
            _make_batter("Frelick", "MIL", ["OF"], score=2.6),   # has game (MIL not in odds but let's assume)
            _make_batter("Castro",  "NYY", ["2B", "3B"], score=2.0),  # has game
        ]
        # MIL not in fake_team_odds — also off day in this test
        opt.rank_batters = lambda *a, **kw: sorted(roster, key=lambda b: b.lineup_score, reverse=True)

        results, warnings = opt.solve_lineup(roster=roster, projections=[], game_date="2026-04-01")
        slot_map = {r.slot: r.player_name for r in results if r.slot != "BN"}

        # Chapman (off day) should NOT fill 3B; Nimmo or Castro should
        assert slot_map.get("3B") != "Chapman", (
            f"Off-day Chapman must not start at 3B; got {slot_map.get('3B')}"
        )

    def test_bench_contains_surplus(self):
        """Extra players beyond the 9 slots are bench entries."""
        opt = _make_opt()
        _patch_no_odds(opt)
        roster = [
            _make_batter("Diaz",    "NYM", ["C"],  score=2.0),
            _make_batter("Alonso",  "NYM", ["1B"], score=3.5),
            _make_batter("Pasq",    "KCR", ["1B"], score=3.0),   # surplus 1B
            _make_batter("Torkel",  "DET", ["1B"], score=2.8),   # surplus 1B
            _make_batter("Semien",  "TEX", ["2B"], score=2.5),
            _make_batter("Chapman", "TOR", ["3B"], score=2.2),
            _make_batter("Perdomo", "ARI", ["SS"], score=1.8),
            _make_batter("Soto",    "NYM", ["OF"], score=4.5),
            _make_batter("Nimmo",   "NYM", ["OF"], score=2.4),
            _make_batter("Buxton",  "MIN", ["OF"], score=3.2),
            _make_batter("Castro",  "NYY", ["2B", "3B"], score=2.0),
        ]
        _patch_ranked(opt, roster)

        results, _ = opt.solve_lineup(roster=roster, projections=[], game_date="2026-04-01")
        bench = [r for r in results if r.slot == "BN"]

        # With 11 batters and 9 slots, at least 2 should be on bench
        assert len(bench) >= 2, f"Expected ≥2 bench entries, got {len(bench)}"

    def test_empty_slot_when_no_eligible_player(self):
        """If no C is on roster, solver emits a warning and an EMPTY slot."""
        opt = _make_opt()
        _patch_no_odds(opt)
        roster = [
            # No catcher!
            _make_batter("Alonso",  "NYM", ["1B"], score=3.5),
            _make_batter("Semien",  "TEX", ["2B"], score=2.5),
            _make_batter("Chapman", "TOR", ["3B"], score=2.2),
            _make_batter("Perdomo", "ARI", ["SS"], score=1.8),
            _make_batter("Soto",    "NYM", ["OF"], score=4.5),
            _make_batter("Nimmo",   "NYM", ["OF"], score=2.4),
            _make_batter("Buxton",  "MIN", ["OF"], score=3.2),
            _make_batter("Castro",  "NYY", ["2B", "3B"], score=2.0),
        ]
        _patch_ranked(opt, roster)

        results, warnings = opt.solve_lineup(roster=roster, projections=[], game_date="2026-04-01")
        empty_slots = [r for r in results if r.player_name == "EMPTY"]
        c_slot = next((r for r in results if r.slot == "C"), None)

        assert c_slot is not None
        assert c_slot.player_name == "EMPTY"
        assert any("C" in w for w in warnings), f"Expected C-slot warning, got: {warnings}"


# ---------------------------------------------------------------------------
# Helpers (module-level, shared)
# ---------------------------------------------------------------------------

def slot_map_from(results, slot_label):
    """Return the player name assigned to the given slot label."""
    for r in results:
        if r.slot == slot_label:
            return r.player_name
    return None


def test_schedule_fallback_uses_probable_pitcher_snapshot(monkeypatch):
    """Missing odds should still produce game context from the snapshot table."""
    opt = DailyLineupOptimizer()
    opt._api_key = ""

    rows = [
        SimpleNamespace(team="NYY", opponent="BOS", is_home=True, park_factor=1.04),
        SimpleNamespace(team="BOS", opponent="NYY", is_home=False, park_factor=1.04),
    ]

    class FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return rows

    class FakeSession:
        def query(self, *args, **kwargs):
            return FakeQuery()

        def close(self):
            return None

    monkeypatch.setattr("backend.fantasy_baseball.daily_lineup_optimizer.SessionLocal", lambda: FakeSession())

    games = opt.fetch_mlb_odds("2026-04-20")

    assert len(games) == 1
    assert games[0].home_abbrev == "NYY"
    assert games[0].away_abbrev == "BOS"
    assert games[0].park_factor == pytest.approx(1.04)
    assert games[0].implied_home_runs != 4.5


def test_smart_lineup_assignments_include_positions(monkeypatch):
    """Smart selector assignments must preserve eligible positions for API payloads."""
    from backend.fantasy_baseball.smart_lineup_selector import SmartLineupSelector, SmartBatterRanking

    selector = SmartLineupSelector()

    def fake_select_optimal_lineup(roster, projections, game_date, category_needs):
        return [
            SmartBatterRanking(
                name="Pete Alonso",
                player_id="alonso",
                team="NYM",
                positions=["1B"],
                has_game=True,
                implied_team_runs=5.2,
                park_factor=1.03,
                smart_score=7.4,
            )
        ], []

    monkeypatch.setattr(selector, "select_optimal_lineup", fake_select_optimal_lineup)

    assignments, warnings = selector.solve_smart_lineup(
        roster=[],
        projections=[],
        game_date="2026-04-20",
        slot_config=[("1B", ["1B"])],
    )

    assert warnings == []
    assert assignments[0]["positions"] == ["1B"]
    assert assignments[0]["slot"] == "1B"
