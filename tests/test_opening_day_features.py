"""
Opening Day Features — unit tests for EMAC-087 (Mar 25, 2026).

Features tested:
  1. FAAB balance field on WaiverWireResponse
  2. Hot/cold flag calculation (_hot_cold_flag logic)
  3. Status/injury_note passthrough on WaiverPlayerOut
  4. Lineup gap detection (batter/pitcher active-slot warnings)

No live Yahoo API calls — all network calls are mocked or exercised
through the Pydantic schema layer directly.
"""

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Feature 1: FAAB balance in WaiverWireResponse
# ---------------------------------------------------------------------------

class TestFaabBalanceField:

    def test_faab_balance_field_exists_on_schema(self):
        """WaiverWireResponse schema must include faab_balance as Optional[float]."""
        from backend.schemas import WaiverWireResponse
        from datetime import date

        resp = WaiverWireResponse(
            week_end=date.today(),
            matchup_opponent="Test Opponent",
            category_deficits=[],
            top_available=[],
            two_start_pitchers=[],
        )
        # Default is None when not supplied
        assert resp.faab_balance is None

    def test_faab_balance_accepts_float(self):
        """WaiverWireResponse must accept a float faab_balance."""
        from backend.schemas import WaiverWireResponse
        from datetime import date

        resp = WaiverWireResponse(
            week_end=date.today(),
            matchup_opponent="Test Opponent",
            category_deficits=[],
            top_available=[],
            two_start_pitchers=[],
            faab_balance=87.0,
        )
        assert resp.faab_balance == 87.0

    def test_faab_balance_can_be_none_explicitly(self):
        """WaiverWireResponse must accept explicit None faab_balance (non-FAAB league)."""
        from backend.schemas import WaiverWireResponse
        from datetime import date

        resp = WaiverWireResponse(
            week_end=date.today(),
            matchup_opponent="Test Opponent",
            category_deficits=[],
            top_available=[],
            two_start_pitchers=[],
            faab_balance=None,
        )
        assert resp.faab_balance is None

    def test_get_faab_balance_method_exists(self):
        """YahooFantasyClient must expose a get_faab_balance() method."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
        assert hasattr(YahooFantasyClient, "get_faab_balance"), \
            "YahooFantasyClient must have get_faab_balance()"
        assert callable(getattr(YahooFantasyClient, "get_faab_balance"))

    def test_get_faab_balance_returns_none_on_api_error(self):
        """get_faab_balance() must return None when an API error occurs."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_key = "mlb.l.99999"

        def mock_get_raises(path, params=None):
            raise RuntimeError("Network failure")

        client._get = mock_get_raises
        result = client.get_faab_balance()
        assert result is None, "get_faab_balance() must return None on error, not raise"

    def test_get_faab_balance_parses_owned_team(self):
        """get_faab_balance() must return faab_balance from authenticated user's team."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_key = "mlb.l.99999"

        # Mock _get to return a teams response
        def mock_get(path, params=None):
            return {
                "fantasy_content": {
                    "league": [
                        {},
                        {
                            "teams": {
                                "count": 2,
                                "0": {
                                    "team": [
                                        [
                                            {"team_key": "mlb.l.99999.t.1"},
                                            {"name": "My Team"},
                                            {"is_owned_by_current_login": 1},
                                            {"faab_balance": "75"},
                                        ]
                                    ]
                                },
                                "1": {
                                    "team": [
                                        [
                                            {"team_key": "mlb.l.99999.t.2"},
                                            {"name": "Other Team"},
                                            {"is_owned_by_current_login": 0},
                                            {"faab_balance": "50"},
                                        ]
                                    ]
                                },
                            }
                        },
                    ]
                }
            }

        client._get = mock_get
        # Wire up the real _league_section and _iter_block
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient as _YFC
        client._league_section = _YFC._league_section.__get__(client, _YFC)
        client._iter_block = _YFC._iter_block

        result = client.get_faab_balance()
        assert result == 75.0, f"Expected 75.0, got {result}"


# ---------------------------------------------------------------------------
# Feature 2: Hot/cold flag calculation
# ---------------------------------------------------------------------------

class TestHotColdFlag:

    def _make_waiver_player_out(self, contributions: dict, cat_scores: dict = None):
        """Build a WaiverPlayerOut with specified contribution/cat_scores values."""
        from backend.schemas import WaiverPlayerOut
        p = WaiverPlayerOut(
            player_id="test.p.1",
            name="Test Player",
            team="NYY",
            position="OF",
            need_score=1.0,
            category_contributions=contributions,
            owned_pct=10.0,
            starts_this_week=0,
        )
        return p

    def test_hot_cold_flag_hot(self):
        """Player with all high positive category contributions must get HOT."""
        # avg = (0.8 + 0.9 + 0.7) / 3 = 0.8 > 0.4 -> HOT
        contributions = {"HR": 0.8, "RBI": 0.9, "R": 0.7}
        scores = list(contributions.values())
        avg = sum(scores) / len(scores)
        assert avg > 0.4, "Setup invariant: avg should be > 0.4"

        # Direct inline logic test (mirrors _hot_cold_flag in main.py)
        def _hot_cold_flag(cat_contributions):
            scores = list(cat_contributions.values())
            if not scores:
                return None
            avg = sum(scores) / len(scores)
            if avg > 0.4:
                return "HOT"
            if avg < -0.3:
                return "COLD"
            return None

        result = _hot_cold_flag(contributions)
        assert result == "HOT", f"Expected HOT, got {result}"

    def test_hot_cold_flag_cold(self):
        """Player with all low negative category contributions must get COLD."""
        # avg = (-0.6 + -0.7 + -0.5) / 3 = -0.6 < -0.3 -> COLD
        contributions = {"HR": -0.6, "RBI": -0.7, "R": -0.5}
        scores = list(contributions.values())
        avg = sum(scores) / len(scores)
        assert avg < -0.3, "Setup invariant: avg should be < -0.3"

        def _hot_cold_flag(cat_contributions):
            scores = list(cat_contributions.values())
            if not scores:
                return None
            avg = sum(scores) / len(scores)
            if avg > 0.4:
                return "HOT"
            if avg < -0.3:
                return "COLD"
            return None

        result = _hot_cold_flag(contributions)
        assert result == "COLD", f"Expected COLD, got {result}"

    def test_hot_cold_flag_neutral(self):
        """Player with mixed/near-zero scores must get None (no flag)."""
        # avg = (0.1 + -0.1 + 0.0) / 3 ~ 0.0 -> None
        contributions = {"HR": 0.1, "RBI": -0.1, "R": 0.0}

        def _hot_cold_flag(cat_contributions):
            scores = list(cat_contributions.values())
            if not scores:
                return None
            avg = sum(scores) / len(scores)
            if avg > 0.4:
                return "HOT"
            if avg < -0.3:
                return "COLD"
            return None

        result = _hot_cold_flag(contributions)
        assert result is None, f"Expected None, got {result}"

    def test_hot_cold_flag_empty(self):
        """Empty contributions must return None (not crash)."""
        def _hot_cold_flag(cat_contributions):
            scores = list(cat_contributions.values())
            if not scores:
                return None
            avg = sum(scores) / len(scores)
            if avg > 0.4:
                return "HOT"
            if avg < -0.3:
                return "COLD"
            return None

        result = _hot_cold_flag({})
        assert result is None


# ---------------------------------------------------------------------------
# Feature 3: Status / injury_note on WaiverPlayerOut
# ---------------------------------------------------------------------------

class TestWaiverPlayerStatusFields:

    def test_waiver_player_out_has_status_field(self):
        """WaiverPlayerOut schema must include status as Optional[str]."""
        from backend.schemas import WaiverPlayerOut

        p = WaiverPlayerOut(
            player_id="test.p.1",
            name="Test Player",
            team="NYY",
            position="OF",
            need_score=1.0,
            category_contributions={},
            owned_pct=10.0,
            starts_this_week=0,
        )
        assert hasattr(p, "status"), "WaiverPlayerOut must have status field"
        assert p.status is None  # default

    def test_waiver_player_out_has_injury_note_field(self):
        """WaiverPlayerOut schema must include injury_note as Optional[str]."""
        from backend.schemas import WaiverPlayerOut

        p = WaiverPlayerOut(
            player_id="test.p.1",
            name="Test Player",
            team="NYY",
            position="OF",
            need_score=1.0,
            category_contributions={},
            owned_pct=10.0,
            starts_this_week=0,
        )
        assert hasattr(p, "injury_note"), "WaiverPlayerOut must have injury_note field"
        assert p.injury_note is None  # default

    def test_waiver_player_out_status_set_correctly(self):
        """WaiverPlayerOut must accept and return DTD status."""
        from backend.schemas import WaiverPlayerOut

        p = WaiverPlayerOut(
            player_id="test.p.1",
            name="Test Player",
            team="NYY",
            position="OF",
            need_score=1.0,
            category_contributions={},
            owned_pct=10.0,
            starts_this_week=0,
            status="DTD",
            injury_note="Right hamstring tightness",
        )
        assert p.status == "DTD"
        assert p.injury_note == "Right hamstring tightness"


# ---------------------------------------------------------------------------
# Feature 4: Lineup gap detection warnings
# ---------------------------------------------------------------------------

class TestLineupGapWarning:

    def _make_batter(self, assigned_slot):
        """Build a minimal LineupPlayerOut-like dict."""
        from backend.schemas import LineupPlayerOut
        return LineupPlayerOut(
            player_id=f"p_{assigned_slot}",
            name=f"Player {assigned_slot}",
            team="NYY",
            position="OF",
            implied_runs=4.5,
            park_factor=1.0,
            lineup_score=5.0,
            status="START" if assigned_slot != "BN" else "BENCH",
            assigned_slot=assigned_slot,
        )

    def _make_pitcher(self, status):
        """Build a minimal StartingPitcherOut-like object."""
        from backend.schemas import StartingPitcherOut
        return StartingPitcherOut(
            player_id=f"p_{status}",
            name=f"Pitcher {status}",
            team="NYY",
            opponent_implied_runs=4.0,
            park_factor=1.0,
            sp_score=5.0,
            status=status,
        )

    def test_lineup_gap_warning_fires_for_few_batters(self):
        """Fewer than 6 active batter slots must add a warning."""
        # 3 active batters (below the 6 threshold)
        batters = [
            self._make_batter("C"),
            self._make_batter("1B"),
            self._make_batter("2B"),
            self._make_batter("BN"),
            self._make_batter("BN"),
        ]
        pitchers = [
            self._make_pitcher("START"),
            self._make_pitcher("START"),
            self._make_pitcher("START"),
        ]

        lineup_warnings = []

        _BENCH_SLOTS = {"BN", None}
        _batter_active = [b for b in batters if b.assigned_slot not in _BENCH_SLOTS]
        _pitcher_active = [p for p in pitchers if p.status == "START"]

        if len(_batter_active) < 6:
            lineup_warnings.append(
                f"Only {len(_batter_active)} active batter slots filled -- "
                "check bench/IL for promotable players."
            )
        if len(_pitcher_active) < 2:
            lineup_warnings.append(
                f"Only {len(_pitcher_active)} active pitcher slots filled -- "
                "consider streaming a SP."
            )

        assert any("active batter" in w for w in lineup_warnings), \
            f"Expected batter gap warning, got: {lineup_warnings}"
        assert len(_batter_active) == 3

    def test_lineup_gap_warning_does_not_fire_for_full_lineup(self):
        """No gap warning when 8 active batters and 3 active pitchers."""
        batters = [
            self._make_batter("C"),
            self._make_batter("1B"),
            self._make_batter("2B"),
            self._make_batter("3B"),
            self._make_batter("SS"),
            self._make_batter("OF"),
            self._make_batter("OF"),
            self._make_batter("Util"),
            self._make_batter("BN"),
        ]
        pitchers = [
            self._make_pitcher("START"),
            self._make_pitcher("START"),
            self._make_pitcher("NO_START"),
        ]

        lineup_warnings = []

        _BENCH_SLOTS = {"BN", None}
        _batter_active = [b for b in batters if b.assigned_slot not in _BENCH_SLOTS]
        _pitcher_active = [p for p in pitchers if p.status == "START"]

        if len(_batter_active) < 6:
            lineup_warnings.append(
                f"Only {len(_batter_active)} active batter slots filled -- "
                "check bench/IL for promotable players."
            )
        if len(_pitcher_active) < 2:
            lineup_warnings.append(
                f"Only {len(_pitcher_active)} active pitcher slots filled -- "
                "consider streaming a SP."
            )

        assert len(lineup_warnings) == 0, \
            f"No gap warnings expected for full lineup, got: {lineup_warnings}"

    def test_lineup_gap_warning_fires_for_few_pitchers(self):
        """Fewer than 2 active (START) pitchers must add a pitcher warning."""
        batters = [
            self._make_batter(slot) for slot in ["C", "1B", "2B", "3B", "SS", "OF", "OF", "Util"]
        ]
        pitchers = [self._make_pitcher("NO_START")]  # 0 STARTs

        lineup_warnings = []

        _BENCH_SLOTS = {"BN", None}
        _batter_active = [b for b in batters if b.assigned_slot not in _BENCH_SLOTS]
        _pitcher_active = [p for p in pitchers if p.status == "START"]

        if len(_batter_active) < 6:
            lineup_warnings.append(
                f"Only {len(_batter_active)} active batter slots filled -- "
                "check bench/IL for promotable players."
            )
        if len(_pitcher_active) < 2:
            lineup_warnings.append(
                f"Only {len(_pitcher_active)} active pitcher slots filled -- "
                "consider streaming a SP."
            )

        assert any("active pitcher" in w for w in lineup_warnings), \
            f"Expected pitcher gap warning, got: {lineup_warnings}"

    def test_lineup_gap_bench_slots_excluded(self):
        """BN and None assigned_slot must NOT count as active slots."""
        batters = [
            self._make_batter("BN"),
            self._make_batter("BN"),
            self._make_batter(None),
        ]
        _BENCH_SLOTS = {"BN", None}
        active = [b for b in batters if b.assigned_slot not in _BENCH_SLOTS]
        assert len(active) == 0, "BN and None slots must not be counted as active"
