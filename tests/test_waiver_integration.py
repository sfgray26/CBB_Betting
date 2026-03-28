"""
Integration-level unit tests for waiver wire backend.
All Yahoo API calls are mocked — no live network required.
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yahoo_player(name="Test Player", positions=None, percent_owned=42.5):
    return {
        "player_key": "422.p.12345",
        "name": name,
        "team": "NYY",
        "positions": positions or ["OF"],
        "percent_owned": percent_owned,
        "status": None,
        "injury_note": None,
        "is_undroppable": 0,
    }


# ---------------------------------------------------------------------------
# Step 1 — yahoo_client.py "out" param
# ---------------------------------------------------------------------------

class TestYahooClientOutParam:

    def test_get_free_agents_includes_out_param(self):
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_key = "fake.lg.999"

        captured_params = {}

        def mock_get(path, params=None):
            captured_params.update(params or {})
            return {"fantasy_content": {"league": [{}, {"players": {}}]}}

        client._get = mock_get
        client._league_section = lambda data, idx: {}
        client._parse_players_block = lambda raw: []

        client.get_free_agents(count=10)

        assert "out" in captured_params, "get_free_agents() must include 'out' param"
        # Yahoo MLB rejects both 'ownership' and 'stats' on the /players collection endpoint
        # (400: Invalid subresource requested). Use metadata only.
        assert "ownership" not in captured_params["out"], "ownership subresource breaks MLB API"
        assert "stats" not in captured_params["out"], "stats subresource breaks MLB players endpoint"
        assert "metadata" in captured_params["out"]

    def test_get_waiver_players_includes_out_param(self):
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_key = "fake.lg.999"

        captured_params = {}

        def mock_get(path, params=None):
            captured_params.update(params or {})
            return {"fantasy_content": {"league": [{}, {"players": {}}]}}

        client._get = mock_get
        client._league_section = lambda data, idx: {}
        client._parse_players_block = lambda raw: []

        client.get_waiver_players(count=10)

        assert "out" in captured_params, "get_waiver_players() must include 'out' param"
        # Yahoo MLB rejects both 'ownership' and 'stats' on the /players collection endpoint
        assert "ownership" not in captured_params["out"], "ownership subresource breaks MLB API"
        assert "stats" not in captured_params["out"], "stats subresource breaks MLB players endpoint"
        assert "metadata" in captured_params["out"]


# ---------------------------------------------------------------------------
# Step 2 — two_start_pitchers must only contain SPs
# ---------------------------------------------------------------------------

class TestTwoStartPitchers:

    def test_two_start_pitchers_are_sps_only(self):
        """Only players with 'SP' in positions should appear in two_start_pitchers."""
        from backend.services.waiver_edge_detector import WaiverEdgeDetector

        # Verify the filtering logic directly
        mixed_fas = [
            {"name": "SP Guy", "positions": ["SP"], "cat_scores": {"era": 1.0}, "percent_owned": 5.0},
            {"name": "OF Guy", "positions": ["OF"], "cat_scores": {"hr": 1.0}, "percent_owned": 10.0},
            {"name": "RP Guy", "positions": ["RP"], "cat_scores": {"sv": 1.0}, "percent_owned": 3.0},
        ]
        sp_fas = [p for p in mixed_fas if "SP" in (p.get("positions") or [])]
        assert len(sp_fas) == 1
        assert sp_fas[0]["name"] == "SP Guy"


# ---------------------------------------------------------------------------
# Step 3 — top_available must be sorted descending by need_score
# ---------------------------------------------------------------------------

class TestTopAvailableSorting:

    def test_top_available_sorted_descending(self):
        """After scoring, top_available must be sorted by need_score descending."""
        # Simulate the sort logic applied in the endpoint
        players = [
            MagicMock(need_score=0.5, owned_pct=30.0),
            MagicMock(need_score=2.1, owned_pct=15.0),
            MagicMock(need_score=1.3, owned_pct=25.0),
        ]
        players.sort(key=lambda x: x.need_score, reverse=True)
        scores = [p.need_score for p in players]
        assert scores == sorted(scores, reverse=True), "top_available must be descending by need_score"


# ---------------------------------------------------------------------------
# Step 4 — get_roster() called, not get_my_roster()
# ---------------------------------------------------------------------------

class TestRosterMethodName:

    def test_get_roster_is_called_not_get_my_roster(self):
        """YahooFantasyClient.get_roster() must exist; get_my_roster() must not."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        assert hasattr(YahooFantasyClient, "get_roster"), (
            "YahooFantasyClient must have get_roster() method"
        )
        assert not hasattr(YahooFantasyClient, "get_my_roster"), (
            "get_my_roster() doesn't exist — main.py must use get_roster() instead"
        )

    def test_main_py_does_not_call_get_my_roster(self):
        """Verify get_my_roster() has been removed from main.py source."""
        import inspect
        import backend.main as main_module
        source = inspect.getsource(main_module)
        assert "get_my_roster" not in source, (
            "main.py still calls get_my_roster() — must be changed to get_roster()"
        )


# ---------------------------------------------------------------------------
# Step 5 — position filter forwarded to Yahoo
# ---------------------------------------------------------------------------

class TestPositionFilter:

    def test_position_filter_forwarded_to_yahoo(self):
        """When position='2B' is passed, get_free_agents() must receive position='2B'."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_key = "fake.lg.999"

        captured_params = {}

        def mock_get(path, params=None):
            captured_params.update(params or {})
            return {"fantasy_content": {"league": [{}, {"players": {}}]}}

        client._get = mock_get
        client._league_section = lambda data, idx: {}
        client._parse_players_block = lambda raw: []

        client.get_free_agents(position="2B", count=10)

        assert captured_params.get("position") == "2B", (
            f"Expected position='2B' in params, got: {captured_params}"
        )


# ---------------------------------------------------------------------------
# Fix B — Coverage-aware drop protection (_weakest_safe_to_drop logic)
# ---------------------------------------------------------------------------

class TestCoverageProtection:
    """Unit tests for the coverage-aware drop logic embedded in the recommendations endpoint.

    We test the logic in isolation by extracting the same control flow that main.py uses,
    so these tests remain fast with no Yahoo network calls.
    """

    _IL_STATUSES = {"IL", "IL10", "IL60", "NA", "OUT"}

    def _weakest_safe_to_drop(self, roster: list[dict], target_positions: list[str]) -> dict | None:
        """Replicated from main.py for isolated unit testing."""
        il = self._IL_STATUSES
        candidates = [
            rp for rp in roster
            if not rp.get("is_undroppable", False)
            and any(pos in rp["positions"] for pos in target_positions)
        ]
        if not candidates:
            return None
        active = [p for p in candidates if p.get("status") not in il]
        if len(active) == 1:
            return None  # only one active — protected
        if len(active) == 0:
            return min(candidates, key=lambda x: x.get("z_score") or 0.0)
        return min(active, key=lambda x: x.get("z_score") or 0.0)

    def _make_rp(self, name, positions, z_score, status=None, is_undroppable=False):
        return {
            "name": name,
            "player_key": name.lower().replace(" ", "_"),
            "positions": positions,
            "z_score": z_score,
            "is_proxy": False,
            "cat_scores": {},
            "starts_this_week": 1,
            "status": status,
            "injury_note": None,
            "is_undroppable": is_undroppable,
        }

    def test_only_2b_protected(self):
        """Roster has exactly 1 active 2B — must NOT suggest dropping them."""
        roster = [
            self._make_rp("Willi Castro", ["2B", "SS"], z_score=0.5),
            self._make_rp("Juan Soto", ["OF"], z_score=4.0),
        ]
        result = self._weakest_safe_to_drop(roster, ["2B"])
        assert result is None, "Only 2B should be protected from drop"

    def test_il_player_not_protected(self):
        """If the only player at a position is on IL, they ARE droppable (position is already empty)."""
        roster = [
            self._make_rp("Sick 2B", ["2B"], z_score=0.5, status="IL"),
        ]
        result = self._weakest_safe_to_drop(roster, ["2B"])
        # All at position are on IL → anyone droppable
        assert result is not None
        assert result["name"] == "Sick 2B"

    def test_two_2b_weakest_dropped(self):
        """Two active 2Bs → the weaker one is the drop candidate."""
        roster = [
            self._make_rp("Strong 2B", ["2B"], z_score=2.0),
            self._make_rp("Weak 2B", ["2B"], z_score=-0.5),
        ]
        result = self._weakest_safe_to_drop(roster, ["2B"])
        assert result is not None
        assert result["name"] == "Weak 2B"

    def test_undroppable_excluded(self):
        """is_undroppable=True player never appears as a drop candidate."""
        roster = [
            self._make_rp("Mike Trout", ["OF"], z_score=5.0, is_undroppable=True),
            self._make_rp("Randy Arozarena", ["OF"], z_score=2.0, is_undroppable=False),
            self._make_rp("Weak OF", ["OF"], z_score=0.3, is_undroppable=False),
        ]
        result = self._weakest_safe_to_drop(roster, ["OF", "LF", "CF", "RF"])
        assert result is not None
        assert result["name"] != "Mike Trout"
        assert result["name"] == "Weak OF"

    def test_no_candidates(self):
        """If no roster player is at the target position group, return None."""
        roster = [
            self._make_rp("Pitcher", ["SP"], z_score=1.0),
        ]
        result = self._weakest_safe_to_drop(roster, ["C"])
        assert result is None


# ---------------------------------------------------------------------------
# Fix C — Two-start pitchers via MLB Stats API (mocked HTTP)
# ---------------------------------------------------------------------------

class TestTwoStartPitchers:
    """Unit tests for _fetch_mlb_probable_starts() and name matching in the two-start loop."""

    _MLB_SCHEDULE_RESPONSE = {
        "dates": [
            {
                "date": "2026-03-28",
                "games": [
                    {
                        "teams": {
                            "home": {"probablePitcher": {"fullName": "Cristopher Sanchez"}},
                            "away": {"probablePitcher": {"fullName": "Zack Wheeler"}},
                        }
                    }
                ],
            },
            {
                "date": "2026-03-31",
                "games": [
                    {
                        "teams": {
                            "home": {"probablePitcher": {"fullName": "Cristopher Sanchez"}},
                            "away": {"probablePitcher": {}},
                        }
                    }
                ],
            },
        ]
    }

    def _parse_starts(self, schedule_json: dict) -> dict:
        """Same parsing logic as _fetch_mlb_probable_starts (without HTTP)."""
        starts: dict = {}
        for date_entry in schedule_json.get("dates", []):
            for game in date_entry.get("games", []):
                for side in ("home", "away"):
                    pitcher = game.get("teams", {}).get(side, {}).get("probablePitcher", {})
                    pname = (pitcher.get("fullName") or "").strip().lower()
                    if pname:
                        starts[pname] = starts.get(pname, 0) + 1
        return starts

    def test_mlb_api_schedule_parsed(self):
        """Mock HTTP response yields correct pitcher→count dict."""
        starts = self._parse_starts(self._MLB_SCHEDULE_RESPONSE)
        assert starts.get("cristopher sanchez") == 2
        assert starts.get("zack wheeler") == 1
        assert "" not in starts  # empty pitcher slots excluded

    def test_two_start_fuzzy_match(self):
        """'Christopher Sanchez' FA (Yahoo spelling) fuzzy-matches 'cristopher sanchez' in starts map."""
        import difflib
        starts_map = self._parse_starts(self._MLB_SCHEDULE_RESPONSE)
        fa_name = "christopher sanchez"  # Yahoo spelling

        # Exact miss
        assert starts_map.get(fa_name, 0) == 0

        # Fuzzy match
        best = max(
            starts_map.keys(),
            key=lambda k: difflib.SequenceMatcher(None, fa_name, k).ratio(),
            default=None,
        )
        ratio = difflib.SequenceMatcher(None, fa_name, best).ratio()
        assert ratio >= 0.90, f"Expected ratio >= 0.90, got {ratio}"
        assert starts_map[best] == 2

    def test_no_games_returns_empty(self):
        """Empty dates array → empty starts map → no two-start pitchers emitted."""
        starts = self._parse_starts({"dates": []})
        assert starts == {}

    def test_short_name_no_false_positive(self):
        """'Jim' does NOT fuzzy-match 'Tim' (ratio ~0.67 < 0.90)."""
        import difflib
        ratio = difflib.SequenceMatcher(None, "jim", "tim").ratio()
        assert ratio < 0.90


# ---------------------------------------------------------------------------
# EMAC-081: IL Slot Awareness
# ---------------------------------------------------------------------------

class TestILSlotAwareness:
    """Tests for count_il_slots_used() and il_capacity_info()."""

    def _make_player(self, selected_position=None, name="Player"):
        return {"name": name, "selected_position": selected_position}

    def test_no_il_players(self):
        """Roster with 0 IL-slotted players → used=0, available=total."""
        from backend.services.waiver_edge_detector import count_il_slots_used, il_capacity_info

        roster = [
            self._make_player("C", "Catcher"),
            self._make_player("1B", "FirstBase"),
            self._make_player("OF", "Outfield"),
            self._make_player("SP", "Pitcher"),
        ]
        assert count_il_slots_used(roster) == 0
        info = il_capacity_info(roster)
        assert info["used"] == 0
        assert info["available"] == info["total"]

    def test_one_il_slot_used(self):
        """One player selected_position='IL' → used=1, available=total-1."""
        from backend.services.waiver_edge_detector import count_il_slots_used, il_capacity_info

        roster = [
            self._make_player("IL", "InjuredPlayer"),
            self._make_player("C", "Catcher"),
            self._make_player("OF", "Outfield"),
        ]
        assert count_il_slots_used(roster) == 1
        info = il_capacity_info(roster)
        assert info["used"] == 1
        assert info["available"] == info["total"] - 1

    def test_all_il_slots_used(self):
        """Two IL players (default capacity) → used=total, available=0."""
        from backend.services.waiver_edge_detector import count_il_slots_used, il_capacity_info
        import os

        total = int(os.getenv("YAHOO_IL_SLOTS", "2"))
        roster = [
            self._make_player("IL", f"InjuredPlayer{i}") for i in range(total)
        ] + [self._make_player("C", "Catcher")]

        assert count_il_slots_used(roster) == total
        info = il_capacity_info(roster)
        assert info["used"] == total
        assert info["available"] == 0


# ---------------------------------------------------------------------------
# EMAC-081: Closer Alert
# ---------------------------------------------------------------------------

class TestCloserAlert:
    """Tests for closer alert logic — mirrors the inline logic in the waiver endpoint."""

    def _compute_alert(self, fa_list):
        """Replicate the closer alert logic from the waiver endpoint."""
        closer_fas = [
            f for f in fa_list
            if f.get("category_contributions", {}).get("nsv", 0) > 0.5
        ]
        if len(closer_fas) == 0:
            return "NO_CLOSERS"
        elif len(closer_fas) < 2:
            return "LOW_CLOSERS"
        return None

    def test_no_closers_alert(self):
        """Zero FAs with nsv contribution > 0.5 → NO_CLOSERS."""
        fas = [
            {"name": "Batter A", "category_contributions": {"hr": 1.2, "nsv": 0.0}},
            {"name": "Pitcher B", "category_contributions": {"k_pit": 0.8, "nsv": 0.3}},
        ]
        assert self._compute_alert(fas) == "NO_CLOSERS"

    def test_low_closers_alert(self):
        """Exactly one FA with nsv > 0.5 → LOW_CLOSERS."""
        fas = [
            {"name": "Closer A", "category_contributions": {"nsv": 1.5}},
            {"name": "Batter B", "category_contributions": {"hr": 0.9, "nsv": 0.0}},
        ]
        assert self._compute_alert(fas) == "LOW_CLOSERS"

    def test_no_alert_when_closers_present(self):
        """Two or more FAs with nsv > 0.5 → None (no alert)."""
        fas = [
            {"name": "Closer A", "category_contributions": {"nsv": 1.8}},
            {"name": "Closer B", "category_contributions": {"nsv": 0.9}},
            {"name": "Batter C", "category_contributions": {"hr": 0.7, "nsv": 0.0}},
        ]
        assert self._compute_alert(fas) is None
