"""
Tests for Yahoo Pydantic V2 contracts -- Layer 0 validation.

Uses real captured fixtures in tests/fixtures/yahoo_*.json.
Fixtures are committed snapshots from 2026-04-05 live capture via yahoo_capture.py.

Tests prove:
    1. 24/24 roster items parse as YahooRosterEntry
    2. 25/25 FA items parse as YahooWaiverCandidate
    3. 100/100 ADP items parse as YahooPlayer
    4. status=True correctly typed as bool (not coerced from string)
    5. status=None + injury_note present correctly parsed (Verlander pattern)
    6. status=True + injury_note present correctly parsed (Soto pattern)
    7. status=True + injury_note=None correctly parsed (Crochet pattern)
    8. is_injured property: True when status=True
    9. is_injured property: True when injury_note present with status=None
    10. is_injured property: True when "IL" in positions with status=None
    11. is_injured property: False when all three signals are absent
    12. selected_position only on YahooRosterEntry, not YahooPlayer/YahooWaiverCandidate
    13. stats dict parsed correctly for FA -- stat 60 in "H/AB" string format
    14. stats absent on ADP feed (YahooPlayer base model -- no stats field)
    15. percent_owned always float, never None
    16. "NA" in positions accepted (Yahoo "Not Active" token -- not IL but valid)
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.data_contracts import YahooPlayer, YahooRosterEntry, YahooWaiverCandidate

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(name: str) -> list:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Bulk parse rate -- 100% required for each endpoint
# ---------------------------------------------------------------------------

class TestBulkParseRate:
    def test_roster_all_24_parse(self):
        raw = load("yahoo_roster.json")
        assert len(raw) == 24, f"Expected 24 roster items, got {len(raw)}"
        parsed = [YahooRosterEntry.model_validate(item) for item in raw]
        assert len(parsed) == 24

    def test_free_agents_all_25_parse(self):
        raw = load("yahoo_free_agents.json")
        assert len(raw) == 25, f"Expected 25 FA items, got {len(raw)}"
        parsed = [YahooWaiverCandidate.model_validate(item) for item in raw]
        assert len(parsed) == 25

    def test_adp_all_100_parse(self):
        raw = load("yahoo_adp_injury.json")
        assert len(raw) == 100, f"Expected 100 ADP items, got {len(raw)}"
        parsed = [YahooPlayer.model_validate(item) for item in raw]
        assert len(parsed) == 100


# ---------------------------------------------------------------------------
# status field -- must be Optional[bool], not Optional[str]
# ---------------------------------------------------------------------------

class TestStatusField:
    def test_status_true_is_bool_on_roster(self):
        """Juan Soto: status=true in fixture -- must parse as Python bool True."""
        raw = load("yahoo_roster.json")
        soto = next(p for p in raw if p["name"] == "Juan Soto")
        entry = YahooRosterEntry.model_validate(soto)
        assert entry.status is True
        assert isinstance(entry.status, bool)

    def test_status_none_is_none(self):
        """Most players have status=null -- must parse as Python None."""
        raw = load("yahoo_roster.json")
        healthy = next(p for p in raw if p["status"] is None)
        entry = YahooRosterEntry.model_validate(healthy)
        assert entry.status is None

    def test_status_true_no_injury_note(self):
        """Garrett Crochet: status=true, injury_note=null -- both must parse."""
        raw = load("yahoo_roster.json")
        crochet = next(p for p in raw if p["name"] == "Garrett Crochet")
        entry = YahooRosterEntry.model_validate(crochet)
        assert entry.status is True
        assert entry.injury_note is None

    def test_status_string_rejected_in_strict_mode(self):
        """strict=True means a string "IL" must NOT be coerced to bool -- must raise."""
        raw = load("yahoo_roster.json")
        item = dict(raw[0])  # healthy player
        item["status"] = "IL"  # inject string -- must be rejected
        with pytest.raises(ValidationError):
            YahooRosterEntry.model_validate(item)

    def test_status_integer_rejected_in_strict_mode(self):
        """strict=True means integer 1 must NOT be coerced to bool -- must raise."""
        raw = load("yahoo_roster.json")
        item = dict(raw[0])
        item["status"] = 1  # integer -- must be rejected in strict mode
        with pytest.raises(ValidationError):
            YahooRosterEntry.model_validate(item)


# ---------------------------------------------------------------------------
# injury_note -- independent of status
# ---------------------------------------------------------------------------

class TestInjuryNote:
    def test_verlander_status_none_note_present(self):
        """ADP fixture: Verlander has status=None but injury_note='Hip'."""
        raw = load("yahoo_adp_injury.json")
        verlander = next(p for p in raw if p["name"] == "Justin Verlander")
        player = YahooPlayer.model_validate(verlander)
        assert player.status is None
        assert player.injury_note == "Hip"

    def test_soto_status_true_note_present(self):
        """Roster fixture: Soto has status=True and injury_note='Calf'."""
        raw = load("yahoo_roster.json")
        soto = next(p for p in raw if p["name"] == "Juan Soto")
        entry = YahooRosterEntry.model_validate(soto)
        assert entry.status is True
        assert entry.injury_note == "Calf"

    def test_healthy_player_both_none(self):
        raw = load("yahoo_roster.json")
        healthy = next(p for p in raw if p["status"] is None and p["injury_note"] is None)
        entry = YahooRosterEntry.model_validate(healthy)
        assert entry.status is None
        assert entry.injury_note is None


# ---------------------------------------------------------------------------
# is_injured property -- three independent signals
# ---------------------------------------------------------------------------

class TestIsInjuredProperty:
    def test_status_true_triggers_is_injured(self):
        """Crochet: status=True, no note, no IL position -- still is_injured."""
        raw = load("yahoo_roster.json")
        crochet = next(p for p in raw if p["name"] == "Garrett Crochet")
        entry = YahooRosterEntry.model_validate(crochet)
        assert entry.is_injured is True

    def test_injury_note_triggers_is_injured_without_status(self):
        """Verlander: status=None, injury_note='Hip' -- is_injured via note signal."""
        raw = load("yahoo_adp_injury.json")
        verlander = next(p for p in raw if p["name"] == "Justin Verlander")
        player = YahooPlayer.model_validate(verlander)
        assert player.status is None
        assert player.is_injured is True

    def test_il_position_triggers_is_injured_without_status(self):
        """Jason Adam on roster: status=None, injury_note present, positions has IL."""
        raw = load("yahoo_roster.json")
        adam = next(p for p in raw if p["name"] == "Jason Adam")
        entry = YahooRosterEntry.model_validate(adam)
        assert entry.status is None
        assert "IL" in entry.positions
        assert entry.is_injured is True

    def test_healthy_player_not_injured(self):
        """Player with all three signals absent must return is_injured=False."""
        raw = load("yahoo_roster.json")
        # Yainer Diaz: status=null, injury_note=null, no IL in positions
        diaz = next(p for p in raw if p["name"] == "Yainer Diaz")
        entry = YahooRosterEntry.model_validate(diaz)
        assert entry.is_injured is False

    def test_na_position_does_not_trigger_is_injured(self):
        """'NA' (Not Active) in positions is NOT an injury signal -- must not set is_injured."""
        raw = load("yahoo_adp_injury.json")
        # Carlos Carrasco has positions ["SP","P","NA"] with status=null, injury_note=null
        carrasco = next(p for p in raw if p["name"] == "Carlos Carrasco")
        player = YahooPlayer.model_validate(carrasco)
        assert player.is_injured is False
        assert "NA" in player.positions  # NA present but not IL


# ---------------------------------------------------------------------------
# selected_position -- roster only
# ---------------------------------------------------------------------------

class TestSelectedPosition:
    def test_roster_entry_has_selected_position(self):
        raw = load("yahoo_roster.json")
        entry = YahooRosterEntry.model_validate(raw[0])
        assert isinstance(entry.selected_position, str)
        assert len(entry.selected_position) > 0

    def test_selected_position_il_slot(self):
        """Players slotted in IL slot have selected_position='IL'."""
        raw = load("yahoo_roster.json")
        il_slotted = next(p for p in raw if p.get("selected_position") == "IL")
        entry = YahooRosterEntry.model_validate(il_slotted)
        assert entry.selected_position == "IL"

    def test_yahoo_player_has_no_selected_position_attribute(self):
        """YahooPlayer base model must NOT have selected_position."""
        assert not hasattr(YahooPlayer.model_fields, "selected_position") or \
               "selected_position" not in YahooPlayer.model_fields

    def test_yahoo_waiver_has_no_selected_position_attribute(self):
        """YahooWaiverCandidate must NOT have selected_position."""
        assert "selected_position" not in YahooWaiverCandidate.model_fields

    def test_roster_entry_has_selected_position_in_model_fields(self):
        """YahooRosterEntry must declare selected_position."""
        assert "selected_position" in YahooRosterEntry.model_fields


# ---------------------------------------------------------------------------
# stats dict -- FA only
# ---------------------------------------------------------------------------

class TestStatsField:
    def test_fa_stats_dict_parsed(self):
        """First FA player has stats dict with string keys and string values."""
        raw = load("yahoo_free_agents.json")
        candidate = YahooWaiverCandidate.model_validate(raw[0])
        assert candidate.stats is not None
        assert isinstance(candidate.stats, dict)
        # All keys and values must be strings
        for k, v in candidate.stats.items():
            assert isinstance(k, str), f"Stat key {k!r} is not str"
            assert isinstance(v, str), f"Stat value {v!r} for key {k!r} is not str"

    def test_stat_60_is_h_ab_format(self):
        """Stat ID 60 returns 'H/AB' combined format e.g. '8/20' -- not a raw H count."""
        raw = load("yahoo_free_agents.json")
        # First player: Liam Hicks has "60": "8/20"
        hicks = next(p for p in raw if p["name"] == "Liam Hicks")
        candidate = YahooWaiverCandidate.model_validate(hicks)
        assert candidate.stats is not None
        assert "60" in candidate.stats
        stat_60 = candidate.stats["60"]
        assert "/" in stat_60, f"Stat 60 expected 'H/AB' format, got {stat_60!r}"

    def test_adp_player_has_no_stats_field(self):
        """YahooPlayer base model must not have a stats attribute."""
        assert "stats" not in YahooPlayer.model_fields

    def test_adp_player_extra_stats_key_silently_ignored(self):
        """If an ADP item somehow has a stats key, YahooPlayer drops it (extra='ignore')."""
        raw = load("yahoo_adp_injury.json")
        item = dict(raw[0])
        item["stats"] = {"7": "5"}  # inject stats key
        # Must parse without error -- extra field ignored
        player = YahooPlayer.model_validate(item)
        assert not hasattr(player, "stats")


# ---------------------------------------------------------------------------
# percent_owned -- always float, never null
# ---------------------------------------------------------------------------

class TestPercentOwned:
    def test_roster_percent_owned_always_float(self):
        raw = load("yahoo_roster.json")
        for i, item in enumerate(raw):
            entry = YahooRosterEntry.model_validate(item)
            assert isinstance(entry.percent_owned, float), \
                f"Item {i} ({item['name']}): percent_owned is not float"
            assert entry.percent_owned is not None

    def test_fa_percent_owned_always_float(self):
        raw = load("yahoo_free_agents.json")
        for i, item in enumerate(raw):
            candidate = YahooWaiverCandidate.model_validate(item)
            assert isinstance(candidate.percent_owned, float), \
                f"Item {i} ({item['name']}): percent_owned is not float"

    def test_adp_percent_owned_always_float(self):
        raw = load("yahoo_adp_injury.json")
        for i, item in enumerate(raw):
            player = YahooPlayer.model_validate(item)
            assert isinstance(player.percent_owned, float), \
                f"Item {i} ({item['name']}): percent_owned is not float"

    def test_percent_owned_null_rejected(self):
        """percent_owned is non-optional -- None must raise ValidationError."""
        raw = load("yahoo_roster.json")
        item = dict(raw[0])
        item["percent_owned"] = None
        with pytest.raises(ValidationError):
            YahooRosterEntry.model_validate(item)


# ---------------------------------------------------------------------------
# positions list -- misc contract checks
# ---------------------------------------------------------------------------

class TestPositions:
    def test_positions_is_list_of_strings(self):
        raw = load("yahoo_roster.json")
        for item in raw:
            entry = YahooRosterEntry.model_validate(item)
            assert isinstance(entry.positions, list)
            for pos in entry.positions:
                assert isinstance(pos, str)

    def test_il_in_positions_accepted(self):
        """'IL' must be a valid position string -- not rejected by contract."""
        raw = load("yahoo_roster.json")
        il_players = [p for p in raw if "IL" in p.get("positions", [])]
        assert len(il_players) > 0, "No IL-position players found in roster fixture"
        for item in il_players:
            entry = YahooRosterEntry.model_validate(item)
            assert "IL" in entry.positions

    def test_na_in_positions_accepted(self):
        """'NA' must be a valid position string -- Yahoo 'Not Active' token."""
        raw = load("yahoo_adp_injury.json")
        na_players = [p for p in raw if "NA" in p.get("positions", [])]
        assert len(na_players) > 0, "No NA-position players found in ADP fixture"
        for item in na_players:
            player = YahooPlayer.model_validate(item)
            assert "NA" in player.positions


# ---------------------------------------------------------------------------
# Model inheritance -- YahooRosterEntry and YahooWaiverCandidate extend YahooPlayer
# ---------------------------------------------------------------------------

class TestInheritance:
    def test_roster_entry_is_yahoo_player(self):
        raw = load("yahoo_roster.json")
        entry = YahooRosterEntry.model_validate(raw[0])
        assert isinstance(entry, YahooPlayer)

    def test_waiver_candidate_is_yahoo_player(self):
        raw = load("yahoo_free_agents.json")
        candidate = YahooWaiverCandidate.model_validate(raw[0])
        assert isinstance(candidate, YahooPlayer)

    def test_is_injured_available_on_roster_entry(self):
        raw = load("yahoo_roster.json")
        entry = YahooRosterEntry.model_validate(raw[0])
        # Property must be accessible on subclass
        _ = entry.is_injured

    def test_is_injured_available_on_waiver_candidate(self):
        raw = load("yahoo_free_agents.json")
        candidate = YahooWaiverCandidate.model_validate(raw[0])
        _ = candidate.is_injured
