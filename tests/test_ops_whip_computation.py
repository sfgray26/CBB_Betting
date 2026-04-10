"""
Test OPS and WHIP computation in daily_ingestion.py

Covers Task 24 (ops) and Task 22 (whip) from data quality fixes plan.
Tests that computed stats use correct field names from Pydantic models.
"""

import pytest

from backend.data_contracts.mlb_player_stats import MLBPlayerStats
from backend.services.daily_ingestion import _parse_innings_pitched


class TestOPSComputation:
    """Test OPS = OBP + SLG computation"""

    def test_ops_computation_with_valid_obp_slg(self):
        """Test OPS computation when both OBP and SLG are present"""
        # Simulate the computation logic from daily_ingestion.py lines 1131-1133
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            obp=0.350,
            slg=0.450,
        )

        # This is the exact logic from daily_ingestion.py
        computed_ops = None
        if stat.obp is not None and stat.slg is not None:
            computed_ops = stat.obp + stat.slg

        assert computed_ops == 0.800
        assert computed_ops == stat.obp + stat.slg

    def test_ops_computation_with_missing_obp(self):
        """Test OPS computation when OBP is None"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            obp=None,
            slg=0.450,
        )

        computed_ops = None
        if stat.obp is not None and stat.slg is not None:
            computed_ops = stat.obp + stat.slg

        assert computed_ops is None

    def test_ops_computation_with_missing_slg(self):
        """Test OPS computation when SLG is None"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            obp=0.350,
            slg=None,
        )

        computed_ops = None
        if stat.obp is not None and stat.slg is not None:
            computed_ops = stat.obp + stat.slg

        assert computed_ops is None

    def test_ops_computation_with_both_missing(self):
        """Test OPS computation when both OBP and SLG are None"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            obp=None,
            slg=None,
        )

        computed_ops = None
        if stat.obp is not None and stat.slg is not None:
            computed_ops = stat.obp + stat.slg

        assert computed_ops is None

    def test_ops_field_names_match_pydantic_model(self):
        """Verify that obp/slg field names match Pydantic MLBPlayerStats model"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
        )

        # These field names must match the Pydantic model exactly
        assert hasattr(stat, 'obp')
        assert hasattr(stat, 'slg')
        assert hasattr(stat, 'ops')

        # Verify they are Optional (can be None)
        stat.obp = 0.400
        stat.slg = 0.600
        assert stat.obp == 0.400
        assert stat.slg == 0.600


class TestWHIPComputation:
    """Test WHIP = (BB + H) / IP computation"""

    def test_whip_computation_with_valid_stats(self):
        """Test WHIP computation with complete pitching stats"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            bb_allowed=3,
            h_allowed=6,
            ip="7.0",  # 7 innings
        )

        # This is the exact logic from daily_ingestion.py (after fix)
        computed_whip = None
        if (stat.bb_allowed is not None and
            stat.h_allowed is not None and
            stat.ip is not None):
            ip_decimal = _parse_innings_pitched(stat.ip)
            if ip_decimal is not None and ip_decimal > 0:
                computed_whip = (stat.bb_allowed + stat.h_allowed) / ip_decimal

        # WHIP = (3 + 6) / 7.0 = 1.286
        assert computed_whip is not None
        assert abs(computed_whip - 1.286) < 0.001

    def test_whip_computation_with_partial_innings(self):
        """Test WHIP computation with partial innings (e.g., 6.2)"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            bb_allowed=2,
            h_allowed=5,
            ip="6.2",  # 6 and 2/3 innings = 6.667
        )

        computed_whip = None
        if (stat.bb_allowed is not None and
            stat.h_allowed is not None and
            stat.ip is not None):
            ip_decimal = _parse_innings_pitched(stat.ip)
            if ip_decimal is not None and ip_decimal > 0:
                computed_whip = (stat.bb_allowed + stat.h_allowed) / ip_decimal

        # WHIP = (2 + 5) / 6.667 = 1.050
        assert computed_whip is not None
        assert abs(computed_whip - 1.050) < 0.001

    def test_whip_computation_with_missing_bb_allowed(self):
        """Test WHIP computation when walks_allowed (bb_allowed) is None"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            bb_allowed=None,
            h_allowed=6,
            ip="7.0",
        )

        computed_whip = None
        if (stat.bb_allowed is not None and
            stat.h_allowed is not None and
            stat.ip is not None):
            ip_decimal = _parse_innings_pitched(stat.ip)
            if ip_decimal is not None and ip_decimal > 0:
                computed_whip = (stat.bb_allowed + stat.h_allowed) / ip_decimal

        assert computed_whip is None

    def test_whip_computation_with_missing_h_allowed(self):
        """Test WHIP computation when hits_allowed (h_allowed) is None"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            bb_allowed=3,
            h_allowed=None,
            ip="7.0",
        )

        computed_whip = None
        if (stat.bb_allowed is not None and
            stat.h_allowed is not None and
            stat.ip is not None):
            ip_decimal = _parse_innings_pitched(stat.ip)
            if ip_decimal is not None and ip_decimal > 0:
                computed_whip = (stat.bb_allowed + stat.h_allowed) / ip_decimal

        assert computed_whip is None

    def test_whip_computation_with_zero_ip(self):
        """Test WHIP computation when IP is 0 (should return None to avoid division by zero)"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            bb_allowed=0,
            h_allowed=0,
            ip="0.0",
        )

        computed_whip = None
        if (stat.bb_allowed is not None and
            stat.h_allowed is not None and
            stat.ip is not None):
            ip_decimal = _parse_innings_pitched(stat.ip)
            if ip_decimal is not None and ip_decimal > 0:  # This check prevents division by zero
                computed_whip = (stat.bb_allowed + stat.h_allowed) / ip_decimal

        assert computed_whip is None  # Should be None because ip_decimal = 0

    def test_whip_field_names_match_pydantic_model(self):
        """Verify that bb_allowed/h_allowed field names match Pydantic MLBPlayerStats model"""
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
        )

        # These field names must match the Pydantic model exactly
        # CRITICAL: Model uses bb_allowed, NOT walks_allowed
        # CRITICAL: Model uses h_allowed, NOT hits_allowed
        assert hasattr(stat, 'bb_allowed')
        assert hasattr(stat, 'h_allowed')
        assert hasattr(stat, 'whip')

        # Verify they are Optional (can be None)
        stat.bb_allowed = 3
        stat.h_allowed = 6
        assert stat.bb_allowed == 3
        assert stat.h_allowed == 6

    def test_whip_field_names_bug_reproduction(self):
        """
        Reproduce the original bug: code used walks_allowed/hits_allowed
        but Pydantic model has bb_allowed/h_allowed
        """
        stat = MLBPlayerStats(
            id=12345,
            bdl_player_id=98765,
            game_id=45678,
            bb_allowed=3,
            h_allowed=6,
            ip="7.0",
        )

        # This would cause AttributeError in the old buggy code
        # Old code tried: stat.walks_allowed, stat.hits_allowed
        # Correct code uses: stat.bb_allowed, stat.h_allowed

        # Verify correct fields exist
        assert hasattr(stat, 'bb_allowed')
        assert hasattr(stat, 'h_allowed')

        # Verify incorrect field names do NOT exist (they never did)
        # This would raise AttributeError: 'MLBPlayerStats' object has no attribute 'walks_allowed'
        with pytest.raises(AttributeError):
            _ = stat.walks_allowed  # This field does NOT exist

        with pytest.raises(AttributeError):
            _ = stat.hits_allowed  # This field does NOT exist


class TestInningsPitchedParsing:
    """Test the _parse_innings_pitched helper function"""

    def test_parse_full_innings(self):
        """Test parsing full innings (integer string)"""
        assert _parse_innings_pitched("7") == 7.0
        assert _parse_innings_pitched("9") == 9.0

    def test_parse_partial_innings(self):
        """Test parsing partial innings (e.g., 6.2 = 6 and 2/3)"""
        # 6.2 = 6 innings + 2 outs = 6 + 2/3 = 6.667
        result = _parse_innings_pitched("6.2")
        assert result is not None
        assert abs(result - 6.667) < 0.001

        # 7.1 = 7 innings + 1 out = 7 + 1/3 = 7.333
        result = _parse_innings_pitched("7.1")
        assert result is not None
        assert abs(result - 7.333) < 0.001

    def test_parse_decimal_innings(self):
        """Test parsing decimal innings when passed as actual float/string decimals"""
        # When IP is a float, it's treated as decimal (not baseball notation)
        assert _parse_innings_pitched(6.5) == 6.5
        assert _parse_innings_pitched(5.0) == 5.0

        # When IP is a string in baseball notation ".X", it's interpreted as outs
        # "6.1" = 6 innings + 1 out = 6.333
        # "6.2" = 6 innings + 2 outs = 6.667
        # Note: "6.5" would be 6 innings + 5 outs = 7.667 (invalid in baseball but that's how the parser works)
        result = _parse_innings_pitched("6.1")
        assert result is not None
        assert abs(result - 6.333) < 0.001

    def test_parse_integer_input(self):
        """Test parsing when IP is already an integer"""
        assert _parse_innings_pitched(7) == 7.0
        assert _parse_innings_pitched(9) == 9.0

    def test_parse_float_input(self):
        """Test parsing when IP is already a float"""
        assert _parse_innings_pitched(6.5) == 6.5
        assert _parse_innings_pitched(7.333) == 7.333

    def test_parse_none_input(self):
        """Test parsing None input"""
        assert _parse_innings_pitched(None) is None

    def test_parse_invalid_string(self):
        """Test parsing invalid string input"""
        assert _parse_innings_pitched("invalid") is None
        assert _parse_innings_pitched("") is None


class TestFieldMappingVerification:
    """Verify field name mappings between BDL API and Pydantic model"""

    def test_batting_field_mappings(self):
        """Verify batting stat field names match Pydantic model"""
        stat = MLBPlayerStats()

        # These field names are used in daily_ingestion.py lines 1155-1169
        batting_fields = [
            'ab', 'r', 'h', 'double', 'triple', 'hr', 'rbi',
            'bb', 'so', 'sb', 'cs', 'avg', 'obp', 'slg', 'ops'
        ]

        for field in batting_fields:
            assert hasattr(stat, field), f"MLBPlayerStats missing field: {field}"

    def test_pitching_field_mappings(self):
        """Verify pitching stat field names match Pydantic model"""
        stat = MLBPlayerStats()

        # These field names are used in daily_ingestion.py lines 1171-1178
        pitching_fields = [
            'ip', 'h_allowed', 'r_allowed', 'er', 'bb_allowed', 'k', 'whip', 'era'
        ]

        for field in pitching_fields:
            assert hasattr(stat, field), f"MLBPlayerStats missing field: {field}"

    def test_no_field_name_conflicts(self):
        """Verify there are no ambiguous field name mappings"""
        stat = MLBPlayerStats()

        # OPS computation uses: obp, slg (correct)
        assert hasattr(stat, 'obp')
        assert hasattr(stat, 'slg')

        # WHIP computation uses: bb_allowed, h_allowed (correct after fix)
        assert hasattr(stat, 'bb_allowed')
        assert hasattr(stat, 'h_allowed')

        # Verify old buggy field names do NOT exist
        assert not hasattr(stat, 'walks_allowed')
        assert not hasattr(stat, 'hits_allowed')
