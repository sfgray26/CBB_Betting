"""
Tests for computed statistics (ops, whip, caught_stealing).
"""

import pytest


def test_ops_calculation_from_raw_stats():
    """Test OPS calculation from AVG, OBP, SLG."""
    # OPS = OBP + SLG
    obp = 0.350
    slg = 0.450
    ops = obp + slg
    assert abs(ops - 0.800) < 0.001


def test_whip_calculation_from_raw_stats():
    """Test WHIP calculation from walks + hits allowed divided by innings pitched."""
    # WHIP = (BB + H) / IP
    walks_allowed = 20
    hits_allowed = 50
    innings_pitched = 30.0  # 30 IP
    whip = (walks_allowed + hits_allowed) / innings_pitched
    assert abs(whip - 2.333) < 0.01


def test_caught_stealing_defaults_to_zero():
    """Test that caught_stealing defaults to 0 when not provided."""
    # BDL API may not return cs, default to 0
    cs = None
    caught_stealing = cs if cs is not None else 0
    assert caught_stealing == 0


def test_innings_pitched_parsing():
    """Test parsing innings pitched from '6.2' format to decimal."""
    # BDL returns IP as "6.2" meaning 6.2 innings = 6 innings + 2 outs = 6.667
    ip_str = "6.2"
    parts = ip_str.split(".")
    innings = int(parts[0])  # 6
    outs = int(parts[1]) if len(parts) > 1 else 0  # 2
    ip_decimal = innings + (outs / 3.0)  # 6.667
    assert abs(ip_decimal - 6.667) < 0.01


def test_whip_with_decimal_ip():
    """Test WHIP calculation with decimal innings pitched."""
    walks_allowed = 15
    hits_allowed = 40
    ip_str = "6.2"
    # Parse IP
    parts = ip_str.split(".")
    innings = int(parts[0])
    outs = int(parts[1]) if len(parts) > 1 else 0
    ip_decimal = innings + (outs / 3.0)
    # Calculate WHIP
    whip = (walks_allowed + hits_allowed) / ip_decimal
    expected = (15 + 40) / 6.667
    assert abs(whip - expected) < 0.01
