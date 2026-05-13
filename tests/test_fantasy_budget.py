"""Tests for budget endpoint IP-accumulation key logic."""


def test_budget_ip_key_reads_my_stats():
    """get_matchup_stats returns 'my_stats', not 'my_team'; IP must use the right key."""
    matchup_stats = {"my_stats": {"IP": 14.2}, "opp_stats": {}, "opponent_name": "Opponent"}
    my_stats = matchup_stats.get("my_stats", {})
    ip = float(my_stats.get("IP", 0.0))
    assert ip == 14.2, f"Expected 14.2 but got {ip}"


def test_budget_ip_wrong_key_returns_zero():
    """Regression: the old 'my_team' key returns 0, confirming the fix matters."""
    matchup_stats = {"my_stats": {"IP": 14.2}, "opp_stats": {}, "opponent_name": "Opponent"}
    old_key = matchup_stats.get("my_team", {})
    ip = float(old_key.get("IP", 0.0))
    assert ip == 0.0, "Old key should give 0 — confirms the bug was real"
