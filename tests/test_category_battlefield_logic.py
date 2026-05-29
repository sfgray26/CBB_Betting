"""
Unit tests for category battlefield action hint logic.
These mirror the TypeScript logic in category-battlefield.tsx.
"""


def action_hint(win_prob, lower_better, my_current_val=None, opp_current_val=None,
                my_proj=None, opp_proj=None):
    """Python mirror of actionHint() in category-battlefield.tsx."""
    if win_prob is None:
        return ''
    if win_prob > 0.95:
        return 'Protect'
    if win_prob < 0.05:
        return '—'
    if 0.35 <= win_prob <= 0.65:
        if my_proj is not None and opp_proj is not None:
            delta = (my_proj - opp_proj) if lower_better else (opp_proj - my_proj)
            if delta > 0:
                return f"Need -{delta:.2f}" if lower_better else f"Need +{delta:.2f}"
            return 'Ratio risk' if lower_better else 'Hold'
        return 'Close'
    if win_prob > 0.65:
        return 'Hold'
    # Guard: never punt a category currently being won
    if my_current_val is not None and opp_current_val is not None:
        currently_winning = (my_current_val < opp_current_val) if lower_better else (my_current_val > opp_current_val)
        if currently_winning:
            return 'Hold'
    return 'Punt?'


def test_punt_not_emitted_when_winning_avg():
    """AVG .276 vs .248: currently winning → must not suggest Punt?"""
    result = action_hint(
        win_prob=0.20,        # low win_prob (projected end-of-week could flip)
        lower_better=False,   # AVG: higher is better
        my_current_val=0.276,
        opp_current_val=0.248,
    )
    assert result == 'Hold', f"Expected 'Hold' for winning AVG, got '{result}'"


def test_punt_emitted_when_losing():
    """ERA losing (mine 4.50 vs theirs 3.20): low win_prob + losing → Punt? is correct."""
    result = action_hint(
        win_prob=0.15,
        lower_better=True,    # ERA: lower is better
        my_current_val=4.50,
        opp_current_val=3.20,
    )
    assert result == 'Punt?', f"Expected 'Punt?' for losing ERA, got '{result}'"


def test_punt_guard_no_current_vals_falls_through():
    """When current values are unknown, Punt? should still fire (don't break existing behavior)."""
    result = action_hint(
        win_prob=0.20,
        lower_better=False,
        my_current_val=None,
        opp_current_val=None,
    )
    assert result == 'Punt?'


def test_protect_when_win_prob_high():
    result = action_hint(win_prob=0.96, lower_better=False)
    assert result == 'Protect'


def test_bubble_returns_need_delta():
    result = action_hint(
        win_prob=0.50,
        lower_better=False,
        my_proj=8,
        opp_proj=10,
    )
    assert result == 'Need +2.00'
