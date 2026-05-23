"""
Trade Analyzer — projects category-level impact of a fantasy baseball trade.

For each player, z-scores are drawn from the ``cat_scores`` field of the
board-compatible projection dict produced by
``backend.fantasy_baseball.player_board.get_or_create_projection()``.

Usage::

    from backend.fantasy_baseball.player_board import get_or_create_projection
    from backend.fantasy_baseball.trade_analyzer import analyze_trade

    give = [get_or_create_projection({"player_key": k1, "name": n1})]
    recv = [get_or_create_projection({"player_key": k2, "name": n2})]
    result = analyze_trade(give, recv)
    print(result.recommendation)      # 'strong_accept' | 'accept' | 'neutral' | …
"""
from __future__ import annotations

from typing import Any

from backend.contracts import TradeAnalysis, TradeCategoryDelta

# Recommendation thresholds (z-score units across all categories combined).
# Positive total_z_delta means the receive side is stronger.
_STRONG_ACCEPT = 1.5
_ACCEPT = 0.5
_REJECT = -0.5
_STRONG_REJECT = -1.5

# Categories where a lower raw value is better.  A positive delta (receiving
# side has a larger magnitude) is still a "gain" for the receiving manager.
_LOWER_IS_BETTER = frozenset({"era", "whip", "k_bat", "l", "hr_pit", "bb9"})


def _sum_cat_scores(players: list[dict]) -> dict[str, float]:
    """Return {category → total z-score} by summing across all players."""
    totals: dict[str, float] = {}
    for player in players:
        cat_scores = player.get("cat_scores") or {}
        for cat, z in cat_scores.items():
            if isinstance(z, (int, float)):
                totals[cat] = totals.get(cat, 0.0) + float(z)
    return totals


def _player_summary(player: dict) -> dict[str, Any]:
    """Return a compact, serialisable summary of a board projection entry."""
    return {
        "name": player.get("name") or "",
        "player_key": player.get("id") or player.get("player_key") or "",
        "z_score": round(float(player.get("z_score") or 0.0), 3),
        "type": player.get("type") or "unknown",
        "team": player.get("team") or "",
        "positions": player.get("positions") or [],
    }


def analyze_trade(
    give_players: list[dict],
    receive_players: list[dict],
    league_settings: dict | None = None,
) -> TradeAnalysis:
    """Compute per-category trade impact and return an overall recommendation.

    Args:
        give_players: Board-compatible dicts (from ``get_or_create_projection``).
                      Each must have a ``cat_scores`` key → {category: z_score}.
        receive_players: Same format — players being received in the trade.
        league_settings: Optional league configuration (reserved for future
                         category-weighting customization).

    Returns:
        :class:`TradeAnalysis` with per-category deltas, total z-score delta,
        a five-level recommendation, and a human-readable summary.
    """
    give_totals = _sum_cat_scores(give_players)
    receive_totals = _sum_cat_scores(receive_players)

    # Build deltas across the union of all categories touched by either side.
    all_cats = sorted(set(give_totals) | set(receive_totals))
    category_deltas: list[TradeCategoryDelta] = []
    total_z_delta = 0.0

    for cat in all_cats:
        give_z = give_totals.get(cat, 0.0)
        receive_z = receive_totals.get(cat, 0.0)
        delta = receive_z - give_z  # positive → receiving side contributes more
        total_z_delta += delta

        if abs(delta) < 0.05:
            direction = "neutral"
        elif delta > 0:
            direction = "gain"
        else:
            direction = "loss"

        category_deltas.append(TradeCategoryDelta(
            category=cat,
            give_z=round(give_z, 3),
            receive_z=round(receive_z, 3),
            delta=round(delta, 3),
            direction=direction,
        ))

    total_z_delta = round(total_z_delta, 3)

    # Map total delta to a five-level recommendation.
    if total_z_delta >= _STRONG_ACCEPT:
        recommendation = "strong_accept"
    elif total_z_delta >= _ACCEPT:
        recommendation = "accept"
    elif total_z_delta > _REJECT:
        recommendation = "neutral"
    elif total_z_delta > _STRONG_REJECT:
        recommendation = "reject"
    else:
        recommendation = "strong_reject"

    # Human-readable summary.
    gains = [d.category for d in category_deltas if d.direction == "gain"]
    losses = [d.category for d in category_deltas if d.direction == "loss"]
    summary = (
        f"Trade z-score delta: {total_z_delta:+.2f}. "
        f"Categories gained: {', '.join(gains) if gains else 'none'}. "
        f"Categories lost: {', '.join(losses) if losses else 'none'}."
    )

    return TradeAnalysis(
        give_players=[_player_summary(p) for p in give_players],
        receive_players=[_player_summary(p) for p in receive_players],
        category_deltas=category_deltas,
        total_z_delta=total_z_delta,
        recommendation=recommendation,
        summary=summary,
    )
