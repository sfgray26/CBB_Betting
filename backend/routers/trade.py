"""
Trade analyzer router — isolated from fantasy.py per architecture guidelines.

Routes
------
POST /api/fantasy/trade/analyze
    Accepts give/receive player_key lists; returns a TradeAnalysis.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import verify_api_key
from backend.contracts import TradeAnalysis, TradeAnalyzeRequest
from backend.fantasy_baseball.player_board import get_or_create_projection
from backend.fantasy_baseball.trade_analyzer import analyze_trade

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fantasy/trade", tags=["fantasy-trade"])


def _resolve_player(player_key: str, player_name: str | None) -> dict[str, Any]:
    """Fetch a board-compatible projection dict for one player.

    Calls ``get_or_create_projection`` with a minimal Yahoo-style dict
    (player_key + name).  Returns the projection (never raises; falls back to
    an empty-cat-scores entry so the trade analysis can still proceed).
    """
    yahoo_player: dict[str, Any] = {"player_key": player_key}
    if player_name:
        yahoo_player["name"] = player_name
    try:
        return get_or_create_projection(yahoo_player)
    except Exception as exc:
        logger.warning(
            "trade/analyze: could not resolve projection for %s (%s): %s",
            player_key, player_name, exc,
        )
        # Return a minimal stub so the endpoint never hard-crashes on one player.
        return {
            "id": player_key,
            "name": player_name or player_key,
            "type": "unknown",
            "team": "",
            "positions": [],
            "z_score": 0.0,
            "cat_scores": {},
            "is_proxy": True,
            "fusion_source": "error_fallback",
        }


@router.post("/analyze", response_model=TradeAnalysis)
async def analyze_trade_endpoint(
    request: TradeAnalyzeRequest,
    user: str = Depends(verify_api_key),
) -> TradeAnalysis:
    """Analyze projected category impact of a proposed fantasy baseball trade.

    For each player_key in ``give`` and ``receive``, fetches ROS projection
    data from the database (via player_board).  Returns per-category z-score
    deltas and an overall recommendation.

    Recommendation thresholds (total z-score delta):
    - ≥ +1.5  → strong_accept
    - ≥ +0.5  → accept
    - (-0.5, +0.5) → neutral
    - ≤ -0.5  → reject
    - ≤ -1.5  → strong_reject
    """
    if not request.give:
        raise HTTPException(status_code=422, detail="'give' list must not be empty.")
    if not request.receive:
        raise HTTPException(status_code=422, detail="'receive' list must not be empty.")

    give_projections = [_resolve_player(p.player_key, p.player_name) for p in request.give]
    recv_projections = [_resolve_player(p.player_key, p.player_name) for p in request.receive]

    try:
        result = analyze_trade(give_projections, recv_projections, league_settings=None)
    except Exception as exc:
        logger.error("analyze_trade internal error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Trade analysis failed: {exc}")

    return result
