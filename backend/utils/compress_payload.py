"""
Utility: compress_prediction_payload

Strip full four-factor style blocks down to essential fields for multi-agent
prompt pasting.  Reduces payload size ~50% by retaining only the fields needed
to reconstruct analysis context without re-running the model.

Usage:
    from backend.utils.compress_payload import compress_prediction_payload
    slim = compress_prediction_payload(predictions)
    print(json.dumps(slim, indent=2))
"""

import copy


def compress_prediction_payload(predictions: list) -> list:
    """
    Strip full four-factor style blocks down to essential fields
    for multi-agent prompt pasting (reduces payload size ~50%).

    Keeps per-side: pace, efg_pct, def_efg_pct, is_heuristic,
    kenpom_four_factors.  Drops notes from full_analysis.

    Args:
        predictions: List of prediction dicts as returned by
                     GET /api/predictions/today or run_nightly_analysis.

    Returns:
        Deep copy of predictions with style blocks compressed.
    """
    out = copy.deepcopy(predictions)
    for pred in out:
        fa = pred.get("full_analysis", {})
        inp = fa.get("inputs", {})
        for side in ("home_style", "away_style"):
            if side in inp:
                s = inp[side]
                inp[side] = {
                    "pace": s.get("pace"),
                    "efg_pct": s.get("efg_pct"),
                    "def_efg_pct": s.get("def_efg_pct"),
                    "is_heuristic": s.get("is_heuristic"),
                    "kenpom_four_factors": s.get("kenpom_four_factors"),
                }
        fa.pop("notes", None)
    return out
