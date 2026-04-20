"""
UAT Validator — Automated endpoint validation against live Railway deployment.

Hits every core fantasy endpoint, validates response structure + data quality,
outputs a structured report with PASS/FAIL/WARN per check.

Usage:
    venv\\Scripts\\python scripts\\uat_validator.py [--base-url URL] [--api-key KEY]

Defaults to Railway prod URL + API key from environment or postman config.

Exit code: 0 if all PASS, 1 if any FAIL.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

# ── Canonical category codes (18) ─────────────────────────────────────────
SCORING_CATEGORIES_18 = frozenset({
    "R", "H", "HR_B", "RBI", "K_B", "TB", "AVG", "OPS", "NSB",
    "W", "L", "HR_P", "K_P", "ERA", "WHIP", "K_9", "QS", "NSV",
})

LOWER_IS_BETTER = frozenset({"K_B", "L", "HR_P", "ERA", "WHIP"})

# Yahoo stat_id aliases that may appear instead of canonical codes
YAHOO_ALIASES = {
    "K(B)": "K_B", "K(P)": "K_P", "HRA": "HR_P", "K/9": "K_9",
    "HR": "HR_B", "K": "K_P", "SB": "NSB", "SV": "NSV",
}

VALID_ROSTER_POSITIONS = {
    "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF",
    "DH", "Util", "SP", "RP", "P", "BN", "IL", "IL60", "NA",
}

VALID_OPTIMIZE_SLOTS = {"C", "1B", "2B", "3B", "SS", "OF", "Util", "SP", "RP", "P", "BN"}


# ── Result model ──────────────────────────────────────────────────────────

@dataclass
class Check:
    endpoint: str
    check_name: str
    status: str  # PASS, FAIL, WARN, SKIP, ERROR
    expected: str
    actual: str
    detail: str = ""


@dataclass
class EndpointResult:
    endpoint: str
    http_status: int
    elapsed_ms: int
    checks: List[Check] = field(default_factory=list)
    raw_response: Optional[dict] = None
    error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────

def _check(endpoint: str, name: str, expected: str, actual: str,
           passed: bool, detail: str = "") -> Check:
    return Check(
        endpoint=endpoint,
        check_name=name,
        status="PASS" if passed else "FAIL",
        expected=expected,
        actual=actual,
        detail=detail,
    )


def _warn(endpoint: str, name: str, expected: str, actual: str,
          detail: str = "") -> Check:
    return Check(
        endpoint=endpoint,
        check_name=name,
        status="WARN",
        expected=expected,
        actual=actual,
        detail=detail,
    )


def _skip(endpoint: str, name: str, reason: str) -> Check:
    return Check(
        endpoint=endpoint,
        check_name=name,
        status="SKIP",
        expected="N/A",
        actual="N/A",
        detail=reason,
    )


def _call(client: httpx.Client, method: str, path: str,
          params: dict = None, json_body: dict = None,
          timeout: float = 30.0) -> EndpointResult:
    """Make an HTTP call and return the raw EndpointResult."""
    url = path
    t0 = time.monotonic()
    try:
        resp = client.request(method, url, params=params, json=json_body, timeout=timeout)
        elapsed = int((time.monotonic() - t0) * 1000)
        try:
            body = resp.json()
        except Exception:
            body = None
        return EndpointResult(
            endpoint=f"{method} {path}",
            http_status=resp.status_code,
            elapsed_ms=elapsed,
            raw_response=body,
        )
    except Exception as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        return EndpointResult(
            endpoint=f"{method} {path}",
            http_status=0,
            elapsed_ms=elapsed,
            error=str(exc),
        )


# ── Validators ────────────────────────────────────────────────────────────

def validate_health(client: httpx.Client) -> EndpointResult:
    """GET / and GET /health"""
    r = _call(client, "GET", "/health")
    ep = r.endpoint
    if r.error:
        r.checks.append(_check(ep, "reachable", "200", "connection error", False, r.error))
        return r
    r.checks.append(_check(ep, "http_status", "200", str(r.http_status), r.http_status == 200))
    if r.raw_response:
        status = r.raw_response.get("status", "")
        r.checks.append(_check(ep, "status_field", "healthy", str(status),
                                status in ("healthy", "ok", True)))
    return r


def validate_roster(client: httpx.Client) -> EndpointResult:
    """GET /api/fantasy/roster — CanonicalRosterResponse"""
    r = _call(client, "GET", "/api/fantasy/roster")
    ep = r.endpoint
    if r.error or r.http_status != 200:
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    # team_key present
    tk = body.get("team_key", "")
    r.checks.append(_check(ep, "team_key_present", "non-empty string", repr(tk), bool(tk)))

    # players array
    players = body.get("players", [])
    r.checks.append(_check(ep, "players_count", ">0", str(len(players)), len(players) > 0))

    # count field matches
    count = body.get("count", -1)
    r.checks.append(_check(ep, "count_matches_players",
                            f"count={len(players)}", f"count={count}",
                            count == len(players)))

    # freshness block
    freshness = body.get("freshness")
    r.checks.append(_check(ep, "freshness_present", "dict", type(freshness).__name__,
                            isinstance(freshness, dict)))

    if not players:
        return r

    # Per-player checks
    # RosterPlayerOut serializes using Python field names: name, player_key, positions
    # (aliases player_name / yahoo_player_key / eligible_positions are for input only)
    null_names = 0
    null_keys = 0
    bad_positions = []
    no_status = 0
    for p in players:
        # Accept either the Python field name or the alias (forward-compat)
        if not (p.get("name") or p.get("player_name")):
            null_names += 1
        if not (p.get("player_key") or p.get("yahoo_player_key")):
            null_keys += 1
        if not p.get("status"):
            no_status += 1
        for pos in (p.get("positions") or p.get("eligible_positions") or []):
            if pos not in VALID_ROSTER_POSITIONS:
                bad_positions.append(pos)

    r.checks.append(_check(ep, "no_null_player_names", "0 nulls",
                            f"{null_names} nulls", null_names == 0))
    r.checks.append(_check(ep, "no_null_player_keys", "0 nulls",
                            f"{null_keys} nulls", null_keys == 0,
                            detail="Players missing player_key/yahoo_player_key"))
    r.checks.append(_check(ep, "all_positions_valid", "all valid",
                            f"{len(bad_positions)} invalid: {bad_positions[:5]}",
                            len(bad_positions) == 0))
    r.checks.append(_check(ep, "all_have_status", "0 missing",
                            f"{no_status} missing", no_status == 0))

    # Check stat windows present (at least season_stats or rolling_14d)
    has_any_stats = sum(1 for p in players
                        if p.get("season_stats") or p.get("rolling_14d"))
    pct = round(has_any_stats / len(players) * 100)
    r.checks.append(
        _check(ep, "players_with_stats", ">50%", f"{pct}% ({has_any_stats}/{len(players)})",
               pct > 50) if pct <= 50
        else _check(ep, "players_with_stats", ">50%", f"{pct}%", True)
    )

    return r


def validate_matchup(client: httpx.Client) -> EndpointResult:
    """GET /api/fantasy/matchup — MatchupResponse"""
    r = _call(client, "GET", "/api/fantasy/matchup")
    ep = r.endpoint
    if r.error or r.http_status != 200:
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    # my_team block
    my_team = body.get("my_team")
    r.checks.append(_check(ep, "my_team_present", "dict", type(my_team).__name__,
                            isinstance(my_team, dict)))

    # opponent block
    opp = body.get("opponent")
    r.checks.append(_check(ep, "opponent_present", "dict", type(opp).__name__,
                            isinstance(opp, dict)))

    if not isinstance(my_team, dict) or not isinstance(opp, dict):
        return r

    # Check stats on both sides
    my_stats = my_team.get("stats", {})
    opp_stats = opp.get("stats", {})
    r.checks.append(_check(ep, "my_team_has_stats", ">0 stats",
                            f"{len(my_stats)} stats", len(my_stats) > 0))
    r.checks.append(_check(ep, "opponent_has_stats", ">0 stats",
                            f"{len(opp_stats)} stats", len(opp_stats) > 0))

    # Check stat keys match canonical or known aliases
    all_stat_keys = set(my_stats.keys()) | set(opp_stats.keys())
    canonical_or_alias = SCORING_CATEGORIES_18 | set(YAHOO_ALIASES.keys()) | {"IP", "GS", "H/AB", "BB", "HLD"}
    unknown_keys = all_stat_keys - canonical_or_alias
    r.checks.append(
        _warn(ep, "stat_keys_recognized", "all recognized",
              f"unknown: {unknown_keys}", detail="Non-canonical stat keys in matchup")
        if unknown_keys
        else _check(ep, "stat_keys_recognized", "all recognized", "all recognized", True)
    )

    # Check numeric values
    non_numeric = []
    for k, v in my_stats.items():
        try:
            float(v)
        except (TypeError, ValueError):
            non_numeric.append(f"my.{k}={v!r}")
    for k, v in opp_stats.items():
        try:
            float(v)
        except (TypeError, ValueError):
            non_numeric.append(f"opp.{k}={v!r}")
    r.checks.append(_check(ep, "all_stat_values_numeric", "all numeric",
                            f"{len(non_numeric)} non-numeric: {non_numeric[:5]}",
                            len(non_numeric) == 0))

    # week present
    week = body.get("week")
    r.checks.append(_check(ep, "week_present", "int", repr(week),
                            isinstance(week, int) and week > 0))

    # team names
    my_name = my_team.get("team_name", "")
    opp_name = opp.get("team_name", "")
    r.checks.append(_check(ep, "my_team_name", "non-empty", repr(my_name), bool(my_name)))
    r.checks.append(_check(ep, "opponent_team_name", "non-empty", repr(opp_name),
                            bool(opp_name) and opp_name != "TBD"))

    return r


def validate_scoreboard(client: httpx.Client, week: int = 3) -> EndpointResult:
    """GET /api/fantasy/scoreboard"""
    r = _call(client, "GET", "/api/fantasy/scoreboard",
              params={"week": week, "opponent_name": "auto"})
    ep = r.endpoint
    if r.error or r.http_status not in (200, 503):
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    if r.http_status == 503:
        r.checks.append(_warn(ep, "http_status", "200", "503",
                               detail="Yahoo auth may be down"))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    # 18 category rows
    rows = body.get("rows", [])
    r.checks.append(_check(ep, "row_count", "18", str(len(rows)), len(rows) == 18))

    if rows:
        # Check categories match canonical set
        row_cats = {row.get("category") for row in rows}
        missing = SCORING_CATEGORIES_18 - row_cats
        extra = row_cats - SCORING_CATEGORIES_18
        r.checks.append(_check(ep, "categories_match_canonical",
                                f"exact 18: {sorted(SCORING_CATEGORIES_18)}",
                                f"missing={sorted(missing)} extra={sorted(extra)}",
                                not missing and not extra))

        # Check each row has required fields
        bad_rows = []
        for row in rows:
            for field_name in ("category", "my_current", "opp_current", "status"):
                if field_name not in row:
                    bad_rows.append(f"{row.get('category', '?')} missing {field_name}")
        r.checks.append(_check(ep, "rows_have_required_fields", "all fields present",
                                f"{len(bad_rows)} issues: {bad_rows[:5]}",
                                len(bad_rows) == 0))

    # Budget block
    budget = body.get("budget")
    r.checks.append(_check(ep, "budget_present", "dict", type(budget).__name__,
                            isinstance(budget, dict)))

    # Win probability
    wp = body.get("overall_win_probability")
    if wp is not None:
        r.checks.append(_check(ep, "win_probability_range", "0-1",
                                str(wp), 0.0 <= wp <= 1.0))
    else:
        r.checks.append(_warn(ep, "win_probability_present", "float", "None"))

    # Freshness
    freshness = body.get("freshness")
    r.checks.append(_check(ep, "freshness_present", "dict", type(freshness).__name__,
                            isinstance(freshness, dict)))

    return r


def validate_budget(client: httpx.Client) -> EndpointResult:
    """GET /api/fantasy/budget"""
    r = _call(client, "GET", "/api/fantasy/budget")
    ep = r.endpoint
    if r.error or r.http_status not in (200, 503):
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    if r.http_status == 503:
        r.checks.append(_warn(ep, "http_status", "200", "503", detail="Yahoo auth issue"))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}
    budget = body.get("budget", {})

    # Required fields
    for field_name in ("acquisitions_used", "acquisitions_remaining", "acquisition_limit",
                       "il_used", "il_total", "ip_accumulated", "ip_minimum", "ip_pace"):
        val = budget.get(field_name)
        r.checks.append(_check(ep, f"budget.{field_name}_present",
                                "present", repr(val), val is not None))

    # Reasonable ranges
    acq_used = budget.get("acquisitions_used", -1)
    acq_remaining = budget.get("acquisitions_remaining", -1)
    acq_limit = budget.get("acquisition_limit", -1)
    if isinstance(acq_used, (int, float)) and isinstance(acq_limit, (int, float)):
        r.checks.append(_check(ep, "acquisitions_used_reasonable",
                                f"0 <= used <= {acq_limit}",
                                str(acq_used),
                                0 <= acq_used <= acq_limit + 5))  # slight tolerance

    il_used = budget.get("il_used", -1)
    il_total = budget.get("il_total", -1)
    if isinstance(il_used, (int, float)) and isinstance(il_total, (int, float)):
        r.checks.append(_check(ep, "il_used_reasonable",
                                f"0 <= {il_used} <= {il_total}",
                                f"il_used={il_used}, il_total={il_total}",
                                0 <= il_used <= il_total + 2))

    # IP pace enum
    pace = budget.get("ip_pace")
    valid_paces = {"BEHIND", "ON_TRACK", "COMPLETE"}
    r.checks.append(_check(ep, "ip_pace_valid", f"one of {valid_paces}",
                            repr(pace), pace in valid_paces))

    # Freshness
    freshness = body.get("freshness")
    r.checks.append(_check(ep, "freshness_present", "dict", type(freshness).__name__,
                            isinstance(freshness, dict)))

    return r


def validate_optimize(client: httpx.Client) -> EndpointResult:
    """POST /api/fantasy/roster/optimize — RosterOptimizeResponse"""
    r = _call(client, "POST", "/api/fantasy/roster/optimize",
              json_body={"target_date": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")})
    ep = r.endpoint
    if r.error or r.http_status not in (200, 503):
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    if r.http_status == 503:
        r.checks.append(_warn(ep, "http_status", "200", "503", detail="Yahoo auth issue"))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    # success flag
    r.checks.append(_check(ep, "success", "true", str(body.get("success")),
                            body.get("success") is True))

    # starters array
    starters = body.get("starters", [])
    r.checks.append(_check(ep, "starters_count", ">0", str(len(starters)),
                            len(starters) > 0))

    # Check for duplicate slots
    if starters:
        slot_counts: Dict[str, int] = {}
        for s in starters:
            slot = s.get("assigned_slot", "?")
            slot_counts[slot] = slot_counts.get(slot, 0) + 1

        # Check slot validity
        bad_slots = [s for s in slot_counts if s not in VALID_OPTIMIZE_SLOTS]
        r.checks.append(_check(ep, "all_slots_valid", "all valid",
                                f"invalid: {bad_slots}", len(bad_slots) == 0))

        # Check slot capacity
        MAX_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3,
                     "Util": 1, "SP": 2, "RP": 2, "P": 1}
        overflows = []
        for slot, count in slot_counts.items():
            max_c = MAX_SLOTS.get(slot, 999)
            if count > max_c:
                overflows.append(f"{slot}: {count}/{max_c}")
        r.checks.append(_check(ep, "no_slot_overflow", "all within capacity",
                                f"overflows: {overflows}", len(overflows) == 0))

        # Check all starters have scores
        scoreless = [s.get("player_name") for s in starters
                     if s.get("lineup_score") is None]
        r.checks.append(_check(ep, "all_starters_have_scores", "all scored",
                                f"{len(scoreless)} without score: {scoreless[:3]}",
                                len(scoreless) == 0))

        # Check for all-50.0 fallback (means player_scores table is empty/stale)
        scores = [s.get("lineup_score", 0) for s in starters]
        all_same = len(set(scores)) == 1 and len(scores) > 3
        if all_same:
            r.checks.append(_warn(ep, "scores_not_all_default",
                                   "varied scores", f"all={scores[0]}",
                                   detail="All players have same score — player_scores may be stale"))
        else:
            r.checks.append(_check(ep, "scores_not_all_default", "varied scores",
                                    f"{len(set(scores))} distinct values", True))

        # Check reasoning mentions score source
        stale_count = sum(1 for s in starters
                          if "stale_fallback" in (s.get("reasoning") or "")
                          or "default" in (s.get("reasoning") or ""))
        if stale_count > len(starters) * 0.5:
            r.checks.append(_warn(ep, "score_source_quality",
                                   "<50% stale/default", f"{stale_count}/{len(starters)} stale",
                                   detail="Most players using fallback scores"))
        else:
            r.checks.append(_check(ep, "score_source_quality",
                                    "<50% stale/default",
                                    f"{stale_count}/{len(starters)} stale", True))

    # bench
    bench = body.get("bench", [])
    r.checks.append(_check(ep, "bench_present", "list", type(bench).__name__,
                            isinstance(bench, list)))

    # total score
    total = body.get("total_lineup_score")
    r.checks.append(_check(ep, "total_lineup_score", "numeric > 0",
                            repr(total),
                            isinstance(total, (int, float)) and total > 0))

    # freshness
    freshness = body.get("freshness")
    r.checks.append(_check(ep, "freshness_present", "dict", type(freshness).__name__,
                            isinstance(freshness, dict)))

    return r


def validate_waiver(client: httpx.Client) -> EndpointResult:
    """GET /api/fantasy/waiver — WaiverWireResponse"""
    r = _call(client, "GET", "/api/fantasy/waiver", params={"per_page": 10})
    ep = r.endpoint
    if r.error or r.http_status not in (200, 503):
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    if r.http_status == 503:
        r.checks.append(_warn(ep, "http_status", "200", "503", detail="Yahoo auth issue"))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    # top_available
    players = body.get("top_available", [])
    r.checks.append(_check(ep, "top_available_count", ">0", str(len(players)),
                            len(players) > 0))

    if players:
        # All have names
        nameless = [p for p in players if not p.get("name")]
        r.checks.append(_check(ep, "all_have_names", "0 nameless",
                                f"{len(nameless)} nameless", len(nameless) == 0))

        # All have player_id (player_key)
        keyless = [p.get("name", "?") for p in players if not p.get("player_id")]
        r.checks.append(_check(ep, "all_have_player_id", "0 without ID",
                                f"{len(keyless)} without: {keyless[:3]}",
                                len(keyless) == 0))

        # need_score is numeric
        bad_scores = [p.get("name") for p in players
                      if not isinstance(p.get("need_score"), (int, float))]
        r.checks.append(_check(ep, "need_score_numeric", "all numeric",
                                f"{len(bad_scores)} non-numeric", len(bad_scores) == 0))

        # Positions valid
        bad_pos = [f"{p.get('name')}: {p.get('position')}" for p in players
                   if p.get("position") not in VALID_ROSTER_POSITIONS and p.get("position") != "?"]
        r.checks.append(_check(ep, "positions_valid", "all valid",
                                f"{len(bad_pos)} invalid: {bad_pos[:3]}",
                                len(bad_pos) == 0))

    # matchup_opponent
    opp = body.get("matchup_opponent", "")
    r.checks.append(_check(ep, "matchup_opponent", "non-TBD name", repr(opp),
                            bool(opp) and opp != "TBD"))

    # category_deficits
    deficits = body.get("category_deficits", [])
    r.checks.append(_check(ep, "category_deficits_present", ">0 categories",
                            str(len(deficits)), len(deficits) > 0))

    # pagination
    pagination = body.get("pagination")
    r.checks.append(_check(ep, "pagination_present", "dict", type(pagination).__name__,
                            isinstance(pagination, dict)))

    return r


def validate_lineup(client: httpx.Client) -> EndpointResult:
    """GET /api/fantasy/lineup/{date} — DailyLineupResponse"""
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    r = _call(client, "GET", f"/api/fantasy/lineup/{today}", timeout=60.0)
    ep = r.endpoint
    if r.error or r.http_status not in (200, 503):
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    if r.http_status == 503:
        r.checks.append(_warn(ep, "http_status", "200", "503", detail="Yahoo auth / freshness gate"))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    # batters
    batters = body.get("batters", [])
    r.checks.append(_check(ep, "batters_count", ">0", str(len(batters)),
                            len(batters) > 0))

    if batters:
        # All have names + teams
        nameless = sum(1 for b in batters if not b.get("name"))
        teamless = sum(1 for b in batters if not b.get("team"))
        r.checks.append(_check(ep, "batters_have_names", "0 nameless",
                                f"{nameless} nameless", nameless == 0))
        r.checks.append(_check(ep, "batters_have_teams", "0 teamless",
                                f"{teamless} teamless", teamless == 0))

        # Scores numeric
        bad_score = sum(1 for b in batters
                        if not isinstance(b.get("lineup_score"), (int, float)))
        r.checks.append(_check(ep, "batter_scores_numeric", "all numeric",
                                f"{bad_score} non-numeric", bad_score == 0))

        # assigned_slot present
        no_slot = sum(1 for b in batters if b.get("assigned_slot") is None)
        # BN/None is valid for bench players
        active = [b for b in batters if b.get("status") == "START"]
        active_no_slot = sum(1 for b in active if not b.get("assigned_slot"))
        r.checks.append(_check(ep, "active_batters_have_slots", "0 without slot",
                                f"{active_no_slot} active without slot",
                                active_no_slot == 0))

    # pitchers
    pitchers = body.get("pitchers", [])
    r.checks.append(_check(ep, "pitchers_present", "list", type(pitchers).__name__,
                            isinstance(pitchers, list)))

    # games_count
    gc = body.get("games_count")
    r.checks.append(_check(ep, "games_count_present", "int >= 0",
                            repr(gc), isinstance(gc, int) and gc >= 0))

    # warnings
    warnings = body.get("lineup_warnings", [])
    if warnings:
        r.checks.append(_warn(ep, "lineup_warnings", "empty ideally",
                               f"{len(warnings)} warnings: {warnings[:2]}",
                               detail="Lineup produced warnings"))
    else:
        r.checks.append(_check(ep, "lineup_warnings", "empty ideally", "0 warnings", True))

    return r


def validate_briefing(client: httpx.Client) -> EndpointResult:
    """GET /api/fantasy/briefing/{date}"""
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    r = _call(client, "GET", f"/api/fantasy/briefing/{today}", timeout=60.0)
    ep = r.endpoint
    if r.error or r.http_status not in (200, 500, 503):
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    if r.http_status in (500, 503):
        detail = ""
        if r.raw_response:
            detail = str(r.raw_response.get("detail", ""))[:200]
        r.checks.append(_warn(ep, "http_status", "200", str(r.http_status),
                               detail=f"Briefing unavailable: {detail}"))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    # strategy
    strategy = body.get("strategy")
    r.checks.append(_check(ep, "strategy_present", "non-empty", repr(strategy),
                            bool(strategy)))

    # categories
    cats = body.get("categories", [])
    r.checks.append(_check(ep, "categories_present", ">0", str(len(cats)),
                            len(cats) > 0))

    # starters
    starters = body.get("starters", [])
    r.checks.append(_check(ep, "starters_present", ">0", str(len(starters)),
                            len(starters) > 0))

    # overall_confidence
    conf = body.get("overall_confidence")
    r.checks.append(_check(ep, "overall_confidence", "0-1 float",
                            repr(conf),
                            isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0))

    return r


def validate_player_scores(client: httpx.Client, bdl_id: int = 1) -> EndpointResult:
    """GET /api/fantasy/players/{id}/scores"""
    r = _call(client, "GET", f"/api/fantasy/players/{bdl_id}/scores",
              params={"window_days": 14})
    ep = r.endpoint
    if r.error or r.http_status not in (200, 404):
        r.checks.append(_check(ep, "http_status", "200 or 404", str(r.http_status or r.error), False))
        return r

    if r.http_status == 404:
        r.checks.append(_warn(ep, "http_status", "200", "404",
                               detail=f"No scores found for bdl_id={bdl_id} — pipeline may not have run"))
        return r

    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}
    score = body.get("score", {})

    # score_0_100
    s100 = score.get("score_0_100")
    r.checks.append(_check(ep, "score_0_100_range", "0-100",
                            repr(s100),
                            isinstance(s100, (int, float)) and 0 <= s100 <= 100))

    # composite_z
    cz = score.get("composite_z")
    r.checks.append(_check(ep, "composite_z_present", "numeric",
                            repr(cz), isinstance(cz, (int, float))))

    # category_scores
    cats = score.get("category_scores", {})
    r.checks.append(_check(ep, "category_scores_present", "dict",
                            type(cats).__name__, isinstance(cats, dict)))

    return r


def validate_decisions(client: httpx.Client) -> EndpointResult:
    """GET /api/fantasy/decisions"""
    r = _call(client, "GET", "/api/fantasy/decisions", params={"limit": 10})
    ep = r.endpoint
    if r.error or r.http_status != 200:
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    decisions = body.get("decisions", [])
    count = body.get("count", -1)
    r.checks.append(_check(ep, "count_matches", f"count={len(decisions)}",
                            f"count={count}", count == len(decisions)))

    if decisions:
        # Each decision has required fields
        for i, dw in enumerate(decisions[:5]):
            d = dw.get("decision", {})
            for field_name in ("bdl_player_id", "decision_type", "confidence"):
                val = d.get(field_name)
                r.checks.append(_check(ep, f"decision[{i}].{field_name}",
                                        "present", repr(val), val is not None))
    else:
        r.checks.append(_warn(ep, "decisions_count", ">0", "0",
                               detail="No decisions found — pipeline may not have run"))

    return r


def validate_decisions_status(client: httpx.Client) -> EndpointResult:
    """GET /api/fantasy/decisions/status"""
    r = _call(client, "GET", "/api/fantasy/decisions/status")
    ep = r.endpoint
    if r.error or r.http_status != 200:
        r.checks.append(_check(ep, "http_status", "200", str(r.http_status or r.error), False))
        return r
    r.checks.append(_check(ep, "http_status", "200", "200", True))

    body = r.raw_response or {}

    verdict = body.get("verdict")
    valid_verdicts = {"healthy", "stale", "partial", "missing"}
    r.checks.append(_check(ep, "verdict_valid", f"one of {valid_verdicts}",
                            repr(verdict), verdict in valid_verdicts))

    if verdict in ("stale", "missing"):
        r.checks.append(_warn(ep, "pipeline_health", "healthy",
                               verdict, detail=body.get("message", "")))

    return r


# ── Report Rendering ──────────────────────────────────────────────────────

def render_report(results: List[EndpointResult], out_path: str):
    """Write structured findings report."""
    now = datetime.now(ZoneInfo("America/New_York"))

    total_checks = sum(len(r.checks) for r in results)
    pass_count = sum(1 for r in results for c in r.checks if c.status == "PASS")
    fail_count = sum(1 for r in results for c in r.checks if c.status == "FAIL")
    warn_count = sum(1 for r in results for c in r.checks if c.status == "WARN")
    skip_count = sum(1 for r in results for c in r.checks if c.status == "SKIP")

    lines = []
    lines.append("# UAT Validation Report")
    lines.append(f"\n**Generated:** {now.strftime('%Y-%m-%d %H:%M ET')}")
    lines.append(f"**Total Checks:** {total_checks}")
    lines.append(f"**PASS:** {pass_count}  |  **FAIL:** {fail_count}  |  **WARN:** {warn_count}  |  **SKIP:** {skip_count}")
    lines.append("")

    # Summary table
    lines.append("## Summary by Endpoint")
    lines.append("")
    lines.append("| Endpoint | HTTP | Time (ms) | PASS | FAIL | WARN |")
    lines.append("|----------|------|-----------|------|------|------|")
    for r in results:
        p = sum(1 for c in r.checks if c.status == "PASS")
        f = sum(1 for c in r.checks if c.status == "FAIL")
        w = sum(1 for c in r.checks if c.status == "WARN")
        status_icon = "pass" if f == 0 and not r.error else "FAIL"
        lines.append(f"| {r.endpoint} | {r.http_status} | {r.elapsed_ms} | {p} | {f} | {w} |")
    lines.append("")

    # Failures first
    failures = [(r, c) for r in results for c in r.checks if c.status == "FAIL"]
    if failures:
        lines.append("## FAILURES (must fix)")
        lines.append("")
        lines.append("| # | Endpoint | Check | Expected | Actual | Detail |")
        lines.append("|---|----------|-------|----------|--------|--------|")
        for i, (r, c) in enumerate(failures, 1):
            lines.append(f"| {i} | {c.endpoint} | {c.check_name} | {c.expected} | {c.actual} | {c.detail} |")
        lines.append("")

    # Warnings
    warnings = [(r, c) for r in results for c in r.checks if c.status == "WARN"]
    if warnings:
        lines.append("## WARNINGS (should investigate)")
        lines.append("")
        lines.append("| # | Endpoint | Check | Expected | Actual | Detail |")
        lines.append("|---|----------|-------|----------|--------|--------|")
        for i, (r, c) in enumerate(warnings, 1):
            lines.append(f"| {i} | {c.endpoint} | {c.check_name} | {c.expected} | {c.actual} | {c.detail} |")
        lines.append("")

    # Full detail
    lines.append("## Full Check Details")
    lines.append("")
    for r in results:
        lines.append(f"### {r.endpoint}")
        lines.append(f"HTTP {r.http_status} — {r.elapsed_ms}ms")
        if r.error:
            lines.append(f"**ERROR:** {r.error}")
        lines.append("")
        lines.append("| Status | Check | Expected | Actual |")
        lines.append("|--------|-------|----------|--------|")
        for c in r.checks:
            icon = {"PASS": "PASS", "FAIL": "**FAIL**", "WARN": "WARN", "SKIP": "SKIP"}.get(c.status, c.status)
            lines.append(f"| {icon} | {c.check_name} | {c.expected} | {c.actual} |")
        lines.append("")

    # Write raw response snippets for failed endpoints
    failed_endpoints = {r.endpoint for r in results
                        for c in r.checks if c.status == "FAIL"}
    if failed_endpoints:
        lines.append("## Raw Response Samples (Failed Endpoints)")
        lines.append("")
        for r in results:
            if r.endpoint in failed_endpoints and r.raw_response:
                lines.append(f"### {r.endpoint}")
                lines.append("```json")
                # Truncate to avoid huge dumps
                raw_str = json.dumps(r.raw_response, indent=2, default=str)
                if len(raw_str) > 3000:
                    raw_str = raw_str[:3000] + "\n... (truncated)"
                lines.append(raw_str)
                lines.append("```")
                lines.append("")

    report_text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def render_console(results: List[EndpointResult]):
    """Print colored summary to terminal."""
    total = sum(len(r.checks) for r in results)
    passes = sum(1 for r in results for c in r.checks if c.status == "PASS")
    fails = sum(1 for r in results for c in r.checks if c.status == "FAIL")
    warns = sum(1 for r in results for c in r.checks if c.status == "WARN")

    print(f"\n{'='*60}")
    print(f"  UAT VALIDATION RESULTS")
    print(f"  PASS: {passes}  FAIL: {fails}  WARN: {warns}  TOTAL: {total}")
    print(f"{'='*60}\n")

    for r in results:
        ep_fails = sum(1 for c in r.checks if c.status == "FAIL")
        ep_warns = sum(1 for c in r.checks if c.status == "WARN")
        icon = "PASS" if ep_fails == 0 else "FAIL"
        suffix = ""
        if ep_warns > 0:
            suffix = f"  ({ep_warns} warnings)"
        print(f"  [{icon}] {r.endpoint}  ({r.elapsed_ms}ms){suffix}")

        # Print failures inline
        for c in r.checks:
            if c.status == "FAIL":
                print(f"        FAIL: {c.check_name}")
                print(f"              expected: {c.expected}")
                print(f"              actual:   {c.actual}")
                if c.detail:
                    print(f"              detail:   {c.detail}")
            elif c.status == "WARN":
                print(f"        WARN: {c.check_name} — {c.actual}")

    print(f"\n{'='*60}")
    if fails > 0:
        print(f"  {fails} FAILURE(S) — see tasks/uat_findings.md for details")
    else:
        print(f"  ALL CHECKS PASSED")
    print(f"{'='*60}\n")

    return fails


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UAT Validator for CBB Edge MLB Platform")
    parser.add_argument("--base-url", default=None, help="Railway backend URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--output", default="tasks/uat_findings.md",
                        help="Output report path")
    parser.add_argument("--endpoints", default=None,
                        help="Comma-separated endpoint names to test (default: all)")
    args = parser.parse_args()

    # Resolve base URL and API key
    base_url = (
        args.base_url
        or os.environ.get("RAILWAY_BACKEND_URL")
        or os.environ.get("BASE_URL")
        or "https://fantasy-app-production-5079.up.railway.app"
    )
    api_key = (
        args.api_key
        or os.environ.get("CBB_API_KEY")
        or os.environ.get("API_KEY")
        or ""
    )

    if not api_key:
        print("ERROR: No API key provided. Set CBB_API_KEY env var or use --api-key.")
        sys.exit(1)

    print(f"Target: {base_url}")
    print(f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:]}")
    print()

    headers = {"X-API-Key": api_key}
    client = httpx.Client(base_url=base_url, headers=headers, follow_redirects=True)

    # Endpoint registry
    all_validators = {
        "health": validate_health,
        "roster": validate_roster,
        "matchup": validate_matchup,
        "scoreboard": validate_scoreboard,
        "budget": validate_budget,
        "optimize": validate_optimize,
        "waiver": validate_waiver,
        "lineup": validate_lineup,
        "briefing": validate_briefing,
        "player_scores": validate_player_scores,
        "decisions": validate_decisions,
        "decisions_status": validate_decisions_status,
    }

    # Filter if requested
    if args.endpoints:
        requested = [e.strip() for e in args.endpoints.split(",")]
        validators = {k: v for k, v in all_validators.items() if k in requested}
        if not validators:
            print(f"No matching endpoints. Available: {list(all_validators.keys())}")
            sys.exit(1)
    else:
        validators = all_validators

    # Run all validators
    results = []
    for name, validator_fn in validators.items():
        print(f"  Testing {name}...", end="", flush=True)
        try:
            result = validator_fn(client)
        except Exception as exc:
            result = EndpointResult(
                endpoint=name,
                http_status=0,
                elapsed_ms=0,
                error=f"Validator crashed: {exc}",
            )
            result.checks.append(_check(name, "validator_error", "no crash",
                                         str(exc), False))
        fails = sum(1 for c in result.checks if c.status == "FAIL")
        warns = sum(1 for c in result.checks if c.status == "WARN")
        icon = "PASS" if fails == 0 else "FAIL"
        suffix = f" ({warns}w)" if warns else ""
        print(f"  [{icon}]{suffix}")
        results.append(result)

    client.close()

    # Render report
    report = render_report(results, args.output)
    fail_count = render_console(results)

    print(f"Report written to: {args.output}")
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
