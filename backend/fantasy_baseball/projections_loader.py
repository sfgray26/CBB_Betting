"""
Projection System Loader — Steamer / ZiPS / ATC / FantasyPros ADP

Loads real projection CSV files (from FanGraphs or Kimi research output)
and converts them into the player_board dict format.

When real CSV files are available they REPLACE the hardcoded player_board.py
estimates. When not available the hardcoded board serves as fallback.

Expected file locations (drop into data/projections/):
  data/projections/steamer_batting_2026.csv   — FanGraphs Steamer batters
  data/projections/steamer_pitching_2026.csv  — FanGraphs Steamer pitchers
  data/projections/zips_batting_2026.csv      — ZiPS batters (optional)
  data/projections/zips_pitching_2026.csv     — ZiPS pitchers (optional)
  data/projections/adp_yahoo_2026.csv         — FantasyPros Yahoo 12-team ADP

── Steamer batting CSV columns (FanGraphs export) ──────────────────────────
  Name, Team, G, PA, AB, H, 2B, 3B, HR, R, RBI, BB, SO, HBP, SF, AVG,
  OBP, SLG, OPS, wOBA, wRC+, BsR, Off, Def, WAR

── Steamer pitching CSV columns (FanGraphs export) ─────────────────────────
  Name, Team, W, L, ERA, G, GS, IP, H, ER, HR, BB, SO, WHIP,
  K/9, BB/9, K/BB, H/9, HR/9, AVG, BABIP, LOB%, GB%, HR/FB, FIP, xFIP, WAR

── FantasyPros ADP CSV columns ──────────────────────────────────────────────
  PLAYER NAME, TEAM, POS, AVG, BEST, WORST, # TEAMS, STDEV

Run this module standalone to validate loaded data:
  python -m backend.fantasy_baseball.projections_loader
"""

import csv
import logging
import os
import statistics
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "projections"

# ---------------------------------------------------------------------------
# Yahoo position eligibility map — used when parsing position strings
# ---------------------------------------------------------------------------
YAHOO_POS_NORMALIZE = {
    "C": ["C"], "1B": ["1B"], "2B": ["2B"], "3B": ["3B"], "SS": ["SS"],
    "LF": ["LF", "OF"], "CF": ["CF", "OF"], "RF": ["RF", "OF"],
    "OF": ["OF"], "DH": ["DH"],
    "SP": ["SP"], "RP": ["RP"], "P": ["SP", "RP"],
    "C/1B": ["C", "1B"], "C/OF": ["C", "OF"],
    "1B/OF": ["1B", "OF"], "2B/SS": ["2B", "SS"],
    "2B/3B": ["2B", "3B"], "3B/SS": ["3B", "SS"],
    "SS/2B": ["SS", "2B"], "SS/3B": ["SS", "3B"],
    "OF/1B": ["OF", "1B"], "OF/DH": ["OF", "DH"],
    "SP/RP": ["SP", "RP"],
}


def _normalize_positions(pos_str: str) -> list[str]:
    """Convert FanGraphs position string to list of Yahoo-eligible positions."""
    if not pos_str:
        return ["Util"]
    pos_str = pos_str.strip()
    if pos_str in YAHOO_POS_NORMALIZE:
        return YAHOO_POS_NORMALIZE[pos_str]
    # Try splitting on /
    parts = [p.strip() for p in pos_str.replace(",", "/").split("/")]
    result = []
    for p in parts:
        if p in ("LF", "CF", "RF"):
            result.append(p)
            if "OF" not in result:
                result.append("OF")
        elif p in ("SP", "RP", "C", "1B", "2B", "3B", "SS", "DH", "OF"):
            result.append(p)
    return result if result else ["Util"]


def _make_player_id(name: str) -> str:
    return (name.lower()
            .replace(" ", "_").replace(".", "").replace("'", "")
            .replace("é", "e").replace("á", "a").replace("ó", "o")
            .replace("ú", "u").replace("í", "i").replace("ñ", "n")
            .replace(",", "").replace("-", "_"))


# ---------------------------------------------------------------------------
# Steamer batting loader
# ---------------------------------------------------------------------------

def load_steamer_batting(path: Path) -> list[dict]:
    """
    Load FanGraphs Steamer batting projections CSV.
    Returns list of player dicts compatible with player_board format.
    """
    players = []
    if not path.exists():
        logger.warning(f"Steamer batting file not found: {path}")
        return players

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row.get("Name", row.get("name", "")).strip()
                team = row.get("Team", row.get("team", "FA")).strip().upper()
                pos_str = row.get("POS", row.get("Pos", row.get("pos", "OF"))).strip()
                positions = _normalize_positions(pos_str)

                pa = float(row.get("PA", 0) or 0)
                r = float(row.get("R", 0) or 0)
                h = float(row.get("H", 0) or 0)
                hr = float(row.get("HR", 0) or 0)
                rbi = float(row.get("RBI", 0) or 0)
                sb = float(row.get("SB", 0) or 0)
                cs = float(row.get("CS", 0) or 0)
                so = float(row.get("SO", 0) or 0)
                avg = float(row.get("AVG", 0) or 0)
                ops = float(row.get("OPS", 0) or 0)
                slg = float(row.get("SLG", 0) or 0)

                # Compute derived stats
                nsb = max(0, sb - cs)
                tb = round(h * slg / max(avg, 0.001)) if avg > 0 else 0

                player = {
                    "id": _make_player_id(name),
                    "name": name,
                    "team": team,
                    "positions": positions,
                    "type": "batter",
                    "tier": 0,    # Will be assigned after z-score ranking
                    "adp": 999.0, # Will be filled from ADP file
                    "rank": 0,
                    "proj": {
                        "pa": pa, "r": r, "h": h, "hr": hr, "rbi": rbi,
                        "k_bat": so, "tb": tb, "avg": avg, "ops": ops,
                        "nsb": nsb, "slg": slg,
                    },
                    "z_score": 0.0,
                    "cat_scores": {},
                    "source": "steamer",
                }
                players.append(player)
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping row {row.get('Name', '?')}: {e}")

    logger.info(f"Loaded {len(players)} Steamer batters from {path}")
    return players


# ---------------------------------------------------------------------------
# Steamer pitching loader
# ---------------------------------------------------------------------------

def load_steamer_pitching(path: Path) -> list[dict]:
    """
    Load FanGraphs Steamer pitching projections CSV.
    Separates SP (GS >= 10) from RP (GS < 5).
    """
    players = []
    if not path.exists():
        logger.warning(f"Steamer pitching file not found: {path}")
        return players

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row.get("Name", row.get("name", "")).strip()
                team = row.get("Team", row.get("team", "FA")).strip().upper()

                ip = float(row.get("IP", 0) or 0)
                w = float(row.get("W", 0) or 0)
                l = float(row.get("L", 0) or 0)
                sv = float(row.get("SV", 0) or 0)
                bs = float(row.get("BS", 0) or 0)
                gs = float(row.get("GS", 0) or 0)
                k = float(row.get("SO", row.get("K", row.get("SO", 0))) or 0)
                era = float(row.get("ERA", 4.5) or 4.5)
                whip = float(row.get("WHIP", 1.3) or 1.3)
                hr_pit = float(row.get("HR", 0) or 0)
                bb = float(row.get("BB", 0) or 0)

                k9 = (k / ip * 9) if ip > 0 else 0.0
                qs = round(gs * 0.55) if gs >= 10 else 0
                nsv = max(0, sv - bs)

                # Determine position type
                if gs >= 10:
                    positions = ["SP"]
                elif sv > 5 or (sv > 0 and ip < 80):
                    positions = ["RP"]
                else:
                    positions = ["SP", "RP"]

                player = {
                    "id": _make_player_id(name),
                    "name": name,
                    "team": team,
                    "positions": positions,
                    "type": "pitcher",
                    "tier": 0,
                    "adp": 999.0,
                    "rank": 0,
                    "proj": {
                        "ip": ip, "w": w, "l": l, "sv": sv, "bs": bs,
                        "qs": qs, "k_pit": k, "era": era, "whip": whip,
                        "k9": k9, "hr_pit": hr_pit, "nsv": nsv,
                    },
                    "z_score": 0.0,
                    "cat_scores": {},
                    "source": "steamer",
                }
                players.append(player)
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping row {row.get('Name', '?')}: {e}")

    logger.info(f"Loaded {len(players)} Steamer pitchers from {path}")
    return players


# ---------------------------------------------------------------------------
# ADP loader (FantasyPros Yahoo 12-team format)
# ---------------------------------------------------------------------------

def load_adp(path: Path) -> dict[str, float]:
    """
    Load FantasyPros consensus ADP CSV.
    Returns dict mapping normalized player name → ADP float.
    """
    adp_map = {}
    if not path.exists():
        logger.warning(f"ADP file not found: {path}")
        return adp_map

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (
                row.get("PLAYER NAME", row.get("Name", row.get("name", "")))
                .strip()
            )
            try:
                adp_val = float(
                    row.get("AVG", row.get("ADP", row.get("adp", 999))) or 999
                )
                if name:
                    adp_map[_make_player_id(name)] = adp_val
            except (ValueError, TypeError):
                pass

    logger.info(f"Loaded {len(adp_map)} ADP entries from {path}")
    return adp_map


def _apply_adp(players: list[dict], adp_map: dict[str, float]) -> None:
    """Merge ADP data into player list in-place."""
    matched = 0
    for p in players:
        pid = p["id"]
        if pid in adp_map:
            p["adp"] = adp_map[pid]
            matched += 1
        else:
            # Fallback: try partial name match
            name_lower = p["name"].lower()
            for adp_id, adp_val in adp_map.items():
                if adp_id.replace("_", " ") in name_lower:
                    p["adp"] = adp_val
                    matched += 1
                    break
    logger.info(f"ADP matched {matched}/{len(players)} players")


# ---------------------------------------------------------------------------
# Tier assignment based on z-score rank
# ---------------------------------------------------------------------------

def assign_tiers(players: list[dict]) -> None:
    """Assign tier 1-8 based on z-score rank within type, in-place."""
    batters = [p for p in players if p["type"] == "batter"]
    pitchers = [p for p in players if p["type"] == "pitcher"]

    tier_cutoffs_bat = [12, 36, 72, 108, 144, 180, 220, 999]
    tier_cutoffs_pit = [10, 30, 60, 90, 120, 160, 200, 999]

    batters.sort(key=lambda p: p.get("z_score", 0), reverse=True)
    pitchers.sort(key=lambda p: p.get("z_score", 0), reverse=True)

    for i, p in enumerate(batters):
        for tier, cutoff in enumerate(tier_cutoffs_bat, 1):
            if i < cutoff:
                p["tier"] = tier
                break

    for i, p in enumerate(pitchers):
        for tier, cutoff in enumerate(tier_cutoffs_pit, 1):
            if i < cutoff:
                p["tier"] = tier
                break


# ---------------------------------------------------------------------------
# Master loader — tries real CSVs, falls back to hardcoded board
# ---------------------------------------------------------------------------

def load_full_board(data_dir: Optional[Path] = None) -> Optional[list[dict]]:
    """
    Attempt to load real projection data from CSV files.
    Returns None if no CSV files found (caller falls back to player_board.py).

    Priority:
    1. Steamer 2026 (most accurate, publicly available on FanGraphs)
    2. ZiPS 2026 (optional second source for averaging)
    3. ATC (average of all systems — if available)
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)

    bat_path = data_dir / "steamer_batting_2026.csv"
    pit_path = data_dir / "steamer_pitching_2026.csv"
    adp_path = data_dir / "adp_yahoo_2026.csv"

    if not bat_path.exists() and not pit_path.exists():
        logger.info("No Steamer CSV files found — using hardcoded player board")
        return None

    batters = load_steamer_batting(bat_path)
    pitchers = load_steamer_pitching(pit_path)

    if not batters and not pitchers:
        return None

    adp_map = load_adp(adp_path) if adp_path.exists() else {}

    # Compute z-scores using same logic as player_board.py
    from backend.fantasy_baseball.player_board import _compute_zscores
    _compute_zscores(batters, pitchers)

    all_players = batters + pitchers

    # Deduplicate by player ID — keeps first occurrence (batters take priority
    # over pitchers, so two-way players like Ohtani are counted as batters).
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for p in all_players:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            deduped.append(p)
    if len(deduped) < len(all_players):
        logger.info(
            "Removed %d duplicate player entries (two-way players in both CSVs)",
            len(all_players) - len(deduped),
        )
    all_players = deduped

    _apply_adp(all_players, adp_map)
    assign_tiers(all_players)

    # Sort by ADP for final rank
    all_players.sort(key=lambda p: p["adp"])
    for i, p in enumerate(all_players, 1):
        p["rank"] = i

    logger.info(f"Loaded real projection board: {len(batters)} batters, {len(pitchers)} pitchers")
    return all_players


# ---------------------------------------------------------------------------
# CSV template generator — gives Kimi exact format to deliver data in
# ---------------------------------------------------------------------------

def write_csv_templates(data_dir: Optional[Path] = None) -> None:
    """
    Write empty CSV template files with correct column headers.
    Kimi or manual entry can populate these.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    bat_headers = ["Name", "Team", "POS", "G", "PA", "AB", "H", "2B", "3B",
                   "HR", "R", "RBI", "BB", "SO", "SB", "CS", "HBP", "SF",
                   "AVG", "OBP", "SLG", "OPS", "wOBA"]
    pit_headers = ["Name", "Team", "POS", "W", "L", "ERA", "G", "GS", "IP",
                   "H", "ER", "HR", "BB", "SO", "SV", "BS", "HLD", "WHIP",
                   "K/9", "BB/9", "FIP", "xFIP"]
    adp_headers = ["PLAYER NAME", "TEAM", "POS", "AVG", "BEST", "WORST",
                   "# TEAMS", "STDEV"]

    templates = [
        (data_dir / "steamer_batting_2026.csv", bat_headers),
        (data_dir / "steamer_pitching_2026.csv", pit_headers),
        (data_dir / "adp_yahoo_2026.csv", adp_headers),
    ]

    for path, headers in templates:
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"Created template: {path}")
        else:
            print(f"Already exists (not overwriting): {path}")


if __name__ == "__main__":
    print("Writing CSV templates to data/projections/...")
    write_csv_templates()
    print()
    result = load_full_board()
    if result:
        print(f"Loaded {len(result)} players from real projections")
    else:
        print("No real projection CSVs found — templates created.")
        print("Drop Steamer CSV exports from FanGraphs into data/projections/")
        print("and re-run to activate real projection mode.")
