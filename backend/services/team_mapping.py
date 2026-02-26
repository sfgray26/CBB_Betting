"""
Comprehensive mapping from The Odds API team names to KenPom names.
This is the single source of truth for name normalization.
"""

from __future__ import annotations

import logging
from typing import List

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# High-priority manual overrides — checked FIRST, before ODDS_TO_KENPOM and
# before any fuzzy matching.  Use this dict for production-confirmed mismatches
# where the standard dictionary or fuzzy logic would produce a wrong result
# (e.g. short forms that have too little token overlap for a confident score).
#
# All entries here should also appear in ODDS_TO_KENPOM for backwards
# compatibility with callers that iterate the full dict directly.
# ---------------------------------------------------------------------------
_MANUAL_OVERRIDES: dict[str, str] = {
    "Sam Houston St Bearkats":       "Sam Houston State",
    "Sam Houston St":                "Sam Houston State",
    "Florida Int'l Golden Panthers": "FIU",
    "Florida Int'l":                 "FIU",
    "St. Thomas (MN) Tommies":       "St. Thomas - Minnesota",
    "St. Thomas (MN)":               "St. Thomas - Minnesota",
    "CSU Northridge Matadors":       "Cal State Northridge",
    "CSU Northridge":                "Cal State Northridge",
    "Cal Baptist Lancers":           "California Baptist",
    "Cal Baptist":                   "California Baptist",
    "Tenn-Martin Skyhawks":          "UT Martin",
    "Tenn-Martin":                   "UT Martin",
    "IUPUI Jaguars":                 "IU Indianapolis",
    "IUPUI":                         "IU Indianapolis",
}

# Primary, comprehensive mapping from Odds API names to KenPom names.
# KenPom is the "standard" name format we're normalizing to.
ODDS_TO_KENPOM: dict[str, str] = {
    # Direct mismatches from user report
    "Louisiana Ragin' Cajuns": "Louisiana",
    "Southern Miss Golden Eagles": "Southern Mississippi",
    "UConn Huskies": "Connecticut",
    "NC State Wolfpack": "North Carolina St.",
    "Ole Miss Rebels": "Mississippi",
    "Saint Mary's Gaels": "Saint Mary's CA",
    "VCU Rams": "VCU",
    "St. John's Red Storm": "St. John's NY",
    "Florida Atlantic Owls": "Florida Atlantic",
    "FAU Owls": "Florida Atlantic",
    "USC Trojans": "USC",
    "SMU Mustangs": "SMU",
    "LSU Tigers": "LSU",
    "BYU Cougars": "BYU",
    "TCU Horned Frogs": "TCU",
    "UTSA Roadrunners": "UTSA",
    "UTEP Miners": "UTEP",

    # Common Power 5 / Major Program Mappings
    "Alabama Crimson Tide": "Alabama",
    "Arkansas Razorbacks": "Arkansas",
    "Auburn Tigers": "Auburn",
    "Arizona Wildcats": "Arizona",
    "Arizona State Sun Devils": "Arizona St.",
    "Boston College Eagles": "Boston College",
    "Baylor Bears": "Baylor",
    "Clemson Tigers": "Clemson",
    "Colorado Buffaloes": "Colorado",
    "Duke Blue Devils": "Duke",
    "Florida Gators": "Florida",
    "Florida State Seminoles": "Florida St.",
    "Georgia Bulldogs": "Georgia",
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Gonzaga Bulldogs": "Gonzaga",
    "Iowa Hawkeyes": "Iowa",
    "Iowa State Cyclones": "Iowa St.",
    "Illinois Fighting Illini": "Illinois",
    "Indiana Hoosiers": "Indiana",
    "Kansas Jayhawks": "Kansas",
    "Kansas State Wildcats": "Kansas St.",
    "Kentucky Wildcats": "Kentucky",
    "Louisville Cardinals": "Louisville",
    "Maryland Terrapins": "Maryland",
    "Miami Hurricanes": "Miami FL",
    "Michigan Wolverines": "Michigan",
    "Michigan State Spartans": "Michigan St.",
    "Michigan St Spartans": "Michigan St.",
    "Minnesota Golden Gophers": "Minnesota",
    "Mississippi State Bulldogs": "Mississippi St.",
    "Missouri Tigers": "Missouri",
    "Marquette Golden Eagles": "Marquette",
    "Nebraska Cornhuskers": "Nebraska",
    "North Carolina Tar Heels": "North Carolina",
    "Notre Dame Fighting Irish": "Notre Dame",
    "Ohio State Buckeyes": "Ohio St.",
    "Oklahoma Sooners": "Oklahoma",
    "Oklahoma State Cowboys": "Oklahoma St.",
    "Oregon Ducks": "Oregon",
    "Oregon State Beavers": "Oregon St.",
    "Penn State Nittany Lions": "Penn St.",
    "Pittsburgh Panthers": "Pittsburgh",
    "Purdue Boilermakers": "Purdue",
    "Rutgers Scarlet Knights": "Rutgers",
    "South Carolina Gamecocks": "South Carolina",
    "Syracuse Orange": "Syracuse",
    "Stanford Cardinal": "Stanford",
    "Tennessee Volunteers": "Tennessee",
    "Texas Longhorns": "Texas",
    "Texas A&M Aggies": "Texas A&M",
    "Texas Tech Red Raiders": "Texas Tech",
    "UCLA Bruins": "UCLA",
    "Utah Utes": "Utah",
    "Vanderbilt Commodores": "Vanderbilt",
    "Virginia Cavaliers": "Virginia",
    "Virginia Tech Hokies": "Virginia Tech",
    "Wake Forest Demon Deacons": "Wake Forest",
    "Washington Huskies": "Washington",
    "Washington State Cougars": "Washington St.",
    "West Virginia Mountaineers": "West Virginia",
    "Wisconsin Badgers": "Wisconsin",
    "Xavier Musketeers": "Xavier",
    "Villanova Wildcats": "Villanova",
    "Creighton Bluejays": "Creighton",
    "Seton Hall Pirates": "Seton Hall",
    "Providence Friars": "Providence",
    "Butler Bulldogs": "Butler",
    "Houston Cougars": "Houston",
    "Cincinnati Bearcats": "Cincinnati",
    "Memphis Tigers": "Memphis",
    "Wichita State Shockers": "Wichita St.",
    "San Diego State Aztecs": "San Diego St.",
    "Boise State Broncos": "Boise St.",
    "UNLV Rebels": "UNLV",
    "Nevada Wolf Pack": "Nevada",
    "New Mexico Lobos": "New Mexico",
    "Utah State Aggies": "Utah St.",
    "Colorado State Rams": "Colorado St.",
    "Wyoming Cowboys": "Wyoming",
    "Air Force Falcons": "Air Force",
    "Fresno State Bulldogs": "Fresno St.",
    "San Jose State Spartans": "San Jose St.",
    "San Diego Toreros": "San Diego",

    # "Saint" and "St." variations
    "St. Bonaventure Bonnies": "St. Bonaventure",
    "St. Joseph's Hawks": "St. Joseph's",
    "Saint Joseph's Hawks": "St. Joseph's",
    "St. Louis Billikens": "Saint Louis",
    "Saint Louis Billikens": "Saint Louis",
    "St. Peter's Peacocks": "Saint Peter's",
    "Saint Peter's Peacocks": "Saint Peter's",
    "Mt. St. Mary's Mountaineers": "Mount St. Mary's",
    "Mount St. Mary's Mountaineers": "Mount St. Mary's",

    # "Cal St" and "CSU" variations
    "Cal State Fullerton Titans": "Cal St. Fullerton",
    "CSU Fullerton Titans": "Cal St. Fullerton",
    "Cal State Northridge Matadors": "Cal St. Northridge",
    "CSU Northridge Matadors": "Cal State Northridge",
    "CSU Northridge": "Cal State Northridge",
    "Cal State Bakersfield Roadrunners": "Cal St. Bakersfield",
    "CSU Bakersfield Roadrunners": "Cal St. Bakersfield",
    "CSU Bakersfield": "Cal St. Bakersfield",
    "Long Beach State Beach": "Long Beach St.",
    "Long Beach St 49ers": "Long Beach St.",
    "Sacramento State Hornets": "Sacramento St.",

    # "UC" schools
    "UC Berkeley Golden Bears": "California",
    "California Golden Bears": "California",
    "UC Irvine Anteaters": "UC Irvine",
    "UC Riverside Highlanders": "UC Riverside",
    "UC Santa Barbara Gauchos": "UC Santa Barbara",
    "UC San Diego Tritons": "UC San Diego",
    "UC Davis Aggies": "UC Davis",
    "UC Santa Cruz Banana Slugs": "UC Santa Cruz", # :)

    # Texas A&M regional campuses — MUST be listed explicitly so they never
    # fall through to fuzzy and land on "Texas A&M" (the main campus).
    "Texas A&M-CC Islanders": "Texas A&M-CC",
    "Texas A&M-CC": "Texas A&M-CC",
    "Texas A&M-Corpus Christi Islanders": "Texas A&M-CC",
    "Texas A&M-Corpus Christi": "Texas A&M-CC",
    "Texas A&M-Commerce Lions": "Texas A&M-Commerce",
    "Texas A&M-Commerce": "Texas A&M-Commerce",

    # Michigan schools — explicit entries so fuzzy never maps a regional
    # campus to the flagship "Michigan".
    "Central Michigan Chippewas": "Central Michigan",
    "Eastern Michigan Eagles": "Eastern Michigan",
    "Western Michigan Broncos": "Western Michigan",
    "Northern Michigan Wildcats": "Northern Michigan",

    # Patriot League / small-program names observed failing in production logs
    "American Eagles": "American",
    "Lehigh Mountain Hawks": "Lehigh",
    "Holy Cross Crusaders": "Holy Cross",
    "Bucknell Bison": "Bucknell",
    "Colgate Raiders": "Colgate",
    "Lafayette Leopards": "Lafayette",
    "Loyola (MD) Greyhounds": "Loyola MD",
    "Loyola Maryland Greyhounds": "Loyola MD",
    "Boston Univ. Terriers": "Boston University",
    "Boston University Terriers": "Boston University",

    # Horizon / Summit League
    "Cleveland St Vikings": "Cleveland St.",
    "Cleveland State Vikings": "Cleveland St.",
    "Northern Kentucky Norse": "Northern Kentucky",
    "Robert Morris Colonials": "Robert Morris",
    "Detroit Mercy Titans": "Detroit Mercy",
    "IUPUI Jaguars": "IUPUI",
    "Fort Wayne Mastodons": "Fort Wayne",
    "Wright St Raiders": "Wright St.",
    "Oakland Golden Grizzlies": "Oakland",

    # Big South / ASUN
    "Eastern Kentucky Colonels": "Eastern Kentucky",
    "Queens University Royals": "Queens NC",
    "Lipscomb Bisons": "Lipscomb",
    "West Georgia Wolves": "West Georgia",

    # Summit / WAC
    "South Dakota Coyotes": "South Dakota",
    "Omaha Mavericks": "Nebraska Omaha",

    # Southern / SoCon
    "East Tennessee St Buccaneers": "East Tennessee St.",
    "Wofford Terriers": "Wofford",
    "Furman Paladins": "Furman",
    "The Citadel Bulldogs": "The Citadel",
    "Western Carolina Catamounts": "Western Carolina",
    "Mercer Bears": "Mercer",
    "VMI Keydets": "VMI",
    "Samford Bulldogs": "Samford",

    # MEAC / SWAC
    "South Carolina St Bulldogs": "South Carolina St.",
    "Morgan St Bears": "Morgan St.",
    "Navy Midshipmen": "Navy",
    "Army Knights": "Army",

    # MVC
    "Northern Iowa Panthers": "Northern Iowa",
    "Illinois St Redbirds": "Illinois St.",
    "Indiana St Sycamores": "Indiana St.",
    "Missouri St Bears": "Missouri St.",
    "Bradley Braves": "Bradley",
    "Evansville Purple Aces": "Evansville",
    "Southern Illinois Salukis": "Southern Illinois",
    "Valparaiso Beacons": "Valparaiso",

    # A-10 / Atlantic 10
    "Charlotte 49ers": "Charlotte",
    "Saint Joseph's Hawks": "St. Joseph's",
    "Duquesne Dukes": "Duquesne",
    "Davidson Wildcats": "Davidson",
    "George Mason Patriots": "George Mason",
    "North Texas Mean Green": "North Texas",

    # Tricky mid-majors and other common mismatches
    "Appalachian State Mountaineers": "Appalachian St.",
    "Arkansas State Red Wolves": "Arkansas St.",
    "Ball State Cardinals": "Ball St.",
    "Bowling Green Falcons": "Bowling Green",
    "Coastal Carolina Chanticleers": "Coastal Carolina",
    "East Carolina Pirates": "East Carolina",
    "East Tennessee State Buccaneers": "East Tennessee St.",
    "ETSU Buccaneers": "East Tennessee St.",
    "Florida Gulf Coast Eagles": "Florida Gulf Coast",
    "FGCU Eagles": "Florida Gulf Coast",
    "George Mason Patriots": "George Mason",
    "George Washington Revolutionaries": "George Washington",
    "GW Revolutionaries": "George Washington",
    "Grand Canyon Antelopes": "Grand Canyon",
    "Indiana State Sycamores": "Indiana St.",
    "Jacksonville State Gamecocks": "Jacksonville St.",
    "James Madison Dukes": "James Madison",
    "Kennesaw State Owls": "Kennesaw St.",
    "Kent State Golden Flashes": "Kent St.",
    "Liberty Flames": "Liberty",
    "Louisiana Tech Bulldogs": "Louisiana Tech",
    "Louisiana-Monroe Warhawks": "Louisiana Monroe",
    "ULM Warhawks": "Louisiana Monroe",
    "Loyola Chicago Ramblers": "Loyola Chicago",
    "Loyola Marymount Lions": "Loyola Marymount",
    "LMU Lions": "Loyola Marymount",
    "Middle Tennessee Blue Raiders": "Middle Tennessee",
    "Missouri State Bears": "Missouri St.",
    "Murray State Racers": "Murray St.",
    "UNC Asheville Bulldogs": "UNC Asheville",
    "UNC Greensboro Spartans": "UNC Greensboro",
    "UNC Wilmington Seahawks": "UNC Wilmington",
    "North Dakota State Bison": "North Dakota St.",
    "North Texas Mean Green": "North Texas",
    "Northern Iowa Panthers": "Northern Iowa",
    "Ohio Bobcats": "Ohio",
    "Old Dominion Monarchs": "Old Dominion",
    "Oral Roberts Golden Eagles": "Oral Roberts",
    "Rice Owls": "Rice",
    "Richmond Spiders": "Richmond",
    "Sam Houston State Bearkats": "Sam Houston St.",
    "South Dakota State Jackrabbits": "South Dakota St.",
    "South Florida Bulls": "South Florida",
    "Southern Illinois Salukis": "Southern Illinois",
    "Stephen F. Austin Lumberjacks": "Stephen F. Austin",
    "SFA Lumberjacks": "Stephen F. Austin",
    "Texas State Bobcats": "Texas St.",
    "Towson Tigers": "Towson",
    "UAB Blazers": "UAB",
    "UMass Lowell River Hawks": "UMass Lowell",
    "UMBC Retrievers": "UMBC",
    "UT Arlington Mavericks": "UT Arlington",
    "UT-Arlington Mavericks": "UT Arlington",
    "UT Rio Grande Valley Vaqueros": "UT Rio Grande Valley",
    "UTRGV Vaqueros": "UT Rio Grande Valley",
    "Western Carolina Catamounts": "Western Carolina",
    "Western Kentucky Hilltoppers": "Western Kentucky",
    "Wright State Raiders": "Wright St.",
    "Youngstown State Penguins": "Youngstown St.",

    # Mid-major API-to-KenPom/Torvik discrepancies confirmed in production logs
    # (Feb 2026) — short forms and nickname-first variants that bypass fuzzy
    # matching because the token overlap is too low for a confident score.
    "Sam Houston St Bearkats": "Sam Houston State",
    "Sam Houston St": "Sam Houston State",
    "Florida Int'l Golden Panthers": "FIU",
    "Florida Int'l": "FIU",
    "St. Thomas (MN) Tommies": "St. Thomas - Minnesota",
    "St. Thomas (MN)": "St. Thomas - Minnesota",
    "Cal Baptist Lancers": "California Baptist",
    "Cal Baptist": "California Baptist",
    "Tenn-Martin Skyhawks": "UT Martin",
    "Tenn-Martin": "UT Martin",
}

# A list of mascots, used as a fallback if the main dictionary misses.
# More comprehensive than the original list.
COMMON_MASCOTS: list[str] = [
    "Aggies", "Antelopes", "Bears", "Bearcats", "Bearkats", "Beavers", "Big Green",
    "Big Red", "Billikens", "Bison", "Blazers", "Blue Devils", "Blue Hens", "Blue Raiders",
    "Bluejays", "Bobcats", "Boilermakers", "Bonnies", "Braves", "Broncos", "Bruins",
    "Buccaneers", "Buckeyes", "Buffaloes", "Bulldogs", "Bulls", "Camel", "Camels",
    "Cardinals", "Catamounts", "Cavaliers", "Chanticleers", "Chippewas", "Citadel",
    "Cornhuskers", "Cowboys", "Cougars", "Crimson", "Crimson Tide", "Crusaders", "Crusaders",
    "Demon Deacons", "Dragons", "Dukes", "Eagles", "Falcons", "Fighting Camels",
    "Fighting Hawks", "Fighting Illini", "Fighting Irish", "Flames", "Flyers", "Forty-Niners", "49ers",
    "Friars", "Gaels", "Gael", "Gators", "Gauchos", "Golden Eagles", "Golden Flashes", "Golden Gophers",
    "Golden Griffins", "Golden Grizzlies", "Golden Panthers", "Golden Hurricane", "Gophers",
    "Gamecocks", "Great Danes", "Green Wave", "Griffins", "Grizzlies", "Hawkeyes",
    "Hawks", "Highlanders", "Hokies", "Hoosiers", "Horned Frogs", "Hornets", "Huskies",
    "Huskers", "Illini", "Jackrabbits", "Jaguars", "Jaspers", "Jayhawks", "Kangaroos",
    "Keydets", "Knights", "Leopards", "Lions", "Lobos", "Longhorns", "Lumberjacks",
    "Marauders", "Matadors", "Mean Green", "Midshipmen", "Miners", "Minutemen", "Monarchs",
    "Mountaineers", "Musketeers", "Mustangs", "Nittany Lions", "Ospreys", "Orange", "Owls",
    "Paladins", "Panthers", "Peacocks", "Penguins", "Pioneers", "Pirates", "Privateers",
    "Quakers", "Racers", "Raiders", "Ragin' Cajuns", "Ramblers", "Rams", "Razorbacks",
    "Red Flash", "Red Storm", "Red Wolves", "Redbirds", "RedHawks", "Rebels", "Revolutionaries",
    "Rhode Island", "River Hawks", "Roadrunners", "Rockets", "Runnin' Bulldogs", "Salukis",
    "Samford", "Santa Clara", "Scarlet Knights", "Scorpions", "Seahawks", "Seawolves",
    "Seminoles", "Shockers", "Sooners", "Spiders", "Stags", "Sycamores",
    "Sun Devils", "Tigers", "Terrapins", "Texans", "Thunderbirds", "Titans",
    "Toreros", "Trojans", "Tribe", "Tritons", "Utes", "Vandals", "Vaqueros",
    "Vikings", "Volunteers", "Warhawks", "Western Michigan", "Wildcats", "Wolf Pack",
    "Wolverines", "Yellow Jackets", "Zips"
]


def _strip_mascot(name: str) -> str:
    """Helper to remove common mascots from the end of a team name."""
    # Sort mascots by length, longest first, to handle multi-word mascots
    sorted_mascots = sorted(COMMON_MASCOTS, key=len, reverse=True)
    for mascot in sorted_mascots:
        if name.endswith(f" {mascot}"):
            return name[:-(len(mascot) + 1)].strip()
    return name


def _is_dangerous_substring_match(query: str, matched: str) -> bool:
    """
    Returns True when a fuzzy match is likely a false positive caused by
    token_set_ratio's tolerance for extra tokens.

    Two failure patterns are caught:

    1. **Hyphen-suffix regional variants** — "Texas A&M-CC" fuzzy-matches
       "Texas A&M" because the shared prefix dominates the score.
       Detected by: query contains a hyphen, the match does not, and the
       pre-hyphen fragment of the query equals the match.

    2. **Short-base substring** — "Central Michigan Chippewas" fuzzy-matches
       "Michigan" because token_set_ratio ignores the extra "Central" and
       "Chippewas" tokens and returns 100 on the shared "Michigan" token.
       Detected by: one string is a case-insensitive substring of the other
       AND the character-level fuzz.ratio is below 75 (the strings differ
       substantially in length).
    """
    q = query.lower().strip()
    m = matched.lower().strip()

    # Guard 1: hyphen-suffix regional campus
    # e.g. "Texas A&M-CC" must never match "Texas A&M"
    if "-" in q and "-" not in m:
        base = q.split("-")[0].strip()
        if base == m or q.startswith(m + "-"):
            return True

    # Guard 2: one string is a bare substring of the other
    # AND character-level similarity is low (large length disparity)
    if m in q or q in m:
        if fuzz.ratio(q, m) < 75:
            return True

    return False


def normalize_team_name(name: str, valid_choices: list[str]) -> str | None:
    """
    Finds the best match for a team name against a list of valid choices.
    Uses a multi-step strategy for accuracy.

    Args:
        name: The raw team name from an external source (e.g., The Odds API).
        valid_choices: A list of all possible "correct" names (e.g., from KenPom).

    Returns:
        The best matching name from valid_choices, or None if no good match is found.
    """
    # Strip surrounding whitespace first so API names like "  IUPUI " still hit
    # the override dict before any downstream normalisation runs.
    name = name.strip()

    if not name or not valid_choices:
        return None

    # Strategy 0: High-priority manual overrides (O(1) exact lookup).
    # These are production-confirmed mismatches that must never fall through
    # to fuzzy matching.  Checked before ODDS_TO_KENPOM and before any scorer.
    if name in _MANUAL_OVERRIDES:
        mapped_name = _MANUAL_OVERRIDES[name]
        if mapped_name in valid_choices:
            return mapped_name
        # Mapped target not in valid_choices — pass the canonical name into
        # the remaining strategies so fuzzy has a better base to work with.
        name = mapped_name

    # Strategy 1: Direct hit in the primary mapping dictionary
    if name in ODDS_TO_KENPOM:
        mapped_name = ODDS_TO_KENPOM[name]
        if mapped_name in valid_choices:
            return mapped_name
        # If the mapped name isn't a direct hit, it might still be a better base for fuzzy matching
        name = mapped_name

    # Strategy 2: Exact match in the choices list (case-insensitive)
    # This is fast and handles cases where the name is already correct.
    for choice in valid_choices:
        if choice.lower() == name.lower():
            return choice

    # Strategy 3: Mascot stripping + exact match
    # Try cleaning the name by removing a mascot, then check for an exact match.
    clean_name = _strip_mascot(name)
    if clean_name in valid_choices:
        return clean_name

    # Strategy 4: Fuzzy matching (the fallback)
    # Use token_set_ratio, which is good for when one string is a subset of another
    # and word order doesn't matter (e.g., "St. John's NY" vs "St. John's").
    # Threshold: 85 — high enough to avoid cross-conference false positives
    # while still catching "Cleveland St Vikings" → "Cleveland St." etc.
    result = process.extractOne(clean_name, valid_choices, scorer=fuzz.token_set_ratio, score_cutoff=85)

    if result and _is_dangerous_substring_match(clean_name, result[0]):
        logger.warning(
            "Substring guard blocked fuzzy match '%s' → '%s'",
            clean_name, result[0],
        )
        result = None

    if result:
        # result is a tuple: (matched_string, score, index)
        logger.debug(f"Fuzzy matched '{name}' (cleaned: '{clean_name}') to '{result[0]}' with score {result[1]}")
        return result[0]

    # Final attempt with a lower-quality but more permissive scorer
    partial_result = process.extractOne(clean_name, valid_choices, scorer=fuzz.partial_ratio, score_cutoff=85)

    if partial_result and _is_dangerous_substring_match(clean_name, partial_result[0]):
        logger.warning(
            "Substring guard blocked partial match '%s' → '%s'",
            clean_name, partial_result[0],
        )
        partial_result = None

    if partial_result:
        logger.debug(f"Partial matched '{name}' to '{partial_result[0]}' with score {partial_result[1]}")
        return partial_result[0]

    # If all else fails, we couldn't find a confident match.
    return None
