"""
Comprehensive mapping from The Odds API team names to KenPom names.
This is the single source of truth for name normalization.
"""

from __future__ import annotations

import logging
from typing import List

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

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
    "CSU Northridge Matadors": "Cal St. Northridge",
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
    if not name or not valid_choices:
        return None

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
    result = process.extractOne(clean_name, valid_choices, scorer=fuzz.token_set_ratio, score_cutoff=80)

    if result:
        # result is a tuple: (matched_string, score, index)
        logger.debug(f"Fuzzy matched '{name}' (cleaned: '{clean_name}') to '{result[0]}' with score {result[1]}")
        return result[0]

    # Final attempt with a lower-quality but more permissive scorer
    partial_result = process.extractOne(clean_name, valid_choices, scorer=fuzz.partial_ratio, score_cutoff=85)
    if partial_result:
        logger.debug(f"Partial matched '{name}' to '{partial_result[0]}' with score {partial_result[1]}")
        return partial_result[0]

    # If all else fails, we couldn't find a confident match.
    return None
