"""
Tests for backend/services/team_mapping.py

Run:  pytest tests/test_team_mapping.py -v

Covers:
  - All five examples from the Gemini / user audit
  - The full class of abbreviated "St" variants (no period, no "State")
  - _MANUAL_OVERRIDES priority path
  - Fuzzy-matching fallback for previously unseen nicknames
  - Dangerous-substring guard (A&M-CC must never resolve to A&M)
  - Mascot stripping stage
"""

import pytest
from backend.services.team_mapping import (
    ODDS_TO_KENPOM,
    _MANUAL_OVERRIDES,
    normalize_team_name,
    _strip_mascot,
)

# ---------------------------------------------------------------------------
# A realistic subset of KenPom canonical names.
# normalize_team_name() requires a `valid_choices` list; these are the names
# KenPom actually returns for the teams under test.
# ---------------------------------------------------------------------------
KENPOM_NAMES = [
    # Power-conference schools
    "Alabama", "Arizona", "Arizona St.", "Arkansas", "Auburn",
    "Baylor", "Boston College", "BYU", "California", "Clemson",
    "Colorado", "Connecticut", "Creighton", "Duke", "Florida",
    "Florida St.", "Georgia", "Georgia St.", "Georgia Tech",
    "Gonzaga", "Houston", "Illinois", "Indiana", "Iowa",
    "Iowa St.", "Kansas", "Kansas St.", "Kentucky", "LSU",
    "Louisville", "Maryland", "Miami FL", "Michigan", "Michigan St.",
    "Minnesota", "Mississippi", "Mississippi St.", "Missouri",
    "Nebraska", "North Carolina", "North Carolina St.", "Notre Dame",
    "Ohio St.", "Oklahoma", "Oklahoma St.", "Oregon", "Oregon St.",
    "Penn St.", "Pittsburgh", "Purdue", "Rutgers", "SMU",
    "South Carolina", "Stanford", "Syracuse", "TCU", "Tennessee",
    "Texas", "Texas A&M", "Texas Tech", "UCLA", "USC",
    "Utah", "Vanderbilt", "Villanova", "Virginia", "Virginia Tech",
    "Wake Forest", "Washington", "Washington St.", "West Virginia",
    "Wisconsin", "Xavier",
    # Mountain West / AAC
    "Boise St.", "Colorado St.", "Fresno St.", "New Mexico",
    "San Diego St.", "San Jose St.", "UNLV", "Utah St.",
    "Wichita St.", "Wyoming",
    # Mid-majors
    "Appalachian St.", "Arkansas St.", "Ball St.", "Bradley",
    "Butler", "Charlotte", "Cincinnati", "Coastal Carolina",
    "Davidson", "Dayton", "Drake", "Duquesne",
    "East Tennessee St.", "Eastern Kentucky", "Evansville",
    "FIU", "Florida Atlantic", "Florida Gulf Coast",
    "George Mason", "George Washington", "Grand Canyon",
    "Holy Cross", "Illinois St.", "Indiana St.", "Jacksonville St.",
    "James Madison", "Kennesaw St.", "Kent St.",
    "Liberty", "Lipscomb", "Long Beach St.", "Louisiana",
    "Louisiana Monroe", "Louisiana Tech",
    "Loyola Chicago", "Loyola Marymount", "Loyola MD",
    "Marquette", "Memphis", "Middle Tennessee",
    "Missouri St.", "Murray St.", "Navy",
    "Nebraska Omaha", "Northern Iowa",
    "North Dakota", "North Dakota St.", "North Texas",
    "Ohio", "Old Dominion", "Oral Roberts",
    "Pepperdine", "Providence",
    "Rice", "Richmond", "Robert Morris",
    "Sacramento St.", "Saint Louis", "Sam Houston St.",
    "Sam Houston State", "Seton Hall",
    "South Dakota", "South Dakota St.", "South Florida",
    "Southern Illinois", "Southern Mississippi",
    "St. Bonaventure", "St. John's NY", "St. Joseph's",
    "Stephen F. Austin",
    "Texas St.", "Texas A&M-CC", "Texas A&M-Commerce",
    "Towson", "UAB", "UC Irvine", "UC Riverside",
    "UC San Diego", "UC Santa Barbara",
    "UMass Lowell", "UMBC", "UNC Asheville",
    "UNC Greensboro", "UNC Wilmington",
    "UT Arlington", "UT Martin", "UT Rio Grande Valley",
    "UTEP", "UTSA", "VCU",
    "Western Carolina", "Western Kentucky",
    "Wright St.", "Youngstown St.",
    "California Baptist", "SIU Edwardsville",
    "Southeast Missouri St.", "St. Thomas - Minnesota",
    "Georgia St.", "Alabama St.", "Alabama A&M",
    "Austin Peay", "Cal St. Fullerton", "Cal State Northridge",
    "IU Indianapolis",
    "Morgan St.", "South Carolina St.",
    # Additional Big Ten
    "Illinois St.", "Valparaiso",
    # MAC
    "Central Michigan", "Eastern Michigan", "Western Michigan",
    "Northern Michigan",
    # WCC
    "Saint Mary's CA",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def norm(raw: str) -> str | None:
    """Shortcut: normalize raw name against the KENPOM_NAMES list."""
    return normalize_team_name(raw, KENPOM_NAMES)


# ===========================================================================
# 1.  The five exact examples from the user / Gemini audit
# ===========================================================================

class TestUserReportedExamples:
    """Exact names that appeared in the betting export — must resolve correctly."""

    def test_virginia_tech_hokies(self):
        assert norm("Virginia Tech Hokies") == "Virginia Tech"

    def test_florida_intl_golden_panthers(self):
        assert norm("Florida Int'l Golden Panthers") == "FIU"

    def test_kansas_st_wildcats(self):
        """Critical: abbreviated 'St' without period or 'State'."""
        assert norm("Kansas St Wildcats") == "Kansas St."

    def test_penn_state_nittany_lions(self):
        assert norm("Penn State Nittany Lions") == "Penn St."

    def test_smu_mustangs(self):
        assert norm("SMU Mustangs") == "SMU"


# ===========================================================================
# 2.  Full "State" forms (already covered; regression guard)
# ===========================================================================

class TestFullStateForms:
    """Ensure full-spelling versions still work after adding St abbreviations."""

    def test_kansas_state_wildcats(self):
        assert norm("Kansas State Wildcats") == "Kansas St."

    def test_iowa_state_cyclones(self):
        assert norm("Iowa State Cyclones") == "Iowa St."

    def test_ohio_state_buckeyes(self):
        assert norm("Ohio State Buckeyes") == "Ohio St."

    def test_florida_state_seminoles(self):
        assert norm("Florida State Seminoles") == "Florida St."

    def test_arizona_state_sun_devils(self):
        assert norm("Arizona State Sun Devils") == "Arizona St."

    def test_michigan_state_spartans(self):
        assert norm("Michigan State Spartans") == "Michigan St."

    def test_oklahoma_state_cowboys(self):
        assert norm("Oklahoma State Cowboys") == "Oklahoma St."

    def test_oregon_state_beavers(self):
        assert norm("Oregon State Beavers") == "Oregon St."

    def test_washington_state_cougars(self):
        assert norm("Washington State Cougars") == "Washington St."

    def test_mississippi_state_bulldogs(self):
        assert norm("Mississippi State Bulldogs") == "Mississippi St."


# ===========================================================================
# 3.  New abbreviated "St" forms (the whole set added in this PR)
# ===========================================================================

class TestAbbreviatedStForms:
    """Abbreviated 'St' variants — the class of names missing before this fix."""

    @pytest.mark.parametrize("raw,expected", [
        ("Kansas St Wildcats",           "Kansas St."),
        ("Iowa St Cyclones",             "Iowa St."),
        ("Ohio St Buckeyes",             "Ohio St."),
        ("Penn St Nittany Lions",        "Penn St."),
        ("Florida St Seminoles",         "Florida St."),
        ("Arizona St Sun Devils",        "Arizona St."),
        ("Oklahoma St Cowboys",          "Oklahoma St."),
        ("Oregon St Beavers",            "Oregon St."),
        ("Washington St Cougars",        "Washington St."),
        ("Mississippi St Bulldogs",      "Mississippi St."),
        ("Utah St Aggies",               "Utah St."),
        ("Colorado St Rams",             "Colorado St."),
        ("Fresno St Bulldogs",           "Fresno St."),
        ("San Jose St Spartans",         "San Jose St."),
        ("Boise St Broncos",             "Boise St."),
        ("Wichita St Shockers",          "Wichita St."),
        ("Appalachian St Mountaineers",  "Appalachian St."),
        ("Arkansas St Red Wolves",       "Arkansas St."),
        ("Texas St Bobcats",             "Texas St."),
        ("Ball St Cardinals",            "Ball St."),
        ("Kent St Golden Flashes",       "Kent St."),
        ("Jacksonville St Gamecocks",    "Jacksonville St."),
        ("Kennesaw St Owls",             "Kennesaw St."),
        ("East Tennessee St Buccaneers", "East Tennessee St."),
        ("Murray St Racers",             "Murray St."),
        ("Sacramento St Hornets",        "Sacramento St."),
        ("South Dakota St Jackrabbits",  "South Dakota St."),
        ("Youngstown St Penguins",       "Youngstown St."),
        ("North Dakota St Bison",        "North Dakota St."),
    ])
    def test_abbreviated_st(self, raw, expected):
        assert norm(raw) == expected, (
            f"'{raw}' should resolve to '{expected}', got '{norm(raw)}'"
        )


# ===========================================================================
# 4.  _MANUAL_OVERRIDES priority path
# ===========================================================================

class TestManualOverrides:
    """Override dict is checked before ODDS_TO_KENPOM and fuzzy — verify O(1) path."""

    def test_florida_intl_short(self):
        assert norm("Florida Int'l") == "FIU"

    def test_georgia_st_no_mascot(self):
        assert norm("Georgia St Panthers") == "Georgia St."

    def test_georgia_state_with_mascot(self):
        assert norm("Georgia State Panthers") == "Georgia St."

    def test_sam_houston_st(self):
        assert norm("Sam Houston St Bearkats") == "Sam Houston State"

    def test_kansas_st_bare(self):
        """Bare abbreviated form without mascot hits manual override."""
        assert norm("Kansas St") == "Kansas St."

    def test_iupui_modern_name(self):
        assert norm("IUPUI Jaguars") == "IU Indianapolis"


# ===========================================================================
# 5.  Mascot stripping stage
# ===========================================================================

class TestMascotStripping:
    """_strip_mascot removes trailing mascot; normalize_team_name uses the result."""

    def test_strip_hokies(self):
        assert _strip_mascot("Virginia Tech Hokies") == "Virginia Tech"

    def test_strip_wildcats(self):
        assert _strip_mascot("Kansas State Wildcats") == "Kansas State"

    def test_strip_nittany_lions(self):
        # "Nittany Lions" is two words; strip must handle multi-word mascots
        assert _strip_mascot("Penn State Nittany Lions") == "Penn State"

    def test_strip_golden_panthers(self):
        assert _strip_mascot("Florida Int'l Golden Panthers") == "Florida Int'l"

    def test_strip_no_mascot_unchanged(self):
        assert _strip_mascot("Virginia Tech") == "Virginia Tech"


# ===========================================================================
# 6.  Dangerous-substring guard — false positives must be blocked
# ===========================================================================

class TestSubstringGuard:
    """Regional campus / short-base names must never collapse to flagship."""

    def test_texas_am_cc_does_not_map_to_texas_am(self):
        result = norm("Texas A&M-CC Islanders")
        assert result == "Texas A&M-CC"
        assert result != "Texas A&M"

    def test_central_michigan_does_not_map_to_michigan(self):
        result = norm("Central Michigan Chippewas")
        assert result == "Central Michigan"
        assert result != "Michigan"

    def test_alabama_am_does_not_map_to_alabama(self):
        result = norm("Alabama A&M Bulldogs")
        assert result == "Alabama A&M"
        assert result != "Alabama"

    def test_alabama_state_does_not_map_to_alabama(self):
        result = norm("Alabama St Hornets")
        assert result == "Alabama St."
        assert result != "Alabama"

    def test_georgia_st_does_not_map_to_georgia(self):
        result = norm("Georgia St Panthers")
        assert result == "Georgia St."
        assert result != "Georgia"


# ===========================================================================
# 7.  "School + mascot" → correct KenPom abbreviation (regression suite)
# ===========================================================================

class TestSchoolMascotToKenPom:
    """
    Broad regression: common 'School + Mascot' forms the Odds API sends.
    Each should resolve to the KenPom canonical short form.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("Duke Blue Devils",            "Duke"),
        ("North Carolina Tar Heels",    "North Carolina"),
        ("Kentucky Wildcats",           "Kentucky"),
        ("Gonzaga Bulldogs",            "Gonzaga"),
        ("UConn Huskies",               "Connecticut"),
        ("Ole Miss Rebels",             "Mississippi"),
        ("NC State Wolfpack",           "North Carolina St."),
        ("Saint Mary's Gaels",          "Saint Mary's CA"),
        ("VCU Rams",                    "VCU"),
        ("LSU Tigers",                  "LSU"),
        ("BYU Cougars",                 "BYU"),
        ("Michigan Wolverines",         "Michigan"),
        ("Florida Atlantic Owls",       "Florida Atlantic"),
        ("UMBC Retrievers",             "UMBC"),
        ("Georgia Bulldogs",            "Georgia"),
        ("Tennessee Volunteers",        "Tennessee"),
        ("Houston Cougars",             "Houston"),
    ])
    def test_school_mascot_resolution(self, raw, expected):
        result = norm(raw)
        assert result == expected, (
            f"'{raw}' → expected '{expected}', got '{result}'"
        )


# ===========================================================================
# 8.  Verify all _MANUAL_OVERRIDES targets exist in KENPOM_NAMES
#     (catches a stale override pointing at a name KenPom no longer uses)
# ===========================================================================

class TestOverrideTargetsAreValid:
    def test_all_manual_override_targets_in_kenpom_names(self):
        missing = [
            f"'{k}' → '{v}'"
            for k, v in _MANUAL_OVERRIDES.items()
            if v not in KENPOM_NAMES
        ]
        # Some production overrides map to names only present in real KenPom data
        # (e.g. "Sam Houston State") — allow a short known-absent list.
        allowed_absent = {
            "Sam Houston State",    # KenPom uses "Sam Houston St." in some years
            "Saint Mary's CA",      # not included in our KENPOM_NAMES stub
        }
        truly_missing = [
            entry for entry in missing
            if entry.split(" → ")[1].strip("'") not in allowed_absent
        ]
        assert not truly_missing, (
            "These _MANUAL_OVERRIDES targets are absent from KENPOM_NAMES — "
            "update the test stub or fix the mapping:\n" + "\n".join(truly_missing)
        )
