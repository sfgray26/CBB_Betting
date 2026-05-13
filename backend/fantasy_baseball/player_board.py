"""
Player Big Board — 2026 MLB Draft Rankings

200+ players with Steamer/ZiPS consensus projections tuned for
Treemendous league categories (18 cats, H2H One Win).

Each player dict:
  id, name, team, positions, type (batter/pitcher),
  tier (1-10), rank (overall), adp (approximate Yahoo ADP),
  proj (dict of projected stats), z_score, cat_scores

Category directions:
  Batter K  → negative (lower is better for your team)
  Pitcher L  → negative
  Pitcher HR → negative
  ERA, WHIP  → negative (lower is better)
  All others → positive

Run this module standalone to see rankings:
  python -m backend.fantasy_baseball.player_board
"""

import difflib as _difflib
import json
import logging
import math
import statistics
import time
from collections import namedtuple
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from backend.fantasy_baseball.id_resolution_service import IdentityResolutionService, _normalize_name

logger = logging.getLogger(__name__)
_IDENTITY_RESOLUTION_SERVICE = IdentityResolutionService()

# Import the fusion engine for Bayesian projection combination
from backend.fantasy_baseball.fusion_engine import (
    fuse_batter_projection,
    fuse_pitcher_projection,
    PopulationPrior,
    FusionResult,
)

# ---------------------------------------------------------------------------
# Projection name-map TTL cache
# Caches the 9,686-row player_projections scan as plain Python namedtuples so
# SQLAlchemy session-scoping is never a problem.  Rebuilt at most once per 30
# minutes (1800 s) — safe under Python's GIL for concurrent request handling.
# ---------------------------------------------------------------------------

_ProjectionEntry = namedtuple(
    "_ProjectionEntry",
    [
        "player_name",
        "player_id",
        "player_type",
        "cat_scores",
        "z_score",
        # Batting rate stats
        "avg",
        "obp",
        "slg",
        "ops",
        # Batting counting stats
        "hr",
        "sb",
        # Pitching stats
        "era",
        "whip",
        "k_per_nine",
        "bb_per_nine",
        "w",
        "qs",
    ],
)

# Module-level cache: {"built_at": float, "map": dict[str, _ProjectionEntry]}
_PROJ_NAME_CACHE: dict = {}

_PROJ_NAME_CACHE_TTL = 1800  # seconds (30 minutes)


# ---------------------------------------------------------------------------
# Raw player data (2026 Steamer/ZiPS consensus)
# Format: (name, team, positions, type, tier, adp,
#          pa/ip, r/w, h/l, hr/sv, rbi/bs, k_bat/qs, tb/k_pit, avg/era, ops/whip, nsb/k9, nsv/hr_pit)
# ---------------------------------------------------------------------------

_BATTER_RAW = [
    # Tier 1 — Elite multi-cat (pick 1-12)
    ("Ronald Acuna Jr.", "ATL", ["LF", "CF", "RF", "OF"], 1, 1.0,
     680, 115, 190, 37, 95, 145, 345, 0.306, 0.995, 58, 0),
    ("Juan Soto",        "NYM", ["LF", "RF", "OF"],        1, 2.0,
     680, 105, 172, 38, 100, 128, 330, 0.288, 0.977, 7, 0),
    ("Shohei Ohtani",   "LAD", ["DH", "OF"],               1, 3.0,
     650, 100, 165, 45, 115, 150, 365, 0.282, 0.988, 15, 0),
    ("Mookie Betts",    "LAD", ["SS", "RF", "OF"],         1, 4.0,
     650, 110, 180, 32, 92, 130, 318, 0.295, 0.940, 14, 0),
    ("Yordan Alvarez",  "HOU", ["DH", "LF", "OF"],         1, 5.0,
     620, 95, 165, 38, 110, 138, 342, 0.295, 0.975, 3, 0),
    ("Freddie Freeman", "LAD", ["1B"],                      1, 6.0,
     660, 100, 192, 28, 102, 118, 312, 0.305, 0.916, 10, 0),
    ("Bobby Witt Jr.",  "KC",  ["SS"],                      1, 7.0,
     660, 108, 188, 28, 96, 125, 302, 0.295, 0.895, 42, 0),
    ("Corey Seager",    "TEX", ["SS"],                      1, 8.0,
     580, 92, 166, 32, 98, 108, 300, 0.298, 0.928, 5, 0),
    ("Gunnar Henderson","BAL", ["SS", "3B"],                1, 9.0,
     640, 102, 172, 34, 98, 162, 315, 0.280, 0.902, 18, 0),
    ("José Ramirez",    "CLE", ["3B"],                      1, 10.0,
     650, 100, 175, 28, 102, 82, 298, 0.285, 0.888, 26, 0),
    ("Julio Rodriguez", "SEA", ["CF", "OF"],                1, 11.0,
     650, 95, 175, 30, 90, 158, 298, 0.285, 0.872, 30, 0),
    ("Francisco Lindor","NYM", ["SS"],                      1, 12.0,
     640, 95, 166, 30, 97, 140, 298, 0.278, 0.869, 22, 0),

    # Tier 2 — Strong starters (pick 13-36)
    ("Bryce Harper",    "PHI", ["1B", "RF", "OF"],         2, 15.0,
     590, 92, 162, 32, 97, 118, 300, 0.288, 0.938, 8, 0),
    ("Elly De La Cruz", "CIN", ["SS", "3B"],                2, 16.0,
     630, 97, 162, 26, 87, 198, 278, 0.265, 0.835, 52, 0),
    ("Trea Turner",     "PHI", ["SS"],                      2, 17.0,
     620, 96, 176, 22, 80, 128, 285, 0.295, 0.862, 30, 0),
    ("Marcus Semien",   "TEX", ["2B", "SS"],                2, 18.0,
     660, 100, 170, 25, 88, 132, 285, 0.270, 0.842, 12, 0),
    ("Pete Alonso",     "NYM", ["1B"],                      2, 21.0,
     620, 82, 150, 40, 108, 152, 308, 0.256, 0.876, 2, 0),
    ("Paul Goldschmidt","STL", ["1B"],                      2, 22.0,
     600, 88, 162, 26, 92, 128, 280, 0.282, 0.880, 8, 0),
    ("Adley Rutschman", "BAL", ["C"],                       2, 24.0,
     580, 80, 156, 20, 82, 95, 260, 0.278, 0.848, 5, 0),
    ("Jazz Chisholm",   "NYY", ["LF", "2B", "OF"],         2, 28.0,
     560, 84, 148, 24, 80, 152, 255, 0.266, 0.822, 24, 0),
    ("William Contreras","MIL",["C", "1B"],                 2, 30.0,
     570, 78, 152, 24, 82, 102, 258, 0.278, 0.848, 5, 0),
    ("CJ Abrams",       "WSH", ["SS"],                      2, 32.0,
     640, 98, 180, 22, 78, 148, 280, 0.290, 0.852, 36, 0),
    ("Nolan Arenado",   "STL", ["3B"],                      2, 35.0,
     600, 85, 162, 28, 98, 112, 285, 0.278, 0.862, 4, 0),
    ("Rafael Devers",   "BOS", ["3B", "DH"],                2, 36.0,
     610, 88, 162, 28, 98, 138, 285, 0.272, 0.868, 4, 0),

    # Tier 3 — Solid contributors (pick 37-72)
    ("Kyle Tucker",     "CHC", ["RF", "OF"],                3, 40.0,
     580, 85, 155, 28, 88, 142, 275, 0.278, 0.885, 15, 0),
    ("Willy Adames",    "SF",  ["SS"],                      3, 42.0,
     600, 85, 155, 28, 90, 155, 270, 0.268, 0.848, 12, 0),
    ("Matt Olson",      "ATL", ["1B"],                      3, 44.0,
     620, 88, 155, 36, 105, 168, 305, 0.252, 0.870, 2, 0),
    ("Yordan Alvarez",  "HOU", ["DH"],                      3, 45.0,
     610, 90, 158, 35, 102, 142, 325, 0.278, 0.942, 3, 0),
    ("Cody Bellinger",  "CHC", ["1B", "CF", "OF"],          3, 48.0,
     580, 82, 158, 22, 80, 128, 265, 0.278, 0.858, 12, 0),
    ("Brent Rooker",    "OAK", ["LF", "DH", "OF"],          3, 50.0,
     560, 80, 140, 32, 92, 165, 272, 0.265, 0.878, 4, 0),
    ("Christopher Morel","CHC",["3B", "2B"],                3, 52.0,
     560, 78, 142, 24, 80, 162, 250, 0.265, 0.838, 18, 0),
    ("Eugenio Suarez",  "ARI", ["3B"],                      3, 55.0,
     570, 78, 138, 28, 88, 172, 255, 0.252, 0.838, 4, 0),
    ("Manny Machado",   "SD",  ["3B", "SS"],                3, 58.0,
     590, 82, 155, 24, 88, 118, 262, 0.272, 0.842, 6, 0),
    ("Jarred Kelenic",  "ATL", ["LF", "RF", "OF"],          3, 60.0,
     550, 80, 145, 24, 78, 152, 252, 0.272, 0.852, 18, 0),
    ("Seiya Suzuki",    "CHC", ["RF", "OF"],                3, 62.0,
     560, 78, 150, 22, 78, 128, 255, 0.278, 0.855, 8, 0),
    ("Wyatt Langford",  "TEX", ["LF", "CF", "RF", "OF"],   3, 64.0,
     590, 85, 158, 22, 82, 145, 262, 0.278, 0.858, 20, 0),
    ("Riley Greene",    "DET", ["CF", "OF"],                3, 65.0,
     580, 82, 155, 22, 80, 148, 258, 0.275, 0.850, 14, 0),
    ("Maikel Garcia",   "KC",  ["3B", "SS"],                3, 68.0,
     570, 78, 158, 12, 65, 98, 218, 0.290, 0.808, 28, 0),
    ("Ke'Bryan Hayes",  "PIT", ["3B"],                      3, 70.0,
     530, 70, 140, 14, 62, 108, 215, 0.278, 0.808, 16, 0),

    # Tier 4 — Good depth (pick 73-108)
    ("J.T. Realmuto",   "PHI", ["C"],                       4, 75.0,
     520, 68, 132, 18, 72, 108, 228, 0.268, 0.828, 8, 0),
    ("Austin Wells",    "NYY", ["C"],                       4, 80.0,
     510, 65, 128, 20, 70, 118, 232, 0.265, 0.822, 4, 0),
    ("Connor Wong",     "BOS", ["C"],                       4, 85.0,
     490, 62, 122, 18, 65, 122, 218, 0.262, 0.808, 8, 0),
    ("Salvador Perez",  "KC",  ["C", "DH"],                 4, 88.0,
     550, 68, 140, 22, 80, 112, 242, 0.268, 0.822, 2, 0),
    ("Jake Cronenworth","SD",  ["1B", "2B"],                4, 90.0,
     560, 75, 148, 16, 72, 108, 235, 0.275, 0.818, 10, 0),
    ("Zach McKinstry",  "DET", ["2B", "3B", "LF"],          4, 92.0,
     520, 68, 132, 16, 65, 122, 218, 0.265, 0.808, 14, 0),
    ("Jonathan India",  "CIN", ["2B"],                      4, 95.0,
     570, 78, 148, 18, 72, 135, 240, 0.272, 0.832, 15, 0),
    ("Ezequiel Tovar",  "COL", ["SS"],                      4, 98.0,
     560, 75, 148, 20, 75, 128, 248, 0.272, 0.835, 16, 0),
    ("Ian Happ",        "CHC", ["LF", "RF", "OF"],          4, 100.0,
     560, 78, 142, 20, 75, 138, 248, 0.265, 0.838, 8, 0),
    ("Anthony Santander","TOR",["LF", "RF", "DH", "OF"],   4, 102.0,
     580, 80, 148, 28, 88, 145, 268, 0.262, 0.848, 4, 0),
    ("Teoscar Hernandez","LAD",["LF", "RF", "OF"],          4, 105.0,
     570, 78, 145, 28, 88, 158, 262, 0.262, 0.842, 8, 0),
    ("Jackson Merrill",  "SD", ["CF", "OF"],                4, 108.0,
     570, 78, 152, 22, 78, 118, 252, 0.275, 0.845, 14, 0),
    ("Michael Harris",  "ATL", ["CF", "OF"],                4, 110.0,
     580, 82, 155, 22, 78, 135, 252, 0.272, 0.838, 22, 0),

    # Tier 5 — Solid bench/streaming (pick 109-144)
    ("Nolan Jones",     "COL", ["LF", "RF", "3B", "OF"],   5, 115.0,
     530, 72, 130, 20, 72, 148, 228, 0.262, 0.838, 14, 0),
    ("Oswaldo Cabrera", "NYY", ["2B", "SS", "3B", "LF"],   5, 118.0,
     520, 68, 132, 16, 65, 128, 222, 0.262, 0.808, 12, 0),
    ("Spencer Steer",   "CIN", ["1B", "2B", "3B", "OF"],   5, 120.0,
     580, 78, 148, 22, 82, 130, 252, 0.265, 0.802, 12, 0),
    ("David Peralta",   "LAD", ["LF", "RF", "OF"],          5, 122.0,
     480, 62, 128, 14, 62, 92, 202, 0.278, 0.818, 5, 0),
    ("Jorge Mateo",     "BAL", ["SS", "2B", "CF"],          5, 125.0,
     490, 68, 122, 10, 48, 138, 178, 0.258, 0.738, 38, 0),
    ("TJ Friedl",       "CIN", ["CF", "LF", "OF"],          5, 128.0,
     520, 72, 132, 14, 58, 118, 205, 0.268, 0.798, 28, 0),
    ("Esteury Ruiz",    "MIL", ["CF", "OF"],                5, 130.0,
     510, 68, 128, 8, 48, 122, 175, 0.260, 0.735, 52, 0),
    ("Jose Siri",       "TB",  ["CF", "LF", "OF"],          5, 132.0,
     480, 65, 115, 16, 55, 168, 198, 0.248, 0.768, 30, 0),
    ("Jahmai Jones",    "SEA", ["2B"],                      5, 135.0,
     510, 68, 128, 12, 58, 118, 198, 0.262, 0.782, 18, 0),
    ("Davis Schneider", "TOR", ["2B", "3B", "LF"],          5, 138.0,
     510, 68, 128, 18, 65, 138, 222, 0.262, 0.820, 10, 0),
    ("Nick Gonzales",   "PIT", ["2B"],                      5, 140.0,
     520, 70, 132, 16, 65, 125, 218, 0.265, 0.810, 14, 0),
    ("Tyler Stephenson","CIN", ["C"],                       5, 142.0,
     480, 58, 122, 14, 62, 105, 210, 0.265, 0.808, 2, 0),

    # ── MISSED TOP-TIER ADDITIONS ──────────────────────────────────────────
    # Aaron Judge — massive oversight in original board
    ("Aaron Judge",     "NYY", ["RF", "CF", "OF"],          1, 4.5,
     540, 95, 145, 45, 108, 175, 340, 0.278, 0.978, 6, 0),
    # Jarren Duran — breakout bat, speed elite
    ("Jarren Duran",    "BOS", ["CF", "LF", "OF"],          3, 58.0,
     580, 88, 165, 18, 72, 138, 258, 0.292, 0.848, 28, 0),
    # Luis Arraez — AVG/OBP category specialist
    ("Luis Arraez",     "SD",  ["1B", "2B"],                4, 112.0,
     550, 68, 178, 5, 48, 48, 202, 0.332, 0.798, 6, 0),
    # Daulton Varsho — multi-pos C/CF value
    ("Daulton Varsho",  "TOR", ["C", "CF", "OF"],           4, 116.0,
     500, 65, 122, 18, 62, 148, 218, 0.252, 0.782, 12, 0),

    # Tier 5 cont — ADP 145-200
    ("Cal Raleigh",     "SEA", ["C", "1B"],                 5, 148.0,
     520, 68, 130, 25, 72, 148, 238, 0.262, 0.828, 3, 0),
    ("Nathaniel Lowe",  "TEX", ["1B"],                      5, 152.0,
     570, 72, 158, 18, 72, 122, 252, 0.282, 0.818, 4, 0),
    ("Ha-Seong Kim",    "SD",  ["SS", "2B", "3B"],          5, 155.0,
     540, 68, 138, 14, 58, 118, 215, 0.265, 0.782, 22, 0),
    ("Corbin Carroll",  "ARI", ["CF", "LF", "OF"],          5, 158.0,
     560, 78, 148, 18, 68, 148, 252, 0.275, 0.828, 28, 0),
    ("Randy Arozarena", "SEA", ["LF", "RF", "OF"],          5, 162.0,
     540, 72, 138, 20, 70, 148, 248, 0.265, 0.818, 20, 0),
    ("Jackson Chourio", "MIL", ["CF", "LF", "OF"],          5, 165.0,
     580, 80, 155, 18, 68, 145, 248, 0.275, 0.818, 22, 0),
    ("Kyle Schwarber",  "PHI", ["LF", "1B", "OF"],          5, 168.0,
     580, 82, 140, 38, 95, 198, 278, 0.252, 0.885, 4, 0),
    ("Nolan Gorman",    "STL", ["2B", "3B"],                5, 172.0,
     520, 70, 128, 28, 80, 185, 248, 0.252, 0.838, 8, 0),
    ("Jorge Soler",     "MIA", ["LF", "DH", "OF"],          5, 175.0,
     520, 68, 128, 30, 80, 168, 258, 0.252, 0.848, 4, 0),
    ("Nico Hoerner",    "CHC", ["2B", "SS"],                5, 178.0,
     570, 75, 162, 8, 60, 82, 218, 0.292, 0.782, 18, 0),
    ("Christian Walker","HOU", ["1B"],                      5, 180.0,
     560, 72, 138, 28, 82, 165, 255, 0.258, 0.842, 4, 0),
    ("Tommy Edman",     "LAD", ["2B", "SS", "CF", "OF"],   5, 182.0,
     540, 72, 142, 12, 55, 110, 212, 0.272, 0.772, 28, 0),
    ("Max Muncy",       "LAD", ["1B", "2B", "3B"],          5, 185.0,
     530, 72, 122, 28, 75, 165, 248, 0.248, 0.848, 5, 0),
    ("Steven Kwan",     "CLE", ["LF", "CF", "OF"],          5, 188.0,
     580, 80, 172, 8, 55, 88, 218, 0.305, 0.818, 18, 0),
    ("Gleyber Torres",  "NYM", ["2B"],                      5, 190.0,
     550, 72, 148, 18, 70, 118, 242, 0.275, 0.818, 12, 0),
    ("Alejandro Kirk",  "TOR", ["C"],                       5, 193.0,
     490, 55, 132, 12, 60, 95, 218, 0.278, 0.798, 2, 0),
    ("Gabriel Moreno",  "ARI", ["C"],                       5, 195.0,
     480, 55, 128, 14, 60, 98, 218, 0.278, 0.802, 5, 0),
    ("MJ Melendez",     "KC",  ["C", "LF", "OF"],           5, 197.0,
     490, 60, 118, 18, 62, 148, 228, 0.255, 0.808, 8, 0),
    ("Logan O'Hoppe",   "LAA", ["C"],                       5, 199.0,
     480, 55, 122, 16, 60, 115, 218, 0.262, 0.802, 3, 0),
    ("Willson Contreras","STL",["C"],                       5, 202.0,
     490, 60, 125, 16, 65, 108, 228, 0.262, 0.812, 4, 0),

    # Tier 6 — ADP 200-260
    ("Josh Naylor",     "ARI", ["1B", "DH"],                6, 205.0,
     540, 65, 138, 22, 82, 128, 248, 0.268, 0.812, 3, 0),
    ("Andres Gimenez",  "CLE", ["2B"],                      6, 208.0,
     520, 68, 138, 12, 55, 108, 215, 0.272, 0.768, 16, 0),
    ("Triston Casas",   "BOS", ["1B"],                      6, 210.0,
     520, 65, 128, 22, 75, 148, 238, 0.258, 0.828, 3, 0),
    ("Spencer Torkelson","DET",["1B"],                      6, 212.0,
     530, 65, 128, 24, 78, 158, 245, 0.255, 0.828, 2, 0),
    ("Yainer Diaz",     "HOU", ["C", "1B", "DH"],           6, 214.0,
     490, 58, 128, 16, 65, 112, 222, 0.268, 0.802, 4, 0),
    ("Jonah Heim",      "TEX", ["C"],                       6, 216.0,
     460, 52, 112, 14, 55, 108, 202, 0.252, 0.778, 3, 0),
    ("Josh Smith",      "TEX", ["3B", "SS"],                6, 218.0,
     490, 62, 128, 12, 52, 112, 202, 0.272, 0.782, 10, 0),
    ("Patrick Bailey",  "SF",  ["C"],                       6, 220.0,
     440, 48, 108, 10, 48, 88, 188, 0.252, 0.752, 3, 0),
    ("Brendan Donovan", "STL", ["1B","2B","3B","LF","OF"], 6, 222.0,
     500, 62, 132, 10, 55, 98, 202, 0.272, 0.778, 8, 0),
    ("Lane Thomas",     "CLE", ["CF", "RF", "OF"],          6, 225.0,
     490, 65, 122, 16, 58, 135, 222, 0.255, 0.782, 20, 0),
    ("Bo Naylor",       "CLE", ["C", "DH"],                 6, 228.0,
     460, 55, 112, 18, 58, 148, 218, 0.252, 0.798, 4, 0),
    ("Jon Berti",       "NYM", ["2B","3B","SS","OF"],       6, 230.0,
     420, 58, 105, 6, 38, 95, 162, 0.258, 0.718, 28, 0),
    ("Brice Turang",    "MIL", ["2B", "SS"],                6, 232.0,
     490, 62, 125, 6, 42, 95, 172, 0.265, 0.728, 30, 0),
    ("Victor Scott II", "STL", ["CF", "OF"],                6, 235.0,
     440, 58, 108, 4, 32, 82, 148, 0.255, 0.698, 42, 0),
    ("Mike Trout",      "LAA", ["CF", "OF", "DH"],          6, 238.0,
     380, 52, 92, 20, 52, 118, 205, 0.272, 0.892, 4, 0),
    ("Gavin Lux",       "LAD", ["2B", "SS"],                6, 240.0,
     480, 58, 125, 10, 48, 108, 195, 0.268, 0.768, 10, 0),
    ("Ryan McMahon",    "COL", ["2B", "3B"],                6, 242.0,
     490, 58, 120, 16, 58, 115, 218, 0.252, 0.782, 10, 0),
    ("Max Kepler",      "PHI", ["RF", "LF", "OF"],          6, 244.0,
     470, 58, 120, 16, 58, 118, 215, 0.262, 0.782, 8, 0),
    ("Andrew Vaughn",   "CWS", ["1B", "LF", "DH", "OF"],  6, 246.0,
     490, 55, 128, 16, 68, 108, 218, 0.268, 0.792, 3, 0),
    ("DJ LeMahieu",     "NYY", ["1B", "2B", "3B"],          6, 248.0,
     470, 55, 125, 8, 52, 92, 190, 0.272, 0.762, 5, 0),

    # Tier 7 — ADP 250-300 (late-round targets)
    ("Masataka Yoshida","BOS", ["LF", "DH", "OF"],          7, 252.0,
     480, 55, 130, 12, 60, 85, 195, 0.282, 0.808, 4, 0),
    ("Evan Carter",     "TEX", ["CF", "LF", "OF"],          7, 255.0,
     440, 58, 112, 12, 48, 112, 185, 0.268, 0.792, 18, 0),
    ("Colt Keith",      "DET", ["2B", "3B"],                7, 258.0,
     500, 62, 130, 14, 60, 128, 212, 0.268, 0.792, 12, 0),
    ("Daulton Varsho",  "TOR", ["C", "CF", "OF"],           7, 260.0,
     440, 55, 108, 16, 55, 135, 195, 0.252, 0.778, 12, 0),
    ("Oscar Gonzalez",  "CLE", ["RF", "OF"],                7, 262.0,
     450, 52, 115, 14, 58, 95, 195, 0.262, 0.772, 4, 0),
    ("Enrique Hernandez","LAD",["2B","SS","CF","OF"],       7, 264.0,
     410, 52, 102, 14, 50, 105, 188, 0.258, 0.768, 8, 0),
    ("Gavin Sheets",    "CWS", ["1B", "LF", "DH", "OF"],  7, 266.0,
     440, 48, 110, 18, 62, 108, 205, 0.258, 0.782, 3, 0),
    ("CJ Cron",         "COL", ["1B", "DH"],                7, 268.0,
     450, 52, 112, 20, 65, 115, 215, 0.258, 0.788, 2, 0),
    ("Romy Gonzalez",   "CHW", ["SS", "2B", "3B"],          7, 270.0,
     440, 52, 108, 14, 52, 128, 195, 0.255, 0.762, 10, 0),
    ("DJ Stewart",      "FA",  ["LF", "RF", "DH", "OF"],  7, 272.0,
     410, 48, 100, 16, 55, 135, 192, 0.252, 0.768, 6, 0),
    ("Yolmer Sanchez",  "free",["2B", "3B"],                7, 274.0,
     400, 45, 98, 8, 42, 92, 158, 0.258, 0.728, 14, 0),
    ("Joey Loperfido",  "TOR", ["LF", "RF", "OF"],          7, 276.0,
     430, 55, 108, 12, 48, 118, 185, 0.262, 0.758, 20, 0),
    ("Tyrone Taylor",   "NYM", ["CF", "RF", "OF"],          7, 278.0,
     410, 50, 102, 14, 50, 118, 185, 0.258, 0.758, 12, 0),
    ("Greg Allen",      "FA",  ["CF", "LF", "OF"],          7, 280.0,
     380, 52, 95, 4, 32, 82, 135, 0.262, 0.708, 28, 0),
    ("Michael Busch",   "CHC", ["1B", "2B", "OF"],          7, 282.0,
     480, 60, 118, 18, 62, 138, 218, 0.258, 0.802, 6, 0),
    ("Nolan Jones",     "COL", ["LF", "RF", "3B", "OF"],   7, 284.0,
     420, 55, 102, 16, 55, 138, 195, 0.255, 0.808, 10, 0),
    ("Edouard Julien",  "MIN", ["2B", "DH"],                7, 286.0,
     480, 62, 118, 14, 55, 148, 198, 0.258, 0.798, 10, 0),
    ("Michael Siani",   "STL", ["CF", "OF"],                7, 288.0,
     380, 52, 92, 4, 32, 78, 128, 0.258, 0.698, 28, 0),
    ("Joc Pederson",    "SF",  ["LF", "DH", "OF"],          7, 290.0,
     440, 55, 105, 20, 60, 125, 198, 0.255, 0.812, 4, 0),
    ("David Fry",       "CLE", ["C", "1B", "3B", "OF"],   7, 292.0,
     410, 48, 100, 14, 52, 118, 185, 0.255, 0.778, 5, 0),
    ("Tyler O'Neill",   "BOS", ["LF", "RF", "CF", "OF"],  7, 294.0,
     420, 52, 100, 18, 55, 148, 198, 0.252, 0.778, 10, 0),
    ("Henry Davis",     "PIT", ["C"],                       7, 296.0,
     420, 48, 100, 14, 52, 128, 185, 0.252, 0.762, 4, 0),
    ("Avisail Garcia",  "free",["LF", "RF", "DH", "OF"],  7, 298.0,
     380, 42, 95, 12, 48, 105, 172, 0.258, 0.762, 5, 0),
    ("Nick Ahmed",      "ARI", ["SS"],                      8, 300.0,
     360, 40, 88, 8, 38, 92, 145, 0.252, 0.718, 8, 0),
]

_PITCHER_RAW = [
    # Tier 1 — Ace SPs / Elite Closers (pick 14-30)
    # (name, team, positions, tier, adp, ip, w, l, sv, bs, qs, k, era, whip, k9, hr, nsv)
    ("Spencer Strider",  "ATL", ["SP"],     1, 13.0,
     175, 14, 5, 0, 0, 21, 242, 2.68, 0.94, 12.5, 16, 0),
    ("Gerrit Cole",      "NYY", ["SP"],     1, 14.0,
     188, 14, 6, 0, 0, 23, 226, 2.80, 1.00, 10.8, 18, 0),
    ("Paul Skenes",      "PIT", ["SP"],     1, 19.0,
     172, 12, 7, 0, 0, 20, 222, 2.88, 1.02, 11.6, 16, 0),
    ("Zack Wheeler",     "PHI", ["SP"],     1, 20.0,
     195, 14, 6, 0, 0, 24, 216, 2.98, 1.04, 9.9, 18, 0),
    ("Emmanuel Clase",   "CLE", ["RP"],     1, 23.0,
     70, 4, 2, 38, 4, 0, 84, 2.18, 0.88, 10.8, 4, 34),
    ("Logan Webb",       "SF",  ["SP"],     1, 25.0,
     200, 14, 7, 0, 0, 24, 196, 3.08, 1.07, 8.8, 16, 0),
    ("Corbin Burnes",    "BAL", ["SP"],     1, 26.0,
     192, 13, 7, 0, 0, 22, 208, 3.08, 1.05, 9.7, 17, 0),
    ("Josh Hader",       "HOU", ["RP"],     1, 29.0,
     65, 3, 3, 35, 6, 0, 90, 2.38, 0.92, 12.5, 4, 29),
    ("Dylan Cease",      "SD",  ["SP"],     1, 33.0,
     182, 12, 8, 0, 0, 20, 202, 3.38, 1.14, 10.0, 19, 0),
    ("Kevin Gausman",    "TOR", ["SP"],     1, 34.0,
     188, 12, 8, 0, 0, 22, 196, 3.18, 1.04, 9.4, 19, 0),

    # Tier 2 — SP2/Strong Closer (pick 31-72)
    ("Edwin Diaz",       "NYM", ["RP"],     2, 37.0,
     65, 3, 3, 32, 5, 0, 92, 2.48, 0.94, 12.7, 4, 27),
    ("Framber Valdez",   "HOU", ["SP"],     2, 38.0,
     195, 13, 8, 0, 0, 23, 178, 3.22, 1.18, 8.2, 15, 0),
    ("Tarik Skubal",     "DET", ["SP"],     2, 39.0,
     180, 13, 6, 0, 0, 22, 208, 3.02, 1.06, 10.4, 17, 0),
    ("Félix Bautista",   "BAL", ["RP"],     2, 43.0,
     62, 3, 3, 30, 5, 0, 82, 2.58, 0.98, 11.9, 5, 25),
    ("Shane Bieber",     "CLE", ["SP"],     2, 46.0,
     178, 12, 7, 0, 0, 21, 192, 3.12, 1.04, 9.7, 17, 0),
    ("Sonny Gray",       "STL", ["SP"],     2, 47.0,
     175, 12, 7, 0, 0, 20, 188, 3.22, 1.10, 9.7, 18, 0),
    ("Camilo Doval",     "SF",  ["RP"],     2, 49.0,
     65, 3, 4, 30, 7, 0, 74, 2.78, 1.04, 10.2, 5, 23),
    ("Aaron Nola",       "PHI", ["SP"],     2, 51.0,
     192, 13, 8, 0, 0, 22, 198, 3.48, 1.12, 9.3, 22, 0),
    ("Tyler Glasnow",    "LAD", ["SP"],     2, 53.0,
     162, 11, 6, 0, 0, 19, 198, 3.18, 1.06, 11.0, 17, 0),
    ("Yordan Guzman",    "TB",  ["SP"],     2, 57.0,
     172, 11, 7, 0, 0, 19, 185, 3.38, 1.14, 9.7, 18, 0),
    ("Blake Snell",      "SF",  ["SP"],     2, 59.0,
     158, 10, 7, 0, 0, 17, 195, 3.22, 1.12, 11.1, 16, 0),
    ("Jordan Romano",    "TOR", ["RP"],     2, 61.0,
     60, 3, 3, 28, 6, 0, 68, 2.68, 1.02, 10.2, 5, 22),
    ("Andrés Muñoz",     "SEA", ["RP"],     2, 63.0,
     62, 3, 3, 28, 6, 0, 78, 2.48, 0.98, 11.3, 4, 22),
    ("Ryan Helsley",     "STL", ["RP"],     2, 66.0,
     62, 3, 4, 28, 6, 0, 78, 2.78, 1.04, 11.3, 5, 22),

    # Tier 3 — SP3/Middle Closer (pick 73-120)
    ("Pablo Lopez",      "MIN", ["SP"],     3, 73.0,
     185, 11, 8, 0, 0, 20, 185, 3.58, 1.15, 9.0, 20, 0),
    ("Max Fried",        "NYY", ["SP"],     3, 76.0,
     180, 11, 8, 0, 0, 20, 182, 3.48, 1.14, 9.1, 18, 0),
    ("Hunter Greene",    "CIN", ["SP"],     3, 78.0,
     172, 10, 8, 0, 0, 18, 195, 3.58, 1.15, 10.2, 20, 0),
    ("George Kirby",     "SEA", ["SP"],     3, 80.0,
     185, 11, 7, 0, 0, 20, 178, 3.42, 1.09, 8.7, 18, 0),
    ("José Berríos",     "TOR", ["SP"],     3, 82.0,
     188, 12, 9, 0, 0, 20, 188, 3.62, 1.18, 9.0, 21, 0),
    ("Grayson Rodriguez","BAL", ["SP"],     3, 84.0,
     175, 11, 8, 0, 0, 19, 192, 3.48, 1.14, 9.9, 19, 0),
    ("Kodai Senga",      "NYM", ["SP"],     3, 86.0,
     165, 10, 7, 0, 0, 18, 188, 3.28, 1.08, 10.3, 16, 0),
    ("Tanner Bibee",     "CLE", ["SP"],     3, 88.0,
     175, 11, 7, 0, 0, 19, 182, 3.48, 1.16, 9.4, 18, 0),
    ("Devin Williams",   "NYY", ["RP"],     3, 90.0,
     60, 4, 3, 25, 7, 0, 78, 2.88, 1.06, 11.7, 5, 18),
    ("Clay Holmes",      "CLE", ["RP"],     3, 93.0,
     62, 4, 4, 24, 7, 0, 72, 2.98, 1.12, 10.4, 5, 17),
    ("Jordan Hicks",     "TOR", ["RP"],     3, 95.0,
     62, 4, 3, 26, 6, 0, 68, 2.88, 1.08, 9.9, 5, 20),
    ("Reynaldo Lopez",   "ATL", ["RP"],     3, 97.0,
     62, 4, 3, 26, 6, 0, 72, 2.88, 1.05, 10.4, 4, 20),
    ("Michael King",     "SD",  ["SP", "RP"],3, 100.0,
     155, 9, 7, 0, 0, 15, 172, 3.68, 1.14, 10.0, 18, 0),
    ("Cristopher Sanchez","PHI",["SP"],     3, 102.0,
     175, 11, 8, 0, 0, 18, 165, 3.62, 1.18, 8.5, 18, 0),
    ("Brayan Bello",     "BOS", ["SP"],     3, 105.0,
     178, 10, 9, 0, 0, 18, 175, 3.78, 1.22, 8.8, 19, 0),

    # Tier 4 — SP4/Streaming/Closer Handcuff (pick 121-168)
    ("Shota Imanaga",    "CHC", ["SP"],     4, 112.0,
     175, 10, 8, 0, 0, 18, 178, 3.62, 1.14, 9.2, 17, 0),
    ("Joe Ryan",         "MIN", ["SP"],     4, 115.0,
     178, 10, 8, 0, 0, 18, 182, 3.72, 1.14, 9.2, 20, 0),
    ("Eury Pérez",       "MIA", ["SP"],     4, 118.0,
     162, 9, 8, 0, 0, 17, 185, 3.68, 1.18, 10.3, 17, 0),
    ("MacKenzie Gore",   "WSH", ["SP"],     4, 120.0,
     168, 9, 9, 0, 0, 17, 182, 3.78, 1.20, 9.8, 18, 0),
    ("Nick Lodolo",      "CIN", ["SP"],     4, 123.0,
     162, 9, 8, 0, 0, 16, 175, 3.78, 1.18, 9.7, 17, 0),
    ("Gavin Stone",      "LAD", ["SP"],     4, 125.0,
     165, 9, 7, 0, 0, 16, 165, 3.88, 1.18, 9.0, 17, 0),
    ("Clarke Schmidt",   "NYY", ["SP"],     4, 128.0,
     168, 10, 8, 0, 0, 17, 172, 3.82, 1.18, 9.2, 18, 0),
    ("Logan Gilbert",    "SEA", ["SP"],     4, 130.0,
     182, 10, 9, 0, 0, 18, 178, 3.88, 1.18, 8.8, 19, 0),
    ("DL Hall",          "MIL", ["SP", "RP"],4, 133.0,
     155, 9, 8, 0, 0, 15, 172, 3.78, 1.18, 10.0, 17, 0),
    ("Kyle Freeland",    "COL", ["SP"],     4, 136.0,
     175, 9, 10, 0, 0, 16, 152, 4.18, 1.28, 7.8, 22, 0),
    ("Trevor Rogers",    "MIA", ["SP"],     4, 138.0,
     162, 9, 9, 0, 0, 15, 168, 3.98, 1.25, 9.3, 18, 0),
    ("Justin Verlander", "HOU", ["SP"],     4, 140.0,
     155, 10, 7, 0, 0, 16, 155, 3.68, 1.15, 9.0, 18, 0),
    ("Chris Sale",       "ATL", ["SP"],     4, 143.0,
     168, 10, 7, 0, 0, 17, 188, 3.58, 1.10, 10.1, 17, 0),
    ("Ryan Pressly",     "HOU", ["RP"],     4, 146.0,
     58, 3, 3, 20, 6, 0, 62, 3.18, 1.12, 9.6, 6, 14),
    ("Robert Suarez",    "SD",  ["RP"],     4, 148.0,
     58, 3, 4, 22, 6, 0, 64, 2.98, 1.06, 9.9, 5, 16),
    ("Raisel Iglesias",  "LAA", ["RP"],     4, 150.0,
     58, 3, 4, 22, 6, 0, 68, 2.88, 1.08, 10.6, 5, 16),

    # Tier 5 — Streamers / Speculative (pick 169-230)
    ("Patrick Corbin",   "WSH", ["SP"],     5, 165.0,
     158, 8, 12, 0, 0, 13, 138, 4.58, 1.35, 7.9, 22, 0),
    ("Lance Lynn",       "STL", ["SP"],     5, 168.0,
     165, 9, 10, 0, 0, 15, 155, 4.18, 1.28, 8.5, 22, 0),
    ("Freddy Peralta",   "MIL", ["SP"],     5, 170.0,
     155, 8, 8, 0, 0, 14, 175, 3.88, 1.18, 10.2, 17, 0),
    ("Josh Walker",      "NYM", ["SP", "RP"],5, 175.0,
     145, 8, 8, 0, 0, 13, 158, 3.98, 1.24, 9.8, 17, 0),
    ("Matt Brash",       "SEA", ["RP"],     5, 178.0,
     55, 3, 3, 18, 6, 0, 62, 3.08, 1.12, 10.1, 6, 12),
    ("Kenley Jansen",    "BOS", ["RP"],     5, 180.0,
     55, 3, 4, 18, 7, 0, 62, 3.18, 1.15, 10.1, 6, 11),
    ("Pete Fairbanks",   "TB",  ["RP"],     5, 182.0,
     55, 3, 3, 20, 6, 0, 65, 2.98, 1.08, 10.6, 5, 14),
    ("Alex Vesia",       "LAD", ["RP"],     5, 185.0,
     55, 3, 3, 12, 5, 0, 68, 2.98, 1.10, 11.1, 5, 7),

    # Tier 5 cont — SP streamers / handcuff closers (ADP 185-240)
    ("Mason Miller",     "OAK", ["RP"],     5, 188.0,
     60, 3, 3, 28, 5, 0, 80, 2.58, 0.95, 12.0, 4, 23),
    ("Luis Castillo",    "SEA", ["SP"],     5, 190.0,
     185, 11, 8, 0, 0, 19, 185, 3.52, 1.15, 9.0, 19, 0),
    ("Brady Singer",     "KC",  ["SP"],     5, 193.0,
     182, 10, 9, 0, 0, 18, 175, 3.68, 1.18, 8.7, 20, 0),
    ("Mitch Keller",     "PIT", ["SP"],     5, 196.0,
     185, 10, 9, 0, 0, 18, 178, 3.72, 1.19, 8.7, 20, 0),
    ("Hunter Brown",     "HOU", ["SP"],     5, 198.0,
     172, 10, 8, 0, 0, 17, 185, 3.58, 1.16, 9.7, 18, 0),
    ("David Bednar",     "PIT", ["RP"],     5, 200.0,
     58, 3, 4, 24, 7, 0, 68, 2.98, 1.06, 10.6, 5, 17),
    ("Kyle Finnegan",    "WSH", ["RP"],     5, 203.0,
     58, 3, 4, 26, 7, 0, 62, 3.08, 1.12, 9.6, 6, 19),
    ("Paul Sewald",      "ARI", ["RP"],     5, 205.0,
     55, 3, 3, 24, 6, 0, 68, 2.88, 1.04, 11.1, 4, 18),
    ("Hunter Harvey",    "WSH", ["RP"],     5, 208.0,
     55, 3, 4, 22, 6, 0, 62, 3.08, 1.08, 10.1, 5, 16),
    ("Nestor Cortes",    "MIL", ["SP"],     5, 210.0,
     168, 9, 8, 0, 0, 16, 165, 3.78, 1.18, 8.8, 17, 0),
    ("Zach Eflin",       "TB",  ["SP"],     5, 212.0,
     178, 10, 9, 0, 0, 18, 168, 3.82, 1.19, 8.5, 20, 0),
    ("Bailey Ober",      "MIN", ["SP"],     5, 215.0,
     175, 10, 9, 0, 0, 17, 172, 3.78, 1.17, 8.8, 19, 0),
    ("Jose Alvarado",    "PHI", ["RP"],     5, 218.0,
     55, 3, 3, 20, 6, 0, 68, 2.98, 1.10, 11.1, 5, 14),
    ("Ryan Walker",      "SF",  ["RP"],     5, 220.0,
     55, 3, 3, 18, 6, 0, 62, 3.18, 1.14, 10.1, 5, 12),
    ("A.J. Minter",      "ATL", ["RP"],     5, 222.0,
     58, 3, 3, 8, 4, 0, 68, 2.98, 1.08, 10.6, 5, 4),
    ("Ranger Suarez",    "PHI", ["SP"],     5, 225.0,
     172, 10, 8, 0, 0, 17, 162, 3.68, 1.20, 8.5, 16, 0),
    ("Nathan Eovaldi",   "TEX", ["SP"],     5, 228.0,
     158, 9, 8, 0, 0, 16, 155, 3.82, 1.18, 8.8, 18, 0),
    ("Kyle Harrison",    "SF",  ["SP"],     5, 230.0,
     162, 9, 8, 0, 0, 15, 172, 3.88, 1.20, 9.6, 17, 0),

    # Tier 6 — Mid streamers / closer handcuffs (ADP 230-270)
    ("Max Meyer",        "MIA", ["SP"],     6, 232.0,
     155, 8, 8, 0, 0, 14, 168, 3.98, 1.24, 9.8, 17, 0),
    ("Braxton Garrett",  "MIA", ["SP"],     6, 234.0,
     158, 9, 9, 0, 0, 15, 155, 3.98, 1.24, 8.8, 18, 0),
    ("Jose Leclerc",     "TEX", ["RP"],     6, 236.0,
     52, 3, 3, 16, 6, 0, 58, 3.28, 1.15, 10.0, 6, 10),
    ("Kevin Ginkel",     "ARI", ["RP"],     6, 238.0,
     52, 3, 3, 16, 6, 0, 58, 3.18, 1.12, 10.0, 5, 10),
    ("Evan Phillips",    "LAD", ["RP"],     6, 240.0,
     55, 3, 3, 14, 5, 0, 65, 2.98, 1.06, 10.6, 4, 9),
    ("Carlos Rodon",     "NYY", ["SP"],     6, 242.0,
     155, 9, 8, 0, 0, 15, 178, 3.78, 1.20, 10.3, 17, 0),
    ("Triston McKenzie", "CLE", ["SP"],     6, 244.0,
     158, 9, 9, 0, 0, 14, 168, 3.98, 1.22, 9.6, 18, 0),
    ("JP Sears",         "OAK", ["SP"],     6, 246.0,
     165, 9, 10, 0, 0, 15, 152, 4.08, 1.24, 8.3, 20, 0),
    ("Aaron Civale",     "TB",  ["SP"],     6, 248.0,
     162, 9, 9, 0, 0, 15, 148, 4.02, 1.22, 8.2, 19, 0),
    ("Seranthony Dominguez","PHI",["RP"],   6, 250.0,
     52, 3, 3, 14, 5, 0, 58, 3.18, 1.14, 10.0, 5, 9),
    ("Matt Strahm",      "PHI", ["SP","RP"],6, 252.0,
     145, 8, 8, 0, 0, 12, 155, 3.98, 1.20, 9.6, 16, 0),
    ("Aroldis Chapman",  "PIT", ["RP"],     6, 254.0,
     50, 3, 4, 12, 5, 0, 65, 3.28, 1.18, 11.7, 6, 7),
    ("Jake Irvin",       "WSH", ["SP"],     6, 256.0,
     162, 9, 10, 0, 0, 15, 148, 4.18, 1.25, 8.2, 20, 0),
    ("Jameson Taillon",  "CHC", ["SP"],     6, 258.0,
     158, 9, 9, 0, 0, 14, 142, 4.08, 1.22, 8.1, 19, 0),
    ("DL Hall",          "MIL", ["SP","RP"],6, 260.0,
     148, 8, 7, 0, 0, 13, 162, 3.88, 1.20, 9.9, 16, 0),
    # Josh Hader — NOT duplicated here; already in Tier 1 at ADP 29

    # Tier 7 — Deep streamers / handcuffs / speculative (ADP 260-310)
    ("Phil Maton",       "HOU", ["RP"],     7, 264.0,
     50, 3, 3, 10, 5, 0, 55, 3.28, 1.16, 9.9, 5, 5),
    ("Craig Kimbrel",    "PHI", ["RP"],     7, 266.0,
     48, 3, 4, 14, 6, 0, 58, 3.48, 1.18, 10.9, 6, 8),
    ("Andrew Heaney",    "TEX", ["SP"],     7, 268.0,
     145, 8, 8, 0, 0, 12, 162, 3.98, 1.20, 10.1, 16, 0),
    ("Spencer Turnbull", "PHI", ["SP"],     7, 270.0,
     148, 8, 8, 0, 0, 12, 155, 3.92, 1.19, 9.5, 16, 0),
    ("Ben Lively",       "CLE", ["SP"],     7, 272.0,
     162, 9, 9, 0, 0, 14, 145, 4.12, 1.24, 8.1, 20, 0),
    ("Giovanny Gallegos","STL", ["RP"],     7, 274.0,
     50, 2, 3, 10, 5, 0, 58, 3.38, 1.16, 10.4, 5, 5),
    ("Kyle Muller",      "OAK", ["SP"],     7, 276.0,
     148, 8, 9, 0, 0, 13, 152, 4.18, 1.26, 9.2, 18, 0),
    ("Mitchell Parker",  "WSH", ["SP"],     7, 278.0,
     148, 8, 9, 0, 0, 12, 148, 4.08, 1.26, 9.0, 18, 0),
    ("Reid Detmers",     "LAA", ["SP"],     7, 280.0,
     148, 7, 9, 0, 0, 12, 152, 4.02, 1.22, 9.2, 17, 0),
    ("Colin Rea",        "MIL", ["SP"],     7, 282.0,
     155, 8, 9, 0, 0, 13, 138, 4.22, 1.26, 8.0, 20, 0),
    ("Taj Bradley",      "TB",  ["SP"],     7, 284.0,
     148, 8, 9, 0, 0, 13, 155, 4.08, 1.22, 9.4, 18, 0),
    ("Joe Boyle",        "OAK", ["SP"],     7, 286.0,
     140, 7, 9, 0, 0, 11, 145, 4.18, 1.28, 9.3, 17, 0),
    ("Louie Varland",    "MIN", ["SP"],     7, 288.0,
     145, 8, 9, 0, 0, 12, 148, 4.18, 1.28, 9.2, 18, 0),
    ("Gavin Williams",   "CLE", ["SP"],     7, 290.0,
     145, 8, 9, 0, 0, 12, 152, 3.98, 1.22, 9.4, 17, 0),
    ("Reese Olson",      "DET", ["SP"],     7, 292.0,
     148, 8, 9, 0, 0, 12, 148, 4.08, 1.24, 9.0, 18, 0),
    ("Tommy Henry",      "ARI", ["SP"],     7, 294.0,
     140, 7, 9, 0, 0, 11, 138, 4.22, 1.28, 8.9, 18, 0),
    ("Ryan Pepiot",      "TB",  ["SP"],     7, 296.0,
     145, 8, 8, 0, 0, 13, 148, 3.98, 1.22, 9.2, 17, 0),
    ("Yerry De Los Santos","OAK",["RP"],    7, 298.0,
     48, 2, 3, 10, 5, 0, 55, 3.38, 1.18, 10.3, 5, 5),
    ("Cade Cavalli",     "WSH", ["SP"],     7, 300.0,
     138, 7, 9, 0, 0, 11, 145, 4.18, 1.26, 9.5, 17, 0),
    ("Tyler Wells",      "BAL", ["SP","RP"],7, 302.0,
     145, 8, 9, 0, 0, 12, 148, 4.08, 1.24, 9.2, 18, 0),
    # Grayson Rodriguez — NOT duplicated here; already in Tier 3 at ADP 84
]


# ---------------------------------------------------------------------------
# Build the board — compute z-scores and rank
# ---------------------------------------------------------------------------

def _parse_batter(row: tuple) -> dict:
    (name, team, positions, tier, adp,
     pa, r, h, hr, rbi, k_bat, tb, avg, ops, nsb, _) = row
    slg = ops - 0.330
    return {
        "id": name.lower().replace(" ", "_").replace(".", "").replace("'", ""),
        "name": name, "team": team, "positions": positions,
        "type": "batter", "tier": tier, "adp": adp,
        "proj": {
            "pa": pa, "r": r, "h": h, "hr": hr, "rbi": rbi,
            "k_bat": k_bat, "tb": tb, "avg": avg, "ops": ops,
            "nsb": nsb, "slg": slg,
        },
        "z_score": 0.0, "rank": 0, "cat_scores": {},
    }


def _parse_pitcher(row: tuple) -> dict:
    (name, team, positions, tier, adp,
     ip, w, l, sv, bs, qs, k, era, whip, k9, hr_pit, nsv) = row
    return {
        "id": name.lower().replace(" ", "_").replace(".", "").replace("'", "").replace("é", "e").replace("á", "a").replace("ó", "o").replace("ú", "u").replace("í", "i"),
        "name": name, "team": team, "positions": positions,
        "type": "pitcher", "tier": tier, "adp": adp,
        "proj": {
            "ip": ip, "w": w, "l": l, "sv": sv, "bs": bs,
            "qs": qs, "k_pit": k, "era": era, "whip": whip,
            "k9": k9, "hr_pit": hr_pit, "nsv": nsv,
        },
        "z_score": 0.0, "rank": 0, "cat_scores": {},
    }


def _zscore(value: float, values: list[float], direction: int = 1) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    # Use population std (ddof=0) to match scoring_engine.py — consistent scale
    n = len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    if std < 1e-9:
        return 0.0
    return ((value - mean) / std) * direction


def _compute_zscores(batters: list[dict], pitchers: list[dict]) -> None:
    """Compute and assign z-scores for all players in-place."""

    # Batter pools
    bat_pools = {
        "r":     ([p["proj"]["r"]     for p in batters], 1),
        "h":     ([p["proj"]["h"]     for p in batters], 1),
        "hr":    ([p["proj"]["hr"]    for p in batters], 1),
        "rbi":   ([p["proj"]["rbi"]   for p in batters], 1),
        "k_bat": ([p["proj"]["k_bat"] for p in batters], -1),   # negative
        "tb":    ([p["proj"]["tb"]    for p in batters], 1),
        "avg":   ([p["proj"]["avg"]   for p in batters], 1),
        "ops":   ([p["proj"]["ops"]   for p in batters], 1),
        "nsb":   ([p["proj"]["nsb"]   for p in batters], 1),
    }
    bat_weights = {
        "r": 1.0, "h": 0.9, "hr": 1.1, "rbi": 1.1, "k_bat": 0.8,
        "tb": 1.0, "avg": 1.1, "ops": 1.2, "nsb": 1.4,
    }
    for p in batters:
        total = 0.0
        cat_scores = {}
        for cat, (pool, direction) in bat_pools.items():
            z = _zscore(p["proj"][cat], pool, direction)
            w = bat_weights[cat]
            wz = z * w
            cat_scores[cat] = round(wz, 3)
            total += wz
        p["z_score"] = round(total, 3)
        p["cat_scores"] = cat_scores

    # Pitcher pools
    pit_pools = {
        "w":      ([p["proj"]["w"]      for p in pitchers], 1),
        "l":      ([p["proj"]["l"]      for p in pitchers], -1),   # negative
        "hr_pit": ([p["proj"]["hr_pit"] for p in pitchers], -1),   # negative
        "k_pit":  ([p["proj"]["k_pit"]  for p in pitchers], 1),
        "era":    ([p["proj"]["era"]    for p in pitchers], -1),   # negative
        "whip":   ([p["proj"]["whip"]   for p in pitchers], -1),   # negative
        "k9":     ([p["proj"]["k9"]     for p in pitchers], 1),
        "qs":     ([p["proj"]["qs"]     for p in pitchers], 1),
        "nsv":    ([p["proj"]["nsv"]    for p in pitchers], 1),
    }
    pit_weights = {
        "w": 1.1, "l": 0.8, "hr_pit": 0.7, "k_pit": 1.1,
        "era": 1.1, "whip": 1.1, "k9": 1.0, "qs": 1.0, "nsv": 1.5,
    }
    for p in pitchers:
        total = 0.0
        cat_scores = {}
        for cat, (pool, direction) in pit_pools.items():
            z = _zscore(p["proj"][cat], pool, direction)
            w = pit_weights[cat]
            wz = z * w
            cat_scores[cat] = round(wz, 3)
            total += wz
        p["z_score"] = round(total, 3)
        p["cat_scores"] = cat_scores


def build_board() -> list[dict]:
    """
    Build and return the full ranked player board.
    Players are sorted by z_score (within position type), then merged by ADP.
    """
    batters = [_parse_batter(r) for r in _BATTER_RAW]
    pitchers = [_parse_pitcher(r) for r in _PITCHER_RAW]

    _compute_zscores(batters, pitchers)

    # Rank batters and pitchers separately by z_score
    batters.sort(key=lambda p: p["z_score"], reverse=True)
    pitchers.sort(key=lambda p: p["z_score"], reverse=True)
    for i, p in enumerate(batters, 1):
        p["bat_rank"] = i
    for i, p in enumerate(pitchers, 1):
        p["pit_rank"] = i

    # Merge and sort by ADP for overall rank
    all_players = batters + pitchers
    all_players.sort(key=lambda p: p["adp"])

    # Deduplicate by player ID — keep the first occurrence (lowest ADP)
    seen_ids: set[str] = set()
    deduped = []
    for p in all_players:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            deduped.append(p)

    for i, p in enumerate(deduped, 1):
        p["rank"] = i

    return deduped


# Singleton — built once, reused per process
_BOARD: Optional[list[dict]] = None


def invalidate_board() -> None:
    """Force board rebuild on next call (use after loading real CSVs)."""
    global _BOARD
    _BOARD = None


def get_board(apply_park_factors: bool = True) -> list[dict]:
    """
    Return the full ranked player board.

    Priority:
    1. Real Steamer/ZiPS CSV data (if data/projections/ CSVs are present)
    2. Hardcoded estimates (fallback — always available)

    Park factors and risk adjustments are applied on top of either source.
    """
    global _BOARD
    if _BOARD is None:
        # Try real projection data first
        try:
            from backend.fantasy_baseball.projections_loader import load_full_board
            real_board = load_full_board()
            if real_board:
                _BOARD = real_board
        except Exception:
            pass

        if _BOARD is None:
            _BOARD = build_board()

        # Apply park factors and risk flags to whichever board we have
        if apply_park_factors:
            try:
                from backend.fantasy_baseball.ballpark_factors import annotate_board
                annotate_board(_BOARD)
            except Exception:
                pass

        # Stamp keeper flags
        annotate_keepers(_BOARD)

    return _BOARD


def reset_board_cache() -> None:
    """Clear the in-memory board cache so the next get_board() call reloads from disk.

    Called by /admin/board/refresh when new Steamer CSVs are dropped to data/projections/.
    Safe to call at any time -- next request rebuilds the board automatically.
    """
    global _BOARD
    _BOARD = None
    try:
        from backend.fantasy_baseball.projections_loader import load_full_board
        load_full_board.cache_clear()
    except Exception:
        pass


def get_player_by_name(name: str) -> Optional[dict]:
    board = get_board()
    name_lower = name.lower()
    for p in board:
        if name_lower in p["name"].lower():
            return p
    return None


# ---------------------------------------------------------------------------
# Keeper configuration — my keepers for this season
# ---------------------------------------------------------------------------

MY_KEEPERS: dict[str, int] = {
    "juan_soto": 1,   # Keep Juan Soto, costs Round 1
}

# All 14 league-wide keepers (all teams). Used to pre-filter the value board
# before the Yahoo roster API sweep fires at 19:00. Verified from Yahoo lock
# screen 2026-03-23.
ALL_LEAGUE_KEEPERS: frozenset[str] = frozenset({
    "aaron_judge",        # ChippaJone
    "shohei_ohtani",      # Marte Partay
    "bobby_witt_jr",      # Bartolo's Colon
    "juan_soto",          # Lindor Truffles (us)
    "elly_de_la_cruz",    # Mendoza Line
    "kyle_tucker",        # Juiced Balls
    "jose_ramirez",       # Game Blausers
    "ronald_acuna_jr",    # Juiced Balls
    "julio_rodriguez",    # Damn the Torpedoes
    "corbin_carroll",     # Slap Dick Prospects
    "fernando_tatis_jr",  # Mendoza Line
    "francisco_lindor",   # ChippaJone
    "nick_kurtz",         # High&TightyWhitey's
    "jackson_merrill",    # High&TightyWhitey's
})


def annotate_keepers(board: list[dict]) -> None:
    """Stamp is_keeper / keeper_round onto keeper players (in-place)."""
    for p in board:
        if p["id"] in MY_KEEPERS:
            p["is_keeper"] = True
            p["keeper_round"] = MY_KEEPERS[p["id"]]
        elif p["id"] in ALL_LEAGUE_KEEPERS:
            p["is_keeper"] = True
            p["keeper_round"] = None  # other team's keeper — round unknown
        else:
            p.setdefault("is_keeper", False)
            p.setdefault("keeper_round", None)


def available_players(drafted_ids: set[str]) -> list[dict]:
    """Return players not yet drafted and not a league keeper."""
    excluded = ALL_LEAGUE_KEEPERS | drafted_ids
    return [p for p in get_board() if p["id"] not in excluded]


# ---------------------------------------------------------------------------
# Position baseline z-scores (conservative — median of bottom half of board)
# Used as proxy for call-ups / undrafted players not in PLAYER_BOARD.
# Derived from get_board() distribution; update each spring.
# ---------------------------------------------------------------------------

# Runtime cache for on-the-fly projections (cleared on process restart)
_projection_cache: dict[str, dict] = {}


def _query_statcast_proxy(db: Session, player_id: str, player_type: str,
                          name: str = "") -> dict | None:
    """
    Query Statcast metrics tables and return RAW DATA for fusion engine.

    NO LONGER builds projections — fusion_engine.py handles that.
    Returns raw Statcast metrics with sample_size for Bayesian fusion.

    Pitcher qualifications: IP >= 20.0
    Batter qualifications: PA >= 50 (meaningful sample)

    Args:
        db: SQLAlchemy database session
        player_id: Yahoo player ID string
        player_type: 'batter' or 'pitcher'
        name: Player name (for fallback name-based lookup)

    Returns:
        Dict with raw Statcast metrics including 'sample_size', or None
    """
    if db is None:
        return None

    from backend.models import (
        StatcastBatterMetrics, StatcastPitcherMetrics,
        PlayerIDMapping
    )

    # Try to find mlbam_id via PlayerIDMapping
    mlbam_id = None
    id_mapping = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.yahoo_id == player_id
    ).first()

    if id_mapping and id_mapping.mlbam_id:
        mlbam_id = str(id_mapping.mlbam_id)

    # If no ID mapping, try name-based lookup as fallback
    if not mlbam_id and name:
        name_normalized = name.lower().strip()
        mapping = db.query(PlayerIDMapping).filter(
            PlayerIDMapping.normalized_name == name_normalized
        ).first()
        if mapping and mapping.mlbam_id:
            mlbam_id = str(mapping.mlbam_id)

    if not mlbam_id:
        return None

    # Query appropriate Statcast table
    if player_type == "pitcher":
        metrics = db.query(StatcastPitcherMetrics).filter(
            StatcastPitcherMetrics.mlbam_id == mlbam_id
        ).first()

        if not metrics:
            return None
        ip_val = getattr(metrics, 'ip', None)
        if not isinstance(ip_val, (int, float)) or ip_val < 20.0:
            return None  # Insufficient data for reliable projection

        # Return RAW Statcast metrics for fusion engine
        # Include sample_size (IP converted to int for fusion)
        return {
            "era": metrics.era,
            "whip": metrics.whip,
            "k_percent": metrics.k_percent,
            "bb_percent": metrics.bb_percent,
            "k_9": metrics.k_9,
            "xera": metrics.xera,
            "ip": ip_val,
            "w": metrics.w or 0,
            "l": metrics.l or 0,
            "sv": int(metrics.sv or 0),
            "qs": metrics.qs or 0,
            "hr_pit": metrics.hr_pit or 0,
            "k_pit": metrics.k_pit or 0,
            "sample_size": int(ip_val),  # Required by fusion engine
        }

    else:  # batter
        metrics = db.query(StatcastBatterMetrics).filter(
            StatcastBatterMetrics.mlbam_id == mlbam_id
        ).first()

        if not metrics:
            return None
        pa_val = getattr(metrics, 'pa', None)
        if not isinstance(pa_val, (int, float)) or pa_val < 50:
            return None  # Insufficient data for reliable projection

        # Return RAW Statcast metrics for fusion engine
        # Include sample_size (PA)
        raw_data = {
            "avg": metrics.avg,
            "slg": metrics.slg,
            "ops": metrics.ops,
            "xwoba": metrics.xwoba,
            "pa": metrics.pa,
            "hr": metrics.hr or 0,
            "r": metrics.r or 0,
            "rbi": metrics.rbi or 0,
            "sb": metrics.sb or 0,
            "sample_size": metrics.pa or 0,  # Required by fusion engine
        }

        # Handle obp - may not be in StatcastBatterMetrics, estimate if missing
        obp_val = getattr(metrics, 'obp', None)
        if isinstance(obp_val, (int, float)):
            raw_data["obp"] = obp_val
        else:
            if raw_data["avg"] is not None:
                raw_data["obp"] = raw_data["avg"] + 0.070
            else:
                raw_data["obp"] = None

        # Handle woba - may not be in metrics
        woba_val = getattr(metrics, 'woba', None)
        if isinstance(woba_val, (int, float)):
            raw_data["woba"] = woba_val
        else:
            if raw_data["ops"] is not None:
                raw_data["woba"] = raw_data["ops"] * 0.95
            else:
                raw_data["woba"] = None

        return raw_data


def _get_proj_name_map(db) -> dict:
    """
    Return a normalized-name -> _ProjectionEntry dict for all PlayerProjection
    rows that have cat_scores populated.

    The result is module-level cached for _PROJ_NAME_CACHE_TTL seconds (30 min).
    Plain namedtuples are stored — never SQLAlchemy row objects — so the cache
    survives across DB sessions without detached-instance errors.
    """
    from backend.models import PlayerProjection

    now = time.monotonic()
    cached = _PROJ_NAME_CACHE.get("map")
    built_at = _PROJ_NAME_CACHE.get("built_at", 0.0)

    if cached is not None and (now - built_at) < _PROJ_NAME_CACHE_TTL:
        return cached

    rows = db.query(PlayerProjection).filter(
        PlayerProjection.cat_scores.isnot(None)
    ).all()

    name_map: dict = {}
    for row in rows:
        raw_name = row.player_name or ""
        if not raw_name:
            continue
        norm = _normalize_name(raw_name)
        entry = _ProjectionEntry(
            player_name=raw_name,
            player_id=row.player_id,
            player_type=row.player_type,
            cat_scores=row.cat_scores,
            z_score=getattr(row, "z_score", None),
            avg=row.avg,
            obp=row.obp,
            slg=row.slg,
            ops=row.ops,
            hr=row.hr,
            sb=row.sb,
            era=row.era,
            whip=row.whip,
            k_per_nine=row.k_per_nine,
            bb_per_nine=row.bb_per_nine,
            w=row.w,
            qs=row.qs,
        )
        name_map[norm] = entry

    _PROJ_NAME_CACHE["map"] = name_map
    _PROJ_NAME_CACHE["built_at"] = now
    logger.debug("[player_board] Rebuilt projection name map: %d entries", len(name_map))
    return name_map


def _lookup_projection_by_name(db, name: str):
    """
    Find a _ProjectionEntry by player name using the TTL-cached projection map.

    Returns a _ProjectionEntry namedtuple (not a SQLAlchemy row) so the result
    is safe to use after the DB session is closed.  Tries exact normalized-name
    match first, then difflib fuzzy match at cutoff=0.85.
    """
    norm = _normalize_name(name) if name else ""
    if not norm:
        return None

    name_map = _get_proj_name_map(db)

    # Exact match
    if norm in name_map:
        return name_map[norm]

    # Fuzzy match
    matches = _difflib.get_close_matches(norm, name_map.keys(), n=1, cutoff=0.85)
    if matches:
        return name_map[matches[0]]

    return None


def _lookup_canonical_by_name(db, name: str, player_type: str):
    """Return (CanonicalProjection, [CategoryImpact]) for a SAVANT_ADJUSTED row, or None.

    Bridges through PlayerIdentity because CanonicalProjection has no player_name.
    Tries mlbam_id first, then negative yahoo_id fallback namespace.
    """
    from backend.models import CanonicalProjection, CategoryImpact, PlayerIdentity

    norm = _normalize_name(name) if name else ""
    if not norm or not db:
        return None

    identity_row = (
        db.query(PlayerIdentity)
        .filter(PlayerIdentity.normalized_name == norm)
        .first()
    )
    if not identity_row:
        return None

    candidate_ids = []
    if identity_row.mlbam_id:
        candidate_ids.append(identity_row.mlbam_id)
    if identity_row.yahoo_id:
        candidate_ids.append(-(int(identity_row.yahoo_id)))

    if not candidate_ids:
        return None

    cp = (
        db.query(CanonicalProjection)
        .filter(
            CanonicalProjection.player_id.in_(candidate_ids),
            CanonicalProjection.player_type == player_type.upper(),
            CanonicalProjection.source_engine == "SAVANT_ADJUSTED",
        )
        .order_by(CanonicalProjection.projection_date.desc())
        .first()
    )
    if cp is None:
        return None

    impacts = (
        db.query(CategoryImpact)
        .filter(CategoryImpact.canonical_projection_id == cp.id)
        .all()
    )
    return cp, impacts


# CategoryImpact.category → player_board cat_scores key mappings
_CI_BATTER_MAP = {"R": "r", "HR": "hr", "RBI": "rbi", "SB": "sb", "AVG": "avg", "OPS": "ops"}
_CI_PITCHER_MAP = {"W": "w", "K": "k_pit", "SV": "sv", "ERA": "era", "WHIP": "whip", "K9": "k9"}
# Note: k_bat, tb, nsb, l, hr_pit, qs are not emitted by ProjectionAssemblyService yet.


def _category_impacts_to_cat_scores(impacts, player_type: str) -> dict:
    """Convert CategoryImpact rows to player_board-compatible cat_scores dict."""
    impact_map = _CI_BATTER_MAP if player_type == "batter" else _CI_PITCHER_MAP
    cat_scores: dict = {}
    for impact in impacts:
        board_key = impact_map.get(impact.category)
        if board_key is not None and impact.z_score is not None:
            cat_scores[board_key] = round(impact.z_score, 3)
    return cat_scores


def get_or_create_projection(yahoo_player: dict) -> dict:
    """
    Return a board-compatible dict using Bayesian fusion (four-state logic).

    Four-State Logic:
        1. Steamer + Statcast: Full Marcel update via fusion_engine
        2. Steamer only: Return Steamer as-is
        3. Statcast only: Fuse with population prior (double shrinkage)
        4. Neither: Return population prior with generic z-score

    Args:
        yahoo_player: dict from YahooFantasyClient (has name, player_key,
                      positions, team, percent_owned, etc.)

    Returns:
        board-compatible dict with at minimum: name, z_score, positions,
        cat_scores, type, proj, and fusion metadata.
    """
    from backend.models import get_db, PlayerIdentity, PlayerProjection, PlayerIDMapping
    from backend.services.cat_scores_builder import (
        BATTER_WEIGHTS, PITCHER_WEIGHTS, compute_cat_scores
    )

    name = (yahoo_player.get("name") or "").strip()
    name_normalized = _normalize_name(name) if name else None
    player_key = yahoo_player.get("player_key") or ""

    # 1. Check runtime cache first (avoids repeated lookups).
    # Skip the cache when the caller provided explicit cat_scores so test fixtures
    # are never masked by prior runs that cached the same player_key.
    if player_key and player_key in _projection_cache and not yahoo_player.get("cat_scores"):
        return _projection_cache[player_key]

    # 2. Check board by exact name match ONLY if database lookup fails
    # This is a fallback for players not in the database (e.g., Christopher Sanchez)
    # IMPORTANT: Database check happens first (below), draft board is only for missing DB data

    # NOTE: We skip draft board check here and do it AFTER database query
    # This ensures real DB projections are always preferred over draft board data
    positions = yahoo_player.get("positions") or []
    primary_pos = positions[0] if positions else ""

    # Infer type from position
    player_type = "pitcher" if primary_pos in ("SP", "RP", "P") else "batter"

    # Extract Yahoo ID from player_key
    # Standard Yahoo keys: "469.l.X.p.12345" or "mlb.p.12345" → "12345"
    # Non-standard (no .p.): "mlb.12345" → "12345" via last segment
    yahoo_id = None
    if player_key and ".p." in player_key:
        yahoo_id = player_key.split(".p.")[-1]
    elif player_key:
        yahoo_id = player_key.split(".")[-1]
    yahoo_id = yahoo_id or None

    # Check for explicit test data
    existing_cat_scores = yahoo_player.get("cat_scores") or {}
    if existing_cat_scores:
        # Test provided cat_scores directly — use them
        proxy = {
            "id": player_key or name.lower().replace(" ", "_"),
            "name": name,
            "team": yahoo_player.get("team") or yahoo_player.get("editorial_team_abbr") or "",
            "positions": positions,
            "type": player_type,
            "tier": 10,
            "rank": 9999,
            "adp": 9999.0,
            "z_score": sum(existing_cat_scores.values()) if existing_cat_scores else 0.0,
            "cat_scores": existing_cat_scores,
            "proj": {},
            "is_keeper": False,
            "keeper_round": None,
            "is_proxy": True,
            "fusion_source": "test_data",
            "components_fused": 0,
            "xwoba_override": False,
        }
        if player_key:
            _projection_cache[player_key] = proxy
        return proxy

    # Try to query DB for Steamer and Statcast data
    steamer_data = None
    statcast_data = None
    sample_size = 0
    fusion_metadata = {
        "fusion_source": "population_prior",
        "components_fused": 0,
        "xwoba_override": False,
    }

    db = None
    db_gen = None
    projection_row = None
    identity_row = None
    mlbam_id = None
    try:
        db_gen = get_db()
        db = next(db_gen)

        if yahoo_id:
            # Translate Yahoo ID → MLBAM ID via PlayerIDMapping
            id_mapping = db.query(PlayerIDMapping).filter(
                PlayerIDMapping.yahoo_id == yahoo_id
            ).first()
            if id_mapping:
                mlbam_id = id_mapping.mlbam_id or id_mapping.bdl_id

        if name:
            resolved_identity_id = _IDENTITY_RESOLUTION_SERVICE.resolve(
                db,
                yahoo_guid=player_key or None,
                yahoo_id=yahoo_id,
                full_name=name,
                provider="YAHOO",
                raw_id=player_key or yahoo_id or None,
            )
            if resolved_identity_id is not None:
                identity_row = db.query(PlayerIdentity).filter(
                    PlayerIdentity.id == resolved_identity_id,
                ).first()
            elif name_normalized:
                identity_row = db.query(PlayerIdentity).filter(
                    PlayerIdentity.normalized_name == name_normalized
                ).first()
            if identity_row and identity_row.mlbam_id and not mlbam_id:
                mlbam_id = str(identity_row.mlbam_id)

        if mlbam_id:
            # Query PlayerProjection for Steamer data
            projection_row = db.query(PlayerProjection).filter(
                PlayerProjection.player_id == str(mlbam_id)
            ).first()
        if not projection_row and identity_row:
            projection_row = db.query(PlayerProjection).filter(
                PlayerProjection.player_name == identity_row.full_name
            ).first()
            if projection_row:
                logger.debug(
                    f"[player_board] Identity-based exact match: {name} -> {projection_row.player_name}"
                )

    except Exception as e:
        logger.debug(f"[player_board] DB query failed for {name}: {e}")

    # Name-based fallback: query player_projections directly by name when identity
    # chain fails (e.g. player not yet in player_identities table)
    if not projection_row and db and name:
        try:
            projection_row = _lookup_projection_by_name(db, name)
            if projection_row:
                logger.info(
                    "[player_board] Name-fallback found %s → %s (cat_scores=%s)",
                    name, projection_row.player_name, bool(projection_row.cat_scores)
                )
        except Exception as _nf_err:
            logger.debug("[player_board] Name-fallback failed for %s: %s", name, _nf_err)

    # FAST PATH: If projection_row has curated cat_scores (real dict),
    # use them directly without running fusion. This preserves the pre-Phase 9.5
    # contract used by the waiver/optimize callers and yahoo_id translation tests.
    projection_cat_scores = None
    if projection_row is not None:
        pcs = getattr(projection_row, 'cat_scores', None)
        if isinstance(pcs, dict) and pcs:
            projection_cat_scores = pcs

    if projection_cat_scores is not None:
        # Close DB session before returning
        if db_gen is not None:
            try:
                next(db_gen)
            except (StopIteration, Exception):
                pass
        proxy = {
            "id": player_key or name.lower().replace(" ", "_"),
            "name": name,
            "team": yahoo_player.get("team") or yahoo_player.get("editorial_team_abbr") or "",
            "positions": positions,
            "type": player_type,
            "tier": 10,
            "rank": 9999,
            "adp": 9999.0,
            "z_score": sum(projection_cat_scores.values()),
            "cat_scores": projection_cat_scores,
            "proj": {},
            "is_keeper": False,
            "keeper_round": None,
            "is_proxy": True,
            "fusion_source": "steamer_db",
            "components_fused": 0,
            "xwoba_override": False,
        }
        if player_key:
            _projection_cache[player_key] = proxy
        return proxy

    # FAST PATH 2: SAVANT_ADJUSTED CanonicalProjection
    # PlayerProjection had no curated cat_scores — try the Statcast-fused table.
    # Covers ~238 players (48% of top-500 fantasy-relevant players).
    # Missing categories: k_bat, tb, nsb, l, hr_pit, qs (not yet in CategoryImpact).
    if db is not None:
        try:
            canonical_result = _lookup_canonical_by_name(db, name, player_type)
            if canonical_result is not None:
                cp, impacts = canonical_result
                sa_cat_scores = _category_impacts_to_cat_scores(impacts, player_type)
                if sa_cat_scores:
                    logger.info(
                        "[player_board] SAVANT_ADJUSTED fast path: %s → %s cats",
                        name, list(sa_cat_scores.keys()),
                    )
                    if db_gen is not None:
                        try:
                            next(db_gen)
                        except (StopIteration, Exception):
                            pass
                    proxy = {
                        "id": player_key or name.lower().replace(" ", "_"),
                        "name": name,
                        "team": yahoo_player.get("team") or yahoo_player.get("editorial_team_abbr") or "",
                        "positions": positions,
                        "type": player_type,
                        "tier": 10,
                        "rank": 9999,
                        "adp": 9999.0,
                        "z_score": sum(sa_cat_scores.values()),
                        "cat_scores": sa_cat_scores,
                        "proj": {},
                        "is_keeper": False,
                        "keeper_round": None,
                        "is_proxy": True,
                        "fusion_source": "savant_adjusted_db",
                        "components_fused": 2,
                        "xwoba_override": False,
                    }
                    if player_key:
                        _projection_cache[player_key] = proxy
                    return proxy
        except Exception as _cp_err:
            logger.debug("[player_board] CanonicalProjection lookup failed for %s: %s", name, _cp_err)

    # Always invoke helpers — they handle None inputs gracefully and the
    # tests rely on these being patchable as the entry points.
    try:
        steamer_data = _extract_steamer_data(projection_row, player_type)
    except Exception as e:
        logger.debug(f"[player_board] _extract_steamer_data failed for {name}: {e}")
        steamer_data = None
    try:
        statcast_data = _query_statcast_proxy(db, yahoo_id, player_type, name)
    except Exception as e:
        logger.debug(f"[player_board] _query_statcast_proxy failed for {name}: {e}")
        statcast_data = None

    # If Statcast lookup by yahoo_id failed, try mlbam_id
    if not statcast_data and mlbam_id and db is not None:
        try:
            statcast_data = _query_statcast_proxy_mlbam(db, str(mlbam_id), player_type)
        except Exception:
            pass

    # Close DB session
    if db_gen is not None:
        try:
            next(db_gen)
        except (StopIteration, Exception):
            pass

    # Get sample size from Statcast data
    if statcast_data is not None:
        if isinstance(statcast_data, dict):
            sample_size = statcast_data.get("sample_size", 0)
            if sample_size is None:
                sample_size = statcast_data.get("pa", statcast_data.get("ip", 0))
        else:
            sample_size = getattr(statcast_data, "sample_size", 0)
            if sample_size is None or not isinstance(sample_size, (int, float)):
                sample_size = getattr(statcast_data, "pa", 0) or getattr(statcast_data, "ip", 0) or 0
        if sample_size is None or not isinstance(sample_size, (int, float)):
            sample_size = 0

    # Apply four-state Bayesian fusion
    if player_type == "batter":
        fusion_result = fuse_batter_projection(
            steamer=steamer_data,
            statcast=statcast_data,
            sample_size=max(1, sample_size)  # Ensure at least 1 for fusion
        )
    else:
        fusion_result = fuse_pitcher_projection(
            steamer=steamer_data,
            statcast=statcast_data,
            sample_size=max(1, sample_size)
        )

    # Update metadata from fusion result
    fusion_metadata["fusion_source"] = fusion_result.source
    fusion_metadata["components_fused"] = fusion_result.components_fused
    fusion_metadata["xwoba_override"] = fusion_result.xwoba_override_detected  # Detection only

    # Convert fusion engine output to board-compatible projection format
    fused_proj = _convert_fusion_proj_to_board_format(
        fusion_result.proj,
        player_type,
        steamer_projection=projection_row  # Pass through for counting stats
    )

    # cat_scores and z_score: Only pre-computed DB z-scores are used.
    # Fusion engine does NOT compute cat_scores to avoid scale mismatch.
    # All non-DB proxy players get z_score=0.0 (neutral) until a periodic
    # backfill runs compute_cat_scores() against the full player pool.
    cat_scores = {}
    z_score = 0.0

    # DRAFT BOARD FALLBACK: Only use if we have NO real data from database
    # This ensures players like Christopher Sanchez (not in DB) still get some projection
    # while preventing Gavin Williams draft data from overriding real DB projections.
    if not projection_cat_scores and not steamer_data and not statcast_data:
        # No database data found - use draft board as fallback
        logger.info(f"[player_board] No DB data for {name}, checking draft board fallback")

        board = get_board()
        board_by_name = {p["name"].lower(): p for p in board}
        entry = board_by_name.get(name.lower())

        if entry:
            logger.info(f"[player_board] Using draft board fallback for {name}")
            if player_key:
                _projection_cache[player_key] = entry
            return entry

        # Try fuzzy match for draft board
        import difflib as _difflib
        clean_name = "".join(c for c in name.lower() if c.isalnum() or c == " ")
        for board_name, board_entry in board_by_name.items():
            clean_board = "".join(c for c in board_name if c.isalnum() or c == " ")
            if clean_board == clean_name:
                logger.info(f"[player_board] Using draft board fuzzy match for {name}")
                if player_key:
                    _projection_cache[player_key] = board_entry
                return board_entry

        # Similarity match for draft board
        best_ratio = 0.0
        best_entry = None
        for board_name, board_entry in board_by_name.items():
            clean_board = "".join(c for c in board_name if c.isalnum() or c == " ")
            ratio = _difflib.SequenceMatcher(None, clean_name, clean_board).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_entry = board_entry
        if best_ratio >= 0.90 and best_entry is not None:
            logger.info(f"[player_board] Using draft board similarity match for {name} (ratio={best_ratio:.2f})")
            if player_key:
                _projection_cache[player_key] = best_entry
            return best_entry

    # Log which fusion path was taken
    logger.info(f"[player_board] Fusion for {name}: source={fusion_result.source}, "
                f"components_fused={fusion_result.components_fused}, "
                f"sample_size={sample_size}")

    # Build proxy dict
    proxy = {
        "id": player_key or name.lower().replace(" ", "_"),
        "name": name,
        "team": yahoo_player.get("team") or yahoo_player.get("editorial_team_abbr") or "",
        "positions": positions,
        "type": player_type,
        "tier": 10,
        "rank": 9999,
        "adp": 9999.0,
        "z_score": z_score,
        "cat_scores": cat_scores,
        "proj": fused_proj,
        "is_keeper": False,
        "keeper_round": None,
        "is_proxy": True,
        # Fusion metadata for debugging
        "fusion_source": fusion_metadata["fusion_source"],
        "components_fused": fusion_metadata["components_fused"],
        "xwoba_override": fusion_metadata["xwoba_override"],
    }

    if player_key:
        _projection_cache[player_key] = proxy
    return proxy


def _extract_steamer_data(projection_row, player_type: str) -> dict | None:
    """
    Extract Steamer projection data from PlayerProjection row.

    Returns dict format expected by fusion_engine, or None if no valid data.
    """
    if not projection_row:
        return None

    if player_type == "batter":
        # Require at least one numeric rate stat. Mock attrs (non-numeric) count as missing.
        avg_val = getattr(projection_row, 'avg', None)
        ops_val = getattr(projection_row, 'ops', None)
        if not isinstance(avg_val, (int, float)) and not isinstance(ops_val, (int, float)):
            return None

        def _num(attr, default):
            v = getattr(projection_row, attr, None)
            return v if isinstance(v, (int, float)) else default

        return {
            "avg": _num('avg', 0.250),
            "obp": _num('obp', 0.320),
            "slg": _num('slg', 0.400),
            "ops": _num('ops', 0.720),
            "hr_per_pa": _num('hr', 15) / 550.0,
            "sb_per_pa": _num('sb', 5) / 550.0,
            "k_percent": 0.225,
            "bb_percent": 0.080,
        }
    else:
        # pitcher
        # Reject only if ALL key fields are at their defaults (uninitialized row).
        # A pitcher with ERA 4.00 and K/9 8.5 is a real projection (league average).
        # Mock objects (from tests) count as "unset".
        def _is_unset_or_default(val, default):
            if val is None:
                return True
            if isinstance(val, (int, float)):
                return val == default
            # Non-numeric (e.g. Mock) counts as unset
            return True

        era_default = _is_unset_or_default(projection_row.era, 4.00)
        whip_default = _is_unset_or_default(projection_row.whip, 1.30)
        k9_default = _is_unset_or_default(projection_row.k_per_nine, 8.5)
        bb9_default = _is_unset_or_default(projection_row.bb_per_nine, 3.0)

        if era_default and whip_default and k9_default and bb9_default:
            # All defaults — uninitialized row, not real pitcher data
            return None

        return {
            "era": projection_row.era if projection_row.era is not None else 4.50,
            "whip": projection_row.whip if projection_row.whip is not None else 1.35,
            "k_per_nine": projection_row.k_per_nine if projection_row.k_per_nine is not None else 8.5,
            "bb_per_nine": projection_row.bb_per_nine if projection_row.bb_per_nine is not None else 3.0,
            "k_percent": 0.22,  # Default if not in Steamer
            "bb_percent": 0.07,  # Default if not in Steamer
        }


def _query_statcast_proxy_mlbam(db: Session, mlbam_id: str, player_type: str) -> dict | None:
    """
    Query Statcast metrics using mlbam_id directly.

    Helper for when yahoo_id lookup failed but we have mlbam_id.
    """
    if db is None:
        return None

    from backend.models import StatcastBatterMetrics, StatcastPitcherMetrics

    if player_type == "pitcher":
        metrics = db.query(StatcastPitcherMetrics).filter(
            StatcastPitcherMetrics.mlbam_id == mlbam_id
        ).first()

        if not metrics:
            return None
        ip_val = getattr(metrics, 'ip', None)
        if not isinstance(ip_val, (int, float)) or ip_val < 20.0:
            return None

        return {
            "era": metrics.era,
            "whip": metrics.whip,
            "k_percent": metrics.k_percent,
            "bb_percent": metrics.bb_percent,
            "k_9": metrics.k_9,
            "xera": metrics.xera,
            "ip": ip_val,
            "w": metrics.w or 0,
            "l": metrics.l or 0,
            "sv": int(metrics.sv or 0),
            "qs": metrics.qs or 0,
            "hr_pit": metrics.hr_pit or 0,
            "k_pit": metrics.k_pit or 0,
            "sample_size": int(ip_val),
        }
    else:
        metrics = db.query(StatcastBatterMetrics).filter(
            StatcastBatterMetrics.mlbam_id == mlbam_id
        ).first()

        if not metrics:
            return None
        pa_val = getattr(metrics, 'pa', None)
        if not isinstance(pa_val, (int, float)) or pa_val < 50:
            return None

        raw_data = {
            "avg": metrics.avg,
            "slg": metrics.slg,
            "ops": metrics.ops,
            "xwoba": metrics.xwoba,
            "pa": metrics.pa,
            "hr": metrics.hr or 0,
            "r": metrics.r or 0,
            "rbi": metrics.rbi or 0,
            "sb": metrics.sb or 0,
            "sample_size": metrics.pa or 0,
        }

        # Handle obp - may not be in StatcastBatterMetrics
        obp_val = getattr(metrics, 'obp', None)
        if isinstance(obp_val, (int, float)):
            raw_data["obp"] = obp_val
        else:
            if raw_data["avg"] is not None:
                raw_data["obp"] = raw_data["avg"] + 0.070
            else:
                raw_data["obp"] = None

        # Handle woba
        woba_val = getattr(metrics, 'woba', None)
        if isinstance(woba_val, (int, float)):
            raw_data["woba"] = woba_val
        else:
            if raw_data["ops"] is not None:
                raw_data["woba"] = raw_data["ops"] * 0.95
            else:
                raw_data["woba"] = None

        return raw_data


def _convert_fusion_proj_to_board_format(
    fusion_proj: dict,
    player_type: str,
    steamer_projection=None
) -> dict:
    """
    Convert fusion engine output format to board-compatible projection format.

    The fusion engine produces rate stats. Counting stats come from Steamer
    when available; otherwise use heuristics (for statcast-only / population_prior).

    Args:
        fusion_proj: Fused rate stats from fusion engine
        player_type: 'batter' or 'pitcher'
        steamer_projection: PlayerProjection row with counting stats (optional)

    Returns:
        Board-compatible projection dict with both rate and counting stats.
    """
    if player_type == "batter":
        # Full-season target for rate stat → counting stat conversion
        pa = 550

        # Extract Steamer counting stats when available
        if steamer_projection is not None:
            steamer_hr = getattr(steamer_projection, 'hr', None)
            steamer_r = getattr(steamer_projection, 'r', None)
            steamer_rbi = getattr(steamer_projection, 'rbi', None)
            steamer_sb = getattr(steamer_projection, 'sb', None)
        else:
            steamer_hr = steamer_r = steamer_rbi = steamer_sb = None

        # Use Steamer counting stats when present, otherwise estimate from fused rates
        hr_per_pa = fusion_proj.get("hr_per_pa", 0.035)
        sb_per_pa = fusion_proj.get("sb_per_pa", 0.010)
        k_pct = fusion_proj.get("k_percent", 0.225)

        return {
            "pa": pa,
            "r": steamer_r if isinstance(steamer_r, int) else round(fusion_proj.get("obp", 0.320) * pa * 0.14),
            "h": round(fusion_proj.get("avg", 0.250) * pa * 0.87),
            "hr": steamer_hr if isinstance(steamer_hr, int) else round(hr_per_pa * pa),
            "rbi": steamer_rbi if isinstance(steamer_rbi, int) else round(fusion_proj.get("slg", 0.410) * pa * 0.16),
            "k_bat": round(k_pct * pa),
            "tb": round(fusion_proj.get("slg", 0.410) * pa * 1.85),
            "avg": fusion_proj.get("avg", 0.250),
            "ops": fusion_proj.get("ops", fusion_proj.get("obp", 0.320) + fusion_proj.get("slg", 0.410)),
            "nsb": steamer_sb if isinstance(steamer_sb, int) else round(sb_per_pa * pa),
        }
    else:
        # pitcher - Steamer has counting stats; otherwise estimate from fused rates
        ip = 180

        if steamer_projection is not None:
            steamer_w = getattr(steamer_projection, 'w', None)
            steamer_l = getattr(steamer_projection, 'l', None)
            steamer_qs = getattr(steamer_projection, 'qs', None)
            steamer_hr_pit = getattr(steamer_projection, 'hr_pit', None)
            steamer_k_pit = getattr(steamer_projection, 'k_pit', None)
            steamer_nsv = getattr(steamer_projection, 'nsv', None)
        else:
            steamer_w = steamer_l = steamer_qs = steamer_hr_pit = steamer_k_pit = steamer_nsv = None

        era = fusion_proj.get("era", 4.50)
        k9 = fusion_proj.get("k_per_nine", 8.5)

        # Use mathematically sound formulas when Steamer counting stats unavailable.
        # See PitcherCountingStatFormulas docstring for derivation and constants.
        from backend.fantasy_baseball.fusion_engine import PitcherCountingStatFormulas as _pcf

        _w = _pcf.project_wins(era, ip) if steamer_w is None else steamer_w
        _l = _pcf.project_losses(era, ip) if steamer_l is None else steamer_l
        _qs = _pcf.project_quality_starts(era, ip) if steamer_qs is None else steamer_qs
        _hr_pit = _pcf.project_hr_allowed(era, ip) if steamer_hr_pit is None else steamer_hr_pit

        return {
            "ip": ip,
            "w": _w if isinstance(_w, int) else round(_w),
            "l": _l if isinstance(_l, int) else round(_l),
            "hr_pit": _hr_pit if isinstance(_hr_pit, int) else round(_hr_pit),
            "k_pit": steamer_k_pit if isinstance(steamer_k_pit, int) else round(k9 * ip / 9),
            "era": era,
            "whip": fusion_proj.get("whip", 1.35),
            "k9": k9,
            "qs": _qs if isinstance(_qs, int) else round(_qs),
            "nsv": steamer_nsv if isinstance(steamer_nsv, int) else 0,
        }


# ---------------------------------------------------------------------------
# CLI — print top 50
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    board = get_board()
    print(f"\n{'Rank':>4} {'Tier':>4} {'Name':<24} {'Team':>4} {'Pos':<18} {'Type':>7} {'ADP':>6} {'Z-Score':>8}")
    print("-" * 85)
    for p in board[:60]:
        pos_str = "/".join(p["positions"][:3])
        print(
            f"{p['rank']:>4} {p['tier']:>4}  {p['name']:<24} {p['team']:>4} "
            f"{pos_str:<18} {p['type']:>7} {p['adp']:>6.1f} {p['z_score']:>+8.2f}"
        )
