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

import statistics
from typing import Optional


# ---------------------------------------------------------------------------
# Raw player data (2026 Steamer/ZiPS consensus)
# Format: (name, team, positions, type, tier, adp,
#          pa/ip, r/w, h/l, hr/sv, rbi/bs, k_bat/qs, tb/k_pit, avg/era, ops/whip, nsb/k9, nsv/hr_pit)
# ---------------------------------------------------------------------------

_BATTER_RAW = [
    # Tier 1 — Elite multi-cat (pick 1-12)
    ("Ronald Acuna Jr.", "ATL", ["LF", "CF", "RF", "OF"], 1, 1.0,
     680, 115, 190, 37, 95, 145, 345, 0.306, 0.995, 58, 0),
    ("Juan Soto",        "NYY", ["LF", "RF", "OF"],        1, 2.0,
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
    std = statistics.stdev(values)
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
    for i, p in enumerate(all_players, 1):
        p["rank"] = i

    return all_players


# Singleton — built once, reused
_BOARD: Optional[list[dict]] = None


def get_board() -> list[dict]:
    global _BOARD
    if _BOARD is None:
        _BOARD = build_board()
    return _BOARD


def get_player_by_name(name: str) -> Optional[dict]:
    board = get_board()
    name_lower = name.lower()
    for p in board:
        if name_lower in p["name"].lower():
            return p
    return None


def available_players(drafted_ids: set[str]) -> list[dict]:
    return [p for p in get_board() if p["id"] not in drafted_ids]


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
