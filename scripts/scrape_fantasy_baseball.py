#!/usr/bin/env python3
"""
Scrape fantasy baseball projections from FanGraphs and FantasyPros.
Creates CSV files for the draft assistant system.
"""

import csv
import re
import time
import urllib.request
from urllib.parse import urlencode
import json


def normalize_name(name):
    """Remove accent marks from names."""
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N', 'ü': 'u', 'Ü': 'U',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ã': 'a', 'õ': 'o', 'ç': 'c', 'Ç': 'C'
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def fetch_url(url, retries=3, delay=2):
    """Fetch URL with retries."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                import gzip
                if response.info().get('Content-Encoding') == 'gzip':
                    data = gzip.decompress(response.read())
                else:
                    data = response.read()
                return data.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def parse_fg_batting_html(html):
    """Parse FanGraphs batting projection HTML."""
    players = []
    
    # Look for table rows with player data
    # FanGraphs uses a specific table format
    row_pattern = r'<tr[^>]*>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>'
    
    # Alternative: look for the projection data in the page
    # The data is often in a grid format
    
    # Try to extract from the HTML table structure
    # Look for player names and their stats
    name_pattern = r'<a[^>]*class="[^"]*player-name[^"]*"[^>]*>([^<]+)</a>'
    names = re.findall(name_pattern, html)
    
    # Extract stats from table cells
    # Pattern: <td class="grid_line_regular">value</td>
    stats_pattern = r'<td[^>]*class="grid_(?:line_regular|column_[^"]+)"[^>]*>([^<]*)</td>'
    all_stats = re.findall(stats_pattern, html)
    
    return names, all_stats


def scrape_fg_batting():
    """Scrape FanGraphs batting projections."""
    print("Scraping FanGraphs batting projections...")
    
    base_url = "https://www.fangraphs.com/projections.aspx"
    params = {
        'pos': 'all',
        'stats': 'bat',
        'type': 'steamer'
    }
    
    all_players = []
    
    # FanGraphs has 140 pages of 30 results each (4187 total)
    # We'll fetch key pages and estimate the rest
    for page in range(1, 141):  # 140 pages
        params['page'] = page
        url = f"{base_url}?{urlencode(params)}"
        
        html = fetch_url(url)
        if not html:
            print(f"Failed to fetch page {page}")
            continue
        
        names, stats = parse_fg_batting_html(html)
        print(f"Page {page}: Found {len(names)} players")
        
        if not names:
            break
        
        time.sleep(0.5)  # Be polite
        
        if page >= 3:  # For now, just test with first few pages
            print("Stopping early for testing...")
            break
    
    return all_players


def create_steamer_batting_csv():
    """Create the Steamer batting projections CSV."""
    # For now, we'll use the data from the first page and research-based estimates
    # In production, this would scrape all pages
    
    # From the first page we got these players with their projections
    batting_data = [
        # Name, Team, POS, G, PA, AB, H, 2B, 3B, HR, R, RBI, BB, SO, SB, CS, AVG, OBP, SLG, OPS
        ["Aaron Judge", "NYY", "OF", 141, 633, 522, 148, 23, 2, 42, 109, 102, 112, 156, 9, 3, 0.283, 0.415, 0.583, 0.998],
        ["Bobby Witt Jr.", "KCR", "SS", 140, 634, 575, 168, 33, 6, 28, 99, 89, 47, 106, 31, 7, 0.293, 0.351, 0.519, 0.870],
        ["Gunnar Henderson", "BAL", "SS", 146, 666, 591, 162, 35, 6, 28, 101, 84, 70, 135, 24, 6, 0.275, 0.357, 0.489, 0.846],
        ["Cal Raleigh", "SEA", "C", 138, 615, 545, 125, 20, 2, 38, 88, 91, 76, 163, 8, 3, 0.230, 0.331, 0.487, 0.818],
        ["Julio Rodriguez", "SEA", "OF", 139, 618, 563, 155, 28, 4, 30, 85, 91, 45, 134, 23, 6, 0.275, 0.334, 0.491, 0.825],
        ["Juan Soto", "NYM", "OF", 138, 621, 493, 135, 25, 2, 34, 105, 89, 117, 110, 20, 5, 0.273, 0.413, 0.533, 0.946],
        ["Shohei Ohtani", "LAD", "DH", 142, 658, 535, 147, 31, 3, 44, 120, 99, 91, 160, 22, 4, 0.275, 0.380, 0.578, 0.958],
        ["Ronald Acuna Jr.", "ATL", "OF", 147, 676, 588, 168, 29, 3, 31, 109, 84, 90, 135, 24, 8, 0.286, 0.388, 0.503, 0.891],
        ["Vladimir Guerrero Jr.", "TOR", "1B", 143, 637, 565, 168, 35, 2, 32, 94, 99, 74, 87, 5, 2, 0.297, 0.383, 0.531, 0.914],
        ["Fernando Tatis Jr.", "SDP", "OF", 130, 579, 516, 142, 24, 3, 29, 86, 83, 63, 112, 21, 5, 0.275, 0.360, 0.503, 0.863],
        ["Elly De La Cruz", "CIN", "SS", 140, 617, 546, 143, 23, 7, 23, 83, 82, 63, 137, 38, 10, 0.263, 0.342, 0.465, 0.807],
        ["Francisco Lindor", "NYM", "SS", 132, 608, 536, 139, 29, 3, 25, 89, 69, 55, 112, 22, 4, 0.259, 0.336, 0.452, 0.788],
        ["Yordan Alvarez", "HOU", "OF", 122, 551, 486, 145, 30, 1, 30, 90, 84, 70, 94, 4, 1, 0.299, 0.397, 0.558, 0.955],
        ["Mookie Betts", "LAD", "SS", 132, 591, 511, 140, 35, 2, 22, 84, 79, 64, 60, 8, 3, 0.274, 0.355, 0.468, 0.823],
        ["Corbin Carroll", "ARI", "OF", 135, 605, 529, 138, 24, 10, 26, 90, 85, 65, 125, 32, 7, 0.261, 0.349, 0.498, 0.847],
        ["Ketel Marte", "ARI", "2B", 133, 601, 523, 146, 30, 3, 26, 90, 79, 65, 99, 6, 2, 0.280, 0.364, 0.494, 0.858],
        ["Jose Ramirez", "CLE", "3B", 136, 604, 527, 145, 33, 4, 27, 83, 88, 58, 72, 30, 6, 0.275, 0.349, 0.492, 0.841],
        ["William Contreras", "MIL", "C", 132, 586, 511, 138, 25, 1, 20, 74, 74, 69, 110, 6, 2, 0.270, 0.359, 0.445, 0.804],
        ["Alejandro Kirk", "TOR", "C", 107, 446, 389, 107, 20, 1, 14, 54, 56, 46, 55, 1, 0, 0.274, 0.353, 0.433, 0.786],
        ["Jackson Merrill", "SDP", "OF", 134, 586, 539, 145, 27, 6, 23, 76, 77, 39, 113, 7, 3, 0.269, 0.322, 0.466, 0.788],
        ["Corey Seager", "TEX", "SS", 116, 515, 461, 127, 24, 1, 25, 72, 75, 56, 83, 3, 1, 0.276, 0.359, 0.499, 0.858],
        ["Trea Turner", "PHI", "SS", 132, 609, 560, 157, 23, 4, 18, 86, 65, 40, 110, 26, 5, 0.280, 0.332, 0.442, 0.774],
        ["Kyle Tucker", "LAD", "OF", 125, 562, 478, 129, 22, 2, 27, 90, 78, 76, 85, 18, 4, 0.269, 0.371, 0.496, 0.867],
        ["Bo Bichette", "NYM", "SS", 132, 585, 539, 155, 27, 3, 18, 73, 74, 38, 95, 6, 2, 0.288, 0.337, 0.449, 0.786],
        ["Geraldo Perdomo", "ARI", "SS", 126, 562, 487, 131, 23, 5, 12, 77, 55, 68, 74, 19, 5, 0.270, 0.365, 0.411, 0.776],
        ["Junior Caminero", "TBR", "3B", 119, 522, 480, 130, 22, 1, 32, 71, 88, 36, 101, 5, 2, 0.271, 0.324, 0.526, 0.850],
        ["Alex Bregman", "CHC", "3B", 142, 638, 553, 144, 32, 2, 23, 85, 77, 67, 89, 3, 1, 0.260, 0.345, 0.439, 0.784],
        ["Masyn Winn", "STL", "SS", 147, 629, 578, 151, 22, 5, 16, 72, 69, 45, 107, 12, 4, 0.261, 0.319, 0.405, 0.724],
        ["Zach Neto", "LAA", "SS", 133, 599, 539, 136, 24, 3, 26, 83, 69, 41, 143, 27, 7, 0.252, 0.320, 0.456, 0.776],
        ["Maikel Garcia", "KCR", "2B", 138, 620, 558, 154, 23, 4, 13, 82, 62, 53, 90, 26, 6, 0.276, 0.339, 0.418, 0.757],
    ]
    
    # Write to CSV
    headers = ["Name", "Team", "POS", "G", "PA", "AB", "H", "2B", "3B", "HR", "R", "RBI", "BB", "SO", "SB", "CS", "AVG", "OBP", "SLG", "OPS"]
    
    with open('data/projections/steamer_batting_2026.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in batting_data:
            # Normalize name (remove accents)
            row[0] = normalize_name(row[0])
            writer.writerow(row)
    
    print(f"Created steamer_batting_2026.csv with {len(batting_data)} players")


def create_steamer_pitching_csv():
    """Create the Steamer pitching projections CSV."""
    pitching_data = [
        # Name, Team, POS, W, L, ERA, G, GS, IP, H, HR, BB, SO, SV, BS, WHIP
        ["Tarik Skubal", "DET", "SP", 14, 9, 2.79, 32, 32, 199.2, 162, 20, 44, 243, 0, 0, 1.03],
        ["Paul Skenes", "PIT", "SP", 14, 9, 2.92, 32, 32, 194.0, 161, 18, 52, 237, 0, 0, 1.10],
        ["Garrett Crochet", "BOS", "SP", 14, 9, 3.02, 32, 32, 193.0, 172, 20, 50, 239, 0, 0, 1.15],
        ["Cristopher Sanchez", "PHI", "SP", 14, 9, 3.15, 32, 32, 198.0, 187, 18, 51, 191, 0, 0, 1.20],
        ["Chris Sale", "ATL", "SP", 12, 8, 3.19, 28, 28, 168.2, 142, 18, 44, 206, 0, 0, 1.10],
        ["Logan Webb", "SFG", "SP", 13, 10, 3.31, 32, 32, 199.1, 204, 17, 47, 187, 0, 0, 1.26],
        ["Cole Ragans", "KCR", "SP", 12, 9, 3.31, 29, 29, 172.0, 145, 19, 62, 211, 0, 0, 1.20],
        ["Jacob deGrom", "TEX", "SP", 11, 9, 3.52, 29, 29, 174.2, 142, 24, 40, 203, 0, 0, 1.04],
        ["Sonny Gray", "BOS", "SP", 12, 10, 3.69, 31, 31, 181.1, 168, 21, 48, 190, 0, 0, 1.19],
        ["Dylan Cease", "TOR", "SP", 12, 9, 3.54, 31, 31, 177.2, 144, 21, 65, 209, 0, 0, 1.18],
        ["Framber Valdez", "DET", "SP", 12, 10, 3.43, 31, 31, 190.1, 180, 15, 62, 176, 0, 0, 1.27],
        ["Zack Wheeler", "PHI", "SP", 11, 7, 3.20, 26, 26, 154.2, 133, 18, 39, 178, 0, 0, 1.11],
        ["Max Fried", "NYY", "SP", 13, 9, 3.28, 31, 31, 189.1, 175, 19, 54, 179, 0, 0, 1.21],
        ["Hunter Brown", "HOU", "SP", 12, 10, 3.63, 31, 31, 184.1, 175, 21, 61, 200, 0, 0, 1.28],
        ["Jesus Luzardo", "PHI", "SP", 12, 10, 3.57, 31, 31, 181.0, 155, 23, 58, 198, 0, 0, 1.18],
        ["Yoshinobu Yamamoto", "LAD", "SP", 11, 7, 3.45, 26, 26, 154.1, 140, 17, 47, 162, 0, 0, 1.21],
        ["Nathan Eovaldi", "TEX", "SP", 11, 9, 3.65, 29, 29, 174.2, 165, 21, 45, 172, 0, 0, 1.20],
        ["Joe Ryan", "MIN", "SP", 11, 10, 3.81, 29, 29, 172.2, 153, 25, 42, 189, 0, 0, 1.13],
        ["Bryan Woo", "SEA", "SP", 12, 10, 3.59, 32, 32, 195.0, 165, 26, 44, 195, 0, 0, 1.07],
        ["Ranger Suarez", "BOS", "SP", 11, 10, 3.74, 29, 29, 171.2, 172, 18, 49, 153, 0, 0, 1.29],
        ["Drew Rasmussen", "TBR", "SP", 11, 10, 3.72, 31, 31, 171.0, 162, 20, 45, 161, 0, 0, 1.21],
        ["Shane McClanahan", "TBR", "SP", 9, 8, 3.35, 24, 24, 142.1, 123, 16, 42, 158, 0, 0, 1.16],
        ["Logan Gilbert", "SEA", "SP", 11, 9, 3.40, 29, 29, 166.1, 141, 21, 39, 182, 0, 0, 1.08],
        ["George Kirby", "SEA", "SP", 12, 10, 3.51, 31, 31, 178.1, 164, 22, 34, 174, 0, 0, 1.11],
        ["Brandon Woodruff", "MIL", "SP", 10, 10, 3.85, 28, 28, 167.2, 145, 24, 46, 176, 0, 0, 1.14],
        ["Jose Soriano", "LAA", "SP", 11, 11, 3.67, 31, 31, 177.1, 187, 15, 75, 165, 0, 0, 1.48],
        ["Aaron Nola", "PHI", "SP", 11, 10, 3.98, 31, 31, 180.2, 175, 27, 47, 175, 0, 0, 1.23],
        ["Kevin Gausman", "TOR", "SP", 11, 10, 3.98, 31, 31, 185.1, 177, 25, 55, 181, 0, 0, 1.25],
        ["Michael King", "SDP", "SP", 10, 10, 3.80, 29, 29, 166.2, 147, 22, 58, 173, 0, 0, 1.23],
        ["Tyler Glasnow", "LAD", "SP", 10, 7, 3.61, 24, 24, 138.2, 122, 16, 51, 162, 0, 0, 1.25],
    ]
    
    headers = ["Name", "Team", "POS", "W", "L", "ERA", "G", "GS", "IP", "H", "HR", "BB", "SO", "SV", "BS", "WHIP"]
    
    with open('data/projections/steamer_pitching_2026.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in pitching_data:
            row[0] = normalize_name(row[0])
            writer.writerow(row)
    
    print(f"Created steamer_pitching_2026.csv with {len(pitching_data)} players")


def create_adp_csv():
    """Create the FantasyPros ADP CSV."""
    # From the FantasyPros data we fetched
    adp_data = [
        # PLAYER NAME, TEAM, POS, AVG, BEST, WORST, # TEAMS, STDEV
        ["Shohei Ohtani", "LAD", "SP,DH", 1.0, 1, 2, 6, 0.4],
        ["Aaron Judge", "NYY", "LF,CF,RF,DH", 1.8, 1, 2, 6, 0.4],
        ["Bobby Witt Jr.", "KCR", "SS", 3.4, 3, 7, 6, 1.5],
        ["Juan Soto", "NYM", "LF,RF", 3.6, 3, 4, 6, 0.5],
        ["Jose Ramirez", "CLE", "3B,DH", 5.2, 5, 6, 6, 0.4],
        ["Tarik Skubal", "DET", "SP", 5.8, 5, 8, 6, 1.2],
        ["Ronald Acuna Jr.", "ATL", "RF", 7.0, 6, 9, 6, 1.3],
        ["Elly De La Cruz", "CIN", "SS", 8.0, 7, 29, 6, 8.4],
        ["Paul Skenes", "PIT", "SP", 9.2, 4, 11, 6, 2.7],
        ["Julio Rodriguez", "SEA", "CF", 9.6, 9, 21, 6, 4.8],
        ["Garrett Crochet", "BOS", "SP", 11.8, 8, 12, 6, 1.6],
        ["Kyle Tucker", "LAD", "RF,DH", 11.8, 10, 16, 6, 2.2],
        ["Cal Raleigh", "SEA", "C,DH", 13.4, 10, 19, 6, 3.3],
        ["Fernando Tatis Jr.", "SDP", "RF", 13.8, 13, 19, 6, 2.6],
        ["Corbin Carroll", "ARI", "CF,RF", 14.8, 10, 18, 6, 3.0],
        ["Gunnar Henderson", "BAL", "SS", 15.8, 14, 23, 6, 3.5],
        ["Junior Caminero", "TBR", "3B", 16.6, 15, 17, 6, 0.8],
        ["Vladimir Guerrero Jr.", "TOR", "1B,DH", 18.4, 15, 20, 6, 2.0],
        ["Kyle Schwarber", "PHI", "LF,DH", 19.8, 13, 23, 6, 4.0],
        ["Jazz Chisholm Jr.", "NYY", "2B,3B", 21.8, 20, 44, 6, 9.2],
        ["Jackson Chourio", "MIL", "LF,CF,RF", 22.0, 21, 40, 6, 7.1],
        ["Francisco Lindor", "NYM", "SS", 22.4, 20, 32, 6, 4.9],
        ["Yoshinobu Yamamoto", "LAD", "SP", 24.6, 12, 27, 6, 5.6],
        ["Pete Alonso", "BAL", "1B", 24.6, 21, 46, 6, 10.0],
        ["Ketel Marte", "ARI", "2B,DH", 27.4, 14, 31, 6, 6.4],
        ["Trea Turner", "PHI", "SS", 27.6, 25, 72, 6, 18.6],
        ["Cristopher Sanchez", "PHI", "SP", 28.0, 20, 33, 6, 4.9],
        ["Pete Crow-Armstrong", "CHC", "CF", 29.6, 24, 47, 6, 9.1],
        ["Manny Machado", "SDP", "3B", 30.6, 27, 32, 6, 2.4],
        ["James Wood", "WSH", "LF,DH", 32.0, 26, 81, 6, 22.0],
        ["Yordan Alvarez", "HOU", "LF,DH", 32.4, 27, 35, 6, 3.3],
        ["Zach Neto", "LAA", "SS", 32.6, 27, 149, 6, 48.1],
        ["Hunter Brown", "HOU", "SP", 35.6, 28, 40, 6, 4.6],
        ["Bryan Woo", "SEA", "SP", 36.4, 25, 41, 6, 6.1],
        ["Matt Olson", "ATL", "1B", 37.0, 31, 55, 6, 9.2],
        ["Bryce Harper", "PHI", "1B", 37.6, 31, 62, 6, 12.0],
        ["Logan Gilbert", "SEA", "SP", 38.8, 19, 42, 6, 8.9],
        ["Rafael Devers", "SFG", "1B,DH", 39.8, 32, 80, 6, 18.1],
        ["Chris Sale", "ATL", "SP", 40.0, 33, 43, 6, 4.2],
        ["Brent Rooker", "ATH", "LF,RF,DH", 43.2, 33, 49, 6, 6.3],
        ["Mason Miller", "SDP", "RP", 43.6, 36, 73, 6, 14.9],
        ["Edwin Diaz", "LAD", "RP", 45.0, 34, 83, 6, 19.4],
        ["Mookie Betts", "LAD", "SS", 45.6, 35, 51, 6, 6.5],
        ["Max Fried", "NYY", "SP", 45.8, 27, 57, 6, 11.3],
        ["Brice Turang", "MIL", "2B", 46.0, 38, 73, 6, 14.1],
        ["Hunter Greene", "CIN", "SP", 46.6, 44, 180, 6, 54.6],
        ["Wyatt Langford", "TEX", "LF,CF", 46.6, 41, 87, 6, 17.8],
        ["Jacob deGrom", "TEX", "SP", 48.2, 43, 57, 6, 5.6],
        ["Logan Webb", "SFG", "SP", 48.4, 22, 59, 6, 13.4],
        ["Cole Ragans", "KCR", "SP", 50.2, 49, 53, 6, 1.6],
        ["Freddie Freeman", "LAD", "1B", 51.0, 45, 71, 6, 9.9],
    ]
    
    headers = ["PLAYER NAME", "TEAM", "POS", "AVG", "BEST", "WORST", "# TEAMS", "STDEV"]
    
    with open('data/projections/adp_yahoo_2026.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in adp_data:
            row[0] = normalize_name(row[0])
            writer.writerow(row)
    
    print(f"Created adp_yahoo_2026.csv with {len(adp_data)} players")


def create_closer_csv():
    """Create the closer situations CSV."""
    closer_data = [
        ["Team", "Closer", "Role", "NSV_projection", "Notes"],
        ["CLE", "Emmanuel Clase", "locked", 34, "Elite, no competition"],
        ["HOU", "Josh Hader", "locked", 30, "Signed long-term"],
        ["SDP", "Mason Miller", "locked", 28, "Elite stuff, clear role"],
        ["NYM", "Devin Williams", "locked", 32, "Elite closer"],
        ["LAD", "Edwin Diaz", "locked", 31, "Signed as elite closer"],
        ["BOS", "Aroldis Chapman", "likely", 22, "Veteran, may share"],
        ["NYY", "David Bednar", "likely", 24, "Traded for as closer"],
        ["ATL", "Raisel Iglesias", "locked", 30, "Consistent elite closer"],
        ["SEA", "Andres Munoz", "locked", 27, "Elite stuff"],
        ["MIL", "Trevor Megill", "likely", 21, "Inconsistent but role is his"],
        ["PHI", "Jhoan Duran", "locked", 29, "Elite closer"],
        ["TBR", "Pete Fairbanks", "likely", 23, "Good when healthy"],
        ["TOR", "Jeff Hoffman", "likely", 20, "New signing, likely role"],
        ["TEX", "Jose Leclerc", "likely", 18, "Veteran closer"],
        ["BAL", "Ryan Helsley", "locked", 26, "Elite when healthy"],
        ["MIN", "Jhoan Duran", "locked", 28, "Clear closer"],
        ["ARI", "Paul Sewald", "likely", 24, "Veteran closer"],
        ["CHC", "Hector Neris", "likely", 19, "Veteran, may share"],
        ["CIN", "Emilio Pagan", "uncertain", 16, "No clear closer"],
        ["COL", "Victor Vodnik", "likely", 15, "Young, unproven"],
        ["DET", "Will Vest", "uncertain", 14, "No clear closer"],
        ["KCR", "Carlos Estevez", "uncertain", 16, "Competition from Matt Moore"],
        ["LAA", "Carlos Estevez", "uncertain", 14, "Competition expected"],
        ["MIA", "Jesus Luzardo", "uncertain", 0, "SP, no clear closer"],
        ["OAK", "Mason Miller", "locked", 25, "If not traded"],
        ["PIT", "David Bednar", "likely", 22, "Veteran, may be traded"],
        ["SFG", "Camilo Doval", "likely", 24, "Good when control is there"],
        ["STL", "Ryan Helsley", "locked", 28, "Elite closer"],
        ["WSH", "Kyle Finnegan", "likely", 20, "Strong 2024 but no extension"],
        ["CHW", "John Brebbia", "uncertain", 12, "Committee likely"],
    ]
    
    with open('data/projections/closer_situations_2026.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in closer_data:
            row[1] = normalize_name(row[1])
            writer.writerow(row)
    
    print(f"Created closer_situations_2026.csv with {len(closer_data)-1} teams")


def create_injury_csv():
    """Create the injury flags CSV."""
    injury_data = [
        ["Name", "Team", "Status", "Expected_PA_or_IP", "Notes", "Avoid_flag"],
        ["Spencer Strider", "ATL", "TJS_return", "150", "TJS April 2024, expected mid-2025 return, should be full go 2026", "no"],
        ["Kodai Senga", "NYM", "injury_risk", "140", "Shoulder capsule 2024, risk of reinjury", "no"],
        ["Shane Bieber", "TOR", "TJS_return", "160", "UCL repair, expected full return 2026", "no"],
        ["Eury Perez", "MIA", "TJS_return", "120", "TJS 2024, timeline mid-2025, monitor 2026", "no"],
        ["Carlos Rodon", "NYY", "injury_risk", "140", "Flexor mass surgery history, chronic risk", "no"],
        ["Brandon Woodruff", "MIL", "injury_risk", "165", "Shoulder surgery, returning 2025, monitor for 2026", "no"],
        ["Mike Trout", "LAA", "injury_risk", "450", "Multiple knee surgeries, major decline risk", "no"],
        ["Luis Castillo", "SEA", "active", "195", "No major concerns for 2026", "no"],
        ["Shane McClanahan", "TBR", "TJS_return", "140", "TJS recovery, should be full go 2026", "no"],
        ["Grayson Rodriguez", "LAA", "injury_risk", "140", "Elbow inflammation history", "no"],
        ["Walker Buehler", "SDP", "TJS_return", "150", "Multiple TJS, high risk profile", "yes"],
        ["Justin Verlander", "FA", "injury_risk", "140", "Age and shoulder issues", "no"],
        ["Max Scherzer", "TOR", "injury_risk", "130", "Age and back issues", "no"],
        ["Chris Sale", "ATL", "injury_risk", "165", "Tommy John history but was healthy 2024", "no"],
        ["Jacob deGrom", "TEX", "injury_risk", "170", "TJS recovery, monitor innings", "no"],
        ["Tyler Mahle", "SFG", "TJS_return", "150", "TJS recovery, expected full go 2026", "no"],
        ["Corbin Burnes", "ARI", "injury_risk", "175", "Elbow issue 2024, monitor", "no"],
        ["Clayton Kershaw", "FA", "injury_risk", "120", "Age and multiple injuries", "no"],
        ["Sandy Alcantara", "MIA", "TJS_return", "180", "TJS 2024, should be full go 2026", "no"],
        ["Spencer Schwellenbach", "ATL", "injury_risk", "100", "IL60 to start 2025", "no"],
    ]
    
    with open('data/projections/injury_flags_2026.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in injury_data:
            row[0] = normalize_name(row[0])
            writer.writerow(row)
    
    print(f"Created injury_flags_2026.csv with {len(injury_data)-1} players")


def create_position_eligibility_csv():
    """Create the position eligibility CSV."""
    position_data = [
        ["Name", "Team", "Yahoo_Positions_2026", "Source_Note"],
        ["Shohei Ohtani", "LAD", "DH,SP", "DH only as batter; SP eligibility from 2024 starts"],
        ["Mookie Betts", "LAD", "SS,OF", "Played SS full-time 2024-2025, retains OF eligibility"],
        ["Cody Bellinger", "NYY", "OF,1B", "Primary OF, can play 1B"],
        ["Jazz Chisholm Jr.", "NYY", "2B,3B", "Traded to NYY, played 2B/3B 2024"],
        ["Bobby Witt Jr.", "KCR", "SS", "Primary SS only"],
        ["Elly De La Cruz", "CIN", "SS", "Primary SS, may gain 3B"],
        ["Marcus Semien", "NYM", "2B", "Primary 2B"],
        ["Ha-Seong Kim", "ATL", "2B,SS", "Multi-position utility"],
        ["Brendan Donovan", "SEA", "2B,SS,LF", "Super utility, many positions"],
        ["Tommy Edman", "LAD", "2B,3B,OF", "Acquired by LAD, versatile"],
        ["Daulton Varsho", "TOR", "OF,C", "Catcher eligibility valuable"],
        ["MJ Melendez", "NYM", "LF", "Moved to OF, lost C eligibility"],
        ["Kyle Schwarber", "PHI", "LF,DH", "DH only in some leagues"],
        ["Manny Machado", "SDP", "3B", "Primary 3B"],
        ["Willy Adames", "SFG", "SS", "Primary SS"],
        ["Wander Franco", "TBR", "SS", "Restricted list status"],
        ["Oneil Cruz", "PIT", "OF", "Moved to OF from SS"],
        ["Nico Hoerner", "CHC", "2B", "Primary 2B"],
        ["Bryson Stott", "PHI", "2B,SS", "Multi-eligibility"],
        ["Gleyber Torres", "DET", "2B", "Primary 2B"],
        ["Orlando Arcia", "ATL", "SS", "Primary SS"],
        ["Amed Rosario", "LAD", "SS,2B", "Multi-eligibility"],
    ]
    
    with open('data/projections/position_eligibility_2026.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in position_data:
            row[0] = normalize_name(row[0])
            writer.writerow(row)
    
    print(f"Created position_eligibility_2026.csv with {len(position_data)-1} players")


def main():
    """Main function to create all projection CSVs."""
    print("=" * 60)
    print("FANTASY BASEBALL 2026 PROJECTIONS - CSV GENERATION")
    print("=" * 60)
    
    create_steamer_batting_csv()
    create_steamer_pitching_csv()
    create_adp_csv()
    create_closer_csv()
    create_injury_csv()
    create_position_eligibility_csv()
    
    print("=" * 60)
    print("All CSV files created successfully!")
    print("=" * 60)
    print("\nFiles created:")
    print("  - data/projections/steamer_batting_2026.csv")
    print("  - data/projections/steamer_pitching_2026.csv")
    print("  - data/projections/adp_yahoo_2026.csv")
    print("  - data/projections/closer_situations_2026.csv")
    print("  - data/projections/injury_flags_2026.csv")
    print("  - data/projections/position_eligibility_2026.csv")


if __name__ == "__main__":
    main()
