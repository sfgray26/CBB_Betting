"""
Inspect the FanGraphs Steamer (RoS) JSON structure and map fields.
"""
import json
import requests
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fangraphs.com/projections",
    "Origin": "https://www.fangraphs.com",
    "X-Requested-With": "XMLHttpRequest",
}

# Fetch batters
url = "https://www.fangraphs.com/api/projections"
params = {
    "type": "steamerr",
    "stats": "bat",
    "pos": "all",
    "team": "0",
    "players": "0",
    "lg": "all",
    "z": "1778047498",
}
resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
data = resp.json()
df_bat = pd.DataFrame(data)

# Fetch pitchers
params["stats"] = "pit"
resp2 = requests.get(url, params=params, headers=HEADERS, timeout=30)
data2 = resp2.json()
df_pit = pd.DataFrame(data2)

print("=" * 60)
print("BATTERS")
print("=" * 60)
print(f"Rows: {len(df_bat)}")
print(f"Columns ({len(df_bat.columns)}):")
for c in df_bat.columns:
    print(f"  {c}")
print("\nFirst 3 rows (key fields):")
print(df_bat[["PlayerName", "Team", "G", "AB", "PA", "HR", "R", "RBI", "SB", "AVG", "OBP", "SLG", "wOBA", "wRC+"]].head(3).to_string(index=False))

print("\n" + "=" * 60)
print("PITCHERS")
print("=" * 60)
print(f"Rows: {len(df_pit)}")
print(f"Columns ({len(df_pit.columns)}):")
for c in df_pit.columns:
    print(f"  {c}")
print("\nFirst 3 rows (key fields):")
pit_cols = [c for c in ["PlayerName", "Team", "G", "GS", "IP", "K", "BB", "ERA", "WHIP", "FIP", "xFIP", "K/9", "BB/9"] if c in df_pit.columns]
print(df_pit[pit_cols].head(3).to_string(index=False))

# Save sample for reference
df_bat.head(100).to_json("fangraphs_ros_batters_sample.json", orient="records", indent=2)
df_pit.head(100).to_json("fangraphs_ros_pitchers_sample.json", orient="records", indent=2)
print("\nSaved sample files: fangraphs_ros_batters_sample.json, fangraphs_ros_pitchers_sample.json")
