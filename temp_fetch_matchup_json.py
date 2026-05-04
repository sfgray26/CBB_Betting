"""
Fetch raw matchup stats from the /api/fantasy/scoreboard endpoint
and display the complete JSON response for manual stat mapping verification.
"""
import json
import sys
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

def main():
    client = YahooFantasyClient()
    
    # Get raw scoreboard data (week 3 to match production)
    print("=" * 80)
    print("RAW YAHOO SCOREBOARD DATA (Week 3)")
    print("=" * 80)
    raw_scoreboard = client.get_scoreboard(week=3)
    print(json.dumps(raw_scoreboard, indent=2))
    
    print("\n" + "=" * 80)
    print("PARSED MATCHUP STATS (via get_matchup_stats)")
    print("=" * 80)
    matchup_stats = client.get_matchup_stats(week=3)
    print(json.dumps(matchup_stats, indent=2))
    
    print("\n" + "=" * 80)
    print("STAT ID MAPPING TABLE")
    print("=" * 80)
    
    # Extract stats from first matchup
    if raw_scoreboard and len(raw_scoreboard) > 0:
        matchup = raw_scoreboard[0]
        teams = matchup.get("0", {}).get("teams", {})
        
        for team_idx in ["0", "1"]:
            team = teams.get(team_idx, {})
            team_info = team.get("team", [[]])[0]
            team_name = None
            for item in team_info:
                if isinstance(item, dict) and "name" in item:
                    team_name = item["name"]
                    break
            
            team_stats = team.get("team", [{}])[1].get("team_stats", {}).get("stats", [])
            
            print(f"\n{team_name or f'Team {team_idx}'} Stats:")
            print(f"{'Stat ID':<10} {'Value':<15} {'Current Mapping':<20} {'Correct Mapping'}")
            print("-" * 80)
            
            # Current mapping from code
            yahoo_to_canonical = {
                "7": "R", "8": "H", "12": "HR_B", "13": "RBI", "16": "TB", "60": "NSB",
                "10": "K_B", "25": "AVG", "26": "OPS",
                "23": "W", "24": "L", "28": "K_P", "29": "QS", "57": "K_9", "83": "NSV",
                "35": "HR_P", "44": "ERA", "45": "WHIP",
            }
            
            for stat_entry in team_stats:
                if isinstance(stat_entry, dict):
                    stat = stat_entry.get("stat", {})
                    stat_id = stat.get("stat_id", "")
                    value = stat.get("value", "")
                    current_map = yahoo_to_canonical.get(stat_id, "UNMAPPED")
                    print(f"{stat_id:<10} {value:<15} {current_map:<20} ???")

if __name__ == "__main__":
    main()
