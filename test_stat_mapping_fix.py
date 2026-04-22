"""
Test the fixed Yahoo stat_id to canonical mapping.
Verifies against real Yahoo API data from Week 3.
"""
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
import json

def test_mapping():
    client = YahooFantasyClient()
    result = client.get_matchup_stats(week=3)
    
    print("=" * 80)
    print("FIXED STAT MAPPING TEST (Week 3)")
    print("=" * 80)
    
    my_stats = result["my_stats"]
    opp_stats = result["opp_stats"]
    opponent_name = result["opponent_name"]
    
    print(f"\nMy Team Stats:")
    print(json.dumps(my_stats, indent=2, sort_keys=True))
    
    print(f"\nOpponent ({opponent_name}) Stats:")
    print(json.dumps(opp_stats, indent=2, sort_keys=True))
    
    # Verify expected stats are present
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    expected_batting = ["H_AB", "R", "H", "HR_B", "RBI", "K_B", "TB", "AVG", "OPS", "NSB"]
    expected_pitching = ["IP", "W", "L", "HR_P", "K_P", "ERA", "WHIP", "K_9", "QS", "NSV"]
    expected_all = expected_batting + expected_pitching
    
    missing = []
    for stat in expected_all:
        if stat not in my_stats and stat not in opp_stats:
            missing.append(stat)
    
    if missing:
        print(f"\n❌ MISSING STATS: {missing}")
    else:
        print(f"\n✅ ALL EXPECTED STATS PRESENT")
    
    # Verify H_AB is a string
    if "H_AB" in my_stats:
        if isinstance(my_stats["H_AB"], str) and "/" in my_stats["H_AB"]:
            print(f"✅ H_AB correctly formatted as string: {my_stats['H_AB']}")
        else:
            print(f"❌ H_AB wrong format: {my_stats['H_AB']} (type: {type(my_stats['H_AB'])})")
    
    # Verify expected ordering in contract
    from backend.stat_contract import MATCHUP_DISPLAY_ORDER
    print(f"\n✅ MATCHUP_DISPLAY_ORDER:")
    print(f"   Batting: {MATCHUP_DISPLAY_ORDER[:10]}")
    print(f"   Pitching: {MATCHUP_DISPLAY_ORDER[10:]}")
    
    expected_order = ["H_AB", "R", "H", "HR_B", "RBI", "K_B", "TB", "AVG", "OPS", "NSB",
                      "IP", "W", "L", "HR_P", "K_P", "ERA", "WHIP", "K_9", "QS", "NSV"]
    
    if MATCHUP_DISPLAY_ORDER == expected_order:
        print(f"✅ Display order matches Yahoo UI")
    else:
        print(f"❌ Display order mismatch:")
        print(f"   Expected: {expected_order}")
        print(f"   Actual: {MATCHUP_DISPLAY_ORDER}")

if __name__ == "__main__":
    test_mapping()
