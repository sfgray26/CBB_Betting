"""Quick check: statsapi playerInfo fullName lookup."""
import statsapi
box = statsapi.boxscore_data(824457)
pi = box['playerInfo']
# Check first away batter
bat = box['awayBatters'][1]  # skip header row (index 0 is totals)
pid = bat['personId']
info = pi.get(f'ID{pid}', {})
print(f"personId={pid}, fullName={info.get('fullName')}, boxscoreName={info.get('boxscoreName')}")

# Check Salvador Perez (BDL#150)
print("\n--- All player names ---")
for key, val in pi.items():
    print(f"  {val.get('id')}: {val.get('fullName')}")
