"""
Check raw BDL API response for pitchers to see if p_hits/p_bb are provided
"""
import sys
sys.path.insert(0, '.')

with open('pitcher_raw_check.txt', 'w') as f:
    try:
        f.write("Checking raw BDL API response for pitchers...\n")
        import requests
        import os

        # Make raw API call to get stats
        api_key = os.environ.get('BALLDONTLIE_API_KEY')
        headers = {'Authorization': api_key}
        params = {'per_page': 100}

        response = requests.get(
            'https://api.balldontlie.io/mlb/v1/stats',
            headers=headers,
            params=params
        )

        data = response.json()
        f.write(f"Status: {response.status_code}\n")
        f.write(f"Total stats: {len(data.get('data', []))}\n\n")

        # Find pitchers (stats with ip but not ab)
        pitchers = [s for s in data.get('data', []) if s.get('ip') is not None and s.get('at_bats') is None]
        f.write(f"Found {len(pitchers)} pitchers\n\n")

        # Show first 5 pitchers with detailed fields
        for i, pitcher in enumerate(pitchers[:5]):
            f.write(f"Pitcher {i+1}:\n")
            f.write(f"  ip: {pitcher.get('ip')}\n")
            f.write(f"  era: {pitcher.get('era')}\n")
            f.write(f"  whip: {pitcher.get('whip')} (API field)\n")
            f.write(f"  p_hits: {pitcher.get('p_hits')}\n")
            f.write(f"  p_bb: {pitcher.get('p_bb')}\n")
            f.write(f"  er: {pitcher.get('er')}\n")
            f.write(f"  k: {pitcher.get('k')}\n")
            f.write(f"  player: {pitcher.get('player', {}).get('full_name') if pitcher.get('player') else 'Unknown'}\n")
            f.write("\n")

        # Check field availability across all pitchers
        f.write("Field availability across all pitchers:\n")
        f.write(f"  pitchers with whip field: {sum(1 for p in pitchers if 'whip' in p)}\n")
        f.write(f"  pitchers with p_hits: {sum(1 for p in pitchers if p.get('p_hits') is not None)}\n")
        f.write(f"  pitchers with p_bb: {sum(1 for p in pitchers if p.get('p_bb') is not None)}\n")
        f.write(f"  pitchers with ip: {sum(1 for p in pitchers if p.get('ip') is not None)}\n")
        f.write(f"  pitchers with er: {sum(1 for p in pitchers if p.get('er') is not None)}\n")

        f.write("\nTest completed successfully\n")

    except Exception as e:
        f.write(f"Error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        sys.exit(1)