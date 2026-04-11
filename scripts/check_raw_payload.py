"""
Check raw BDL API response to confirm field availability
"""
import sys
sys.path.insert(0, '.')

with open('raw_payload_check.txt', 'w') as f:
    try:
        f.write("Checking raw BDL API response structure...\n")
        import requests
        import os

        # Make raw API call to see exact response
        api_key = os.environ.get('BALLDONTLIE_API_KEY')
        headers = {'Authorization': api_key}
        params = {'per_page': 2}

        response = requests.get(
            'https://api.balldontlie.io/mlb/v1/stats',
            headers=headers,
            params=params
        )

        data = response.json()
        f.write(f"Status: {response.status_code}\n")
        f.write(f"Data keys: {list(data.keys())}\n\n")

        if 'data' in data and data['data']:
            first_stat = data['data'][0]
            f.write("First stat raw keys:\n")
            for key in sorted(first_stat.keys()):
                value = first_stat[key]
                # Truncate long values
                if isinstance(value, dict) and len(str(value)) > 100:
                    f.write(f"  {key}: {type(value).__name__} (truncated)\n")
                elif isinstance(value, str) and len(value) > 100:
                    f.write(f"  {key}: {value[:100]}...\n")
                else:
                    f.write(f"  {key}: {value}\n")

        f.write("\nTest completed successfully\n")

    except Exception as e:
        f.write(f"Error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        sys.exit(1)