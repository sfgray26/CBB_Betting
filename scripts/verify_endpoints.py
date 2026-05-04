import requests
import time
import json
import os

API_KEY = "j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg"
BASE_URL = "https://fantasy-app-production-5079.up.railway.app"
HEADERS = {"X-API-KEY": API_KEY}

def test_endpoint(name, method, url, data=None):
    print(f"=== Testing {name}: {method} {url} ===")
    start = time.time()
    try:
        if method == "GET":
            resp = requests.get(url, headers=HEADERS, timeout=40)
        else:
            resp = requests.post(url, headers=HEADERS, json=data, timeout=40)
        elapsed = time.time() - start
        print(f"Status: {resp.status_code}")
        print(f"Response Time: {elapsed:.2f}s")
        if resp.status_code == 200:
            print("Content Snippet:", json.dumps(resp.json(), indent=2)[:500])
        else:
            print("Error:", resp.text[:500])
        return resp.status_code, elapsed, resp.json() if resp.status_code == 200 else None
    except requests.exceptions.Timeout:
        print("TIMEOUT after 40s")
        return "TIMEOUT", 40.0, None
    except Exception as e:
        print(f"FAILED: {e}")
        return "ERROR", 0.0, None

results = []

# Test 1
status, elapsed, data = test_endpoint("Lineup", "GET", f"{BASE_URL}/api/fantasy/lineup/2026-05-02")
results.append(("GET /api/fantasy/lineup/2026-05-02", status, elapsed))

# Test 2
status, elapsed, data = test_endpoint("Roster Optimize", "POST", f"{BASE_URL}/api/fantasy/roster/optimize", 
                                     data={"target_date": "2026-05-02", "yahoo_league_id": "72586"})
results.append(("POST /api/fantasy/roster/optimize", status, elapsed))

# Test 3
status, elapsed, data = test_endpoint("Matchup", "GET", f"{BASE_URL}/api/fantasy/matchup?date=2026-05-02")
results.append(("GET /api/fantasy/matchup", status, elapsed))

# Test 4
status, elapsed, data = test_endpoint("Waiver recommendations", "GET", f"{BASE_URL}/api/fantasy/waiver/recommendations?league_id=72586")
results.append(("GET /api/fantasy/waiver/recommendations", status, elapsed))

# Test 5
status, elapsed, data = test_endpoint("Decisions", "GET", f"{BASE_URL}/api/fantasy/decisions?limit=5")
results.append(("GET /api/fantasy/decisions", status, elapsed))

print("\n\nSUMMARY TABLE")
print("| Endpoint | Status | Time |")
print("|----------|--------|------|")
for name, status, elapsed in results:
    print(f"| {name} | {status} | {elapsed:.2f}s |")
