"""
Run CBB Edge UAT Collection and collect responses for analysis.
"""
import json
import time
import requests
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "https://fantasy-app-production-5079.up.railway.app"
API_KEY = "j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

OUTPUT_DIR = Path(__file__).parent / "responses"
OUTPUT_DIR.mkdir(exist_ok=True)

ENDPOINTS = [
    ("GET", "/", None, "health_root"),
    ("GET", "/health", None, "health_check"),
    ("GET", "/api/fantasy/draft-board?limit=200", None, "draft_board"),
    ("GET", "/api/fantasy/roster", None, "roster"),
    ("GET", "/api/fantasy/lineup/2026-04-20", None, "lineup"),
    ("GET", "/api/fantasy/waiver?position=ALL&player_type=ALL", None, "waiver"),
    ("GET", "/api/fantasy/waiver/recommendations", None, "waiver_recommendations"),
    ("GET", "/api/fantasy/matchup", None, "matchup"),
    ("GET", "/api/fantasy/player-scores?period=season", None, "player_scores"),
    ("GET", "/api/fantasy/decisions", None, "decisions"),
    ("GET", "/api/fantasy/briefing/2026-04-20", None, "briefing"),
    ("GET", "/admin/pipeline-health", None, "pipeline_health"),
    ("GET", "/admin/scheduler/status", None, "scheduler_status"),
    ("GET", "/admin/validate-system", None, "validate_system"),
    ("POST", "/api/fantasy/roster/optimize", {"date": "2026-04-20", "force_stale": False}, "roster_optimize"),
    ("POST", "/api/fantasy/matchup/simulate", {"week": 3, "n_sims": 1000}, "matchup_simulate"),
]

results = []

for method, path, body, name in ENDPOINTS:
    url = urljoin(BASE_URL, path)
    print(f"[{method}] {url} ...", end=" ")
    try:
        if method == "GET":
            resp = requests.get(url, headers=HEADERS, timeout=30)
        else:
            resp = requests.post(url, headers=HEADERS, json=body, timeout=30)
        print(f"{resp.status_code} ({len(resp.content)} bytes)")
        
        # Save response
        out_file = OUTPUT_DIR / f"{name}_{resp.status_code}.json"
        try:
            data = resp.json()
            out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            out_file = OUTPUT_DIR / f"{name}_{resp.status_code}.txt"
            out_file.write_text(resp.text, encoding="utf-8")
        
        results.append({
            "name": name,
            "method": method,
            "url": url,
            "status": resp.status_code,
            "size": len(resp.content),
            "content_type": resp.headers.get("Content-Type", ""),
            "data": data if "data" in dir() else None,
        })
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "name": name,
            "method": method,
            "url": url,
            "status": None,
            "error": str(e),
        })
    time.sleep(0.5)

# Save summary
summary_file = OUTPUT_DIR / "summary.json"
summary_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print(f"\nDone. Responses saved to {OUTPUT_DIR}")
