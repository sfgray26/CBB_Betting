import requests, json, os, re
from datetime import datetime

BASE = "https://fantasy-app-production-5079.up.railway.app"
HEADERS = {"X-API-Key": "j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg", "Content-Type": "application/json"}

def slugify(path):
    s = re.sub(r"[^\w]", "_", path.strip("/"))
    s = s.replace("__", "_").strip("_")
    return s[:60]

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "responses")
    os.makedirs(out_dir, exist_ok=True)

    summary = []

    # Helper to save response
    def save_resp(method, path, resp):
        fname = f"{slugify(path)}_{ts}.json"
        fpath = os.path.join(out_dir, fname)
        try:
            data = resp.json()
        except Exception:
            data = {"_raw_text": resp.text[:500], "_status": resp.status_code}
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        size = os.path.getsize(fpath)
        line = f"{method} {path} -> {resp.status_code} ({size} bytes) -> {fname}"
        print(line)
        summary.append(line)
        return data

    # 1. GET endpoints
    get_endpoints = [
        "/",
        "/health",
        "/api/fantasy/draft-board?limit=200",
        "/api/fantasy/roster",
        "/api/fantasy/lineup/2026-04-22",
        "/api/fantasy/waiver?position=ALL&player_type=ALL",
        "/api/fantasy/waiver/recommendations",
        "/api/fantasy/matchup",
        "/api/fantasy/player-scores?period=season",
        "/api/fantasy/decisions",
        "/api/fantasy/briefing/2026-04-22",
        "/admin/pipeline-health",
        "/admin/scheduler/status",
        "/admin/validate-system",
    ]

    for path in get_endpoints:
        try:
            resp = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=60)
            save_resp("GET", path, resp)
        except Exception as e:
            line = f"GET {path} -> ERROR: {e}"
            print(line)
            summary.append(line)

    # 2. POST roster/optimize with proper payload
    try:
        path = "/api/fantasy/roster/optimize"
        payload = {"target_date": "2026-04-22"}
        resp = requests.post(f"{BASE}{path}", headers=HEADERS, json=payload, timeout=60)
        save_resp("POST", path, resp)
    except Exception as e:
        line = f"POST {path} -> ERROR: {e}"
        print(line)
        summary.append(line)

    # 3. POST matchup/simulate with proper payload (minimal roster dicts)
    try:
        path = "/api/fantasy/matchup/simulate"
        payload = {
            "my_roster": [],
            "opponent_roster": [],
            "n_sims": 100,
            "week": "2026-W17"
        }
        resp = requests.post(f"{BASE}{path}", headers=HEADERS, json=payload, timeout=60)
        save_resp("POST", path, resp)
    except Exception as e:
        line = f"POST {path} -> ERROR: {e}"
        print(line)
        summary.append(line)

    summary_path = os.path.join(out_dir, f"summary_{ts}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
