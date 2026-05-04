import requests, json, os, re
from datetime import datetime

BASE = "https://fantasy-app-production-5079.up.railway.app"
HEADERS = {"X-API-Key": "j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg"}

ENDPOINTS = [
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/api/fantasy/draft-board?limit=200"),
    ("GET", "/api/fantasy/roster"),
    ("GET", "/api/fantasy/lineup/2026-04-22"),
    ("GET", "/api/fantasy/waiver?position=ALL&player_type=ALL"),
    ("GET", "/api/fantasy/waiver/recommendations"),
    ("GET", "/api/fantasy/matchup"),
    ("GET", "/api/fantasy/player-scores?period=season"),
    ("GET", "/api/fantasy/decisions"),
    ("GET", "/api/fantasy/briefing/2026-04-22"),
    ("GET", "/admin/pipeline-health"),
    ("GET", "/admin/scheduler/status"),
    ("GET", "/admin/validate-system"),
    ("POST", "/api/fantasy/roster/optimize"),
    ("POST", "/api/fantasy/matchup/simulate"),
]

def slugify(path):
    s = re.sub(r"[^\w]", "_", path.strip("/"))
    s = s.replace("__", "_").strip("_")
    return s[:60]

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "responses")
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for method, path in ENDPOINTS:
        url = f"{BASE}{path}"
        fname = f"{slugify(path)}_{ts}.json"
        fpath = os.path.join(out_dir, fname)
        try:
            if method == "GET":
                resp = requests.get(url, headers=HEADERS, timeout=30)
            else:
                resp = requests.post(url, headers=HEADERS, timeout=30)
            status = resp.status_code
            try:
                data = resp.json()
            except Exception:
                data = {"_raw_text": resp.text[:500]}
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            size = os.path.getsize(fpath)
            line = f"{method} {path} -> {status} ({size} bytes) -> {fname}"
            print(line)
            summary.append(line)
        except Exception as e:
            line = f"{method} {path} -> ERROR: {e}"
            print(line)
            summary.append(line)

    summary_path = os.path.join(out_dir, f"summary_{ts}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
