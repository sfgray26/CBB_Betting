"""
Yahoo payload capture -- live roster, free agents, and ADP/injury feed.

Run via:
    railway run "C:/Users/sfgra/repos/Fixed/cbb-edge/venv/Scripts/python.exe" scripts/yahoo_capture.py

Writes live API responses to tests/fixtures/ for schema discovery.
Does NOT modify any application code or database state.
"""
import sys
import json
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


def _inspect(label: str, data: object) -> None:
    """Print field summary for the first item of a list response."""
    if isinstance(data, list):
        print(f"{label}: list len={len(data)}", flush=True)
        if data and isinstance(data[0], dict):
            print(f"  first keys: {list(data[0].keys())}", flush=True)
            print(f"  first item: {repr(data[0])[:500]}", flush=True)
    elif isinstance(data, dict):
        print(f"{label}: dict keys={list(data.keys())}", flush=True)
    else:
        print(f"{label}: {type(data).__name__} = {repr(data)[:200]}", flush=True)


try:
    from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
    print("import ok", flush=True)
    c = YahooFantasyClient()
    print("client ok", flush=True)

    # -----------------------------------------------------------------------
    # Task 1: My team roster
    # -----------------------------------------------------------------------
    roster = c.get_roster()
    _inspect("roster", roster)
    (FIXTURES_DIR / "yahoo_roster.json").write_text(
        json.dumps(roster, indent=2, default=str), encoding="utf-8"
    )
    print("roster saved", flush=True)

    # -----------------------------------------------------------------------
    # Task 2: Free agents (first page)
    # -----------------------------------------------------------------------
    free_agents = c.get_free_agents()
    _inspect("free_agents", free_agents)
    (FIXTURES_DIR / "yahoo_free_agents.json").write_text(
        json.dumps(free_agents, indent=2, default=str), encoding="utf-8"
    )
    print("free_agents saved", flush=True)

    # -----------------------------------------------------------------------
    # Task 3: ADP + injury feed (embargo-critical path)
    # -----------------------------------------------------------------------
    adp_feed = c.get_adp_and_injury_feed()
    _inspect("adp_injury_feed", adp_feed)
    (FIXTURES_DIR / "yahoo_adp_injury.json").write_text(
        json.dumps(adp_feed, indent=2, default=str), encoding="utf-8"
    )
    print("adp_injury_feed saved", flush=True)

    print("\nAll Yahoo fixtures saved to tests/fixtures/", flush=True)

except Exception:
    traceback.print_exc()
    sys.exit(1)
