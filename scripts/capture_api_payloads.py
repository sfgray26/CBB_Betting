"""
Payload capture script — Raw API reconnaissance.

Run via:
    railway run python scripts/capture_api_payloads.py

Writes live API responses to tests/fixtures/ for schema discovery.
Does NOT modify any application code or database state.

Auth note: BallDontLie API uses bare key auth (no "Bearer" prefix),
matching the established pattern in backend/services/balldontlie.py line 49.
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Bootstrap path so backend imports resolve when run from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils.time_utils import today_et  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://api.balldontlie.io"
MLB_PREFIX = "/mlb/v1"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
if not API_KEY:
    logger.error(
        "BALLDONTLIE_API_KEY is not set. "
        "Add it to Railway env vars and re-run via: railway run python scripts/capture_api_payloads.py"
    )
    sys.exit(1)

SESSION = requests.Session()
# BDL auth: bare key, no "Bearer" prefix — matches production client in balldontlie.py
SESSION.headers.update({"Authorization": API_KEY})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None) -> dict:
    """Single GET against the BDL MLB API. Raises on non-2xx."""
    url = BASE_URL + MLB_PREFIX + path
    logger.info("GET %s  params=%s", url, params)
    resp = SESSION.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _save(fixture_name: str, payload: dict) -> None:
    """Write payload to tests/fixtures/<fixture_name>."""
    path = FIXTURES_DIR / fixture_name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %s (%d bytes)", path, path.stat().st_size)


def _print_field_summary(label: str, payload: dict) -> None:
    """
    For each top-level key in the first data item, print:
        field_name: type(value) = repr(value)[:80]
    """
    data = payload.get("data", [])
    if not data:
        logger.info("[%s] field summary: data array is empty — no fields to inspect", label)
        return
    first = data[0]
    print(f"\n--- Field summary: {label} (first item) ---")
    for key, value in first.items():
        type_name = type(value).__name__
        snippet = repr(value)[:80]
        print(f"  {key}: {type_name} = {snippet}")
    print()


# ---------------------------------------------------------------------------
# Capture tasks
# ---------------------------------------------------------------------------

def capture_games(today: str) -> dict:
    """Task 1: GET /mlb/v1/games?dates[]={today}"""
    logger.info("=== Task 1: MLB games for %s ===", today)
    try:
        payload = _get("/games", {"dates[]": today})
        _save("bdl_mlb_games.json", payload)
        _print_field_summary("bdl_mlb_games", payload)
        return payload
    except requests.HTTPError as exc:
        logger.error("games endpoint failed: %s", exc)
        return {}


def capture_odds(games_payload: dict) -> None:
    """Task 2: GET /mlb/v1/odds?game_ids[]={first_game_id}
    OpenAPI spec: param is 'game_ids' (array), not 'game_id'.
    Spread values are strings (e.g. '+1.5'), odds are integers (American).
    """
    logger.info("=== Task 2: MLB odds ===")
    data = games_payload.get("data", [])
    if not data:
        logger.warning("No games in payload — skipping odds capture")
        _save("bdl_mlb_odds.json", {})
        return
    game_id = data[0].get("id")
    if game_id is None:
        logger.warning("First game has no 'id' field — skipping odds capture")
        _save("bdl_mlb_odds.json", {})
        return
    logger.info("Using game_id=%s for odds lookup", game_id)
    try:
        # Spec: param name is 'game_ids[]' (array), not 'game_id'
        payload = _get("/odds", {"game_ids[]": game_id})
        _save("bdl_mlb_odds.json", payload)
        _print_field_summary("bdl_mlb_odds", payload)
    except requests.HTTPError as exc:
        logger.error("odds endpoint failed: %s", exc)
        _save("bdl_mlb_odds.json", {})


def capture_injuries() -> None:
    """Task 3: GET /mlb/v1/player_injuries
    OpenAPI spec: endpoint is '/player_injuries', NOT '/injuries'.
    Returns player object + injury detail fields.
    """
    logger.info("=== Task 3: MLB player injuries ===")
    try:
        payload = _get("/player_injuries")
        _save("bdl_mlb_injuries.json", payload)
        _print_field_summary("bdl_mlb_injuries", payload)
        if not payload.get("data"):
            logger.info("player_injuries endpoint returned empty data array")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status in (404, 403, 401):
            logger.info(
                "player_injuries endpoint not available (HTTP %s) — "
                "may not be on current BDL tier",
                status,
            )
        else:
            logger.error("player_injuries endpoint failed with HTTP %s: %s", status, exc)
        _save("bdl_mlb_injuries.json", {})


def capture_players() -> None:
    """Task 4: GET /mlb/v1/players?search=Ohtani"""
    logger.info("=== Task 4: MLB players (search=Ohtani) ===")
    try:
        payload = _get("/players", {"search": "Ohtani"})
        _save("bdl_mlb_players.json", payload)
        _print_field_summary("bdl_mlb_players", payload)
    except requests.HTTPError as exc:
        logger.error("players endpoint failed: %s", exc)
        _save("bdl_mlb_players.json", {})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    today = today_et().isoformat()
    logger.info("Starting payload capture for %s", today)
    logger.info("Fixtures will be written to: %s", FIXTURES_DIR)

    games_payload = capture_games(today)
    capture_odds(games_payload)
    capture_injuries()
    capture_players()

    logger.info("Payload capture complete. Review tests/fixtures/ for raw responses.")
    logger.info(
        "Next step: fill in reports/SCHEMA_DISCOVERY.md field tables "
        "from the captured JSON files."
    )


if __name__ == "__main__":
    main()
