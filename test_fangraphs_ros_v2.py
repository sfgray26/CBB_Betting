"""
Round 2: FanGraphs ROS fetch test with corrected parameters.

Key findings from Round 1:
- API needs: team=0&players=0&lg=all&z=<timestamp>
- pybaseball's batting_stats() signature changed (season kwarg removed)
- Playwright intercept worked but parsing failed; page needs lighter wait
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fangraphs.com/projections",
    "Origin": "https://www.fangraphs.com",
    "X-Requested-With": "XMLHttpRequest",
}

RESULTS: list[dict] = []


def _log(name: str, ok: bool, rows: int = 0, err: str = "", extra: dict | None = None):
    RESULTS.append({"name": name, "ok": ok, "rows": rows, "err": err, "extra": extra or {}})
    status = "✅" if ok else "❌"
    logger.info(f"{status} {name}: rows={rows}, err={err or 'none'}")


# ---------------------------------------------------------------------------
# Test A: FanGraphs API with full query params
# ---------------------------------------------------------------------------

def test_api_full_params():
    """Hit the API with the exact params observed in Playwright intercept."""
    base = "https://www.fangraphs.com/api/projections"
    params = {
        "type": "rostest",
        "stats": "bat",
        "pos": "all",
        "team": "0",
        "players": "0",
        "lg": "all",
        "z": str(int(time.time())),  # cache buster
    }

    try:
        resp = requests.get(base, params=params, headers=_BROWSER_HEADERS, timeout=30)
        logger.info(f"  Status: {resp.status_code}, Content-Type: {resp.headers.get('Content-Type', 'N/A')}")
        logger.info(f"  URL: {resp.url}")
        logger.info(f"  Body preview (first 500 chars): {resp.text[:500]}")

        resp.raise_for_status()

        # Try JSON
        try:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                _log("api_full_params", True, len(df), extra={"cols": list(df.columns[:10]), "player": data[0].get("Name")})
                return df
            elif isinstance(data, dict):
                _log("api_full_params", False, err=f"dict response keys: {list(data.keys())}")
                return None
        except Exception:
            pass

        # Try CSV
        if "text/csv" in resp.headers.get("Content-Type", "") or resp.text.count(",") > 10:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            _log("api_full_params_csv", True, len(df), extra={"cols": list(df.columns[:10])})
            return df

        _log("api_full_params", False, err="unparseable response")
        return None

    except Exception as exc:
        _log("api_full_params", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Test B: FanGraphs API for pitchers
# ---------------------------------------------------------------------------

def test_api_pitchers():
    base = "https://www.fangraphs.com/api/projections"
    params = {
        "type": "rostest",
        "stats": "pit",
        "pos": "all",
        "team": "0",
        "players": "0",
        "lg": "all",
        "z": str(int(time.time())),
    }
    try:
        resp = requests.get(base, params=params, headers=_BROWSER_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
            _log("api_pitchers", True, len(df), extra={"cols": list(df.columns[:10])})
            return df
        _log("api_pitchers", False, err="non-list response")
        return None
    except Exception as exc:
        _log("api_pitchers", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Test C: pybaseball without season kwarg
# ---------------------------------------------------------------------------

def test_pybaseball_no_season():
    try:
        import pybaseball
        from backend.fantasy_baseball.pybaseball_loader import _patch_pybaseball_user_agent
        _patch_pybaseball_user_agent()
    except Exception as exc:
        _log("pybaseball_no_season", False, err=f"import: {exc}")
        return None

    try:
        # Newer pybaseball might use a different signature or require no season param for ROS
        df = pybaseball.batting_stats(2026, qual=10)
        _log("pybaseball_no_season", True, len(df), extra={"cols": list(df.columns[:10])})
        return df
    except Exception as exc:
        _log("pybaseball_no_season", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Test D: pybaseball projection_fangraphs_batters
# ---------------------------------------------------------------------------

def test_pybaseball_projections():
    try:
        import pybaseball
        from backend.fantasy_baseball.pybaseball_loader import _patch_pybaseball_user_agent
        _patch_pybaseball_user_agent()
    except Exception as exc:
        _log("pybaseball_projections", False, err=f"import: {exc}")
        return None

    # Try the explicit projection function if available
    try:
        df = pybaseball.projections_fangraphs_batters(2026)
        _log("pybaseball_projections", True, len(df), extra={"cols": list(df.columns[:10])})
        return df
    except AttributeError:
        _log("pybaseball_projections", False, err="projections_fangraphs_batters not found")
    except Exception as exc:
        _log("pybaseball_projections", False, err=str(exc))
    return None


# ---------------------------------------------------------------------------
# Test E: Playwright with lighter wait + raw text capture
# ---------------------------------------------------------------------------

def test_playwright_v2():
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        _log("playwright_v2", False, err=f"import: {exc}")
        return None

    captured: list[tuple[str, str]] = []  # url, body_text

    def handle_route(route, request):
        url = request.url
        if "fangraphs.com/api/projections" in url:
            logger.info(f"  Intercepted: {url}")
            try:
                response = route.fetch()
                text = response.text()
                captured.append((url, text))
                logger.info(f"  Body length: {len(text)}")
            except Exception as exc:
                logger.warning(f"  Intercept error: {exc}")
        route.continue_()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=_BROWSER_HEADERS["User-Agent"],
                viewport={"width": 1920, "height": 1080},
            )
            page.route("**/*", handle_route)

            url = "https://www.fangraphs.com/projections?pos=all&stats=bat&type=rostest"
            logger.info(f"Navigating to {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(5)  # Let JS table hydrate

            # Screenshot for debug
            page.screenshot(path="fangraphs_debug.png")
            logger.info("Screenshot saved to fangraphs_debug.png")

            browser.close()

        if captured:
            url, text = captured[0]
            # Try JSON parse
            try:
                data = json.loads(text)
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    _log("playwright_v2", True, len(df), extra={
                        "url": url,
                        "cols": list(df.columns[:10]),
                        "sample": data[0].get("Name"),
                    })
                    return df
            except json.JSONDecodeError:
                _log("playwright_v2", False, err="intercepted response is not JSON", extra={
                    "text_preview": text[:200],
                    "url": url,
                })
                return None

        _log("playwright_v2", False, err="no API calls intercepted")
        return None

    except Exception as exc:
        _log("playwright_v2", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Test F: MLB-StatsAPI for active rosters (fallback)
# ---------------------------------------------------------------------------

def test_mlb_statsapi():
    try:
        import statsapi
    except Exception as exc:
        _log("mlb_statsapi", False, err=f"import: {exc}")
        return None

    try:
        # StatsAPI doesn't have ROS projections, but let's verify it's available
        player = statsapi.lookup_player("Mike Trout")
        _log("mlb_statsapi", True, len(player), extra={"sample": player[0] if player else None})
        return None  # Not ROS projections, just availability check
    except Exception as exc:
        _log("mlb_statsapi", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("FanGraphs ROS Fetch Test — Round 2")
    logger.info("=" * 60)

    test_api_full_params()
    test_api_pitchers()
    test_pybaseball_no_season()
    test_pybaseball_projections()
    test_playwright_v2()
    test_mlb_statsapi()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    ok = [r for r in RESULTS if r["ok"]]
    bad = [r for r in RESULTS if not r["ok"]]

    logger.info(f"Passed: {len(ok)}")
    for r in ok:
        logger.info(f"  ✅ {r['name']}: {r['rows']} rows")

    logger.info(f"Failed: {len(bad)}")
    for r in bad:
        logger.info(f"  ❌ {r['name']}: {r['err']}")

    Path("test_fangraphs_ros_v2_report.json").write_text(json.dumps(RESULTS, indent=2, default=str))

    return len(ok) > 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
