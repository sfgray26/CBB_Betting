"""
Round 4: FanGraphs changed their API parameter from 'rostest' to 'steamerr'.

Key finding from Round 3 error:
  href="/projections?pos=all&stats=bat&type=steamerr"

Let's test the corrected endpoint.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS: list[dict] = []

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


def _log(name: str, ok: bool, rows: int = 0, err: str = "", extra: dict | None = None):
    RESULTS.append({"name": name, "ok": ok, "rows": rows, "err": err, "extra": extra or {}})
    status = "✅" if ok else "❌"
    logger.info(f"{status} {name}: rows={rows}, err={err or 'none'}")


# ---------------------------------------------------------------------------
# Test A: Direct API with 'steamerr' (new parameter)
# ---------------------------------------------------------------------------

def test_api_steamerr():
    base = "https://www.fangraphs.com/api/projections"
    params = {
        "type": "steamerr",
        "stats": "bat",
        "pos": "all",
        "team": "0",
        "players": "0",
        "lg": "all",
        "z": "1778047498",
    }
    try:
        resp = requests.get(base, params=params, headers=_BROWSER_HEADERS, timeout=30)
        logger.info(f"  Status: {resp.status_code}")
        logger.info(f"  Preview: {resp.text[:300]}")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            _log("api_steamerr_bat", True, len(df), extra={
                "cols": list(df.columns[:15]),
                "sample": data[0].get("Name"),
            })
            return df
        _log("api_steamerr_bat", False, err="empty or non-list response")
        return None
    except Exception as exc:
        _log("api_steamerr_bat", False, err=str(exc))
        return None


def test_api_steamerr_pit():
    base = "https://www.fangraphs.com/api/projections"
    params = {
        "type": "steamerr",
        "stats": "pit",
        "pos": "all",
        "team": "0",
        "players": "0",
        "lg": "all",
        "z": "1778047498",
    }
    try:
        resp = requests.get(base, params=params, headers=_BROWSER_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            _log("api_steamerr_pit", True, len(df), extra={"cols": list(df.columns[:15])})
            return df
        _log("api_steamerr_pit", False, err="empty or non-list response")
        return None
    except Exception as exc:
        _log("api_steamerr_pit", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Test B: pybaseball with corrected season parameter
# ---------------------------------------------------------------------------

def test_pybaseball_steamerr():
    try:
        import pybaseball
        from backend.fantasy_baseball.pybaseball_loader import _patch_pybaseball_user_agent
        _patch_pybaseball_user_agent()
    except Exception as exc:
        _log("pybaseball_steamerr", False, err=f"import: {exc}")
        return None

    try:
        # Try without season kwarg, just year
        df = pybaseball.batting_stats(2026, qual=10)
        _log("pybaseball_batting_stats", True, len(df), extra={"cols": list(df.columns[:10])})
        return df
    except Exception as exc:
        _log("pybaseball_batting_stats", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Test C: Playwright + response capture with 'steamerr'
# ---------------------------------------------------------------------------

def test_playwright_steamerr():
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        _log("playwright_steamerr", False, err=f"import: {exc}")
        return None

    captured: list[dict] = []

    def handle_response(response):
        url = response.url
        if "api/projections" in url and "steamerr" in url:
            logger.info(f"  Captured: {url} (status={response.status})")
            try:
                body = response.json()
                captured.append({"url": url, "body": body, "status": response.status})
            except Exception:
                try:
                    text = response.text()
                    captured.append({"url": url, "text": text[:500], "status": response.status})
                except Exception:
                    pass

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=_BROWSER_HEADERS["User-Agent"],
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()
            page.on("response", handle_response)

            url = "https://www.fangraphs.com/projections?pos=all&stats=bat&type=steamerr"
            logger.info(f"Navigating: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(5)

            page.screenshot(path="fangraphs_v4.png")
            logger.info("Screenshot: fangraphs_v4.png")
            browser.close()

        for cap in captured:
            body = cap.get("body")
            if isinstance(body, list) and len(body) > 0:
                df = pd.DataFrame(body)
                _log("playwright_steamerr", True, len(df), extra={
                    "url": cap["url"],
                    "cols": list(df.columns[:15]),
                    "sample": body[0].get("Name"),
                })
                return df
            text = cap.get("text")
            if text:
                _log("playwright_steamerr_text", False, err="non-JSON", extra={"preview": text[:200]})

        _log("playwright_steamerr", False, err="no captures")
        return None

    except Exception as exc:
        _log("playwright_steamerr", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("FanGraphs ROS Fetch Test — Round 4 (steamerr endpoint)")
    logger.info("=" * 60)

    test_api_steamerr()
    test_api_steamerr_pit()
    test_pybaseball_steamerr()
    test_playwright_steamerr()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    ok = [r for r in RESULTS if r["ok"]]
    bad = [r for r in RESULTS if not r["ok"]]

    for r in ok:
        logger.info(f"  ✅ {r['name']}: {r['rows']} rows")
    for r in bad:
        logger.info(f"  ❌ {r['name']}: {r['err']}")

    Path("test_fangraphs_ros_v4_report.json").write_text(json.dumps(RESULTS, indent=2, default=str))

    if ok:
        logger.info("\n✅ WORKING PATH FOUND.")
        for r in ok:
            logger.info(f"   Strategy: {r['name']} — {r['rows']} rows")
    else:
        logger.info("\n❌ All paths failed.")

    return len(ok) > 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
