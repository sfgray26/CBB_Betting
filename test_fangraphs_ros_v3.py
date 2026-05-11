"""
Round 3: Capture FanGraphs ROS data via Playwright response interception.

Key insight from Round 2:
- The API endpoint is correct but needs the browser's cookie context.
- route.fetch() fails because it re-issues the request without the page's cookies.
- Solution: use page.on('response') to capture the response body AFTER the browser receives it.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS: list[dict] = []


def _log(name: str, ok: bool, rows: int = 0, err: str = "", extra: dict | None = None):
    RESULTS.append({"name": name, "ok": ok, "rows": rows, "err": err, "extra": extra or {}})
    status = "✅" if ok else "❌"
    logger.info(f"{status} {name}: rows={rows}, err={err or 'none'}")


# ---------------------------------------------------------------------------
# Playwright response capture
# ---------------------------------------------------------------------------

def test_playwright_response_capture():
    """
    Navigate to FanGraphs projections page, wait for the API call to complete,
    and capture the response body via page.on('response').
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        _log("playwright_response", False, err=f"import: {exc}")
        return None

    captured_responses: list[dict] = []

    def handle_response(response):
        url = response.url
        if "fangraphs.com/api/projections" in url and "rostest" in url:
            logger.info(f"  Response captured: {url}")
            try:
                body = response.json()
                captured_responses.append({"url": url, "body": body, "status": response.status})
                logger.info(f"  Status: {response.status}, Items: {len(body) if isinstance(body, list) else 'N/A'}")
            except Exception as exc:
                # Maybe it's not JSON
                try:
                    text = response.text()
                    captured_responses.append({"url": url, "text": text[:500], "status": response.status})
                    logger.info(f"  Status: {response.status}, Text preview: {text[:200]}")
                except Exception as exc2:
                    logger.warning(f"  Could not read response body: {exc2}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()
            page.on("response", handle_response)

            # Step 1: Load the projections hub page
            hub_url = "https://www.fangraphs.com/projections"
            logger.info(f"Navigating to hub: {hub_url}")
            page.goto(hub_url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)

            # Step 2: Click "Steamer (RoS)" link for batters
            # The page has tabs/links for different projection systems
            # Try to find and click the Steamer (RoS) link
            try:
                steamer_link = page.locator("a:has-text('Steamer (RoS)')").first
                if steamer_link.count() > 0:
                    logger.info("Clicking 'Steamer (RoS)' link...")
                    steamer_link.click()
                    time.sleep(4)
                else:
                    logger.info("Steamer (RoS) link not found, proceeding with URL params...")
            except Exception as exc:
                logger.info(f"Click failed: {exc}, proceeding...")

            # Step 3: Also try direct URL with all params
            direct_url = (
                "https://www.fangraphs.com/projections?"
                "pos=all&stats=bat&type=rostest&team=0&lg=all"
            )
            logger.info(f"Navigating to direct URL: {direct_url}")
            page.goto(direct_url, wait_until="networkidle", timeout=60000)
            time.sleep(5)

            # Screenshot for debugging
            page.screenshot(path="fangraphs_v3.png")
            logger.info("Screenshot saved to fangraphs_v3.png")

            browser.close()

        # Process captured responses
        for cap in captured_responses:
            body = cap.get("body")
            if isinstance(body, list) and len(body) > 0:
                df = pd.DataFrame(body)
                _log("playwright_response", True, len(df), extra={
                    "url": cap["url"],
                    "status": cap["status"],
                    "cols": list(df.columns[:15]),
                    "sample": body[0].get("Name") if isinstance(body[0], dict) else "N/A",
                })
                return df
            elif isinstance(body, dict):
                _log("playwright_response", False, err=f"dict keys: {list(body.keys())}", extra={"url": cap["url"]})
            else:
                _log("playwright_response", False, err="non-list body", extra={"url": cap["url"], "type": type(body).__name__})

        if captured_responses:
            # Check text responses
            for cap in captured_responses:
                text = cap.get("text", "")
                if text:
                    _log("playwright_response_text", False, err="non-JSON response", extra={
                        "preview": text[:300],
                        "url": cap["url"],
                    })
        else:
            _log("playwright_response", False, err="no API responses captured")

        return None

    except Exception as exc:
        _log("playwright_response", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Alternative: Try to use the exact intercepted URL with requests but full cookie jar
# ---------------------------------------------------------------------------

def test_requests_with_playwright_cookies():
    """
    Use Playwright to establish a session, extract cookies, then use requests
    with those cookies to call the API.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        _log("requests_with_cookies", False, err=f"import: {exc}")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            # Load the page to get cookies
            page.goto("https://www.fangraphs.com/projections?pos=all&stats=bat&type=rostest", timeout=30000)
            time.sleep(3)

            # Extract cookies
            cookies = context.cookies()
            browser.close()

        # Convert to requests format
        session = requests.Session()
        for c in cookies:
            session.cookies.set(c["name"], c["value"], domain=c["domain"], path=c["path"])

        # Also set all headers
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.fangraphs.com/projections?pos=all&stats=bat&type=rostest",
            "Origin": "https://www.fangraphs.com",
            "X-Requested-With": "XMLHttpRequest",
        }

        url = "https://www.fangraphs.com/api/projections"
        params = {
            "type": "rostest",
            "stats": "bat",
            "pos": "all",
            "team": "0",
            "players": "0",
            "lg": "all",
            "z": "1778047498",  # exact value from interception
        }

        resp = session.get(url, params=params, headers=headers, timeout=30)
        logger.info(f"  Cookie-enhanced request status: {resp.status_code}")
        logger.info(f"  Body preview: {resp.text[:300]}")
        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            _log("requests_with_cookies", True, len(df), extra={"cols": list(df.columns[:10])})
            return df

        _log("requests_with_cookies", False, err="empty or invalid response")
        return None

    except Exception as exc:
        _log("requests_with_cookies", False, err=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("FanGraphs ROS Fetch Test — Round 3")
    logger.info("=" * 60)

    test_playwright_response_capture()
    test_requests_with_playwright_cookies()

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

    Path("test_fangraphs_ros_v3_report.json").write_text(json.dumps(RESULTS, indent=2, default=str))

    if ok:
        logger.info("\n✅ RELIABLE PATH FOUND. Playwright + response capture works.")
    else:
        logger.info("\n❌ All paths failed. FanGraphs may have hardened their anti-bot.")

    return len(ok) > 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
