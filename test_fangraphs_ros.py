"""
Test script to validate FanGraphs ROS projection fetching.
Tries multiple strategies and reports which ones work.

Strategies:
1. pybaseball (existing library)
2. Direct requests with browser headers
3. cloudscraper (Cloudflare bypass)
4. Playwright (full browser automation)

Usage:
    venv\Scripts\python test_fangraphs_ros.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_LOG: list[dict] = []


def _log_result(strategy: str, success: bool, rows: int = 0, error: str = "", extra: dict | None = None):
    RESULTS_LOG.append({
        "strategy": strategy,
        "success": success,
        "rows": rows,
        "error": error,
        "extra": extra or {},
        "timestamp": datetime.now().isoformat(),
    })
    status = "✅ SUCCESS" if success else "❌ FAILED"
    logger.info(f"[{status}] {strategy}: rows={rows}, error={error or 'none'}")


# ---------------------------------------------------------------------------
# Strategy 1: pybaseball
# ---------------------------------------------------------------------------

def test_pybaseball() -> Optional[pd.DataFrame]:
    """Use pybaseball batting_stats/pitching_stats with season='rostest'."""
    try:
        from backend.fantasy_baseball.pybaseball_loader import _patch_pybaseball_user_agent
        _patch_pybaseball_user_agent()
        import pybaseball
    except Exception as exc:
        _log_result("pybaseball", False, error=f"import failed: {exc}")
        return None

    try:
        df_bat = pybaseball.batting_stats(2026, season="rostest", qual=10)
        time.sleep(2)
        df_pit = pybaseball.pitching_stats(2026, season="rostest", qual=5)

        total_rows = len(df_bat) + len(df_pit)
        _log_result("pybaseball", True, rows=total_rows, extra={
            "batters": len(df_bat),
            "pitchers": len(df_pit),
            "bat_cols": list(df_bat.columns[:10]) if len(df_bat) > 0 else [],
        })
        return df_bat
    except Exception as exc:
        _log_result("pybaseball", False, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Strategy 2: Direct requests (browser headers)
# ---------------------------------------------------------------------------

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
}


def test_direct_api() -> Optional[pd.DataFrame]:
    """
    Hit the FanGraphs projections API directly.
    The page loads data from an internal API endpoint.
    """
    # FanGraphs uses a grid/fetch endpoint for projections
    url = "https://www.fangraphs.com/api/projections"
    params = {
        "type": "rostest",
        "stats": "bat",
        "pos": "all",
    }

    try:
        resp = requests.get(url, params=params, headers=_BROWSER_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            _log_result("direct_api", True, rows=len(df), extra={
                "columns": list(df.columns[:10]),
                "sample_player": data[0].get("Name") if isinstance(data[0], dict) else "N/A",
            })
            return df
        elif isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
            _log_result("direct_api", True, rows=len(df), extra={"nested": True})
            return df
        else:
            _log_result("direct_api", False, error=f"unexpected response shape: {type(data)}")
            return None
    except Exception as exc:
        _log_result("direct_api", False, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Strategy 3: cloudscraper
# ---------------------------------------------------------------------------

def test_cloudscraper() -> Optional[pd.DataFrame]:
    """Use cloudscraper to bypass any Cloudflare/anti-bot checks."""
    try:
        import cloudscraper
    except Exception as exc:
        _log_result("cloudscraper", False, error=f"import failed: {exc}")
        return None

    scraper = cloudscraper.create_scraper()
    url = "https://www.fangraphs.com/api/projections"
    params = {"type": "rostest", "stats": "bat", "pos": "all"}

    try:
        resp = scraper.get(url, params=params, headers=_BROWSER_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            _log_result("cloudscraper", True, rows=len(df))
            return df
        else:
            _log_result("cloudscraper", False, error=f"unexpected response shape: {type(data)}")
            return None
    except Exception as exc:
        _log_result("cloudscraper", False, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Strategy 4: Playwright (full browser)
# ---------------------------------------------------------------------------

def test_playwright() -> Optional[pd.DataFrame]:
    """
    Use Playwright to load the FanGraphs projections page,
    intercept the API response, and extract the JSON data.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        _log_result("playwright", False, error=f"import failed: {exc}")
        return None

    intercepted_data: list[dict] = []

    def handle_route(route, request):
        """Intercept API calls to the projections endpoint."""
        url = request.url
        if "fangraphs.com/api/projections" in url or "fangraphs.com/api/leaderboards" in url:
            logger.info(f"Playwright intercepted: {url}")
            try:
                response = route.fetch()
                body = response.json()
                intercepted_data.append({"url": url, "data": body})
                logger.info(f"  -> captured {len(body) if isinstance(body, list) else 'N/A'} items")
            except Exception as exc:
                logger.warning(f"  -> failed to parse intercepted response: {exc}")
        route.continue_()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=_BROWSER_HEADERS["User-Agent"],
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()
            page.route("**/*", handle_route)

            # Navigate to the projections page
            proj_url = "https://www.fangraphs.com/projections?pos=all&stats=bat&type=rostest"
            logger.info(f"Playwright navigating to {proj_url}")
            page.goto(proj_url, wait_until="networkidle", timeout=60000)

            # Wait a moment for any lazy/XHR loads
            time.sleep(3)

            # Also try to grab data directly from page if the table is rendered server-side
            table_data = page.evaluate("""
                () => {
                    const rows = document.querySelectorAll('table tbody tr');
                    return Array.from(rows).slice(0, 5).map(r => {
                        const cells = r.querySelectorAll('td');
                        return Array.from(cells).slice(0, 5).map(c => c.textContent.trim());
                    });
                }
            """)

            browser.close()

        if intercepted_data:
            body = intercepted_data[0]["data"]
            if isinstance(body, list):
                df = pd.DataFrame(body)
                _log_result("playwright", True, rows=len(df), extra={
                    "intercepted_urls": len(intercepted_data),
                    "table_preview": table_data,
                })
                return df
            elif isinstance(body, dict) and "data" in body:
                df = pd.DataFrame(body["data"])
                _log_result("playwright", True, rows=len(df), extra={
                    "intercepted_urls": len(intercepted_data),
                })
                return df

        # If no interception but table rendered in DOM
        if table_data and any(table_data):
            _log_result("playwright", True, rows=len(table_data), extra={
                "mode": "dom_table",
                "sample": table_data[:3],
            })
            return pd.DataFrame(table_data)

        _log_result("playwright", False, error="no data intercepted or rendered")
        return None

    except Exception as exc:
        _log_result("playwright", False, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Strategy 5: Playwright + direct API call from browser context
# ---------------------------------------------------------------------------

def test_playwright_api() -> Optional[pd.DataFrame]:
    """
    Use Playwright to execute a fetch() inside the browser context.
    This carries the browser's cookies and fingerprint.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        _log_result("playwright_api", False, error=f"import failed: {exc}")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=_BROWSER_HEADERS["User-Agent"],
            )
            page = context.new_page()

            # Go to the base site first to establish cookies/session
            page.goto("https://www.fangraphs.com", wait_until="domcontentloaded", timeout=30000)
            time.sleep(1)

            # Execute fetch in the page context
            api_url = "https://www.fangraphs.com/api/projections?type=rostest&stats=bat&pos=all"
            result = page.evaluate(f"""
                async () => {{
                    try {{
                        const resp = await fetch("{api_url}", {{
                            headers: {{
                                "Accept": "application/json",
                                "X-Requested-With": "XMLHttpRequest"
                            }}
                        }});
                        if (!resp.ok) return {{error: resp.status + " " + resp.statusText}};
                        const data = await resp.json();
                        return {{success: true, count: Array.isArray(data) ? data.length : (data.data ? data.data.length : 0), first: Array.isArray(data) && data.length > 0 ? data[0] : null}};
                    }} catch (e) {{
                        return {{error: e.message}};
                    }}
                }}
            """)

            browser.close()

        if result.get("success"):
            first = result.get("first")
            count = result.get("count", 0)
            _log_result("playwright_api", True, rows=count, extra={
                "sample_player": first.get("Name") if isinstance(first, dict) else "N/A",
            })
            return pd.DataFrame([first]) if first else pd.DataFrame()
        else:
            _log_result("playwright_api", False, error=result.get("error", "unknown"))
            return None

    except Exception as exc:
        _log_result("playwright_api", False, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Strategy 6: pybaseball direct internal call (stealth mode)
# ---------------------------------------------------------------------------

def test_pybaseball_direct() -> Optional[pd.DataFrame]:
    """
    Call pybaseball's internal FanGraphs endpoint directly without
    going through the batting_stats wrapper.
    """
    try:
        import pybaseball.fangraphs as fg
        from backend.fantasy_baseball.pybaseball_loader import _patch_pybaseball_user_agent
        _patch_pybaseball_user_agent()
    except Exception as exc:
        _log_result("pybaseball_direct", False, error=f"import failed: {exc}")
        return None

    try:
        # pybaseball's fangraphs module has a batting_stats_range or similar
        # Let's try the standard batting_stats with explicit retry
        import pybaseball
        df = pybaseball.batting_stats(2026, season="rostest", qual=10)
        _log_result("pybaseball_direct", True, rows=len(df), extra={
            "columns": list(df.columns[:10]),
        })
        return df
    except Exception as exc:
        _log_result("pybaseball_direct", False, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("FanGraphs ROS Projection Fetch Test")
    logger.info("=" * 60)

    # Strategy 1: pybaseball (existing)
    logger.info("\n--- Strategy 1: pybaseball ---")
    df1 = test_pybaseball()

    # Strategy 2: Direct API
    logger.info("\n--- Strategy 2: direct_api ---")
    df2 = test_direct_api()

    # Strategy 3: cloudscraper
    logger.info("\n--- Strategy 3: cloudscraper ---")
    df3 = test_cloudscraper()

    # Strategy 4: Playwright intercept
    logger.info("\n--- Strategy 4: playwright (intercept) ---")
    df4 = test_playwright()

    # Strategy 5: Playwright in-page fetch
    logger.info("\n--- Strategy 5: playwright_api ---")
    df5 = test_playwright_api()

    # Strategy 6: pybaseball direct
    logger.info("\n--- Strategy 6: pybaseball_direct ---")
    df6 = test_pybaseball_direct()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    successes = [r for r in RESULTS_LOG if r["success"]]
    failures = [r for r in RESULTS_LOG if not r["success"]]

    logger.info(f"Successes: {len(successes)}")
    for r in successes:
        logger.info(f"  ✅ {r['strategy']}: {r['rows']} rows")

    logger.info(f"Failures: {len(failures)}")
    for r in failures:
        logger.info(f"  ❌ {r['strategy']}: {r['error']}")

    # Save detailed report
    report_path = Path("test_fangraphs_ros_report.json")
    report_path.write_text(json.dumps(RESULTS_LOG, indent=2, default=str))
    logger.info(f"\nDetailed report saved to: {report_path}")

    # If any strategy worked, show sample data
    for df, name in [(df1, "pybaseball"), (df2, "direct_api"), (df3, "cloudscraper"), (df4, "playwright"), (df5, "playwright_api"), (df6, "pybaseball_direct")]:
        if df is not None and len(df) > 0:
            logger.info(f"\nSample from {name}:")
            logger.info(df.head(3).to_string())
            break

    return len(successes) > 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
