#!/usr/bin/env python3
"""
Production uptime monitoring script.

Checks critical health endpoints and returns exit code 0 if all pass,
1 if any fail. Designed for cron or UptimeRobot integration.

Usage:
    python scripts/uptime_check.py
    python scripts/uptime_check.py --url https://custom-url.railway.app
"""

import argparse
import json
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def check_endpoint(base_url: str, path: str, timeout: int = 10) -> dict:
    """Check a single endpoint and return status details.

    Returns:
        {
            "endpoint": "/health",
            "status_code": 200,
            "response_time_ms": 123,
            "data_ok": True,
            "error": None
        }
    """
    url = f"{base_url}{path}"
    result = {
        "endpoint": path,
        "status_code": None,
        "response_time_ms": None,
        "data_ok": False,
        "error": None,
    }

    try:
        start = time.time()
        req = Request(url, method="GET")
        req.add_header("User-Agent", "CBB-Edge-UptimeMonitor/1.0")

        with urlopen(req, timeout=timeout) as response:
            elapsed_ms = int((time.time() - start) * 1000)
            result["status_code"] = response.status
            result["response_time_ms"] = elapsed_ms

            # Parse response for data validation
            try:
                data = json.loads(response.read().decode())

                # /health and /health/db must have status="healthy" or "connected"
                if path == "/health":
                    result["data_ok"] = data.get("status") in ("healthy", "degraded")
                    # Degraded is OK for uptime checks (service is running)
                    if data.get("status") == "degraded":
                        result["note"] = "Service degraded but running"
                elif path == "/health/db":
                    result["data_ok"] = data.get("status") == "connected"
                elif path == "/health/pipeline":
                    # 503 is returned if critical jobs are unhealthy
                    result["data_ok"] = response.status == 200
                else:
                    # For other endpoints, any 2xx is OK
                    result["data_ok"] = 200 <= response.status < 300
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Non-JSON response is OK if status code is good
                result["data_ok"] = 200 <= response.status < 300

    except HTTPError as e:
        result["status_code"] = e.code
        result["error"] = f"HTTP {e.code}: {e.reason}"
    except URLError as e:
        result["error"] = f"URL Error: {e.reason}"
    except Exception as e:
        result["error"] = str(e)

    return result


def print_result(result: dict):
    """Print a single result in a readable format."""
    status_symbol = "✓" if result["data_ok"] else "✗"
    print(f"{status_symbol} {result['endpoint']}")
    print(f"  Status: {result['status_code'] or 'NO RESPONSE'}")
    print(f"  Time: {result['response_time_ms'] or 'N/A'}ms")

    if result["error"]:
        print(f"  Error: {result['error']}")
    if result.get("note"):
        print(f"  Note: {result['note']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Check CBB Edge production uptime")
    parser.add_argument(
        "--url",
        default="https://fantasy-app-production-5079.up.railway.app",
        help="Base URL to check (default: production Railway app)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON for machine parsing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)",
    )
    args = parser.parse_args()

    # Ensure URL doesn't end with slash
    base_url = args.url.rstrip("/")

    # Critical endpoints to check
    endpoints = [
        "/health",
        "/health/db",
        "/api/fantasy/roster",  # May fail without auth, but checks routing
    ]

    results = []
    all_ok = True

    for path in endpoints:
        result = check_endpoint(base_url, path, args.timeout)
        results.append(result)

        if not result["data_ok"]:
            all_ok = False

    if args.json:
        print(json.dumps({
            "url": base_url,
            "all_ok": all_ok,
            "results": results,
        }))
    else:
        print(f"CBB Edge Uptime Check: {base_url}")
        print("=" * 60)
        for result in results:
            print_result(result)

        if all_ok:
            print("All endpoints passed ✓")
        else:
            print("Some endpoints failed ✗")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
