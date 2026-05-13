"""
Verify Christopher Sanchez fix after yahoo_id_sync job.

This script checks if Christopher Sanchez has been added to the database
and now has proper projection data instead of draft board fallback.
"""
import sys
import os
import requests
import time

# Railway API configuration
RAILWAY_URL = "https://fantasy-app-production-5079.up.railway.app"
ADMIN_API_KEY = "j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg"

def check_waiver_api():
    """Check the waiver API to see if Christopher Sanchez appears with real data."""
    print("Checking waiver API for Christopher Sanchez...")

    try:
        response = requests.get(
            f"{RAILWAY_URL}/api/fantasy/waiver",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            top_available = data.get("top_available", [])

            # Look for Christopher Sanchez in the results
            sanchez_found = False
            for player in top_available:
                name = player.get("name", "")
                if "sanchez" in name.lower() and "christopher" in name.lower():
                    sanchez_found = True
                    print(f"\n✅ FOUND Christopher Sanchez in waiver wire:")
                    print(f"   Name: {player.get('name')}")
                    print(f"   Need Score: {player.get('need_score', 'N/A')}")
                    print(f"   Ownership: {player.get('percent_owned', 'N/A')}%")
                    print(f"   Team: {player.get('team', 'N/A')}")
                    print(f"   Positions: {player.get('positions', [])}")

                    # Check if he has real data (not draft board fallback)
                    if player.get('need_score', 0) > 0:
                        print(f"   [OK] Has need_score > 0 - REAL PROJECTION DATA")
                    else:
                        print(f"   [PROBLEM] need_score = 0 - STILL USING FALLBACK")
                    break

            if not sanchez_found:
                print("\n[X] Christopher Sanchez NOT found in top available players")
                print("   (May be deeper in results or not added yet)")

        else:
            print(f"[X] API returned status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[X] Error checking waiver API: {e}")

def main():
    print("=" * 80)
    print("VERIFYING CHRISTOPHER SANCHEZ FIX")
    print("=" * 80)
    print("\nThe yahoo_id_sync job has been triggered and is now running in the background.")
    print("This typically takes 3+ minutes to complete.")
    print("\nThis script will check the waiver API to see if Christopher Sanchez")
    print("appears with real projection data after the job completes.\n")

    # Check immediately
    check_waiver_api()

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("1. Wait 3-5 minutes for yahoo_id_sync job to complete")
    print("2. Run this script again to verify the fix:")
    print("   railway run --environment production python scripts/verify_sanchez_fix.py")
    print("3. If Christopher Sanchez still has need_score=0, the job may need to be")
    print("   rerun or there may be a different issue.")
    print("=" * 80)

if __name__ == "__main__":
    main()
