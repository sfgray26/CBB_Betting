"""
Test script to verify weather API functionality before EPIC 5 implementation.

This script tests the weather_fetcher.py module for 4 major parks.
If all tests pass, weather API is ready for PR 5.2 implementation.
"""
import os
import sys
from datetime import date

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.fantasy_baseball.weather_fetcher import get_weather_fetcher, validate_weather_api_key

def main():
    print("=" * 70)
    print("WEATHER API VERIFICATION TEST")
    print("=" * 70)
    print()

    # Step 1: Validate API key
    print("Step 1: Validating OpenWeather API key...")
    is_valid, message = validate_weather_api_key()
    if is_valid:
        print(f"[PASS] API key valid: {message}")
    else:
        print(f"[FAIL] API key issue: {message}")
        print("[WARN] Will test with fallback mode (seasonal estimates)")
    print()

    # Step 2: Test weather fetch for 4 major parks
    parks = ["Yankee Stadium", "Fenway Park", "Dodger Stadium", "Wrigley Field"]
    today = date.today()

    print("Step 2: Testing weather fetch for today's games...")
    print(f"Test date: {today}")
    print()

    fetcher = get_weather_fetcher()
    all_passed = True

    for park in parks:
        try:
            import datetime as dt
            game_time = dt.datetime.combine(today, dt.datetime.min.time()) + dt.timedelta(hours=19)

            weather = fetcher.get_game_weather(park, game_time)

            # Verify we got data back
            if weather.temperature is not None and weather.venue == park:
                print(f"[PASS] {park}: {weather.temperature}F, wind {weather.wind_speed}mph {weather.wind_direction}")
                print(f"       HR factor: {weather.hr_factor}, Hitter score: {weather.hitter_friendly_score:.1f}/10")
                if weather.fallback_mode:
                    print(f"       [WARN] Using fallback mode (seasonal estimates)")
                else:
                    print(f"       [PASS] Live API data")
            else:
                print(f"[FAIL] {park}: Invalid weather data returned")
                all_passed = False
        except Exception as e:
            print(f"[FAIL] {park}: ERROR - {e}")
            all_passed = False
        print()

    # Summary
    print("=" * 70)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED")
        print()
        print("Weather API is ready for EPIC 5 implementation:")
        print("- PR 5.2 (Data Collection) can proceed")
        print("- PR 5.3 (Basic Matchup Score) can use weather data")
        print()
        print("Note: If using fallback mode, consider adding OPENWEATHER_API_KEY")
        print("to Railway environment variables for live weather data.")
    else:
        print("[FAILURE] SOME TESTS FAILED")
        print()
        print("Weather API has issues. Fix required before EPIC 5 PR 5.2:")
        print("- Check OPENWEATHER_API_KEY in Railway environment variables")
        print("- Verify internet connectivity")
        print("- Review weather_fetcher.py error logs above")
    print("=" * 70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
