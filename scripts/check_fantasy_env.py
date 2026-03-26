"""
Verify environment configuration for fantasy baseball features.
Run this to check if all required variables are set.
"""

import os
import sys


def check_var(name: str, required: bool = True, hint: str = "") -> bool:
    """Check if an environment variable is set."""
    value = os.getenv(name)
    if value:
        # Mask sensitive values
        display = value[:8] + "..." if len(value) > 10 else value
        print(f"  ✅ {name}: {display}")
        return True
    else:
        if required:
            print(f"  ❌ {name}: NOT SET {hint}")
        else:
            print(f"  ⚠️  {name}: not set (optional) {hint}")
        return not required


def main():
    print("=" * 50)
    print("Fantasy Baseball Environment Check")
    print("=" * 50)
    
    all_ok = True
    
    # Core fantasy baseball
    print("\n📊 Core Fantasy Baseball:")
    all_ok &= check_var("YAHOO_CLIENT_ID", hint="(from Yahoo Developer)")
    all_ok &= check_var("YAHOO_CLIENT_SECRET", hint="(from Yahoo Developer)")
    
    # Weather
    print("\n🌤️  Weather Integration:")
    has_weather = check_var(
        "OPENWEATHER_API_KEY", 
        required=False,
        hint="(get at openweathermap.org - free tier sufficient)"
    )
    if not has_weather:
        print("     ↳ Without this, system uses seasonal temperature estimates")
    
    # Odds API
    print("\n💰 Odds API:")
    all_ok &= check_var("THE_ODDS_API_KEY", hint="(from the-odds-api.com)")
    
    # Database
    print("\n🗄️  Database:")
    all_ok &= check_var("DATABASE_URL", hint="(Railway provides this automatically)")
    
    # Decision tracking
    print("\n📈 Decision Tracking:")
    print("  ℹ️  Decisions stored to: data/decisions.jsonl")
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("✅ All required variables set!")
        if not has_weather:
            print("⚠️  Consider adding OPENWEATHER_API_KEY for live weather data")
    else:
        print("❌ Some required variables missing - see above")
        sys.exit(1)
    
    # Test weather if available
    if has_weather:
        print("\n🌤️  Testing weather fetch...")
        try:
            from backend.fantasy_baseball.weather_fetcher import get_weather_fetcher
            from datetime import datetime
            
            fetcher = get_weather_fetcher()
            
            # Test with Wrigley Field
            weather = fetcher.get_game_weather(
                venue="Wrigley Field",
                game_time=datetime.now()
            )
            
            print(f"  ✅ Weather fetch working!")
            print(f"     Wrigley Field: {weather.summary}")
            print(f"     HR factor: {weather.hr_factor:.2f}x")
            
            # Test park analyzer
            from backend.fantasy_baseball.park_weather import get_park_analyzer
            analyzer = get_park_analyzer()
            analysis = analyzer.analyze_game("Wrigley Field", weather)
            
            print(f"     Park analysis: {analysis['description']}")
            
        except Exception as e:
            print(f"  ❌ Weather test failed: {e}")
            sys.exit(1)
    
    print("\n✅ Environment ready for fantasy baseball!")


if __name__ == "__main__":
    main()
