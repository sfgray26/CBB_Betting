#!/usr/bin/env python3
"""
Pre-flight check script for Railway deployment.
Verifies all critical dependencies are installed before starting the app.
"""

import sys

def check_dependencies():
    """Check that all required packages are installed."""
    errors = []
    warnings = []
    
    # Critical dependencies that must be present
    critical = [
        ("sqlalchemy", "SQLAlchemy"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("requests", "Requests"),
    ]
    
    # Optional dependencies
    optional = [
        ("duckduckgo_search", "DuckDuckGo Search"),
        ("apscheduler", "APScheduler"),
        ("twilio", "Twilio"),
        ("sendgrid", "SendGrid"),
    ]
    
    print("🔍 Checking critical dependencies...")
    for module, name in critical:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            errors.append(f"❌ {name} ({module}) - {e}")
    
    print("\n🔍 Checking optional dependencies...")
    for module, name in optional:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            warnings.append(f"⚠️  {name} ({module}) - optional, will use fallback")
    
    # Report results
    print("\n" + "="*60)
    if errors:
        print("❌ CRITICAL DEPENDENCIES MISSING:")
        for e in errors:
            print(f"   {e}")
        print("\n🔧 Fix: Run 'pip install -r requirements.txt'")
        return False
    
    if warnings:
        print("⚠️  Optional dependencies missing:")
        for w in warnings:
            print(f"   {w}")
        print("\nℹ️  App will start with reduced functionality")
    
    print("\n✅ All critical dependencies present!")
    print("="*60)
    return True


def check_env_vars():
    """Check that required environment variables are set."""
    import os
    
    print("\n🔍 Checking environment variables...")
    
    required = [
        ("DATABASE_URL", "PostgreSQL connection string"),
    ]
    
    optional = [
        ("THE_ODDS_API_KEY", "Odds data"),
        ("KENPOM_API_KEY", "KenPom ratings"),
        ("DISCORD_BOT_TOKEN", "Discord notifications"),
    ]
    
    missing_required = []
    
    for var, description in required:
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"   ✅ {var}")
        else:
            missing_required.append(var)
            print(f"   ❌ {var} ({description})")
    
    for var, description in optional:
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"   ✅ {var} ({description})")
        else:
            print(f"   ⚠️  {var} ({description}) - optional")
    
    if missing_required:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    return True


if __name__ == "__main__":
    deps_ok = check_dependencies()
    env_ok = check_env_vars()
    
    if not deps_ok:
        sys.exit(1)
    
    print("\n🚀 Pre-flight checks passed. Starting application...\n")
    sys.exit(0)
