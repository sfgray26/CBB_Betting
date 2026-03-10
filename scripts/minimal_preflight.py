#!/usr/bin/env python3
"""
Simplified preflight check for Railway - just the essentials.
"""

import sys

def minimal_check():
    """Minimal check that doesn't fail on optional deps."""
    print("🔍 Minimal preflight check...")
    
    # Only check critical deps
    critical = [
        "fastapi",
        "uvicorn", 
        "sqlalchemy",
        "pydantic",
        "numpy",
    ]
    
    missing = []
    for mod in critical:
        try:
            __import__(mod)
            print(f"  ✅ {mod}")
        except ImportError:
            print(f"  ❌ {mod}")
            missing.append(mod)
    
    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        return False
    
    print("\n✅ Critical deps OK")
    return True

if __name__ == "__main__":
    if not minimal_check():
        sys.exit(1)
    print("🚀 Starting...")
    sys.exit(0)
