import sys
import os
sys.path.insert(0, '.')

from backend.main import app

target = "/admin/pipeline/box-stats-health"
found = [r.path for r in app.routes if target in r.path]

if found:
    print(f"✅ Found: {found}")
else:
    print(f"❌ Route '{target}' not found!")
    print("Listing all /admin routes:")
    admin_routes = [r.path for r in app.routes if "/admin" in r.path]
    for r in sorted(admin_routes):
        print(f"  - {r}")
