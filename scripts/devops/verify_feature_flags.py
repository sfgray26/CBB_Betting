#!/usr/bin/env python3
"""
Verify production feature-flag coverage for fantasy pipelines.

Run locally against a DB URL, or in Railway via:
  railway ssh --service <backend-service> python scripts/devops/verify_feature_flags.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import text

from backend.models import SessionLocal


REQUIRED_TRUE_FLAGS = [
    "CANONICAL_PROJECTION_V1",
    "market_signals_enabled",
    "feature_matchup_enabled",
    "opportunity_enabled",
]


def normalize_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def main() -> int:
    db = SessionLocal()
    try:
        rows = db.execute(text("SELECT flag_name, enabled FROM feature_flags")).fetchall()
    finally:
        db.close()

    flags = {str(name): normalize_bool(enabled) for name, enabled in rows}
    missing = [name for name in REQUIRED_TRUE_FLAGS if name not in flags]
    disabled = [name for name in REQUIRED_TRUE_FLAGS if name in flags and not flags[name]]

    print("Feature flag status:")
    for name in REQUIRED_TRUE_FLAGS:
        if name not in flags:
            print(f"  - {name}: MISSING")
        else:
            print(f"  - {name}: {'ENABLED' if flags[name] else 'DISABLED'}")

    season = os.getenv("CURRENT_MLB_SEASON")
    if season:
        print(f"CURRENT_MLB_SEASON={season}")
    else:
        print("CURRENT_MLB_SEASON not set (runtime default may apply)")

    if missing or disabled:
        if missing:
            print(f"[FAIL] missing flags: {', '.join(missing)}")
        if disabled:
            print(f"[FAIL] disabled flags: {', '.join(disabled)}")
        return 1

    print("[PASS] required feature flags are present and enabled")
    return 0


if __name__ == "__main__":
    sys.exit(main())

