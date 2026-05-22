#!/usr/bin/env python3
"""
Guarded Railway deploy for backend + frontend.

This script hardens the deploy path against the recent failure modes:
1) wrong snapshot context / missing repo files
2) root-directory mismatches in monorepo deploys
3) known dependency resolver conflicts
4) missing runtime feature-flag coverage
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_BACKEND_SERVICE = "a302e57f-16c4-4c3e-9000-e8a588468d7f"
DEFAULT_FRONTEND_SERVICE = "ccecd21a-ecc9-4ea1-a445-9a4dc49a8ee8"

REQUIRED_PATHS = [
    ".git",
    "railway.json",
    "Dockerfile",
    "requirements.txt",
    "frontend/railway.json",
    "frontend/Dockerfile",
    "frontend/package.json",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def railway_cmd() -> list[str]:
    if shutil.which("railway"):
        return ["railway"]
    return ["npx", "@railway/cli"]


def run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(f"\n[run] ({cwd}) {' '.join(cmd)}")
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def assert_repo_layout(root: Path) -> None:
    missing = [p for p in REQUIRED_PATHS if not (root / p).exists()]
    if missing:
        print("[FAIL] deploy context check failed.")
        print("Missing required paths from repo root:")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(2)
    print("[PASS] deploy context check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded deploy for Railway backend + frontend")
    parser.add_argument(
        "--backend-service",
        default=DEFAULT_BACKEND_SERVICE,
        help="Railway backend service name/ID",
    )
    parser.add_argument(
        "--frontend-service",
        default=DEFAULT_FRONTEND_SERVICE,
        help="Railway frontend service name/ID",
    )
    parser.add_argument(
        "--environment",
        default="production",
        help="Railway environment name/ID (default: production)",
    )
    parser.add_argument(
        "--skip-frontend-checks",
        action="store_true",
        help="Skip frontend type/lint checks",
    )
    parser.add_argument(
        "--skip-feature-flag-check",
        action="store_true",
        help="Skip post-deploy feature-flag verification over railway ssh",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print checks/commands without executing deploy commands",
    )
    args = parser.parse_args()

    root = repo_root()
    assert_repo_layout(root)

    # 1) Fail-fast on known dependency conflicts.
    run(
        [sys.executable, "scripts/devops/check_requirements_guardrails.py"],
        cwd=root,
        dry_run=args.dry_run,
    )

    # 2) Frontend checks before deploy.
    if not args.skip_frontend_checks:
        run(["npm", "run", "lint"], cwd=root / "frontend", dry_run=args.dry_run)
        run(["npx", "tsc", "--noEmit"], cwd=root / "frontend", dry_run=args.dry_run)
    else:
        print("[skip] frontend checks")

    cli = railway_cmd()

    # 3) Deploy backend and frontend from the same validated repo root.
    run(
        [
            *cli,
            "up",
            "--service",
            args.backend_service,
            "--environment",
            args.environment,
            "--ci",
        ],
        cwd=root,
        dry_run=args.dry_run,
    )
    run(
        [
            *cli,
            "up",
            "--service",
            args.frontend_service,
            "--environment",
            args.environment,
            "--ci",
        ],
        cwd=root,
        dry_run=args.dry_run,
    )

    # 4) Validate critical runtime feature flags on backend container.
    if not args.skip_feature_flag_check:
        run(
            [
                *cli,
                "ssh",
                "--service",
                args.backend_service,
                "--environment",
                args.environment,
                "python",
                "scripts/devops/verify_feature_flags.py",
            ],
            cwd=root,
            dry_run=args.dry_run,
        )
    else:
        print("[skip] feature-flag check")

    print("\n[DONE] guarded deployment completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
