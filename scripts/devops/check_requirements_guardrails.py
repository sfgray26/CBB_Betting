#!/usr/bin/env python3
"""
Fail-fast dependency guardrails for known Railway resolver regressions.

This check is intentionally narrow: it only enforces minimum versions for the
packages that have repeatedly broken production deployments in this repo.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


class RequirementSpec(NamedTuple):
    name: str
    operator: str
    version: str
    raw: str


MIN_PINS: dict[str, tuple[int, int, int]] = {
    "anyio": (4, 5, 0),
    "pydantic": (2, 7, 0),
    "pydantic-settings": (2, 5, 2),
}

REQ_RE = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(==|!=|>=|<=|>|<|~=)\s*([A-Za-z0-9_.+-]+)\s*$"
)


def parse_version(v: str) -> tuple[int, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", v)]
    nums += [0, 0, 0]
    return nums[0], nums[1], nums[2]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def load_requirements(path: Path) -> dict[str, RequirementSpec]:
    out: dict[str, RequirementSpec] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--"):
            # Keep this script simple and deterministic.
            continue
        # Strip trailing inline comments.
        line = line.split(" #", 1)[0].strip()
        m = REQ_RE.match(line)
        if not m:
            continue
        pkg = normalize_name(m.group(1))
        out[pkg] = RequirementSpec(
            name=pkg,
            operator=m.group(2),
            version=m.group(3),
            raw=raw_line,
        )
    return out


def check_min_pins(specs: dict[str, RequirementSpec]) -> list[str]:
    errors: list[str] = []
    for pkg, min_v in MIN_PINS.items():
        spec = specs.get(pkg)
        if not spec:
            continue
        if spec.operator != "==":
            # Pin checks are only deterministic for exact pins.
            continue
        if parse_version(spec.version) < min_v:
            min_s = ".".join(str(x) for x in min_v)
            errors.append(
                f"{pkg} pinned too low: {spec.raw.strip()} (minimum safe pin is {pkg}=={min_s})"
            )
    return errors


def check_known_pair_conflicts(specs: dict[str, RequirementSpec]) -> list[str]:
    errors: list[str] = []

    fastapi_mcp = specs.get("fastapi-mcp")
    pyd_settings = specs.get("pydantic-settings")
    pydantic = specs.get("pydantic")
    anyio = specs.get("anyio")

    if fastapi_mcp and pyd_settings and pyd_settings.operator == "==":
        if parse_version(pyd_settings.version) < MIN_PINS["pydantic-settings"]:
            errors.append(
                "fastapi-mcp is present but pydantic-settings is pinned below 2.5.2"
            )

    if pyd_settings and pyd_settings.operator == "==" and pydantic and pydantic.operator == "==":
        if parse_version(pyd_settings.version) >= MIN_PINS["pydantic-settings"] and parse_version(
            pydantic.version
        ) < MIN_PINS["pydantic"]:
            errors.append(
                "pydantic-settings>=2.5.2 with pydantic<2.7.0 is a known resolver conflict"
            )

    if anyio and anyio.operator == "==" and fastapi_mcp:
        if parse_version(anyio.version) < MIN_PINS["anyio"]:
            errors.append(
                "anyio pinned below 4.5.0 with fastapi-mcp present is a known resolver conflict"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate known-safe dependency pins for Railway deploys.")
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Path to requirements file (default: requirements.txt)",
    )
    args = parser.parse_args()

    req_path = Path(args.requirements).resolve()
    if not req_path.exists():
        print(f"[FAIL] requirements file not found: {req_path}")
        return 2

    specs = load_requirements(req_path)
    errors = []
    errors.extend(check_min_pins(specs))
    errors.extend(check_known_pair_conflicts(specs))

    if errors:
        print("[FAIL] dependency guardrails failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("[PASS] dependency guardrails passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

