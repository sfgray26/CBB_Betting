#!/usr/bin/env python3
"""
doc_retriever.py — On-demand documentation fetcher for Claude Code.

Usage:
    python scripts/doc_retriever.py <relative-path>
    python scripts/doc_retriever.py HANDOFF.md
    python scripts/doc_retriever.py reports/2026-04-15-comprehensive-application-audit.md
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fetch(path_str: str) -> str:
    target = PROJECT_ROOT / path_str
    # Security: disallow traversal outside project root
    try:
        target.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        print(f"ERROR: Path must be inside project root. Got: {path_str}", file=sys.stderr)
        sys.exit(1)

    if not target.exists():
        print(f"ERROR: File not found: {target}", file=sys.stderr)
        sys.exit(1)

    return target.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch project documentation on demand.")
    parser.add_argument("path", help="Relative path to the document (e.g., IDENTITY.md)")
    args = parser.parse_args()
    # Ensure UTF-8 output on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(fetch(args.path))


if __name__ == "__main__":
    main()
