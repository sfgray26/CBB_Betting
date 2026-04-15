#!/usr/bin/env python
"""
DevOps Railway log filter — tail logs and grep for relevant patterns.

This is a convenience wrapper because Gemini is permitted to run Python scripts
but complex bash pipelines can be brittle across shells.

Usage (Railway production):
    railway run python scripts/devops/railway_logs_filter.py --job player_id_mapping
    railway run python scripts/devops/railway_logs_filter.py --service Fantasy-App --lines 50

Note: This script invokes `railway logs` locally, so `railway run` is fine here.
DB scripts should use `railway ssh` instead.

Note: This script assumes it is running INSIDE the Railway container (via railway run).
If run locally, it reads from stdin or a provided log file.
"""
import argparse
import re
import subprocess
import sys


def tail_railway_logs(service: str | None, lines: int, patterns: list[str]) -> None:
    cmd = ["railway", "logs", "--lines", str(lines)]
    if service:
        cmd.extend(["--service", service])

    compiled = [re.compile(p, re.IGNORECASE) for p in patterns] if patterns else []

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if not compiled or any(p.search(line) for p in compiled):
                print(line, end="")
    except KeyboardInterrupt:
        proc.terminate()
        print("\n[Interrupted]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail and filter Railway logs")
    parser.add_argument("--service", help="Railway service name")
    parser.add_argument("--lines", type=int, default=30, help="Number of log lines to fetch")
    parser.add_argument("--job", help="Quick filter for a known job name (e.g., player_id_mapping)")
    args = parser.parse_args()

    patterns = []
    if args.job:
        patterns.append(args.job.replace("_", "_"))
        patterns.append(args.job)

    tail_railway_logs(args.service, args.lines, patterns)


if __name__ == "__main__":
    main()
