# generate_stat_contract.py
"""
Generate fantasy_stat_contract.json for a given Yahoo league.

Usage:
    python -m apps.workers.contracts.generate_stat_contract \
        --league-key mlb.l.12345 \
        --season 2026 \
        --out packages/contracts/fantasy_stat_contract.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .contract_builder import (
    ContractBuildError,
    UnknownYahooStatError,
    build_contract,
)
from .yahoo_league_client import YahooLeagueClient, YahooLeagueClientError


logger = logging.getLogger("generate_stat_contract")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-key", required=True,
                        help="Yahoo league key (e.g. mlb.l.12345)")
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--league-id", default=None,
                        help="Internal league id. Defaults to yahoo_{season}_{key}.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output path for fantasy_stat_contract.json")
    parser.add_argument("--fixture", type=Path, default=None,
                        help="Optional local Yahoo settings JSON for offline runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print, but do not write the file.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else (
            logging.INFO if args.verbose == 1 else logging.WARNING
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    league_id = args.league_id or f"yahoo_{args.season}_{args.league_key}"

    try:
        if args.fixture:
            logger.info("Loading Yahoo settings from fixture: %s", args.fixture)
            yahoo_settings = json.loads(args.fixture.read_text())
        else:
            client = YahooLeagueClient.from_env()
            yahoo_settings = client.fetch_league_settings(args.league_key)
    except YahooLeagueClientError as exc:
        logger.error("Failed to fetch Yahoo settings: %s", exc)
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load fixture: %s", exc)
        return 2

    try:
        contract = build_contract(
            yahoo_settings=yahoo_settings,
            league_id=league_id,
            season=args.season,
        )
    except UnknownYahooStatError as exc:
        logger.error("Unknown Yahoo stat encountered: %s", exc)
        logger.error("Update master_stat_registry.py and re-run.")
        return 3
    except ContractBuildError as exc:
        logger.error("Contract build failed: %s", exc)
        return 4

    payload = contract.model_dump(by_alias=True, mode="json")
    serialized = json.dumps(payload, indent=2, sort_keys=False) + "\n"

    if args.dry_run:
        sys.stdout.write(serialized)
        logger.info("Dry run — file not written.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(serialized)
    tmp.replace(args.out)
    logger.info("Wrote %d bytes to %s", len(serialized), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())