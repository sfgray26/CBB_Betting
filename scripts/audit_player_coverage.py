"""
Audit projection coverage for top Yahoo free agents.

Usage:
  railway run python scripts/audit_player_coverage.py --count 200
  python scripts/audit_player_coverage.py --count 50
"""
import argparse
import difflib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def run_audit(top_n: int = 200):
    from backend.database import SessionLocal
    from backend.models import PlayerProjection, PlayerIDMapping, PlayerIdentity
    from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
    from backend.fantasy_baseball.id_resolution_service import _normalize_name

    print(f"\n{'='*60}")
    print(f"PLAYER COVERAGE AUDIT — Top {top_n} Free Agents")
    print(f"{'='*60}\n")

    client = YahooFantasyClient()
    db = SessionLocal()

    try:
        proj_map = {
            _normalize_name(r.player_name or ""): r
            for r in db.query(PlayerProjection).filter(
                PlayerProjection.cat_scores.isnot(None)
            ).all()
        }
        mapping_yahoo_ids = {
            r.yahoo_id for r in db.query(PlayerIDMapping).filter(
                PlayerIDMapping.yahoo_id.isnot(None)
            ).all()
        }
        identity_names = {
            r.normalized_name for r in db.query(PlayerIdentity).all()
        }

        players = []
        for start in range(0, top_n, 25):
            batch = client.get_free_agents(start=start, count=25)
            players.extend(batch)
            if len(players) >= top_n:
                break
        players = players[:top_n]

        results = []
        for p in players:
            name = p.get("name", "")
            norm = _normalize_name(name)
            yahoo_id = p.get("player_key", "").split(".p.")[-1] if ".p." in p.get("player_key", "") else None

            has_fangraphs = norm in proj_map
            # Also check fuzzy match (mirrors the fallback in get_or_create_projection)
            if not has_fangraphs:
                fuzzy = difflib.get_close_matches(norm, proj_map.keys(), n=1, cutoff=0.85)
                has_fangraphs = bool(fuzzy)

            has_yahoo_mapping = yahoo_id in mapping_yahoo_ids if yahoo_id else False
            has_identity = norm in identity_names
            owned_pct = p.get("percent_owned", 0.0) or 0.0

            results.append({
                "name": name,
                "owned_pct": owned_pct,
                "has_fangraphs": has_fangraphs,
                "has_yahoo_mapping": has_yahoo_mapping,
                "has_identity": has_identity,
            })

        results.sort(key=lambda x: x["owned_pct"], reverse=True)

        total = len(results)
        with_fg = sum(1 for r in results if r["has_fangraphs"])
        with_mapping = sum(1 for r in results if r["has_yahoo_mapping"])
        with_identity = sum(1 for r in results if r["has_identity"])

        print(f"Total players checked:      {total}")
        print(f"FanGraphs RoS cat_scores:   {with_fg}/{total} ({100*with_fg//max(total,1)}%)")
        print(f"Yahoo ID mapped:             {with_mapping}/{total} ({100*with_mapping//max(total,1)}%)")
        print(f"In player_identities:        {with_identity}/{total} ({100*with_identity//max(total,1)}%)")

        missing = [r for r in results if not r["has_fangraphs"]][:30]
        print(f"\n--- PLAYERS WITHOUT FANGRAPHS DATA (top {min(30, len(missing))}) ---")
        if not missing:
            print("All top players have FanGraphs data!")
        else:
            for r in missing:
                flags = []
                if not r["has_yahoo_mapping"]:
                    flags.append("NO_MAPPING")
                if not r["has_identity"]:
                    flags.append("NO_IDENTITY")
                print(f"  {r['name']:30s} {r['owned_pct']:5.1f}% owned  {' '.join(flags)}")

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    run_audit(args.count)
