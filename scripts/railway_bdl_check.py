"""
Quick Railway BDL API investigation
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

print("Starting BDL investigation...", flush=True)

try:
    from backend.services.balldontlie import BallDontLieClient

    print("Creating BDL client...", flush=True)
    bdl = BallDontLieClient()

    print("Fetching stats...", flush=True)
    stats = bdl.get_mlb_stats(limit=3)

    print(f"Got {len(stats)} stats", flush=True)

    for i, stat in enumerate(stats):
        data = stat.model_dump()
        print(f"Stat {i+1}:", flush=True)
        print(f"  obp: {data.get('obp')}", flush=True)
        print(f"  slg: {data.get('slg')}", flush=True)
        print(f"  ops: {data.get('ops')}", flush=True)
        print(f"  walks_allowed: {data.get('bb_allowed')}", flush=True)
        print(f"  hits_allowed: {data.get('h_allowed')}", flush=True)
        print(f"  whip: {data.get('whip')}", flush=True)
        print(flush=True)

    print("BDL API check complete", flush=True)

except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)