"""
Railway BDL API test - writes results to file
"""
import sys
sys.path.insert(0, '.')

with open('bdl_test_results.txt', 'w') as f:
    try:
        f.write("Starting BDL API test...\n")
        from backend.services.balldontlie import BallDontLieClient

        f.write("Creating BDL client...\n")
        bdl = BallDontLieClient()

        f.write("Fetching stats...\n")
        stats = bdl.get_mlb_stats(per_page=3)

        f.write(f"Got {len(stats)} stats\n\n")

        for i, stat in enumerate(stats):
            data = stat.model_dump()
            f.write(f"Stat {i+1}:\n")
            f.write(f"  obp: {data.get('obp')}\n")
            f.write(f"  slg: {data.get('slg')}\n")
            f.write(f"  ops: {data.get('ops')}\n")
            f.write(f"  bb_allowed: {data.get('bb_allowed')}\n")
            f.write(f"  h_allowed: {data.get('h_allowed')}\n")
            f.write(f"  whip: {data.get('whip')}\n")
            f.write("\n")

        f.write("Test completed successfully\n")

    except Exception as e:
        f.write(f"Error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        sys.exit(1)