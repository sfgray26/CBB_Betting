"""
Check if BDL API provides whip for pitchers
"""
import sys
sys.path.insert(0, '.')

with open('pitching_test_results.txt', 'w') as f:
    try:
        f.write("Checking BDL API for pitching stats (whip)...\n")
        from backend.services.balldontlie import BallDontLieClient

        bdl = BallDontLieClient()
        stats = bdl.get_mlb_stats(per_page=100)

        f.write(f"Got {len(stats)} stats\n\n")

        # Find pitchers (players with IP but no OBP)
        pitchers = [s for s in stats if s.ip is not None and s.obp is None]
        f.write(f"Found {len(pitchers)} pitchers\n\n")

        # Show first 10 pitchers
        for i, stat in enumerate(pitchers[:10]):
            data = stat.model_dump()
            f.write(f"Pitcher {i+1}:\n")
            f.write(f"  ip: {data.get('ip')}\n")
            f.write(f"  h_allowed: {data.get('h_allowed')}\n")
            f.write(f"  bb_allowed: {data.get('bb_allowed')}\n")
            f.write(f"  whip: {data.get('whip')}\n")
            f.write(f"  era: {data.get('era')}\n")
            f.write("\n")

        f.write(f"\nSummary: Out of {len(stats)} total stats, {len(pitchers)} are pitchers\n")
        f.write(f"Pitchers with whip: {sum(1 for p in pitchers if p.whip is not None)}\n")
        f.write(f"Pitchers with era: {sum(1 for p in pitchers if p.era is not None)}\n")

        f.write("\nTest completed successfully\n")

    except Exception as e:
        f.write(f"Error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        sys.exit(1)