"""Diagnose why downstream pipeline tables are empty."""
import os
from sqlalchemy import create_engine, text

e = create_engine(os.environ["DATABASE_URL"])
with e.connect() as c:
    # 1. Daily snapshot details
    print("=== DAILY SNAPSHOTS ===")
    rows = c.execute(text(
        "SELECT as_of_date, pipeline_health, pipeline_jobs_run, "
        "n_players_scored, n_simulation_records, health_reasons, summary "
        "FROM daily_snapshots ORDER BY as_of_date DESC LIMIT 5"
    )).fetchall()
    for r in rows:
        print(f"  {r.as_of_date}: status={r.pipeline_health}, jobs={r.pipeline_jobs_run}")
        print(f"    scored={r.n_players_scored}, sims={r.n_simulation_records}")
        print(f"    reasons={r.health_reasons}")
        print(f"    summary={r.summary}")
    if not rows:
        print("  (no snapshots)")

    # 2. Sample raw data - check columns available
    print("\n=== mlb_player_stats COLUMNS ===")
    cols = c.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'mlb_player_stats' ORDER BY ordinal_position"
    )).fetchall()
    for col in cols:
        print(f"  {col.column_name}")

    # 3. Top players by games (use bdl_player_id since no player_name col)
    print("\n=== TOP 10 PLAYERS BY GAMES ===")
    players = c.execute(text(
        "SELECT bdl_player_id, COUNT(*) as games, "
        "MIN(game_date) as first, MAX(game_date) as last "
        "FROM mlb_player_stats GROUP BY bdl_player_id "
        "ORDER BY COUNT(*) DESC LIMIT 10"
    )).fetchall()
    for p in players:
        print(f"  BDL#{p.bdl_player_id}: {p.games} games ({p.first} to {p.last})")

    # 4. Check what stat columns have non-null values
    print("\n=== NON-NULL STAT COVERAGE (sample of 821 rows) ===")
    stats = c.execute(text("""
        SELECT
            COUNT(*) as total,
            COUNT(ab) as has_ab,
            COUNT(hits) as has_hits,
            COUNT(home_runs) as has_hr,
            COUNT(innings_pitched) as has_ip,
            COUNT(earned_runs) as has_er,
            COUNT(strikeouts_pit) as has_k_pit,
            COUNT(strikeouts_bat) as has_k_bat
        FROM mlb_player_stats
    """)).fetchone()
    print(f"  total={stats.total}")
    print(f"  has_ab={stats.has_ab}, has_hits={stats.has_hits}, has_hr={stats.has_hr}")
    print(f"  has_ip={stats.has_ip}, has_er={stats.has_er}, has_k_pit={stats.has_k_pit}")
    print(f"  has_k_bat={stats.has_k_bat}")

    # 5. Check mlb_game_log  
    print("\n=== mlb_game_log SAMPLE ===")
    games = c.execute(text(
        "SELECT game_date, COUNT(*) as count "
        "FROM mlb_game_log GROUP BY game_date ORDER BY game_date DESC LIMIT 5"
    )).fetchall()
    for g in games:
        print(f"  {g.game_date}: {g.count} games")

    # 6. Check player_rolling_stats table (should be empty but verify structure)
    print("\n=== player_rolling_stats COLUMNS ===")
    cols2 = c.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'player_rolling_stats' ORDER BY ordinal_position"
    )).fetchall()
    for col in cols2:
        print(f"  {col.column_name}")
