"""
Backfill the entire MLB 2026 season from Opening Day to yesterday.

Stages (in order):
  1. BDL game_log — fetch games for every date, upsert to mlb_game_log
  2. BDL box stats — fetch player stats by game_id, upsert to mlb_player_stats
  3. statsapi supplement — patch NULL counting stats using MLB-StatsAPI
  4. Rolling windows — compute for each date that has raw stats
  5. Downstream — player_scores, momentum, simulation, etc.

Usage:
  python scripts/backfill_season.py                  # Full season
  python scripts/backfill_season.py --start 2026-04-01  # From Apr 1
  python scripts/backfill_season.py --stage game_log  # Just game_log
  python scripts/backfill_season.py --stage box_stats # Just box stats
  python scripts/backfill_season.py --stage statsapi  # Just statsapi supplement
  python scripts/backfill_season.py --stage pipeline  # Just downstream pipeline
"""
import argparse
import asyncio
import os
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env before any imports that need env vars
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: Set DATABASE_URL environment variable")
    sys.exit(1)

SEASON_OPEN = date(2026, 3, 27)  # First day games occurred (Opening Day games Mar 27)


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def date_range(start: date, end: date):
    """Yield each date from start to end inclusive."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------------------
# Stage 1: Game Log
# ---------------------------------------------------------------------------
def backfill_game_log(start: date, end: date):
    """Fetch BDL games for each date and upsert to mlb_game_log + mlb_team."""
    from backend.services.balldontlie import BallDontLieClient
    from zoneinfo import ZoneInfo

    bdl = BallDontLieClient()
    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    total_games = 0
    for d in date_range(start, end):
        date_str = d.isoformat()
        print(f"  game_log {date_str} ... ", end="", flush=True)
        try:
            games = bdl.get_mlb_games(date_str)
        except Exception as e:
            print(f"BDL error: {e}")
            continue

        if not games:
            print("0 games (off-day)")
            continue

        for game in games:
            # Upsert teams
            for team in (game.home_team, game.away_team):
                cur.execute("""
                    INSERT INTO mlb_team (team_id, abbreviation, name, display_name,
                        short_name, location, slug, league, division, ingested_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
                    ON CONFLICT (team_id) DO UPDATE SET
                        abbreviation=EXCLUDED.abbreviation, name=EXCLUDED.name,
                        display_name=EXCLUDED.display_name, short_name=EXCLUDED.short_name,
                        location=EXCLUDED.location, slug=EXCLUDED.slug,
                        league=EXCLUDED.league, division=EXCLUDED.division
                """, (team.id, team.abbreviation, team.name, team.display_name,
                      team.short_display_name, team.location, team.slug,
                      team.league, team.division))

            # Calculate ET date from game timestamp
            dt_utc = datetime.fromisoformat(game.date.replace("Z", "+00:00"))
            game_date_et = dt_utc.astimezone(ZoneInfo("America/New_York")).date()

            is_active = game.status in {"STATUS_FINAL", "STATUS_IN_PROGRESS"}
            import json
            payload = json.loads(game.model_dump_json())

            cur.execute("""
                INSERT INTO mlb_game_log (game_id, game_date, season, season_type, status,
                    home_team_id, away_team_id, home_runs, away_runs, home_hits, away_hits,
                    home_errors, away_errors, venue, attendance, period,
                    raw_payload, ingested_at, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
                ON CONFLICT (game_id) DO UPDATE SET
                    status=EXCLUDED.status, home_runs=EXCLUDED.home_runs,
                    away_runs=EXCLUDED.away_runs, home_hits=EXCLUDED.home_hits,
                    away_hits=EXCLUDED.away_hits, home_errors=EXCLUDED.home_errors,
                    away_errors=EXCLUDED.away_errors, attendance=EXCLUDED.attendance,
                    period=EXCLUDED.period, raw_payload=EXCLUDED.raw_payload,
                    updated_at=NOW()
            """, (
                game.id, game_date_et, game.season, game.season_type, game.status,
                game.home_team.id, game.away_team.id,
                game.home_team_data.runs if is_active else None,
                game.away_team_data.runs if is_active else None,
                game.home_team_data.hits if is_active else None,
                game.away_team_data.hits if is_active else None,
                game.home_team_data.errors if is_active else None,
                game.away_team_data.errors if is_active else None,
                game.venue, game.attendance if is_active else None, game.period,
                json.dumps(payload),
            ))
            total_games += 1

        conn.commit()
        print(f"{len(games)} games")
        time.sleep(0.15)  # BDL rate limit courtesy

    cur.close()
    conn.close()
    print(f"  Total: {total_games} games upserted")
    return total_games


# ---------------------------------------------------------------------------
# Stage 2: Box Stats
# ---------------------------------------------------------------------------
def backfill_box_stats(start: date, end: date):
    """Fetch BDL player stats by game_id and upsert to mlb_player_stats."""
    from backend.services.balldontlie import BallDontLieClient
    import json

    bdl = BallDontLieClient()
    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    # Get all game_ids for date range
    cur.execute("""
        SELECT game_id, game_date FROM mlb_game_log
        WHERE game_date BETWEEN %s AND %s AND status = 'STATUS_FINAL'
        ORDER BY game_date, game_id
    """, (start, end))
    game_rows = cur.fetchall()

    if not game_rows:
        print("  No final games in mlb_game_log for this range")
        conn.close()
        return 0

    game_date_map = {gid: gd for gid, gd in game_rows}
    all_game_ids = [gid for gid, _ in game_rows]

    print(f"  Found {len(all_game_ids)} final games across {start} to {end}")

    # Fetch stats in batches of game_ids (BDL paginates internally)
    total_upserted = 0
    batch_size = 25  # Don't send too many game_ids at once
    for i in range(0, len(all_game_ids), batch_size):
        batch = all_game_ids[i:i + batch_size]
        batch_dates = sorted(set(game_date_map[gid].isoformat() for gid in batch))
        print(f"  box_stats batch {i // batch_size + 1} ({len(batch)} games, "
              f"{batch_dates[0]} to {batch_dates[-1]}) ... ", end="", flush=True)

        try:
            stats = bdl.get_mlb_stats(game_ids=batch)
        except Exception as e:
            print(f"BDL error: {e}")
            continue

        if not stats:
            print("0 stats")
            continue

        batch_upserted = 0
        for stat in stats:
            if stat.bdl_player_id is None:
                continue

            game_date = game_date_map.get(stat.game_id, date.today())
            payload = json.loads(stat.model_dump_json())

            cur.execute("""
                INSERT INTO mlb_player_stats (
                    bdl_stat_id, bdl_player_id, game_id, game_date, season,
                    ab, runs, hits, doubles, triples, home_runs, rbi, walks,
                    strikeouts_bat, stolen_bases, caught_stealing,
                    avg, obp, slg, ops,
                    innings_pitched, hits_allowed, runs_allowed, earned_runs,
                    walks_allowed, strikeouts_pit, whip, era,
                    raw_payload, ingested_at
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,NOW()
                )
                ON CONFLICT ON CONSTRAINT _mps_player_game_uc DO UPDATE SET
                    bdl_stat_id=EXCLUDED.bdl_stat_id, season=EXCLUDED.season,
                    ab=EXCLUDED.ab, runs=EXCLUDED.runs, hits=EXCLUDED.hits,
                    doubles=EXCLUDED.doubles, triples=EXCLUDED.triples,
                    home_runs=EXCLUDED.home_runs, rbi=EXCLUDED.rbi,
                    walks=EXCLUDED.walks, strikeouts_bat=EXCLUDED.strikeouts_bat,
                    stolen_bases=EXCLUDED.stolen_bases,
                    caught_stealing=EXCLUDED.caught_stealing,
                    avg=EXCLUDED.avg, obp=EXCLUDED.obp, slg=EXCLUDED.slg, ops=EXCLUDED.ops,
                    innings_pitched=EXCLUDED.innings_pitched,
                    hits_allowed=EXCLUDED.hits_allowed,
                    runs_allowed=EXCLUDED.runs_allowed,
                    earned_runs=EXCLUDED.earned_runs,
                    walks_allowed=EXCLUDED.walks_allowed,
                    strikeouts_pit=EXCLUDED.strikeouts_pit,
                    whip=EXCLUDED.whip, era=EXCLUDED.era,
                    raw_payload=EXCLUDED.raw_payload
            """, (
                stat.id, stat.bdl_player_id, stat.game_id, game_date,
                stat.season if stat.season is not None else 2026,
                stat.ab, stat.r, stat.h, stat.double, stat.triple,
                stat.hr, stat.rbi, stat.bb, stat.so, stat.sb, stat.cs,
                stat.avg, stat.obp, stat.slg, stat.ops,
                stat.ip, stat.h_allowed, stat.r_allowed, stat.er,
                stat.bb_allowed, stat.k, stat.whip, stat.era,
                json.dumps(payload),
            ))
            batch_upserted += 1

        conn.commit()
        total_upserted += batch_upserted
        print(f"{batch_upserted} stats")
        time.sleep(0.2)

    cur.close()
    conn.close()
    print(f"  Total: {total_upserted} stat rows upserted")
    return total_upserted


# ---------------------------------------------------------------------------
# Stage 3: statsapi supplement (patch NULL counting stats)
# ---------------------------------------------------------------------------
def _norm(name: str) -> str:
    """Accent-strip and lowercase for fuzzy name matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", name.lower()) if unicodedata.category(c) != "Mn"
    )


def backfill_statsapi(start: date, end: date):
    """Patch NULL-ab rows using MLB Stats API boxscore data."""
    import statsapi

    conn = get_conn()
    cur = conn.cursor()

    # Find dates with NULL-ab hitters that need patching
    cur.execute("""
        SELECT DISTINCT game_date FROM mlb_player_stats
        WHERE season = 2026 AND game_date BETWEEN %s AND %s
        ORDER BY game_date
    """, (start, end))
    dates_in_db = [r[0] for r in cur.fetchall()]

    total_patched = 0
    for d in dates_in_db:
        # Get NULL-ab rows for this date
        cur.execute("""
            SELECT id, bdl_player_id,
                   raw_payload->'player'->>'full_name' AS name
            FROM mlb_player_stats
            WHERE game_date = %s AND ab IS NULL AND innings_pitched IS NULL
        """, (d,))
        null_rows = cur.fetchall()
        if not null_rows:
            # Also check pitchers with NULL strikeouts_pit
            cur.execute("""
                SELECT COUNT(*) FROM mlb_player_stats
                WHERE game_date = %s AND innings_pitched IS NOT NULL AND strikeouts_pit IS NULL
            """, (d,))
            null_pit = cur.fetchone()[0]
            if null_pit == 0:
                print(f"  statsapi {d}: no NULLs to patch")
                continue

        # Build name lookup for NULL rows
        cur.execute("""
            SELECT id, bdl_player_id,
                   raw_payload->'player'->>'full_name' AS name,
                   innings_pitched
            FROM mlb_player_stats
            WHERE game_date = %s AND (
                (ab IS NULL AND innings_pitched IS NULL) OR
                (innings_pitched IS NOT NULL AND strikeouts_pit IS NULL)
            )
        """, (d,))
        rows_to_patch = cur.fetchall()
        name_lookup = {}
        for row_id, bdl_id, name, ip in rows_to_patch:
            if name:
                key = _norm(name)
                name_lookup[key] = (row_id, bdl_id, ip is not None)

        # Fetch statsapi games for this date
        date_str = d.strftime("%m/%d/%Y")
        try:
            sched = statsapi.schedule(sportId=1, date=date_str)
        except Exception as e:
            print(f"  statsapi {d}: schedule error: {e}")
            continue

        day_patched = 0
        for game_info in sched:
            game_pk = game_info.get("game_id")
            if not game_pk:
                continue
            status = game_info.get("status", "")
            if status not in ("Final", "Game Over", "Completed Early"):
                continue
            try:
                box = statsapi.boxscore_data(game_pk)
            except Exception as e:
                print(f"  statsapi {d}: boxscore {game_pk} error: {e}")
                continue

            player_info = box.get("playerInfo", {})

            # Process batters (list of dicts with personId, ab, r, h, etc.)
            for side in ("away", "home"):
                for batter in box.get(f"{side}Batters", []):
                    person_id = batter.get("personId")
                    if not person_id:
                        continue
                    info = player_info.get(f"ID{person_id}", {})
                    full_name = info.get("fullName", "")
                    if not full_name:
                        continue
                    key = _norm(full_name)
                    if key not in name_lookup:
                        continue
                    row_id, bdl_id, is_pitcher = name_lookup[key]
                    if is_pitcher:
                        continue
                    try:
                        cur.execute("""
                            UPDATE mlb_player_stats SET
                                ab=%s, runs=%s, hits=%s, doubles=%s, triples=%s,
                                home_runs=%s, rbi=%s, walks=%s, strikeouts_bat=%s, stolen_bases=%s
                            WHERE id=%s
                        """, (
                            int(batter.get("ab", 0)), int(batter.get("r", 0)),
                            int(batter.get("h", 0)), int(batter.get("doubles", 0)),
                            int(batter.get("triples", 0)), int(batter.get("hr", 0)),
                            int(batter.get("rbi", 0)), int(batter.get("bb", 0)),
                            int(batter.get("k", 0)), int(batter.get("sb", 0)),
                            row_id,
                        ))
                        day_patched += 1
                        del name_lookup[key]
                    except (ValueError, TypeError):
                        pass

            # Process pitchers (list of dicts with personId, ip, h, er, etc.)
            for side in ("away", "home"):
                for pitcher in box.get(f"{side}Pitchers", []):
                    person_id = pitcher.get("personId")
                    if not person_id:
                        continue
                    info = player_info.get(f"ID{person_id}", {})
                    full_name = info.get("fullName", "")
                    if not full_name:
                        continue
                    key = _norm(full_name)
                    if key not in name_lookup:
                        continue
                    row_id, bdl_id, is_pitcher = name_lookup[key]
                    if not is_pitcher:
                        continue
                    try:
                        cur.execute("""
                            UPDATE mlb_player_stats SET
                                hits_allowed=%s, earned_runs=%s, walks_allowed=%s,
                                strikeouts_pit=%s, runs_allowed=%s
                            WHERE id=%s AND (hits_allowed IS NULL OR strikeouts_pit IS NULL)
                        """, (
                            int(pitcher.get("h", 0)), int(pitcher.get("er", 0)),
                            int(pitcher.get("bb", 0)), int(pitcher.get("k", 0)),
                            int(pitcher.get("r", 0)), row_id,
                        ))
                        day_patched += 1
                        del name_lookup[key]
                    except (ValueError, TypeError):
                        pass

        conn.commit()
        total_patched += day_patched
        print(f"  statsapi {d}: {day_patched} rows patched")
        time.sleep(0.1)

    cur.close()
    conn.close()
    print(f"  Total: {total_patched} rows patched")
    return total_patched


# ---------------------------------------------------------------------------
# Stage 4: Downstream pipeline (rolling windows + all stages)
# ---------------------------------------------------------------------------
def run_pipeline(start: date, end: date):
    """Run rolling windows + downstream for each date with data."""
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    os.environ.setdefault("ENABLE_INGESTION_ORCHESTRATOR", "true")

    conn = get_conn()
    cur = conn.cursor()

    # Get all dates that have raw stats
    cur.execute("""
        SELECT DISTINCT game_date FROM mlb_player_stats
        WHERE season = 2026 AND game_date BETWEEN %s AND %s
        ORDER BY game_date
    """, (start, end))
    stat_dates = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    if not stat_dates:
        print("  No raw stat dates found to process")
        return

    print(f"  Pipeline dates: {stat_dates[0]} to {stat_dates[-1]} ({len(stat_dates)} days)")

    # Pipeline methods compute: datetime.now(tz).date() - timedelta(days=1)
    # So we need datetime.now() to return (target_date + 1 day) to get target_date as "yesterday"
    from datetime import datetime as real_datetime

    for target_date in stat_dates:
        print(f"  pipeline {target_date} ... ", end="", flush=True)
        t0 = time.time()

        # Make datetime.now() return a fake "today" so yesterday = target_date
        fake_now = datetime.combine(
            target_date + timedelta(days=1),
            datetime.min.time(),
            tzinfo=ZoneInfo("America/New_York"),
        ).replace(hour=12)  # noon so we're safely in the right date

        class FakeDatetime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                if tz:
                    return fake_now.astimezone(tz)
                return fake_now.replace(tzinfo=None)

        # We need to patch datetime in the daily_ingestion module
        # AND in backend.utils.time_utils (which today_et/now_et use)
        with patch("backend.services.daily_ingestion.datetime", FakeDatetime), \
             patch("backend.utils.time_utils.datetime", FakeDatetime):

            # Re-import after patching so the module sees the patched datetime
            from backend.services.daily_ingestion import DailyIngestionOrchestrator
            orch = DailyIngestionOrchestrator()

            async def _run_stages():
                stages = [
                    ("rolling_windows", orch._compute_rolling_windows),
                    ("player_scores", orch._compute_player_scores),
                    ("player_momentum", orch._compute_player_momentum),
                    ("ros_simulation", orch._run_ros_simulation),
                    ("decision_optimization", orch._run_decision_optimization),
                    ("backtesting", orch._run_backtesting),
                    ("explainability", orch._run_explainability),
                    ("snapshot", orch._run_snapshot),
                ]
                results = {}
                for name, method in stages:
                    try:
                        result = await method()
                        if isinstance(result, dict):
                            status = result.get("status") or result.get("pipeline_health", "?")
                        else:
                            status = "?"
                        results[name] = status
                    except Exception as e:
                        results[name] = f"ERROR: {e}"
                return results

            results = asyncio.run(_run_stages())

        elapsed = time.time() - t0
        issues = [f"{k}={v}" for k, v in results.items() if v not in ("success",)]
        if issues:
            print(f"{elapsed:.0f}s  issues: {', '.join(issues)}")
        else:
            print(f"{elapsed:.0f}s  OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Backfill MLB 2026 season data")
    parser.add_argument("--start", default=SEASON_OPEN.isoformat(),
                        help="Start date (YYYY-MM-DD, default: season open)")
    parser.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat(),
                        help="End date (YYYY-MM-DD, default: yesterday)")
    parser.add_argument("--stage", choices=["game_log", "box_stats", "statsapi", "pipeline", "all"],
                        default="all", help="Which stage to run")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"=== MLB 2026 Season Backfill: {start} to {end} ===\n")

    stages = args.stage

    if stages in ("all", "game_log"):
        print("STAGE 1: Game Log (BDL)")
        backfill_game_log(start, end)
        print()

    if stages in ("all", "box_stats"):
        print("STAGE 2: Box Stats (BDL)")
        backfill_box_stats(start, end)
        print()

    if stages in ("all", "statsapi"):
        print("STAGE 3: statsapi Supplement")
        backfill_statsapi(start, end)
        print()

    if stages in ("all", "pipeline"):
        print("STAGE 4: Downstream Pipeline")
        run_pipeline(start, end)
        print()

    print("=== DONE ===")


if __name__ == "__main__":
    main()
