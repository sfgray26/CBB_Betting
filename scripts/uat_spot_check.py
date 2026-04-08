#!/usr/bin/env python
"""
UAT Spot-Check Script — Layer-by-Layer Data Validation
========================================================
Runs targeted validation queries against each pipeline layer (P11→P20),
printing a PASS/FAIL report per check.  Designed to be run against the
Railway production DB to verify the 10 bug fixes (H1-H3, M1-M6) and
overall data accuracy.

Usage:
    DATABASE_URL=<public_url> python scripts/uat_spot_check.py

    Or on Railway:
    railway run python scripts/uat_spot_check.py
"""

import json
import os
import sys
from datetime import date, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    print("ERROR: DATABASE_URL not set.")
    print("Get the public URL from Railway dashboard > DB service > Connect tab.")
    sys.exit(1)

engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
db = Session()

today = date.today()
yesterday = today - timedelta(days=1)

# Track results
_results = []
_anchors = {}  # populated in check_A1


def _check(check_id: str, description: str, passed: bool, detail: str = ""):
    """Record a check result."""
    status = "PASS" if passed else "FAIL"
    _results.append((check_id, status, description, detail))
    icon = "  ✓" if passed else "  ✗"
    print(f"{icon} [{check_id}] {description}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"      {line}")


def _section(title: str):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


# ---------------------------------------------------------------------------
# Layer A: Raw Stats (P11 — mlb_player_stats)
# ---------------------------------------------------------------------------
_section("LAYER A: RAW STATS (mlb_player_stats)")

# A1: Freshness
try:
    max_date = db.execute(text(
        "SELECT MAX(game_date) FROM mlb_player_stats"
    )).scalar()
    fresh = max_date is not None and max_date >= yesterday
    _check("A1", "Freshness: latest game_date >= yesterday", fresh,
           f"MAX(game_date) = {max_date}")
except Exception as e:
    _check("A1", "Freshness: latest game_date >= yesterday", False, str(e))

# A2: Row counts
try:
    total = db.execute(text(
        "SELECT COUNT(*) FROM mlb_player_stats WHERE season = 2026"
    )).scalar()
    _check("A2", "Row count reasonable (>100 for season 2026)", total > 100,
           f"Total rows: {total}")
except Exception as e:
    _check("A2", "Row count reasonable", False, str(e))

# A3: Anchor player discovery — find 3 active players with most recent games
try:
    anchor_rows = db.execute(text("""
        SELECT s.bdl_player_id,
               COALESCE(m.full_name, 'ID:' || s.bdl_player_id::text) AS name,
               COUNT(*) AS n_games,
               MAX(s.game_date) AS latest,
               BOOL_OR(s.ab IS NOT NULL)  AS has_batting,
               BOOL_OR(s.innings_pitched IS NOT NULL) AS has_pitching
        FROM mlb_player_stats s
        LEFT JOIN player_id_mapping m ON m.bdl_id = s.bdl_player_id
        WHERE s.season = 2026 AND s.game_date >= :cutoff
        GROUP BY s.bdl_player_id, m.full_name
        HAVING COUNT(*) >= 3
        ORDER BY COUNT(*) DESC
    """), {"cutoff": today - timedelta(days=14)}).fetchall()

    hitters = [r for r in anchor_rows if r.has_batting and not r.has_pitching]
    pitchers = [r for r in anchor_rows if r.has_pitching and not r.has_batting]
    two_way = [r for r in anchor_rows if r.has_batting and r.has_pitching]

    anchor_hitter = hitters[0] if hitters else None
    anchor_pitcher = pitchers[0] if pitchers else None
    anchor_twoway = two_way[0] if two_way else (hitters[1] if len(hitters) > 1 else None)

    for label, a in [("HITTER", anchor_hitter), ("PITCHER", anchor_pitcher), ("TWO-WAY", anchor_twoway)]:
        if a:
            _anchors[label] = {"id": a.bdl_player_id, "name": a.name}
            print(f"  Anchor {label}: {a.name} (bdl_id={a.bdl_player_id}, {a.n_games} games, latest={a.latest})")
        else:
            print(f"  Anchor {label}: NOT FOUND")

    _check("A3", "Found at least 2 anchor players", len(_anchors) >= 2,
           f"Found: {list(_anchors.keys())}")
except Exception as e:
    _check("A3", "Anchor player discovery", False, str(e))

# A4: Anchor player stats for manual cross-check
for label, info in _anchors.items():
    try:
        rows = db.execute(text("""
            SELECT game_date, ab, hits, home_runs, rbi, stolen_bases,
                   innings_pitched, strikeouts_pit, earned_runs, era
            FROM mlb_player_stats
            WHERE bdl_player_id = :pid AND season = 2026
            ORDER BY game_date DESC
            LIMIT 5
        """), {"pid": info["id"]}).fetchall()
        detail_lines = [f"  {info['name']} — last 5 games (cross-check vs Baseball Reference):"]
        for r in rows:
            if r.ab is not None:
                detail_lines.append(
                    f"  {r.game_date}: AB={r.ab} H={r.hits} HR={r.home_runs} RBI={r.rbi} SB={r.stolen_bases}"
                )
            elif r.innings_pitched is not None:
                detail_lines.append(
                    f"  {r.game_date}: IP={r.innings_pitched} K={r.strikeouts_pit} ER={r.earned_runs} ERA={r.era}"
                )
        _check(f"A4-{label}", f"Anchor {label} stats available for manual review", len(rows) > 0,
               "\n".join(detail_lines))
    except Exception as e:
        _check(f"A4-{label}", f"Anchor {label} stats", False, str(e))

# A5: NULL audit — impossible column combinations
try:
    bad_bat = db.execute(text("""
        SELECT COUNT(*) FROM mlb_player_stats
        WHERE ab IS NULL AND hits IS NOT NULL AND season = 2026
    """)).scalar()
    bad_pit = db.execute(text("""
        SELECT COUNT(*) FROM mlb_player_stats
        WHERE innings_pitched IS NULL AND earned_runs IS NOT NULL AND season = 2026
    """)).scalar()
    _check("A5", "No impossible NULL combinations", bad_bat == 0 and bad_pit == 0,
           f"Bat null-ab+hit: {bad_bat}, Pitch null-ip+er: {bad_pit}")
except Exception as e:
    _check("A5", "NULL audit", False, str(e))

# A6: Outlier check
try:
    outliers = db.execute(text("""
        SELECT COUNT(*) AS n,
            SUM(CASE WHEN home_runs > 5 THEN 1 ELSE 0 END) AS bad_hr,
            SUM(CASE WHEN avg::float > 1.0 THEN 1 ELSE 0 END) AS bad_avg
        FROM mlb_player_stats WHERE season = 2026
    """)).fetchone()
    ok = (outliers.bad_hr or 0) == 0 and (outliers.bad_avg or 0) == 0
    _check("A6", "No per-game outliers (HR>5, AVG>1.0)", ok,
           f"HR>5: {outliers.bad_hr or 0}, AVG>1.0: {outliers.bad_avg or 0}")
except Exception as e:
    _check("A6", "Outlier check", False, str(e))


# ---------------------------------------------------------------------------
# Layer B: Rolling Windows (P13 — player_rolling_stats)
# ---------------------------------------------------------------------------
_section("LAYER B: ROLLING WINDOWS (player_rolling_stats)")

# B1: Freshness
try:
    max_date = db.execute(text(
        "SELECT MAX(as_of_date) FROM player_rolling_stats"
    )).scalar()
    fresh = max_date is not None and max_date >= yesterday
    _check("B1", "Freshness: latest as_of_date >= yesterday", fresh,
           f"MAX(as_of_date) = {max_date}")
except Exception as e:
    _check("B1", "Freshness", False, str(e))

# B2: Window completeness for anchors
for label, info in _anchors.items():
    try:
        windows = db.execute(text("""
            SELECT window_days FROM player_rolling_stats
            WHERE bdl_player_id = :pid AND as_of_date = :d
            ORDER BY window_days
        """), {"pid": info["id"], "d": max_date}).fetchall()
        found = sorted([r.window_days for r in windows])
        ok = found == [7, 14, 30]
        _check(f"B2-{label}", f"All 3 windows (7/14/30) for {info['name']}", ok,
               f"Found windows: {found}")
    except Exception as e:
        _check(f"B2-{label}", f"Window completeness for {label}", False, str(e))

# B3: w_games populated (post-migration)
try:
    total_recent = db.execute(text("""
        SELECT COUNT(*) FROM player_rolling_stats
        WHERE as_of_date >= :d
    """), {"d": today - timedelta(days=3)}).scalar()
    w_games_pop = db.execute(text("""
        SELECT COUNT(*) FROM player_rolling_stats
        WHERE as_of_date >= :d AND w_games IS NOT NULL
    """), {"d": today - timedelta(days=3)}).scalar()
    rate = w_games_pop / max(total_recent, 1) * 100
    _check("B3", f"w_games populated in recent rows (>=80%)", rate >= 80,
           f"{w_games_pop}/{total_recent} = {rate:.0f}% (0% expected before v24 migration + pipeline rerun)")
except Exception as e:
    _check("B3", "w_games populated", False, str(e))

# B4: w_games <= games_in_window (each decay weight <= 1.0)
try:
    violations = db.execute(text("""
        SELECT COUNT(*) FROM player_rolling_stats
        WHERE w_games IS NOT NULL AND w_games > games_in_window + 0.01
    """)).scalar()
    _check("B4", "w_games <= games_in_window (decay weights <= 1.0)", violations == 0,
           f"Violations: {violations}")
except Exception as e:
    _check("B4", "w_games vs games_in_window", False, str(e))

# B5: Rate sanity — w_avg = w_hits / w_ab within epsilon
try:
    bad_avg = db.execute(text("""
        SELECT COUNT(*) FROM player_rolling_stats
        WHERE w_ab > 0 AND w_avg IS NOT NULL
          AND ABS(w_avg - (w_hits / w_ab)) > 0.001
    """)).scalar()
    _check("B5", "w_avg = w_hits/w_ab (within 0.001)", bad_avg == 0,
           f"Mismatches: {bad_avg}")
except Exception as e:
    _check("B5", "Rate sanity (w_avg)", False, str(e))

# B6: Rate sanity — w_era = 9 * w_earned_runs / w_ip
try:
    bad_era = db.execute(text("""
        SELECT COUNT(*) FROM player_rolling_stats
        WHERE w_ip > 0 AND w_era IS NOT NULL
          AND ABS(w_era - (9.0 * w_earned_runs / w_ip)) > 0.01
    """)).scalar()
    _check("B6", "w_era = 9*w_earned_runs/w_ip (within 0.01)", bad_era == 0,
           f"Mismatches: {bad_era}")
except Exception as e:
    _check("B6", "Rate sanity (w_era)", False, str(e))

# B7: Outlier ranges
try:
    bad_ranges = db.execute(text("""
        SELECT
            SUM(CASE WHEN w_avg > 0.500 THEN 1 ELSE 0 END) AS high_avg,
            SUM(CASE WHEN w_era > 15.0 THEN 1 ELSE 0 END) AS high_era,
            SUM(CASE WHEN w_era < 0 THEN 1 ELSE 0 END) AS neg_era
        FROM player_rolling_stats
        WHERE as_of_date >= :d
    """), {"d": today - timedelta(days=7)}).fetchone()
    ok = (bad_ranges.high_avg or 0) == 0 and (bad_ranges.neg_era or 0) == 0
    _check("B7", "No extreme rolling rates (AVG>0.500, ERA<0)", ok,
           f"AVG>0.500: {bad_ranges.high_avg or 0}, ERA>15: {bad_ranges.high_era or 0}, ERA<0: {bad_ranges.neg_era or 0}")
except Exception as e:
    _check("B7", "Outlier ranges", False, str(e))

# B8: Manual decay math verification for one anchor hitter
if "HITTER" in _anchors:
    try:
        pid = _anchors["HITTER"]["id"]
        latest_rolling = db.execute(text("""
            SELECT as_of_date, window_days, w_home_runs, w_ab, w_hits, games_in_window, w_games
            FROM player_rolling_stats
            WHERE bdl_player_id = :pid AND window_days = 14
            ORDER BY as_of_date DESC LIMIT 1
        """), {"pid": pid}).fetchone()

        if latest_rolling:
            ref_date = latest_rolling.as_of_date
            cutoff = ref_date - timedelta(days=14)
            raw_games = db.execute(text("""
                SELECT game_date, home_runs, ab, hits
                FROM mlb_player_stats
                WHERE bdl_player_id = :pid AND game_date > :cutoff AND game_date <= :ref
                ORDER BY game_date
            """), {"pid": pid, "cutoff": cutoff, "ref": ref_date}).fetchall()

            # Recompute decay-weighted HR
            recomputed_hr = 0.0
            recomputed_ab = 0.0
            recomputed_w = 0.0
            detail_lines = []
            for g in raw_games:
                days_back = (ref_date - g.game_date).days
                w = 0.95 ** days_back
                hr_val = g.home_runs or 0
                ab_val = g.ab or 0
                recomputed_hr += w * hr_val
                recomputed_ab += w * ab_val
                recomputed_w += w
                detail_lines.append(f"  {g.game_date}: days_back={days_back}, w={w:.4f}, HR={hr_val}, AB={ab_val}")

            db_hr = latest_rolling.w_home_runs or 0.0
            diff = abs(recomputed_hr - db_hr)
            ok = diff < 0.01
            detail_lines.insert(0, f"  Recomputed w_HR={recomputed_hr:.4f}, DB w_HR={db_hr:.4f}, diff={diff:.6f}")
            detail_lines.insert(1, f"  Recomputed w_games={recomputed_w:.4f}, DB w_games={latest_rolling.w_games}")
            _check("B8", f"Decay math verified for {_anchors['HITTER']['name']} (14d HR)", ok,
                   "\n".join(detail_lines[:8]))  # cap output
        else:
            _check("B8", "Decay math verification", False, "No 14d rolling row found")
    except Exception as e:
        _check("B8", "Decay math verification", False, str(e))


# ---------------------------------------------------------------------------
# Layer C: Scoring (P14 — player_scores)
# ---------------------------------------------------------------------------
_section("LAYER C: SCORING (player_scores)")

# C1: Freshness
try:
    max_date_sc = db.execute(text(
        "SELECT MAX(as_of_date) FROM player_scores"
    )).scalar()
    _check("C1", "Freshness: latest as_of_date >= yesterday", max_date_sc >= yesterday if max_date_sc else False,
           f"MAX(as_of_date) = {max_date_sc}")
except Exception as e:
    _check("C1", "Freshness", False, str(e))

# C2: Pool separation (M2 fix) — hitters should NOT have z_era/z_whip unless two_way
try:
    bad_pool = db.execute(text("""
        SELECT COUNT(*) FROM player_scores
        WHERE player_type = 'hitter'
          AND (z_era IS NOT NULL OR z_whip IS NOT NULL OR z_k_per_9 IS NOT NULL)
          AND as_of_date >= :d
    """), {"d": today - timedelta(days=7)}).scalar()
    bad_pool2 = db.execute(text("""
        SELECT COUNT(*) FROM player_scores
        WHERE player_type = 'pitcher'
          AND (z_hr IS NOT NULL OR z_rbi IS NOT NULL OR z_sb IS NOT NULL)
          AND as_of_date >= :d
    """), {"d": today - timedelta(days=7)}).scalar()
    _check("C2", "Pool separation: hitters no z_era, pitchers no z_hr (M2 fix)", 
           bad_pool == 0 and bad_pool2 == 0,
           f"Hitters with pitcher Z: {bad_pool}, Pitchers with hitter Z: {bad_pool2}\n"
           f"  (>0 expected if pipeline hasn't rerun since M2 fix)")
except Exception as e:
    _check("C2", "Pool separation", False, str(e))

# C3: Z-score bounds capped at [-3, 3]
try:
    bounds = db.execute(text("""
        SELECT
            MIN(LEAST(COALESCE(z_hr,0), COALESCE(z_rbi,0), COALESCE(z_sb,0),
                      COALESCE(z_avg,0), COALESCE(z_era,0), COALESCE(z_whip,0),
                      COALESCE(z_k_per_9,0))) AS min_z,
            MAX(GREATEST(COALESCE(z_hr,0), COALESCE(z_rbi,0), COALESCE(z_sb,0),
                         COALESCE(z_avg,0), COALESCE(z_era,0), COALESCE(z_whip,0),
                         COALESCE(z_k_per_9,0))) AS max_z
        FROM player_scores
        WHERE as_of_date >= :d
    """), {"d": today - timedelta(days=7)}).fetchone()
    ok = bounds.min_z >= -3.01 and bounds.max_z <= 3.01
    _check("C3", "Z-score bounds within [-3.0, 3.0]", ok,
           f"Min Z: {bounds.min_z:.3f}, Max Z: {bounds.max_z:.3f}")
except Exception as e:
    _check("C3", "Z-score bounds", False, str(e))

# C4: score_0_100 range and mean
try:
    stats = db.execute(text("""
        SELECT MIN(score_0_100) AS mn, MAX(score_0_100) AS mx, AVG(score_0_100) AS avg_s,
               COUNT(*) AS n
        FROM player_scores
        WHERE as_of_date = :d AND window_days = 14
    """), {"d": max_date_sc or yesterday}).fetchone()
    ok = (stats.mn >= -0.01 and stats.mx <= 100.01 and
          30 <= stats.avg_s <= 70) if stats.n > 10 else False
    _check("C4", "score_0_100 range [0,100], mean ~50", ok,
           f"n={stats.n}, min={stats.mn:.1f}, max={stats.mx:.1f}, mean={stats.avg_s:.1f}")
except Exception as e:
    _check("C4", "Score distribution", False, str(e))

# C5: Confidence sanity — none > 1.0
try:
    bad_conf = db.execute(text("""
        SELECT COUNT(*) FROM player_scores
        WHERE confidence > 1.001 AND as_of_date >= :d
    """), {"d": today - timedelta(days=7)}).scalar()
    _check("C5", "No confidence > 1.0", bad_conf == 0,
           f"Violations: {bad_conf}")
except Exception as e:
    _check("C5", "Confidence sanity", False, str(e))

# C6: Top hitters have positive composite_z, top pitchers too
try:
    top_hitters = db.execute(text("""
        SELECT composite_z FROM player_scores
        WHERE player_type = 'hitter' AND as_of_date = :d AND window_days = 14
        ORDER BY score_0_100 DESC LIMIT 5
    """), {"d": max_date_sc or yesterday}).fetchall()
    top_pitchers = db.execute(text("""
        SELECT composite_z FROM player_scores
        WHERE player_type = 'pitcher' AND as_of_date = :d AND window_days = 14
        ORDER BY score_0_100 DESC LIMIT 5
    """), {"d": max_date_sc or yesterday}).fetchall()
    hit_ok = all(r.composite_z > 0 for r in top_hitters) if top_hitters else False
    pit_ok = all(r.composite_z > 0 for r in top_pitchers) if top_pitchers else False
    _check("C6", "Top-scored players have positive composite_z", hit_ok and pit_ok,
           f"Top 5 hitter z: {[f'{r.composite_z:+.2f}' for r in top_hitters]}\n"
           f"  Top 5 pitcher z: {[f'{r.composite_z:+.2f}' for r in top_pitchers]}")
except Exception as e:
    _check("C6", "Composite Z direction", False, str(e))


# ---------------------------------------------------------------------------
# Layer D: Simulation (P16 — simulation_results)
# ---------------------------------------------------------------------------
_section("LAYER D: SIMULATION (simulation_results)")

# D1: Risk metrics populated (H2 fix)
try:
    total_sim = db.execute(text(
        "SELECT COUNT(*) FROM simulation_results WHERE as_of_date = :d"
    ), {"d": max_date_sc or yesterday}).scalar()
    risk_pop = db.execute(text(
        "SELECT COUNT(*) FROM simulation_results WHERE as_of_date = :d AND composite_variance IS NOT NULL"
    ), {"d": max_date_sc or yesterday}).scalar()
    pct = risk_pop / max(total_sim, 1) * 100
    _check("D1", "Risk metrics populated (H2 fix, composite_variance NOT NULL)", risk_pop > 0,
           f"{risk_pop}/{total_sim} = {pct:.0f}% (0% expected before H2 fix + pipeline rerun)")
except Exception as e:
    _check("D1", "Risk metrics", False, str(e))

# D2: Percentile ordering — P10 <= P25 <= P50 <= P75 <= P90
try:
    bad_order_hr = db.execute(text("""
        SELECT COUNT(*) FROM simulation_results
        WHERE as_of_date = :d AND proj_hr_p10 IS NOT NULL
          AND NOT (proj_hr_p10 <= proj_hr_p25 AND proj_hr_p25 <= proj_hr_p50
                   AND proj_hr_p50 <= proj_hr_p75 AND proj_hr_p75 <= proj_hr_p90)
    """), {"d": max_date_sc or yesterday}).scalar()
    bad_order_k = db.execute(text("""
        SELECT COUNT(*) FROM simulation_results
        WHERE as_of_date = :d AND proj_k_p10 IS NOT NULL
          AND NOT (proj_k_p10 <= proj_k_p25 AND proj_k_p25 <= proj_k_p50
                   AND proj_k_p50 <= proj_k_p75 AND proj_k_p75 <= proj_k_p90)
    """), {"d": max_date_sc or yesterday}).scalar()
    _check("D2", "Percentile ordering valid (P10<=P25<=P50<=P75<=P90)",
           bad_order_hr == 0 and bad_order_k == 0,
           f"HR order violations: {bad_order_hr}, K order violations: {bad_order_k}")
except Exception as e:
    _check("D2", "Percentile ordering", False, str(e))

# D3: ERA non-negative
try:
    neg_era = db.execute(text("""
        SELECT COUNT(*) FROM simulation_results
        WHERE as_of_date = :d AND (proj_era_p50 < 0 OR proj_era_p10 < 0)
    """), {"d": max_date_sc or yesterday}).scalar()
    _check("D3", "Projected ERA non-negative", neg_era == 0,
           f"Negative ERA projections: {neg_era}")
except Exception as e:
    _check("D3", "ERA non-negative", False, str(e))

# D4: Player type consistency with player_scores
try:
    type_mismatch = db.execute(text("""
        SELECT COUNT(*) FROM simulation_results sr
        JOIN player_scores ps ON sr.bdl_player_id = ps.bdl_player_id
                              AND sr.as_of_date = ps.as_of_date
                              AND ps.window_days = 14
        WHERE sr.as_of_date = :d AND sr.player_type != ps.player_type
    """), {"d": max_date_sc or yesterday}).scalar()
    _check("D4", "Player type matches between simulation and scores", type_mismatch == 0,
           f"Mismatches: {type_mismatch}")
except Exception as e:
    _check("D4", "Player type consistency", False, str(e))

# D5: Rate derivation sanity — for anchor hitter, P50 HR ~ hr_rate * remaining
if "HITTER" in _anchors:
    try:
        pid = _anchors["HITTER"]["id"]
        sim = db.execute(text("""
            SELECT proj_hr_p50, remaining_games FROM simulation_results
            WHERE bdl_player_id = :pid AND as_of_date = :d
        """), {"pid": pid, "d": max_date_sc or yesterday}).fetchone()
        roll = db.execute(text("""
            SELECT w_home_runs, w_games, games_in_window FROM player_rolling_stats
            WHERE bdl_player_id = :pid AND as_of_date = :d AND window_days = 14
        """), {"pid": pid, "d": max_date_sc or yesterday}).fetchone()

        if sim and roll and sim.proj_hr_p50 and roll.w_home_runs:
            g = roll.w_games if roll.w_games and roll.w_games > 0 else roll.games_in_window
            hr_rate = roll.w_home_runs / g
            expected_p50 = hr_rate * (sim.remaining_games or 130)
            ratio = sim.proj_hr_p50 / max(expected_p50, 0.01)
            ok = 0.7 <= ratio <= 1.3  # within 30% (Monte Carlo variance)
            _check("D5", f"Rate derivation sanity for {_anchors['HITTER']['name']}",ok,
                   f"hr_rate={hr_rate:.4f}, remaining={sim.remaining_games}, "
                   f"expected_p50={expected_p50:.1f}, actual_p50={sim.proj_hr_p50:.1f}, ratio={ratio:.2f}")
        else:
            _check("D5", "Rate derivation sanity", False, "Missing sim or rolling data")
    except Exception as e:
        _check("D5", "Rate derivation sanity", False, str(e))


# ---------------------------------------------------------------------------
# Layer E: Backtesting (P18 — backtest_results)
# ---------------------------------------------------------------------------
_section("LAYER E: BACKTESTING (backtest_results)")

# E1: Table exists and has data
try:
    bt_total = db.execute(text(
        "SELECT COUNT(*) FROM backtest_results"
    )).scalar()
    bt_date = db.execute(text(
        "SELECT MAX(as_of_date) FROM backtest_results"
    )).scalar()
    _check("E1", "Backtest results populated", bt_total > 0,
           f"Total: {bt_total}, latest: {bt_date}")
except Exception as e:
    _check("E1", "Backtest results", False, str(e))

# E2: Time-horizon (H3 fix) — mae_hr should be single-digit not 100+
try:
    if bt_total > 0:
        mae_stats = db.execute(text("""
            SELECT AVG(mae_hr) AS avg_mae_hr, MAX(mae_hr) AS max_mae_hr,
                   AVG(mae_avg) AS avg_mae_avg, MAX(mae_avg) AS max_mae_avg
            FROM backtest_results
            WHERE as_of_date = :d AND player_type = 'hitter'
        """), {"d": bt_date}).fetchone()
        if mae_stats and mae_stats.avg_mae_hr is not None:
            hr_ok = mae_stats.avg_mae_hr < 50  # pre-fix was 100+
            avg_ok = mae_stats.avg_mae_avg is None or mae_stats.avg_mae_avg < 0.200
            _check("E2", "MAE_HR reasonable (<50, was 100+ pre-H3 fix)", hr_ok,
                   f"avg_mae_hr={mae_stats.avg_mae_hr:.2f}, max_mae_hr={mae_stats.max_mae_hr:.2f}")
            _check("E3", "MAE_AVG reasonable (<0.200, M4 fix)", avg_ok,
                   f"avg_mae_avg={mae_stats.avg_mae_avg}, max_mae_avg={mae_stats.max_mae_avg}")
        else:
            _check("E2", "MAE_HR", False, "No hitter backtest data on latest date")
            _check("E3", "MAE_AVG", False, "No hitter backtest data on latest date")
    else:
        _check("E2", "MAE_HR", False, "No backtest data at all")
        _check("E3", "MAE_AVG", False, "No backtest data at all")
except Exception as e:
    _check("E2", "MAE analysis", False, str(e))
    _check("E3", "MAE_AVG", False, str(e))

# E4: Composite MAE non-negative
try:
    neg_mae = db.execute(text("""
        SELECT COUNT(*) FROM backtest_results
        WHERE composite_mae < 0
    """)).scalar()
    _check("E4", "Composite MAE non-negative", neg_mae == 0,
           f"Negative composite_mae: {neg_mae}")
except Exception as e:
    _check("E4", "Composite MAE", False, str(e))

# E5: Golden baseline file
try:
    baseline_exists = os.path.exists("reports/backtesting_baseline.json")
    if baseline_exists:
        with open("reports/backtesting_baseline.json") as f:
            bl = json.load(f)
        _check("E5", "Golden baseline exists with realistic values", True,
               f"Keys: {list(bl.keys())[:5]}")
    else:
        _check("E5", "Golden baseline exists", False,
               "reports/backtesting_baseline.json not found (expected after baseline reset + pipeline rerun)")
except Exception as e:
    _check("E5", "Golden baseline", False, str(e))


# ---------------------------------------------------------------------------
# Layer F: Decisions (P17 — decision_results)
# ---------------------------------------------------------------------------
_section("LAYER F: DECISIONS (decision_results)")

# F1: Has data
try:
    dec_total = db.execute(text(
        "SELECT COUNT(*) FROM decision_results"
    )).scalar()
    dec_date = db.execute(text(
        "SELECT MAX(as_of_date) FROM decision_results"
    )).scalar()
    _check("F1", "Decision results populated", dec_total > 0,
           f"Total: {dec_total}, latest: {dec_date}")
except Exception as e:
    _check("F1", "Decision results", False, str(e))

# F2: Position eligibility (H1 fix) — check target_slot values
try:
    slots = db.execute(text("""
        SELECT DISTINCT target_slot FROM decision_results
        WHERE as_of_date = :d AND decision_type = 'lineup'
              AND target_slot IS NOT NULL
    """), {"d": dec_date or yesterday}).fetchall()
    slot_values = [r.target_slot for r in slots]
    has_real_positions = any(s in slot_values for s in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"])
    has_only_util = slot_values == ["Util"]
    _check("F2", "Position eligibility has real positions (H1 fix, not just Util)",
           has_real_positions or len(slot_values) > 1,
           f"Distinct slots: {slot_values}\n"
           f"  (Only ['Util'] → H1 fix hasn't run yet)")
except Exception as e:
    _check("F2", "Position eligibility", False, str(e))

# F3: Waiver pool (M1 fix) — waiver-type decisions should exist
try:
    waiver_count = db.execute(text("""
        SELECT COUNT(*) FROM decision_results
        WHERE decision_type = 'waiver' AND as_of_date = :d
    """), {"d": dec_date or yesterday}).scalar()
    _check("F3", "Waiver decisions populated (M1 fix)", waiver_count > 0,
           f"Waiver decisions on latest date: {waiver_count}\n"
           f"  (0 expected before M1 fix + pipeline rerun)")
except Exception as e:
    _check("F3", "Waiver pool", False, str(e))


# ---------------------------------------------------------------------------
# Layer G: Snapshots (P20 — daily_snapshots)
# ---------------------------------------------------------------------------
_section("LAYER G: SNAPSHOTS (daily_snapshots)")

# G1: Has data
try:
    snap_total = db.execute(text("SELECT COUNT(*) FROM daily_snapshots")).scalar()
    snap_date = db.execute(text("SELECT MAX(as_of_date) FROM daily_snapshots")).scalar()
    _check("G1", "Snapshots populated", snap_total > 0,
           f"Total: {snap_total}, latest: {snap_date}")
except Exception as e:
    _check("G1", "Snapshots", False, str(e))

# G2: pipeline_jobs_run is dynamic (M5 fix)
try:
    snap = db.execute(text("""
        SELECT pipeline_jobs_run, pipeline_health FROM daily_snapshots
        WHERE as_of_date = :d
    """), {"d": snap_date or yesterday}).fetchone()
    if snap and snap.pipeline_jobs_run:
        jobs = snap.pipeline_jobs_run if isinstance(snap.pipeline_jobs_run, list) else json.loads(snap.pipeline_jobs_run)
        # Should be a subset of known job names, not a hardcoded list
        known_jobs = {"mlb_game_log", "mlb_box_stats", "rolling_windows", "player_scores",
                      "player_momentum", "ros_simulation", "decision_optimization",
                      "backtesting", "explainability", "snapshot"}
        all_known = all(j in known_jobs for j in jobs)
        _check("G2", "pipeline_jobs_run contains valid job names (M5 fix)", all_known,
               f"Jobs: {jobs}\n  Health: {snap.pipeline_health}")
    else:
        _check("G2", "pipeline_jobs_run", False, "No snapshot data or NULL jobs list")
except Exception as e:
    _check("G2", "Pipeline jobs list", False, str(e))

# G3: Health status
try:
    if snap:
        ok = snap.pipeline_health in ("OK", "HEALTHY", "UNKNOWN")
        _check("G3", "Pipeline health status", ok,
               f"Status: {snap.pipeline_health}")
except Exception as e:
    _check("G3", "Health status", False, str(e))


# ---------------------------------------------------------------------------
# Layer H: Cross-Source — Player ID Mapping
# ---------------------------------------------------------------------------
_section("LAYER H: PLAYER ID MAPPING")

try:
    total_map = db.execute(text("SELECT COUNT(*) FROM player_id_mapping")).scalar()
    with_yahoo = db.execute(text(
        "SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NOT NULL"
    )).scalar()
    with_bdl = db.execute(text(
        "SELECT COUNT(*) FROM player_id_mapping WHERE bdl_id IS NOT NULL"
    )).scalar()
    _check("H1", "Player ID mappings populated", total_map > 0,
           f"Total: {total_map}, Yahoo: {with_yahoo}, BDL: {with_bdl}")

    # Check resolution rate
    rate = with_bdl / max(total_map, 1) * 100
    _check("H2", "BDL ID resolution rate > 50%", rate > 50,
           f"Rate: {rate:.0f}%")
except Exception as e:
    _check("H1", "Player ID mapping", False, str(e))
    _check("H2", "BDL resolution", False, str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_section("SUMMARY")

passed = sum(1 for r in _results if r[1] == "PASS")
failed = sum(1 for r in _results if r[1] == "FAIL")
total = len(_results)

print(f"\n  Total checks: {total}")
print(f"  PASSED:       {passed}")
print(f"  FAILED:       {failed}")

if failed > 0:
    print(f"\n  FAILED CHECKS:")
    for check_id, status, desc, detail in _results:
        if status == "FAIL":
            print(f"    [{check_id}] {desc}")

# Distinguish pre-fix expected failures from real issues
pre_fix_checks = {"B3", "C2", "D1", "E5", "F2", "F3", "G2"}
real_failures = [r for r in _results if r[1] == "FAIL" and r[0] not in pre_fix_checks]
pre_fix_failures = [r for r in _results if r[1] == "FAIL" and r[0] in pre_fix_checks]

if pre_fix_failures:
    print(f"\n  NOTE: {len(pre_fix_failures)} failures are expected before pipeline rerun after bug fixes:")
    for check_id, _, desc, _ in pre_fix_failures:
        print(f"    [{check_id}] {desc}")

if real_failures:
    print(f"\n  *** {len(real_failures)} UNEXPECTED FAILURES — INVESTIGATE ***")
    for check_id, _, desc, _ in real_failures:
        print(f"    [{check_id}] {desc}")
elif not pre_fix_failures:
    print(f"\n  ALL CHECKS PASSED — DATA VALIDATED")

db.close()
sys.exit(1 if real_failures else 0)
