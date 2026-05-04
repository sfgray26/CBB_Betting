"""
Statcast data loader — per-player lookup with 6-hour in-memory cache.

Load priority (first found wins):
  1. Real Baseball Savant CSVs in data/cache/  (manual download — see statcast_scraper.py)
  2. Sample projected CSVs in data/projections/ (generated on first run)
  3. Empty dict — endpoint continues without Statcast enrichment (silent fail)

Public API:
  get_statcast_batter(name)  -> Optional[StatcastBatter]
  get_statcast_pitcher(name) -> Optional[StatcastPitcher]
  enrich_recommendations(recs, fa_yahoo_players, roster_players) -> list  (main entry point)
"""

import csv
import logging
import time
from pathlib import Path
from typing import Optional

from backend.fantasy_baseball.advanced_metrics import (
    StatcastBatter,
    StatcastPitcher,
    analyze_batter_regression,
    analyze_pitcher_regression,
    is_breakout_candidate_batter,
    calculate_injury_risk_score,
)

logger = logging.getLogger(__name__)

_CACHE_TTL = 6 * 3600  # seconds

_batter_cache: dict[str, StatcastBatter] = {}
_pitcher_cache: dict[str, StatcastPitcher] = {}
_loaded_at: float = 0.0

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "projections"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Default age to use in breakout detection when not available
_DEFAULT_AGE = 27


# ---------------------------------------------------------------------------
# Internal CSV parsers
# ---------------------------------------------------------------------------

def _parse_sample_batting_csv(csv_path: Path) -> dict[str, StatcastBatter]:
    """Parse the generated advanced_batting_2026.csv format."""
    result: dict[str, StatcastBatter] = {}
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Name") or "").strip()
                if not name:
                    continue
                xwoba_diff = _float(row.get("xwOBA_Diff", 0))
                b = StatcastBatter(
                    name=name,
                    barrel_pct=_float(row.get("Barrel_Pct", 0)),
                    exit_velo_avg=_float(row.get("Exit_Velo", 0)),
                    hard_hit_pct=_float(row.get("Hard_Hit_Pct", 0)),
                    sweet_spot_pct=_float(row.get("Sweet_Spot_Pct", 0)),
                    xba=_float(row.get("xBA", 0)),
                    xslg=_float(row.get("xSLG", 0)),
                    xwoba=_float(row.get("xwOBA", 0)),
                    xwoba_diff=xwoba_diff,
                    o_swing_pct=_float(row.get("O_Swing_Pct", 0)),
                    z_contact_pct=_float(row.get("Z_Contact_Pct", 0)),
                    swstr_pct=_float(row.get("SwStr_Pct", 0)),
                    sprint_speed=_float(row.get("Sprint_Speed", 0)),
                    power_score=_float(row.get("Power_Score", 0)),
                    contact_score=_float(row.get("Contact_Score", 0)),
                    discipline_score=_float(row.get("Discipline_Score", 0)),
                    speed_score=_float(row.get("Speed_Score", 0)),
                    overall_score=_float(row.get("Overall_Score", 0)),
                    regression_up=xwoba_diff < -0.020,
                    regression_down=xwoba_diff > 0.030,
                )
                result[name.lower()] = b
    except Exception as e:
        logger.warning(f"Failed to parse batting CSV {csv_path}: {e}")
    return result


def _parse_sample_pitching_csv(csv_path: Path) -> dict[str, StatcastPitcher]:
    """Parse the generated advanced_pitching_2026.csv format."""
    result: dict[str, StatcastPitcher] = {}
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Name") or "").strip()
                if not name:
                    continue
                xera_diff = _float(row.get("xERA_Diff", 0))
                p = StatcastPitcher(
                    name=name,
                    stuff_plus=_float(row.get("Stuff_Plus", 100)),
                    location_plus=_float(row.get("Location_Plus", 100)),
                    fb_velo_avg=_float(row.get("FB_Velo", 0)),
                    spin_rate_fb=int(_float(row.get("Spin_Rate_FB", 0))),
                    whiff_pct=_float(row.get("Whiff_Pct", 0)),
                    chase_pct=_float(row.get("Chase_Pct", 0)),
                    barrel_allowed_pct=_float(row.get("Barrel_Allowed_Pct", 0)),
                    xera=_float(row.get("xERA", 0)),
                    xera_diff=xera_diff,
                    injury_risk_score=_float(row.get("Injury_Risk_Score", 0)),
                    stuff_score=_float(row.get("Stuff_Score", 0)),
                    whiff_score=_float(row.get("Whiff_Score", 0)),
                    overall_score=_float(row.get("Overall_Score", 0)),
                    luck_regression=(xera_diff > 0.40),
                    breakout_candidate=(row.get("Breakout_Flag", "").strip() == "BREAKOUT"),
                    velo_concern=(row.get("Injury_Risk_Flag", "").strip() in ("INJURY_RISK", "HIGH_INJURY_RISK")),
                )
                result[name.lower()] = p
    except Exception as e:
        logger.warning(f"Failed to parse pitching CSV {csv_path}: {e}")
    return result


def _parse_savant_batting_csv(csv_path: Path) -> dict[str, StatcastBatter]:
    """Parse a real Baseball Savant CSV export using statcast_scraper's parser."""
    from backend.fantasy_baseball.statcast_scraper import parse_statcast_batting_csv
    result: dict[str, StatcastBatter] = {}
    try:
        rows = parse_statcast_batting_csv(csv_path)
        for row in rows:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            xwoba_diff = row.get("xwoba_diff", 0.0)
            b = StatcastBatter(
                name=name,
                barrel_pct=row.get("barrel_pct", 0.0),
                exit_velo_avg=row.get("exit_velo_avg", 0.0),
                hard_hit_pct=row.get("hard_hit_pct", 0.0),
                sweet_spot_pct=row.get("sweet_spot_pct", 0.0),
                xba=row.get("xba", 0.0),
                xslg=row.get("xslg", 0.0),
                xwoba=row.get("xwoba", 0.0),
                xwoba_diff=xwoba_diff,
                o_swing_pct=row.get("o_swing_pct", 0.0),
                z_contact_pct=row.get("z_contact_pct", 0.0),
                swstr_pct=row.get("swstr_pct", 0.0),
                gb_pct=row.get("gb_pct", 0.0),
                fb_pct=row.get("fb_pct", 0.0),
                ld_pct=row.get("ld_pct", 0.0),
                pull_pct=row.get("pull_pct", 0.0),
                regression_up=xwoba_diff < -0.020,
                regression_down=xwoba_diff > 0.030,
            )
            result[name.lower()] = b
    except Exception as e:
        logger.warning(f"Failed to parse Savant batting CSV {csv_path}: {e}")
    return result


def _parse_savant_pitching_csv(csv_path: Path) -> dict[str, StatcastPitcher]:
    """Parse a real Baseball Savant pitching CSV export."""
    from backend.fantasy_baseball.statcast_scraper import parse_statcast_pitching_csv
    result: dict[str, StatcastPitcher] = {}
    try:
        rows = parse_statcast_pitching_csv(csv_path)
        for row in rows:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            xera_diff = row.get("xera_diff", 0.0)
            p = StatcastPitcher(
                name=name,
                stuff_plus=row.get("stuff_plus", 100.0),
                location_plus=row.get("location_plus", 100.0),
                fb_velo_avg=row.get("fb_velo_avg", 0.0),
                spin_rate_fb=row.get("spin_rate_fb", 0),
                whiff_pct=row.get("whiff_pct", 0.0),
                chase_pct=row.get("chase_pct", 0.0),
                barrel_allowed_pct=row.get("barrel_allowed_pct", 0.0),
                xera=row.get("xera", 0.0),
                xera_diff=xera_diff,
                luck_regression=(xera_diff > 0.40),
            )
            result[name.lower()] = p
    except Exception as e:
        logger.warning(f"Failed to parse Savant pitching CSV {csv_path}: {e}")
    return result


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _float(val) -> float:
    try:
        return float(val or 0)
    except (ValueError, TypeError):
        return 0.0


def _load_from_db(season: int = 2026) -> tuple:
    """
    Load batter and pitcher metrics from statcast_batter_metrics /
    statcast_pitcher_metrics tables (populated daily by savant_ingestion at 6 AM).

    Returns (batters_dict, pitchers_dict) keyed by stripped lowercase name.
    Returns ({}, {}) on any error so callers can fall through gracefully.
    """
    from backend.fantasy_baseball.advanced_metrics import StatcastBatter, StatcastPitcher
    try:
        import unicodedata, re as _re
        _sfx = _re.compile(r"\s+(Jr\.?|Sr\.?|II+|III+|IV|V)$", _re.IGNORECASE)

        def _key(name: str) -> str:
            nfkd = unicodedata.normalize("NFKD", name)
            ascii_ = "".join(c for c in nfkd if not unicodedata.combining(c))
            return " ".join(_sfx.sub("", ascii_.strip()).lower().split())

        from sqlalchemy import text as _text
        from backend.models import SessionLocal
        db = SessionLocal()
        try:
            # --- Batters ---
            # Join with statcast_performances to compute xwoba_diff (xwOBA - wOBA).
            # statcast_batter_metrics has xwoba but not woba; statcast_performances
            # has per-game woba that we aggregate into a season mean.
            batter_rows = db.execute(_text("""
                SELECT
                    sbm.player_name,
                    sbm.xwoba,
                    sbm.barrel_percent,
                    sbm.hard_hit_percent,
                    sbm.avg_exit_velocity,
                    sbm.whiff_percent,
                    sp_agg.avg_woba
                FROM statcast_batter_metrics sbm
                LEFT JOIN (
                    SELECT player_id, AVG(woba) AS avg_woba
                    FROM statcast_performances
                    WHERE woba > 0
                    GROUP BY player_id
                ) sp_agg ON sp_agg.player_id = sbm.mlbam_id
                WHERE sbm.season = :season
            """), {"season": season}).fetchall()

            batters: dict = {}
            for row in batter_rows:
                xwoba = _float(row.xwoba)
                avg_woba = _float(row.avg_woba) if row.avg_woba else xwoba
                xwoba_diff = xwoba - avg_woba
                b = StatcastBatter(
                    name=row.player_name,
                    xwoba=xwoba,
                    xwoba_diff=xwoba_diff,
                    barrel_pct=_float(row.barrel_percent),
                    exit_velo_avg=_float(row.avg_exit_velocity),
                    hard_hit_pct=_float(row.hard_hit_percent),
                    swstr_pct=_float(row.whiff_percent),
                    regression_up=xwoba_diff < -0.020,
                    regression_down=xwoba_diff > 0.030,
                )
                batters[_key(row.player_name)] = b

            # --- Pitchers ---
            # xera_diff = xERA - ERA: positive means ERA should rise (BUY_LOW for opposing
            # batters; pitcher got lucky with low ERA vs poor underlying metrics).
            pitcher_rows = db.execute(_text("""
                SELECT
                    spm.player_name,
                    spm.xera,
                    spm.xwoba,
                    spm.barrel_percent_allowed,
                    spm.hard_hit_percent_allowed,
                    spm.avg_exit_velocity_allowed,
                    spm.whiff_percent,
                    spm.k_percent,
                    (
                        SELECT CASE
                            WHEN SUM(sp.ip) > 0
                            THEN ROUND((SUM(sp.er)::numeric / NULLIF(SUM(sp.ip), 0)) * 9, 2)
                            ELSE NULL
                        END
                        FROM statcast_performances sp
                        WHERE sp.player_id = spm.mlbam_id AND sp.ip > 0
                    ) AS computed_era
                FROM statcast_pitcher_metrics spm
                WHERE spm.season = :season
            """), {"season": season}).fetchall()

            pitchers: dict = {}
            for row in pitcher_rows:
                xera = _float(row.xera)
                era = _float(row.computed_era) if row.computed_era else None
                xera_diff = (xera - era) if era else 0.0
                p = StatcastPitcher(
                    name=row.player_name,
                    xera=xera,
                    xera_diff=xera_diff,
                    xwoba_allowed=_float(row.xwoba),
                    barrel_allowed_pct=_float(row.barrel_percent_allowed),
                    hard_hit_allowed_pct=_float(row.hard_hit_percent_allowed),
                    exit_velo_allowed=_float(row.avg_exit_velocity_allowed),
                    whiff_pct=_float(row.whiff_percent),
                    luck_regression=(xera_diff > 0.40),
                )
                pitchers[_key(row.player_name)] = p

            if batters or pitchers:
                logger.info(
                    "statcast_loader DB tier: %d batters, %d pitchers from statcast_*_metrics",
                    len(batters), len(pitchers),
                )
            return batters, pitchers

        finally:
            db.close()

    except Exception as exc:
        logger.warning("statcast_loader DB tier failed (non-fatal): %s", exc)
        return {}, {}


def _ensure_loaded() -> None:
    """Load caches if empty or stale (TTL expired)."""
    global _loaded_at
    now = time.time()
    if _batter_cache and (now - _loaded_at) < _CACHE_TTL:
        return

    # --- Batters ---
    year = 2026
    batters: dict[str, StatcastBatter] = {}

    # --- Tier 0: DB tables (statcast_batter_metrics / statcast_pitcher_metrics) ---
    # Populated daily at 6 AM by _ingest_savant_leaderboards. This is the primary
    # source on Railway where the CSV/JSON cache paths are ephemeral.
    # NOTE: DB tier applied AFTER all lower-tier sources (see below) to ensure
    # it takes priority over pybaseball/CSV data.
    _db_batters, _db_pitchers = _load_from_db(season=year)

    # --- Tier 1: pybaseball JSON cache (24h, 400+ players) ---
    try:
        from backend.fantasy_baseball.pybaseball_loader import (
            fetch_all_statcast_leaderboards,
            load_pybaseball_batters,
        )
        fetch_all_statcast_leaderboards(year=2025)
        pb = load_pybaseball_batters(year=2025)
        if pb:
            batters.update(pb)
            logger.info(f"Tier 1: {len(pb)} pybaseball batters loaded")
    except Exception as e:
        logger.warning(f"pybaseball batter tier skipped: {e}")

    # Real Savant CSVs (user-downloaded)
    for fname in [
        f"statcast_batting_expected_{year}.csv",
        f"statcast_batting_ev_{year}.csv",
        f"statcast_batting_expected_{year - 1}.csv",
    ]:
        p = CACHE_DIR / fname
        if p.exists():
            parsed = _parse_savant_batting_csv(p)
            if parsed:
                batters.update(parsed)
                logger.info(f"Loaded {len(parsed)} Savant batters from {p.name}")
                break

    # --- DB tier override (applies AFTER pybaseball/CSV to take priority) ---
    if _db_batters:
        batters.update(_db_batters)
        logger.info(f"DB tier override: {len(_db_batters)} batters from statcast_batter_metrics")

    # Sample CSV fallback
    if not batters:
        sample = DATA_DIR / "advanced_batting_2026.csv"
        if not sample.exists():
            try:
                from backend.fantasy_baseball.advanced_metrics import generate_advanced_batting_csv
                sample = generate_advanced_batting_csv()
            except Exception as exc:
                logger.warning(f"Could not generate sample batting CSV: {exc}")
        if sample.exists():
            batters = _parse_sample_batting_csv(sample)
            logger.info(f"Loaded {len(batters)} sample batters (Savant CSV not present)")

    # --- Pitchers ---
    pitchers: dict[str, StatcastPitcher] = {}

    # NOTE: DB tier applied AFTER all lower-tier sources (see below) to ensure
    # it takes priority over pybaseball/CSV data.

    # --- Tier 1: pybaseball JSON cache (batters already triggered the fetch above) ---
    try:
        from backend.fantasy_baseball.pybaseball_loader import load_pybaseball_pitchers
        pb_p = load_pybaseball_pitchers(year=2025)
        if pb_p:
            pitchers.update(pb_p)
            logger.info(f"Tier 1: {len(pb_p)} pybaseball pitchers loaded")
    except Exception as e:
        logger.warning(f"pybaseball pitcher tier skipped: {e}")

    for fname in [
        f"statcast_pitching_expected_{year}.csv",
        f"statcast_pitching_arsenal_{year}.csv",
        f"statcast_pitching_expected_{year - 1}.csv",
    ]:
        p = CACHE_DIR / fname
        if p.exists():
            parsed = _parse_savant_pitching_csv(p)
            if parsed:
                pitchers.update(parsed)
                logger.info(f"Loaded {len(parsed)} Savant pitchers from {p.name}")
                break

    # --- DB tier override (applies AFTER pybaseball/CSV to take priority) ---
    if _db_pitchers:
        pitchers.update(_db_pitchers)
        logger.info(f"DB tier override: {len(_db_pitchers)} pitchers from statcast_pitcher_metrics")

    if not pitchers:
        sample = DATA_DIR / "advanced_pitching_2026.csv"
        if not sample.exists():
            try:
                from backend.fantasy_baseball.advanced_metrics import generate_advanced_pitching_csv
                sample = generate_advanced_pitching_csv()
            except Exception as exc:
                logger.warning(f"Could not generate sample pitching CSV: {exc}")
        if sample.exists():
            pitchers = _parse_sample_pitching_csv(sample)
            logger.info(f"Loaded {len(pitchers)} sample pitchers (Savant CSV not present)")

    _batter_cache.clear()
    _batter_cache.update(batters)
    _pitcher_cache.clear()
    _pitcher_cache.update(pitchers)
    _loaded_at = now


def get_statcast_batter(name: str) -> Optional[StatcastBatter]:
    """Return StatcastBatter for player name, or None if no data."""
    try:
        _ensure_loaded()
    except Exception:
        return None
    key = name.strip().lower()
    result = _batter_cache.get(key)
    if result is None:
        try:
            from backend.fantasy_baseball.pybaseball_loader import match_yahoo_to_statcast
            matched = match_yahoo_to_statcast(name, _batter_cache)
            if matched:
                result = _batter_cache.get(matched)
            else:
                # Log data quality issue: name matching failed
                import json
                logger.info(json.dumps({
                    "event": "data_quality_issue",
                    "issue_type": "statcast_name_match_failure",
                    "yahoo_name": name,
                    "cache_keys_sample": list(_batter_cache.keys())[:5]
                }))
        except Exception:
            pass
    return result


def get_statcast_pitcher(name: str) -> Optional[StatcastPitcher]:
    """Return StatcastPitcher for player name, or None if no data."""
    try:
        _ensure_loaded()
    except Exception:
        return None
    key = name.strip().lower()
    result = _pitcher_cache.get(key)
    if result is None:
        try:
            from backend.fantasy_baseball.pybaseball_loader import match_yahoo_to_statcast
            matched = match_yahoo_to_statcast(name, _pitcher_cache)
            if matched:
                result = _pitcher_cache.get(matched)
        except Exception:
            pass
    return result


def cache_age_seconds() -> float:
    """Seconds since last cache load (0 if never loaded)."""
    return time.time() - _loaded_at if _loaded_at else 0.0


# ---------------------------------------------------------------------------
# Enrichment entry point
# ---------------------------------------------------------------------------

def build_statcast_signals(
    player_name: str,
    is_pitcher: bool,
    owned_pct: float = 100.0,
) -> tuple[list[str], float]:
    """
    Return (signals, regression_delta) for a player.

    signals: list of strings like ["BUY_LOW", "BREAKOUT", "HIGH_INJURY_RISK"]
    regression_delta: xwOBA - wOBA for batters, xERA - ERA for pitchers (0.0 if no data)

    Never raises — returns ([], 0.0) on any error.
    """
    signals: list[str] = []
    regression_delta = 0.0

    try:
        if is_pitcher:
            metrics = get_statcast_pitcher(player_name)
            if metrics is None:
                return signals, regression_delta

            regression_delta = metrics.xera_diff
            verdict, _ = analyze_pitcher_regression(metrics)
            if verdict == "BUY_LOW":
                signals.append("BUY_LOW")
            elif verdict == "SELL_HIGH":
                signals.append("SELL_HIGH")

            injury_risk = calculate_injury_risk_score(metrics)
            if injury_risk >= 60:
                signals.append("HIGH_INJURY_RISK")
            elif injury_risk <= 20:
                signals.append("LOW_INJURY_RISK")

            if metrics.breakout_candidate:
                signals.append("BREAKOUT")

        else:
            metrics = get_statcast_batter(player_name)
            if metrics is None:
                return signals, regression_delta

            regression_delta = metrics.xwoba_diff
            verdict, _ = analyze_batter_regression(metrics)
            if verdict == "BUY_LOW":
                signals.append("BUY_LOW")
            elif verdict == "SELL_HIGH":
                signals.append("SELL_HIGH")

            # Breakout detection — only for lower-owned players (hidden gems)
            if owned_pct < 60:
                is_breakout, _ = is_breakout_candidate_batter(metrics, age=_DEFAULT_AGE)
                if is_breakout:
                    signals.append("BREAKOUT")

    except Exception as exc:
        logger.debug(f"Statcast signal build failed for {player_name}: {exc}")

    return signals, regression_delta


def statcast_need_score_boost(signals: list[str]) -> float:
    """
    Return need_score delta based on Statcast signals.

    BUY_LOW:     +0.4 (positive regression expected)
    BREAKOUT:    +0.5 (upside play)
    SELL_HIGH:   -0.3 (likely to regress)
    HIGH_INJURY_RISK: -0.2 (injury concern)
    """
    boost = 0.0
    if "BUY_LOW" in signals:
        boost += 0.4
    if "BREAKOUT" in signals:
        boost += 0.5
    if "SELL_HIGH" in signals:
        boost -= 0.3
    if "HIGH_INJURY_RISK" in signals:
        boost -= 0.2
    return boost
