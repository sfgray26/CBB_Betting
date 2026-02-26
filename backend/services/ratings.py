"""
CBB Ratings data from multiple sources.

Sources
-------
KenPom    (Official API, Bearer token) — AdjEM (adjusted efficiency margin)
BartTorvik (barttorvik.com CSV)        — adjusted efficiency column
EvanMiya   (evanmiya.com HTML table)   — BPR (Box Plus-Minus for teams)

Weights in CBBEdgeModel: KenPom 34.2%, BartTorvik 33.3%, EvanMiya 32.5%.
Missing weights are renormalized at model runtime — see betting_model.py.

EvanMiya scraper notes
-----------------------
The public evanmiya.com site renders a sortable HTML table of team BPR values.
Set EVANMIYA_URL in .env to override the endpoint if the URL changes.
If the scrape fails for any reason the function returns {} and the model
falls back to KenPom+BartTorvik with renormalized weights + a small SD penalty.

BartTorvik robustness
----------------------
Column detection uses CSV header names rather than hardcoded column indexes.
If the site restructures its CSV, the scraper warns and falls back to positional
parsing (original column 1 = team, column 5 = efficiency).
"""

import csv
import io
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from backend.services.team_mapping import normalize_team_name

load_dotenv()

logger = logging.getLogger(__name__)

_KENPOM_URL   = "https://kenpom.com/api.php"
_CURRENT_YEAR = int(os.getenv("SEASON_YEAR", "2026"))

_BARTTORVIK_URL = os.getenv(
    "BARTTORVIK_URL",
    f"https://barttorvik.com/{_CURRENT_YEAR}_team_results.csv",
)
_EVANMIYA_URL = os.getenv("EVANMIYA_URL", "https://evanmiya.com/")


class RatingsService:
    """Multi-source ratings aggregator with automated team name matching."""

    def __init__(self):
        self.kenpom_key      = os.getenv("KENPOM_API_KEY")
        self.cache: Dict     = {}
        self.cache_timestamp: Optional[datetime] = None
        # EvanMiya auto-drop state — set by get_evanmiya_ratings() after
        # EVANMIYA_AUTO_DROP_AFTER consecutive failures (default 3).
        self._evanmiya_fail_count: int = 0
        self._evanmiya_dropped: bool = False

    # -----------------------------------------------------------------------
    # Team name lookup
    # -----------------------------------------------------------------------

    def get_team_rating(
        self, team_name: str, source_data: Dict[str, float]
    ) -> Optional[float]:
        """Safely extract a rating using the centralised normalisation service."""
        if not source_data:
            return None

        normalized = normalize_team_name(team_name, list(source_data.keys()))
        if normalized is None:
            logger.warning("Mismatch: '%s' not found in source data.", team_name)
            return None

        rating = source_data.get(normalized)
        if rating is None:
            logger.warning(
                "Mismatch: '%s' (normalized: '%s') absent post-normalisation.",
                team_name, normalized,
            )
        return rating

    # -----------------------------------------------------------------------
    # KenPom
    # -----------------------------------------------------------------------

    def get_kenpom_ratings(self) -> Dict[str, float]:
        """Fetch AdjEM ratings from the official KenPom API."""
        if not self.kenpom_key:
            logger.warning("KenPom API key not set — skipping KenPom.")
            return {}

        headers = {
            "Authorization": f"Bearer {self.kenpom_key}",
            "Accept":        "application/json",
        }
        params = {"endpoint": "ratings", "y": _CURRENT_YEAR}

        try:
            response = requests.get(
                _KENPOM_URL, headers=headers, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json()

            ratings: Dict[str, float] = {}
            for team in data:
                name   = team.get("TeamName")
                adj_em = team.get("AdjEM")
                if name and adj_em is not None:
                    try:
                        ratings[name.strip()] = float(adj_em)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not parse AdjEM for %s: %r", name, adj_em
                        )

            logger.info("KenPom: loaded %d teams", len(ratings))
            return ratings

        except Exception as exc:
            logger.error("KenPom API error: %s", exc)
            return {}

    # -----------------------------------------------------------------------
    # BartTorvik
    # -----------------------------------------------------------------------

    def get_barttorvik_ratings(self) -> Dict[str, float]:
        """
        Fetch adjusted efficiency margin (AdjEM = AdjOE - AdjDE) from the
        BartTorvik CSV export.

        BartTorvik stores offensive efficiency (AdjOE, ~90-130) and defensive
        efficiency (AdjDE, ~90-130) as separate columns.  The model needs an
        *efficiency margin* on the same scale as KenPom AdjEM (~-30 to +30),
        so we compute AdjEM = AdjOE - AdjDE for every team.

        Column detection uses header names first, falling back to positional
        indexes (team=1, AdjOE=5, AdjDE=6) when headers are absent/renamed.
        """
        url = _BARTTORVIK_URL

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            reader = csv.reader(io.StringIO(response.text))
            rows   = [row for row in reader if any(cell.strip() for cell in row)]

            if not rows:
                logger.warning("BartTorvik CSV is empty")
                return {}

            header = [h.strip().lower() for h in rows[0]]

            # Locate team-name column
            team_col: Optional[int] = None
            for candidate in ("team", "teamname", "team_name", "school"):
                if candidate in header:
                    team_col = header.index(candidate)
                    break

            # Locate offensive efficiency column (AdjOE)
            oe_col: Optional[int] = None
            for candidate in ("adjoe", "adj_oe", "adj oe", "off eff", "off_eff",
                               "o.adjoe", "adjoe.1"):
                if candidate in header:
                    oe_col = header.index(candidate)
                    break

            # Locate defensive efficiency column (AdjDE)
            de_col: Optional[int] = None
            for candidate in ("adjde", "adj_de", "adj de", "def eff", "def_eff",
                               "o.adjde", "adjde.1"):
                if candidate in header:
                    de_col = header.index(candidate)
                    break

            # Legacy fallback: a single net-efficiency column
            legacy_em_col: Optional[int] = None
            for candidate in ("adjem", "adj_em", "adj em", "eff margin"):
                if candidate in header:
                    legacy_em_col = header.index(candidate)
                    break

            # Fallback to hard-coded positional indexes for known CSV layout:
            #   Rk(0) team(1) conf(2) G(3) Rec(4) AdjOE(5) AdjDE(6) Barthag(7)…
            if team_col is None:
                team_col = 1
                logger.debug("BartTorvik: team column not in header; using index 1")
            if oe_col is None and legacy_em_col is None:
                oe_col = 5
                logger.debug(
                    "BartTorvik: AdjOE column not in header; using index 5"
                )
            if de_col is None and legacy_em_col is None:
                de_col = 6
                logger.debug(
                    "BartTorvik: AdjDE column not in header; using index 6"
                )

            ratings: Dict[str, float] = {}
            for row in rows[1:]:
                required = [team_col]
                if legacy_em_col is not None:
                    required.append(legacy_em_col)
                else:
                    required += [oe_col, de_col]
                if len(row) <= max(required):
                    continue

                name = row[team_col].strip()
                if not name:
                    continue

                try:
                    if legacy_em_col is not None:
                        # Direct net-efficiency column (rare but forward-compatible)
                        ratings[name] = float(row[legacy_em_col])
                    else:
                        # Compute AdjEM = AdjOE - AdjDE  (typical BartTorvik layout)
                        adj_oe = float(row[oe_col])
                        adj_de = float(row[de_col])
                        ratings[name] = adj_oe - adj_de
                except (ValueError, TypeError):
                    continue

            logger.info(
                "BartTorvik: loaded %d teams (AdjEM = AdjOE - AdjDE, "
                "team_col=%d, oe_col=%s, de_col=%s)",
                len(ratings), team_col,
                oe_col if oe_col is not None else "legacy",
                de_col if de_col is not None else "legacy",
            )
            return ratings

        except Exception as exc:
            logger.warning("BartTorvik CSV error: %s", exc)
            return {}

    def _parse_barttorvik_json(
        self, data: list
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Parse BartTorvik ``super_standings`` JSON into the standard stats dict.

        The JSON endpoint returns a list of team objects with field names that
        vary slightly across seasons.  Multiple candidate keys are tried for
        each stat so the parser survives minor schema changes.

        All four-factor rates are stored as percentages in the JSON (e.g.,
        50.5 for 50.5 % eFG) and are converted to decimals here so the output
        is identical to the CSV path.  AdjOE / AdjDE / pace are absolute and
        are not rescaled.

        Returns {} if the input is empty or cannot be parsed as expected.
        """
        def _get_f(item: dict, keys: List[str]) -> Optional[float]:
            """Return the first matching key as float, or None."""
            for k in keys:
                v = item.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return None

        def _get_pct(item: dict, keys: List[str]) -> Optional[float]:
            """Like _get_f but converts pct→decimal when value > 1.5."""
            v = _get_f(item, keys)
            if v is None:
                return None
            return v / 100.0 if v > 1.5 else v

        stats: Dict[str, Dict[str, Optional[float]]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue

            team_name: Optional[str] = None
            for k in ("team", "Team", "teamname", "TeamName", "school"):
                if item.get(k):
                    team_name = str(item[k]).strip()
                    break
            if not team_name:
                continue

            adj_oe = _get_f(item, ["adjoe", "AdjOE", "adj_o", "off_eff"])
            adj_de = _get_f(item, ["adjde", "AdjDE", "adj_d", "def_eff"])
            adj_em = (
                (adj_oe - adj_de)
                if (adj_oe is not None and adj_de is not None)
                else None
            )

            stats[team_name] = {
                "adj_oe":        adj_oe,
                "adj_de":        adj_de,
                "adj_em":        adj_em,
                # Offensive four factors
                "efg_pct":       _get_pct(item, [
                    "efg_pct", "efgpct", "efg%", "EFG%", "efg", "EFG_Pct",
                    "off_efg", "o_efg",
                ]),
                "to_pct":        _get_pct(item, [
                    "tor", "TOR", "torp", "torpp", "tov%", "to_rate", "TO_Pct",
                    "off_tor", "o_tor",
                ]),
                "ft_rate":       _get_pct(item, [
                    "ftr", "FTR", "ft_rate", "FT_Rate", "off_ftr",
                ]),
                "three_par":     _get_pct(item, [
                    "3par%", "3par", "3parp", "3PA_Rate", "3PAR", "tpar",
                ]),
                # Defensive four factors
                "def_efg_pct":   _get_pct(item, [
                    "efgd_pct", "efgpctd", "efgd%", "EFGd%", "efgd",
                    "def_efg_pct", "EFGd_Pct", "opp_efg", "d_efg", "efg_d",
                    "opp_efgpct",
                ]),
                "def_to_pct":    _get_pct(item, [
                    "tord", "TORd", "torpp_d", "tov%d", "def_to_rate",
                    "TOd_Pct", "def_tor", "opp_tor", "d_tor",
                ]),
                "def_ft_rate":   _get_pct(item, [
                    "ftrd", "FTRd", "def_ft_rate", "FTd_Rate", "def_ftr",
                    "opp_ftr", "d_ftr",
                ]),
                "def_three_par": _get_pct(item, [
                    "3pard%", "3pard", "3parpp_d", "3PAd_Rate",
                    "def_3par", "3pard_pct", "opp_3par",
                ]),
                # Pace (absolute — not rescaled)
                "pace":          _get_f(item, [
                    "adj_t", "adj_tempo", "tempo", "pace",
                    "adjt", "Adj. T.", "adj. t.",
                ]),
            }

        return stats

    def get_barttorvik_full_stats(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Fetch both AdjEM *and* defensive four-factor columns from BartTorvik.

        Returns a dict keyed by team name; each value contains:
            adj_oe, adj_de, adj_em                              (absolute efficiency)
            efg_pct, to_pct, ft_rate, three_par                (offensive, decimal)
            def_efg_pct, def_to_pct, def_ft_rate, def_three_par (defensive, decimal)
            pace                                                (possessions/40 min)

        Fetch strategy
        --------------
        1. JSON endpoint — ``{year}_super_standings.json`` — tried first.
           Natively structured; no header detection or positional fallbacks needed.
           Field names are parsed flexibly via _parse_barttorvik_json().

        2. CSV fallback — ``_BARTTORVIK_URL`` (env-configurable).
           Uses robust header-name detection + hard positional fallbacks so the
           pipeline survives minor CSV schema changes.

        All four-factor rates are returned as DECIMALS (0.505 not 50.5) in
        both paths.  AdjOE / AdjDE / pace are absolute and not rescaled.
        """
        # ----------------------------------------------------------------
        # Primary: JSON endpoint (no header parsing needed)
        # ----------------------------------------------------------------
        _json_url = f"https://barttorvik.com/{_CURRENT_YEAR}_super_standings.json"
        _json_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://barttorvik.com/",
        }
        try:
            _resp = requests.get(_json_url, headers=_json_headers, timeout=10)
            _resp.raise_for_status()
            _raw = _resp.json()
            if isinstance(_raw, list) and _raw:
                _json_stats = self._parse_barttorvik_json(_raw)
                if _json_stats:
                    n_def = sum(
                        1 for s in _json_stats.values()
                        if s.get("def_efg_pct") is not None
                    )
                    logger.info(
                        "BartTorvik JSON: loaded %d teams | def_efg populated=%d",
                        len(_json_stats), n_def,
                    )
                    if n_def == 0:
                        logger.warning(
                            "BartTorvik JSON: ZERO teams have def_efg_pct — "
                            "field name mapping may need updating.  "
                            "Falling back to CSV."
                        )
                    else:
                        return _json_stats
        except Exception as exc:
            logger.debug(
                "BartTorvik JSON endpoint (%s) unavailable: %s — "
                "falling back to CSV", _json_url, exc
            )

        # ----------------------------------------------------------------
        # Fallback: CSV path (original implementation)
        # ----------------------------------------------------------------
        url = _BARTTORVIK_URL
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            reader = csv.reader(io.StringIO(response.text))
            rows = [r for r in reader if any(c.strip() for c in r)]
        except Exception as exc:
            logger.warning("BartTorvik full-stats CSV error: %s", exc)
            return {}

        if not rows:
            return {}

        header = [h.strip().lower() for h in rows[0]]

        def _find_col(candidates: List[str]) -> Optional[int]:
            for c in candidates:
                if c in header:
                    return header.index(c)
            return None

        # ----------------------------------------------------------------
        # Column index detection (expanded candidate lists for robustness)
        # ----------------------------------------------------------------
        team_col = _find_col(["team", "teamname", "team_name", "school"]) or 1

        # Efficiency (absolute, ~90-130)
        oe_col = _find_col(["adjoe", "adj_oe", "adj oe", "o.adjoe", "adjoe.1"])
        de_col = _find_col(["adjde", "adj_de", "adj de", "o.adjde", "adjde.1"])

        # Offensive four factors (percentage scale in BartTorvik CSV)
        # NOTE: "3p%" is intentionally excluded from tpar_col — it is the
        # 3-point *shooting percentage*, not the attempt rate (3PA/FGA).
        efg_col  = _find_col(["efg%", "efg", "efg_pct", "efgpct", "off efg%", "o efg%",
                               "eff fg%", "efg pct"])
        tor_col  = _find_col(["tor", "to%", "to rate", "to_rate", "torate", "off to%",
                               "tov%", "tov", "turnover%"])
        ftr_col  = _find_col(["ftr", "ft rate", "ft_rate", "ftrate", "off ftr",
                               "fta/fga", "ft/fga"])
        tpar_col = _find_col(["3par%", "3par", "3p rate", "3p ar", "3pa rate",
                               "three par", "3pa%", "3pr%"])

        # Defensive four factors (percentage scale in BartTorvik CSV)
        defg_col  = _find_col(["efgd%", "efgd", "efgd_pct", "efgd pct",
                                "def efg%", "d efg%", "opp efg%", "efg_d", "efg%d"])
        tord_col  = _find_col(["tord", "to%d", "to rate d", "to_rate_d",
                                "def to%", "d to%", "opp to%", "tov%d", "tovd"])
        ftrd_col  = _find_col(["ftrd", "ftr d", "ftr_d", "ftrate_d",
                                "def ftr", "d ftr", "opp ftr", "ftrd%"])
        tpard_col = _find_col(["3pard%", "3pad%", "3pard", "3pad",
                                "3p rate d", "3par_d", "def 3par%", "opp 3par%", "3prd%"])

        # Pace (absolute — possessions per 40 min, ~60-80; no rescaling).
        # BartTorvik header is "Adj. T." → strip().lower() → "adj. t."
        # (note the period after "adj" that was missing from the original list).
        pace_col = _find_col(["adj. t.", "adj t.", "adj t", "adjt", "tempo", "adj_t",
                               "adj tempo", "poss", "possessions"])

        # ----------------------------------------------------------------
        # Hard positional fallbacks — standard BartTorvik T-Rank CSV layout:
        #   Rk(0) Team(1) Conf(2) G(3) Rec(4) AdjOE(5) AdjDE(6) Barthag(7)
        #   EFG%(8) EFGd%(9) TOR(10) TORd(11) ORB(12) DRB(13) FTR(14) FTRd(15)
        #   2P%(16) 2PD%(17) 3P%(18) 3PD%(19) Adj.T.(20) ...
        #
        # Applied only when header detection returns None so that explicit
        # header matches always take priority.  A WARNING is emitted so
        # operators can detect a CSV format change early.
        # ----------------------------------------------------------------
        _fallback_used: List[str] = []

        def _fallback(current: Optional[int], pos: int, label: str) -> int:
            if current is None:
                _fallback_used.append(f"{label}->col{pos}")
                return pos
            return current

        oe_col   = _fallback(oe_col,   5,  "AdjOE")
        de_col   = _fallback(de_col,   6,  "AdjDE")
        efg_col  = _fallback(efg_col,  8,  "EFG%")
        defg_col = _fallback(defg_col, 9,  "EFGd%")
        tor_col  = _fallback(tor_col,  10, "TOR")
        tord_col = _fallback(tord_col, 11, "TORd")
        ftr_col  = _fallback(ftr_col,  14, "FTR")
        ftrd_col = _fallback(ftrd_col, 15, "FTRd")
        pace_col = _fallback(pace_col, 20, "Adj.T.")
        # tpar_col / tpard_col intentionally have NO positional fallback:
        # columns 18/19 are 3-point *shooting* % (3P%/3PD%), not attempt rate.
        # Missing 3PAR keeps three_par=None so the model uses the 0.36 D1 default.

        if _fallback_used:
            logger.warning(
                "BartTorvik full-stats: header detection missed %d column(s); "
                "falling back to positional indexes: %s  |  "
                "Actual CSV headers (first 25): %s",
                len(_fallback_used),
                ", ".join(_fallback_used),
                header[:25],
            )

        logger.info(
            "BartTorvik full-stats column map: "
            "team=%s oe=%s de=%s efg=%s tor=%s ftr=%s tpar=%s "
            "defg=%s tord=%s ftrd=%s tpard=%s pace=%s",
            team_col, oe_col, de_col, efg_col, tor_col, ftr_col, tpar_col,
            defg_col, tord_col, ftrd_col, tpard_col, pace_col,
        )

        # ----------------------------------------------------------------
        # Scale helpers
        # ----------------------------------------------------------------
        def _safe_float(row: List[str], col: Optional[int]) -> Optional[float]:
            """Raw float — used for absolute values (AdjOE, AdjDE, pace).

            BartTorvik CSV cells can contain the stat followed by its national
            ranking (e.g. ``"105.3 12"`` or ``"52.1 345"``).  Only the first
            whitespace-separated token is the actual statistic; the ranking
            suffix is discarded before conversion.
            """
            if col is None or col >= len(row):
                return None
            cell = row[col].strip()
            if not cell:
                return None
            # Take only the primary stat — everything before the first space.
            first_token = cell.split()[0]
            try:
                return float(first_token)
            except (ValueError, TypeError):
                return None

        def _safe_pct(row: List[str], col: Optional[int]) -> Optional[float]:
            """
            Float converted to decimal fraction — used for all four-factor rates.

            BartTorvik stores rates as percentages (e.g., 50.5 for 50.5%).
            The Markov engine expects decimals (0.505).

            Detection heuristic: any value > 1.5 is unambiguously a percentage
            reading (no valid four-factor rate on the 0-1 scale exceeds 1.0;
            the closest edge case is a turnover rate of ~30% = 0.30).
            """
            val = _safe_float(row, col)
            if val is None:
                return None
            if val > 1.5:
                # Percentage scale → convert to decimal
                return val / 100.0
            # Already on decimal scale (defensive fallback)
            return val

        # ----------------------------------------------------------------
        # Build stats dict
        # ----------------------------------------------------------------
        stats: Dict[str, Dict[str, Optional[float]]] = {}
        for row in rows[1:]:
            if len(row) <= team_col:
                continue
            name = row[team_col].strip()
            if not name:
                continue

            adj_oe = _safe_float(row, oe_col)
            adj_de = _safe_float(row, de_col)
            adj_em = (adj_oe - adj_de) if (adj_oe is not None and adj_de is not None) else None

            stats[name] = {
                "adj_oe":        adj_oe,
                "adj_de":        adj_de,
                "adj_em":        adj_em,
                # Offensive four factors — converted pct→decimal
                "efg_pct":       _safe_pct(row, efg_col),
                "to_pct":        _safe_pct(row, tor_col),
                "ft_rate":       _safe_pct(row, ftr_col),
                "three_par":     _safe_pct(row, tpar_col),
                # Defensive four factors — converted pct→decimal
                "def_efg_pct":   _safe_pct(row, defg_col),
                "def_to_pct":    _safe_pct(row, tord_col),
                "def_ft_rate":   _safe_pct(row, ftrd_col),
                "def_three_par": _safe_pct(row, tpard_col),
                # Pace — absolute (no rescaling)
                "pace":          _safe_float(row, pace_col),
            }

        n_def_efg = sum(1 for s in stats.values() if s.get("def_efg_pct") is not None)
        n_def_to  = sum(1 for s in stats.values() if s.get("def_to_pct")  is not None)
        logger.info(
            "BartTorvik full-stats: %d teams | "
            "def_efg populated=%d, def_to populated=%d | "
            "efg col=%s, defg col=%s, tord col=%s",
            len(stats), n_def_efg, n_def_to,
            efg_col, defg_col, tord_col,
        )
        if n_def_efg == 0:
            logger.warning(
                "BartTorvik: ZERO teams have def_efg_pct — "
                "defensive column headers may have changed. "
                "Known candidates: %s",
                ["efgd%", "efgd", "def efg%", "d efg%", "opp efg%"],
            )
        return stats

    def save_team_profiles(self, db, season_year: Optional[int] = None) -> int:
        """
        Scrape BartTorvik full stats and upsert them into the team_profiles table.

        Returns the number of rows inserted or updated.

        Args:
            db: SQLAlchemy session (caller is responsible for commit/close).
            season_year: Override season year; defaults to _CURRENT_YEAR.
        """
        from backend.models import TeamProfile

        year = season_year or _CURRENT_YEAR
        full_stats = self.get_barttorvik_full_stats()
        if not full_stats:
            logger.warning("save_team_profiles: no BartTorvik data to save")
            return 0

        upserted = 0
        for team_name, stat in full_stats.items():
            existing = (
                db.query(TeamProfile)
                .filter_by(team_name=team_name, season_year=year, source="barttorvik")
                .first()
            )
            if existing is None:
                existing = TeamProfile(
                    team_name=team_name,
                    season_year=year,
                    source="barttorvik",
                )
                db.add(existing)

            # Only overwrite a column when the scrape returned a real value.
            # This prevents a partial scrape (e.g., a CSV missing defensive
            # columns) from nulling out previously-stored good data, which
            # would silently regress the Markov engine back to D1 defaults.
            _FIELD_MAP: List[tuple] = [
                ("adj_oe",       "adj_oe"),
                ("adj_de",       "adj_de"),
                ("adj_em",       "adj_em"),
                ("efg_pct",      "efg_pct"),
                ("to_pct",       "to_pct"),
                ("ft_rate",      "ft_rate"),
                ("three_par",    "three_par"),
                ("def_efg_pct",  "def_efg_pct"),
                ("def_to_pct",   "def_to_pct"),
                ("def_ft_rate",  "def_ft_rate"),
                ("def_three_par","def_three_par"),
                ("pace",         "pace"),
            ]
            # Validity ranges — reject scraped values outside D1 plausible bounds.
            # Ranking columns stored as decimals (e.g. rank 346 → 3.46) would fail
            # these checks and be discarded rather than poisoning the Markov engine.
            _EFG_RANGE = (0.35, 0.65)   # D1 eFG% always 39%–62%
            _TO_RANGE  = (0.10, 0.32)   # D1 TO rate always 10%–32%
            _CLAMPED_FIELDS = {
                "efg_pct":     _EFG_RANGE,
                "def_efg_pct": _EFG_RANGE,
                "to_pct":      _TO_RANGE,
                "def_to_pct":  _TO_RANGE,
            }
            for attr, key in _FIELD_MAP:
                val: Optional[float] = stat.get(key)
                if val is not None:
                    lo, hi = _CLAMPED_FIELDS.get(key, (None, None))
                    if lo is not None and not (lo <= val <= hi):
                        logger.debug(
                            "save_team_profiles: rejected %s=%s for %s "
                            "(valid range [%s, %s]) — likely a ranking column",
                            key, val, team_name, lo, hi,
                        )
                        val = None
                if val is not None:
                    setattr(existing, attr, val)
            existing.fetched_at = datetime.utcnow()

            upserted += 1

        logger.info("save_team_profiles: upserted %d TeamProfile rows", upserted)
        return upserted

    # -----------------------------------------------------------------------
    # EvanMiya
    # -----------------------------------------------------------------------

    def get_evanmiya_ratings(self) -> Dict[str, float]:
        """
        Fetch BPR (Box Plus-Minus) ratings from evanmiya.com.

        EvanMiya.com is a JavaScript-heavy site protected by Cloudflare.
        This method uses a layered approach:

          1. cloudscraper (TLS fingerprinting bypass) — preferred.
          2. tls_client library — fallback if cloudscraper unavailable.
          3. Standard requests — last resort.

        Within each HTTP client, three parse strategies are tried:
          A. Direct JSON API endpoint (fastest, most reliable).
          B. Embedded JSON in <script> tags (React/Next.js initial-state).
          C. Legacy HTML <table> parsing.

        Auto-drop: if scraping fails for the configured threshold of
        consecutive attempts, EvanMiya is marked as unavailable so the
        model stops applying the missing-source SD penalty on every game.
        The counter resets on the next successful fetch.

        BPR is on a similar scale to KenPom AdjEM; positive = above-average.

        Returns {} on any failure so the model gracefully falls back to
        KenPom + BartTorvik with renormalized weights.
        """
        import json as _json
        import re as _re

        # ---- Cloudflare bypass: prefer cloudscraper, fall back to requests --
        _session = None
        try:
            import cloudscraper
            _session = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            logger.debug("EvanMiya: using cloudscraper for Cloudflare bypass")
        except ImportError:
            pass

        if _session is None:
            try:
                import tls_client as _tls
                _session = _tls.Session(
                    client_identifier="chrome_120",
                    random_tls_extension_order=True,
                )
                logger.debug("EvanMiya: cloudscraper unavailable; using tls_client")
            except ImportError:
                pass

        if _session is None:
            logger.debug("EvanMiya: neither cloudscraper nor tls_client installed; using requests")
            _session = requests.Session()

        _HEADERS = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://evanmiya.com/",
        }

        def _get(url: str, timeout: int = 15) -> Optional[requests.Response]:
            """GET via whichever session is available; returns None on failure."""
            try:
                resp = _session.get(url, headers=_HEADERS, timeout=timeout)
                if hasattr(resp, "status_code") and resp.status_code == 200:
                    return resp
            except Exception:
                pass
            return None

        # ---- Strategy A: Known JSON API endpoints ---------------------------
        _api_candidates = [
            os.getenv("EVANMIYA_API_URL", ""),
            f"https://evanmiya.com/api/team_ratings?year={_CURRENT_YEAR}",
            f"https://evanmiya.com/api/ratings?year={_CURRENT_YEAR}",
            f"https://evanmiya.com/team_ratings?year={_CURRENT_YEAR}",
            "https://evanmiya.com/api/team_ratings",
            "https://evanmiya.com/api/ratings",
        ]

        for endpoint in _api_candidates:
            if not endpoint:
                continue
            resp = _get(endpoint, timeout=12)
            if resp is None:
                continue
            try:
                data = resp.json()
                ratings = self._parse_evanmiya_json(data)
                if ratings:
                    self._evanmiya_fail_count = 0
                    logger.info("EvanMiya: %d teams via API (%s)", len(ratings), endpoint)
                    return ratings
            except Exception:
                continue

        # ---- Strategy B & C: Fetch the main page ----------------------------
        page_resp = _get(_EVANMIYA_URL, timeout=20)
        if page_resp is None:
            logger.warning("EvanMiya: all HTTP strategies failed (Cloudflare block?)")
            self._evanmiya_fail_count = getattr(self, "_evanmiya_fail_count", 0) + 1
            _AUTO_DROP_THRESHOLD = int(os.getenv("EVANMIYA_AUTO_DROP_AFTER", "3"))
            if self._evanmiya_fail_count >= _AUTO_DROP_THRESHOLD:
                logger.error(
                    "EvanMiya auto-drop: %d consecutive failures. "
                    "Removing from active config — no SD penalty will be applied. "
                    "Set EVANMIYA_API_URL in .env or install cloudscraper to re-enable.",
                    self._evanmiya_fail_count,
                )
                # Signal to get_all_ratings that EvanMiya should be excluded
                # from weight renormalization (not just treated as missing).
                self._evanmiya_dropped = True
            return {}

        html = page_resp.text

        # Strategy B: Embedded JSON in <script> tags
        _json_patterns = [
            r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});',
            r'window\.__APP_DATA__\s*=\s*(\{.*?\});',
            r'__NEXT_DATA__\s*=\s*(\{.*?\})',
            r'(?:ratings|teams|data)\s*=\s*(\[.*?\])\s*[;,]',
        ]
        for pat in _json_patterns:
            for m in _re.finditer(pat, html, _re.DOTALL):
                try:
                    parsed = _json.loads(m.group(1))
                    ratings = self._parse_evanmiya_json(parsed)
                    if ratings:
                        self._evanmiya_fail_count = 0
                        logger.info("EvanMiya: %d teams from embedded script JSON", len(ratings))
                        return ratings
                except Exception:
                    continue

        # Strategy C: Legacy HTML table
        try:
            soup   = BeautifulSoup(html, "lxml")
            tables = soup.find_all("table")

            if not tables:
                logger.warning(
                    "EvanMiya: no <table> elements found (fully JS-rendered). "
                    "Install cloudscraper or set EVANMIYA_API_URL in .env."
                )
                self._evanmiya_fail_count = getattr(self, "_evanmiya_fail_count", 0) + 1
                return {}

            ratings: Dict[str, float] = {}
            for table in tables:
                header_row = table.find("tr")
                if header_row is None:
                    continue

                col_headers = [
                    th.get_text(strip=True).lower()
                    for th in header_row.find_all(["th", "td"])
                ]

                team_col = bpr_col = obpr_col = dbpr_col = None
                for i, h in enumerate(col_headers):
                    if h in ("team", "school", "teamname", "team name"):
                        team_col = i
                    if h in ("bpr", "net bpr", "net_bpr"):
                        bpr_col = i
                    if h in ("obpr", "off bpr", "off_bpr", "o bpr", "offensive bpr"):
                        obpr_col = i
                    if h in ("dbpr", "def bpr", "def_bpr", "d bpr", "defensive bpr"):
                        dbpr_col = i

                if team_col is None:
                    continue
                if bpr_col is None and (obpr_col is None or dbpr_col is None):
                    continue

                for row in table.find_all("tr")[1:]:
                    cells = row.find_all(["td", "th"])
                    if len(cells) <= team_col:
                        continue
                    team_name = cells[team_col].get_text(strip=True)
                    if not team_name:
                        continue
                    try:
                        if bpr_col is not None and bpr_col < len(cells):
                            bpr = float(cells[bpr_col].get_text(strip=True))
                        elif obpr_col is not None and dbpr_col is not None:
                            obpr = float(cells[obpr_col].get_text(strip=True))
                            dbpr = float(cells[dbpr_col].get_text(strip=True))
                            bpr  = obpr - dbpr
                        else:
                            continue
                        ratings[team_name] = bpr
                    except (ValueError, TypeError, IndexError):
                        continue

                if ratings:
                    break

            if ratings:
                self._evanmiya_fail_count = 0
                logger.info("EvanMiya: %d teams via HTML table", len(ratings))
            else:
                logger.warning(
                    "EvanMiya: page loaded but no BPR data found. "
                    "Set EVANMIYA_API_URL in .env to override the endpoint."
                )
                self._evanmiya_fail_count = getattr(self, "_evanmiya_fail_count", 0) + 1

            return ratings

        except Exception as exc:
            logger.warning("EvanMiya parse error: %s", exc)
            self._evanmiya_fail_count = getattr(self, "_evanmiya_fail_count", 0) + 1
            return {}

    def _parse_evanmiya_json(self, data: object) -> Dict[str, float]:
        """
        Extract {team: bpr} from a variety of JSON shapes that EvanMiya may return.

        Handles:
          - List of dicts: [{"team": "...", "bpr": 5.2}, ...]
          - Dict with a "data" or "ratings" list key
          - Dict with team keys mapping to stat dicts

        Returns an empty dict if the shape is unrecognised.
        """
        _TEAM_KEYS   = ("team", "teamname", "Team", "TeamName", "school")
        _BPR_KEYS    = ("bpr", "BPR", "net_bpr", "NetBPR")
        _OBPR_KEYS   = ("obpr", "OBPR", "off_bpr", "OffBPR")
        _DBPR_KEYS   = ("dbpr", "DBPR", "def_bpr", "DefBPR")

        # Unwrap common envelope shapes
        if isinstance(data, dict):
            for key in ("data", "ratings", "teams", "results"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            return {}

        ratings: Dict[str, float] = {}
        for item in data:
            if not isinstance(item, dict):
                continue

            team_name = None
            for k in _TEAM_KEYS:
                if k in item:
                    team_name = str(item[k]).strip()
                    break
            if not team_name:
                continue

            bpr = None
            for k in _BPR_KEYS:
                if k in item:
                    try:
                        bpr = float(item[k])
                    except (TypeError, ValueError):
                        pass
                    break

            if bpr is None:
                # Try OBPR - DBPR
                try:
                    obpr_key = next((k for k in _OBPR_KEYS if k in item), None)
                    dbpr_key = next((k for k in _DBPR_KEYS if k in item), None)
                    if obpr_key and dbpr_key:
                        bpr = float(item[obpr_key]) - float(item[dbpr_key])
                except (TypeError, ValueError):
                    pass

            if bpr is not None:
                ratings[team_name] = bpr

        return ratings

    # -----------------------------------------------------------------------
    # Aggregator
    # -----------------------------------------------------------------------

    def get_all_ratings(self, use_cache: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Fetch all three sources, caching results for up to 6 hours.

        Includes a special ``"_meta"`` key in the returned dict:
            _meta["evanmiya_dropped"] = True  — EvanMiya auto-dropped after
            repeated Cloudflare failures; callers should NOT apply an
            SD penalty for the missing source since the data is unavailable
            through no fault of the model.
        """
        if use_cache and self.cache and self.cache_timestamp:
            age_hours = (
                datetime.utcnow() - self.cache_timestamp
            ).total_seconds() / 3600
            if age_hours < 6:
                return self.cache

        # Reset the drop flag before each full refresh so a future successful
        # scrape automatically re-enables EvanMiya.
        if self._evanmiya_dropped:
            logger.info(
                "EvanMiya was auto-dropped; retrying this cycle "
                "(fail_count=%d)", self._evanmiya_fail_count,
            )
            self._evanmiya_dropped = False

        evanmiya_data = self.get_evanmiya_ratings()
        evanmiya_dropped = self._evanmiya_dropped  # re-read post-call

        ratings = {
            "kenpom":    self.get_kenpom_ratings(),
            "barttorvik": self.get_barttorvik_ratings(),
            "evanmiya":  evanmiya_data,
            "_meta": {
                "evanmiya_dropped": evanmiya_dropped,
                "evanmiya_fail_count": self._evanmiya_fail_count,
            },
        }

        self.cache           = ratings
        self.cache_timestamp = datetime.utcnow()

        logger.info(
            "Ratings loaded — KenPom: %d, BartTorvik: %d, EvanMiya: %d%s",
            len(ratings["kenpom"]),
            len(ratings["barttorvik"]),
            len(evanmiya_data),
            " [DROPPED]" if evanmiya_dropped else "",
        )
        return ratings


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_ratings_service: Optional[RatingsService] = None


def get_ratings_service() -> RatingsService:
    global _ratings_service
    if _ratings_service is None:
        _ratings_service = RatingsService()
    return _ratings_service


def fetch_current_ratings() -> Dict:
    return get_ratings_service().get_all_ratings()
