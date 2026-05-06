"""
ProjectionAssemblyService -- Sprint 2

Assembles CanonicalProjection rows from two sources:
  1. Static board (player_board.get_board()) -- Steamer season projections.
     source_engine = STATIC_BOARD when no Statcast data available.
  2. Statcast metrics (StatcastBatterMetrics / StatcastPitcherMetrics) -- observation.
     source_engine = SAVANT_ADJUSTED when Statcast data is present for fusion.

Counting stat provenance (hybrid):
  proj_hr, proj_sb  -- derived from fusion rate stats x projected_pa/ip
  proj_r, proj_rbi  -- static board passthrough (lineup-context-dependent)
  proj_w            -- PitcherCountingStatFormulas.project_wins(era, ip)
  proj_sv           -- static board passthrough (closer-role-dependent)
  proj_k            -- k_per_nine x projected_ip / 9

Writes one CanonicalProjection row per player per run (upsert on player_id + projection_date).
Writes CategoryImpact rows (z-scores computed across full player pool).

Does NOT replace daily_ingestion.py scheduler jobs -- that wiring is done separately.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Optional

from sqlalchemy.orm import Session

from backend.models import (
    CanonicalProjection,
    CategoryImpact,
    PlayerIdentity,
    PlayerProjection,
    StatcastBatterMetrics,
    StatcastPitcherMetrics,
    SessionLocal,
)
from backend.fantasy_baseball.player_board import get_board
from backend.fantasy_baseball.fusion_engine import (
    fuse_batter_projection,
    fuse_pitcher_projection,
    FusionResult,
    PitcherCountingStatFormulas,
)
from backend.fantasy_baseball.id_resolution_service import _normalize_name

try:
    from backend.fantasy_baseball.fusion_engine import to_season_counts
except ImportError:
    def to_season_counts(result, projected_pa, projected_ip, board_proj):  # type: ignore[misc]
        return {
            "proj_hr": 0,
            "proj_sb": 0,
            "proj_r": int(board_proj.get("r", 0)),
            "proj_rbi": int(board_proj.get("rbi", 0)),
            "proj_w": 0,
            "proj_sv": int(board_proj.get("sv", 0)),
            "proj_k": 0,
        }


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level z-score helpers
# ---------------------------------------------------------------------------

def _zscore_pool(values: list) -> tuple:
    """Return (mean, std) for a list, ignoring None. Returns (0.0, 1.0) if < 2 values."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return 0.0, 1.0
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / len(clean)
    std = variance ** 0.5
    return mean, max(std, 1e-9)


def _zscore(value, mean: float, std: float, direction: int = 1):
    """Return z-score for value. direction=-1 for negative categories (ERA, WHIP)."""
    if value is None:
        return None
    return ((value - mean) / std) * direction


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ProjectionAssemblyService:
    """Assemble and upsert CanonicalProjection rows for all board players."""

    # Batcher size -- commit every N players
    _BATCH_SIZE = 50

    def __init__(self, db: Session, season: int = 2026):
        self.db = db
        self.season = season

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, projection_date: Optional[date] = None) -> dict:
        """
        Assemble and upsert CanonicalProjection rows for all board players.
        Returns summary: {total, assembled, identity_misses, upserted, errors}
        """
        if projection_date is None:
            projection_date = datetime.now(ZoneInfo("America/New_York")).date()

        summary = {
            "total": 0,
            "assembled": 0,
            "identity_misses": 0,
            "mlbam_null_fallback": 0,  # resolved by name but no MLBAM id — STATIC_BOARD only
            "upserted": 0,
            "errors": 0,
        }

        board = get_board()
        summary["total"] = len(board)

        batters = [p for p in board if p.get("type") == "batter"]
        pitchers = [p for p in board if p.get("type") == "pitcher"]

        # Build z-score pools from board projections
        z_pool = self._build_zscore_pools(batters, pitchers)

        players_in_batch = 0
        for player in board:
            try:
                upserted = self._process_player(player, projection_date, z_pool, summary)
                if upserted is None:
                    summary["identity_misses"] += 1
                    continue
                summary["assembled"] += 1
                summary["upserted"] += 1
                players_in_batch += 1

                if players_in_batch >= self._BATCH_SIZE:
                    self.db.commit()
                    players_in_batch = 0

            except Exception as exc:
                logger.warning("Error assembling projection for %s: %s", player.get("name"), exc)
                self.db.rollback()
                summary["errors"] += 1
                players_in_batch = 0

        # Final commit for remaining batch
        if players_in_batch > 0:
            try:
                self.db.commit()
            except Exception as exc:
                logger.error("Final batch commit failed: %s", exc)
                self.db.rollback()

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_player(
        self,
        player: dict,
        projection_date: date,
        z_pool: dict,
        summary: dict,
    ) -> Optional[int]:
        """
        Assemble one CanonicalProjection row and its CategoryImpact children.
        Returns the canonical_projection.id on success, None on true identity miss.

        Identity miss: no PlayerIdentity row exists for the player name at all.
        MLBAM-null fallback: PlayerIdentity row exists but mlbam_id is NULL.
          → assembled as STATIC_BOARD using -(yahoo_id) as the player_id namespace.
          → increments summary["mlbam_null_fallback"] (informational, not a gate failure).
        Uses a savepoint so a single-player failure rolls back cleanly.
        """
        name = player.get("name", "")
        normalized = _normalize_name(name)

        identity_row = self._resolve_identity_row(self.db, normalized)
        if identity_row is None:
            logger.debug("Identity miss for player: %s (normalized: %s)", name, normalized)
            return None

        # Determine canonical player_id and Statcast lookup key
        mlbam_id: Optional[int] = identity_row.mlbam_id
        if mlbam_id is not None:
            player_id = mlbam_id
        elif identity_row.yahoo_id is not None:
            # No MLBAM mapping — use -(yahoo_id) as a distinct negative namespace.
            # Statcast enrichment is skipped; source_engine will be STATIC_BOARD.
            player_id = -(int(identity_row.yahoo_id))
            mlbam_id = None  # explicitly None so Statcast lookups are skipped
            summary["mlbam_null_fallback"] += 1
            logger.debug(
                "MLBAM-null fallback for %s: yahoo_id=%s → player_id=%s",
                name, identity_row.yahoo_id, player_id,
            )
        else:
            # No usable ID at all — cannot write to canonical_projections
            logger.warning("No usable ID (mlbam_id=None, yahoo_id=None) for %s — skipping", name)
            return None

        player_type_raw = player.get("type", "batter")
        player_type = "BATTER" if player_type_raw == "batter" else "PITCHER"
        board_proj = player.get("proj", {})

        sp = self.db.begin_nested()
        try:
            if player_type == "BATTER":
                cp_id = self._assemble_batter(player_id, mlbam_id, player_type, board_proj, projection_date, z_pool)
            else:
                cp_id = self._assemble_pitcher(player_id, mlbam_id, player_type, board_proj, projection_date, z_pool)
            sp.commit()
            return cp_id
        except Exception:
            sp.rollback()
            raise

    def _assemble_batter(
        self,
        player_id: int,
        mlbam_id: Optional[int],
        player_type: str,
        board_proj: dict,
        projection_date: date,
        z_pool: dict,
    ) -> int:
        # Sprint 5: prefer live DB counting stats over March CSV when fangraphs_ros has run
        live_proj = self._get_live_projection(mlbam_id, "")
        if live_proj is not None and live_proj.hr is not None:
            board_proj = dict(board_proj)
            board_proj["hr"] = live_proj.hr or board_proj.get("hr", 0)
            board_proj["r"] = live_proj.r or board_proj.get("r", 0)
            board_proj["rbi"] = live_proj.rbi or board_proj.get("rbi", 0)
            board_proj["sb"] = live_proj.sb or board_proj.get("sb", 0)

        pa = float(board_proj.get("pa", 0) or 0)
        # mlbam_id may be None for Yahoo-only players — skip Statcast lookup safely
        metrics = self._get_statcast_batter(mlbam_id) if mlbam_id is not None else None
        sample_size = int(metrics.pa) if (metrics and metrics.pa) else 0

        steamer = self._build_batter_steamer(board_proj)
        statcast_dict = self._build_batter_statcast(metrics) if metrics else None

        result = fuse_batter_projection(steamer, statcast_dict, sample_size)
        counts = to_season_counts(result, pa, 0.0, board_proj)

        source_engine = (
            "SAVANT_ADJUSTED"
            if (metrics is not None and result.source == "fusion")
            else "STATIC_BOARD"
        )
        confidence = self._confidence_score(result, sample_size)

        # Rate stats from fusion result
        fproj = result.proj
        proj_avg = fproj.get("avg")
        proj_obp = fproj.get("obp")
        proj_slg = fproj.get("slg")
        proj_ops = fproj.get("ops")

        # Upsert
        self._delete_existing(player_id, projection_date)

        row = CanonicalProjection(
            projection_id=str(uuid.uuid4()),
            player_id=player_id,
            player_type=player_type,
            source_engine=source_engine,
            projection_date=projection_date,
            season=self.season,
            projected_pa=pa,
            projected_ab=pa * 0.88 if pa else None,  # ~88% of PA are AB (walk-adjusted)
            proj_hr=counts.get("proj_hr"),
            proj_sb=counts.get("proj_sb"),
            proj_r=counts.get("proj_r"),
            proj_rbi=counts.get("proj_rbi"),
            proj_avg=proj_avg,
            proj_obp=proj_obp,
            proj_slg=proj_slg,
            proj_ops=proj_ops,
            xwoba=metrics.xwoba if metrics else None,
            barrel_pct=self._barrel_decimal(metrics),
            hardhit_pct=(metrics.hard_hit_percent / 100.0) if (metrics and metrics.hard_hit_percent) else None,
            bb_pct=fproj.get("bb_percent"),
            k_pct=fproj.get("k_percent"),
            confidence_score=confidence,
            sample_size=float(sample_size),
            shrinkage_applied=self._shrinkage_from_result(result, sample_size),
            explainability_metadata={
                "source": result.source,
                "components_fused": result.components_fused,
                "xwoba_override_detected": result.xwoba_override_detected,
            },
        )
        self.db.add(row)
        self.db.flush()  # populate row.id for FK

        # Category impacts
        batter_values = {
            "HR": float(counts.get("proj_hr") or 0),
            "SB": float(counts.get("proj_sb") or 0),
            "R": float(counts.get("proj_r") or 0),
            "RBI": float(counts.get("proj_rbi") or 0),
            "AVG": proj_avg,
            "OBP": proj_obp,
            "OPS": proj_ops,
        }
        impacts = self._build_category_impacts(row.id, player_type, batter_values, z_pool)
        for impact in impacts:
            self.db.add(impact)

        return row.id

    def _assemble_pitcher(
        self,
        player_id: int,
        mlbam_id: Optional[int],
        player_type: str,
        board_proj: dict,
        projection_date: date,
        z_pool: dict,
    ) -> int:
        # Sprint 5: prefer live DB counting stats over March CSV when fangraphs_ros has run
        live_proj = self._get_live_projection(mlbam_id, "")
        if live_proj is not None:
            board_proj = dict(board_proj)
            if live_proj.w is not None:
                board_proj["w"] = live_proj.w
            if live_proj.k_pit is not None:
                board_proj["k"] = live_proj.k_pit
            if live_proj.nsv is not None:
                board_proj["sv"] = live_proj.nsv

        ip = float(board_proj.get("ip", 0) or 0)
        # mlbam_id may be None for Yahoo-only players — skip Statcast lookup safely
        metrics = self._get_statcast_pitcher(mlbam_id) if mlbam_id is not None else None
        sample_size = int(metrics.ip) if (metrics and metrics.ip) else 0

        steamer = self._build_pitcher_steamer(board_proj)
        statcast_dict = self._build_pitcher_statcast(metrics) if metrics else None

        result = fuse_pitcher_projection(steamer, statcast_dict, sample_size)
        counts = to_season_counts(result, 0.0, ip, board_proj)

        source_engine = (
            "SAVANT_ADJUSTED"
            if (metrics is not None and result.source == "fusion")
            else "STATIC_BOARD"
        )
        confidence = self._confidence_score(result, sample_size)

        fproj = result.proj
        proj_era = fproj.get("era")
        proj_whip = fproj.get("whip")
        proj_k9 = fproj.get("k_per_nine")

        self._delete_existing(player_id, projection_date)

        row = CanonicalProjection(
            projection_id=str(uuid.uuid4()),
            player_id=player_id,
            player_type=player_type,
            source_engine=source_engine,
            projection_date=projection_date,
            season=self.season,
            projected_ip=ip,
            proj_w=counts.get("proj_w"),
            proj_sv=counts.get("proj_sv"),
            proj_k=counts.get("proj_k"),
            proj_era=proj_era,
            proj_whip=proj_whip,
            proj_k9=proj_k9,
            xera=metrics.xera if metrics else None,
            csw_pct=None,
            swstr_pct=(metrics.whiff_percent / 100.0) if (metrics and metrics.whiff_percent) else None,
            confidence_score=confidence,
            sample_size=float(sample_size),
            shrinkage_applied=self._shrinkage_from_result(result, sample_size),
            explainability_metadata={
                "source": result.source,
                "components_fused": result.components_fused,
                "xera_override_detected": result.xwoba_override_detected,
            },
        )
        self.db.add(row)
        self.db.flush()

        pitcher_values = {
            "W": float(counts.get("proj_w") or 0),
            "K": float(counts.get("proj_k") or 0),
            "SV": float(counts.get("proj_sv") or 0),
            "ERA": proj_era,
            "WHIP": proj_whip,
            "K9": proj_k9,
        }
        impacts = self._build_category_impacts(row.id, player_type, pitcher_values, z_pool)
        for impact in impacts:
            self.db.add(impact)

        return row.id

    def _delete_existing(self, player_id: int, projection_date: date) -> None:
        existing = (
            self.db.query(CanonicalProjection)
            .filter(
                CanonicalProjection.player_id == player_id,
                CanonicalProjection.projection_date == projection_date,
            )
            .first()
        )
        if existing:
            self.db.delete(existing)
            self.db.flush()

    # ------------------------------------------------------------------
    # Identity + Statcast lookups
    # ------------------------------------------------------------------

    def _resolve_identity_row(self, session: Session, normalized_name: str) -> Optional[PlayerIdentity]:
        """
        Query PlayerIdentity by normalized_name. Returns the full row or None if not found.

        Callers should handle the case where row.mlbam_id is None (Yahoo-only players)
        by falling back to -(yahoo_id) as the canonical player_id namespace.
        """
        return (
            session.query(PlayerIdentity)
            .filter(PlayerIdentity.normalized_name == normalized_name)
            .first()
        )

    def _resolve_mlbam_id(self, session: Session, normalized_name: str) -> Optional[int]:
        """Deprecated: use _resolve_identity_row() instead. Returns mlbam_id or None."""
        row = self._resolve_identity_row(session, normalized_name)
        return row.mlbam_id if row is not None else None

    def _get_statcast_batter(self, mlbam_id: int) -> Optional[StatcastBatterMetrics]:
        """Query StatcastBatterMetrics by str(mlbam_id) for current season."""
        return (
            self.db.query(StatcastBatterMetrics)
            .filter(
                StatcastBatterMetrics.mlbam_id == str(mlbam_id),
                StatcastBatterMetrics.season == self.season,
            )
            .first()
        )

    def _get_statcast_pitcher(self, mlbam_id: int) -> Optional[StatcastPitcherMetrics]:
        """Query StatcastPitcherMetrics by str(mlbam_id) for current season."""
        return (
            self.db.query(StatcastPitcherMetrics)
            .filter(
                StatcastPitcherMetrics.mlbam_id == str(mlbam_id),
                StatcastPitcherMetrics.season == self.season,
            )
            .first()
        )

    def _get_live_projection(
        self, mlbam_id: Optional[int], player_name: str
    ) -> Optional[PlayerProjection]:
        """Return a PlayerProjection row if one exists with update_method != 'prior'.

        Only non-prior rows are trusted over the static March CSV board.
        'prior' means the row was seeded from the March CSV and has not been
        updated by fangraphs_ros or a Bayesian pass yet -- treat as no override.

        Returns None when:
          - No row exists for the player.
          - Row exists but update_method == 'prior' (stale -- treat as no override).
          - Any DB error (never raises).
        """
        try:
            if mlbam_id is not None:
                row = (
                    self.db.query(PlayerProjection)
                    .filter(PlayerProjection.player_id == str(mlbam_id))
                    .first()
                )
                if row is not None:
                    return row if row.update_method != "prior" else None
            # Fallback: exact name match (players without MLBAM mapping)
            row = (
                self.db.query(PlayerProjection)
                .filter(PlayerProjection.player_name.ilike(f"%{player_name}%"))
                .first()
            )
            if row is not None:
                return row if row.update_method != "prior" else None
        except Exception as exc:
            logger.debug("_get_live_projection failed for %s: %s", player_name, exc)
        return None

    # ------------------------------------------------------------------
    # Steamer dict builders
    # ------------------------------------------------------------------

    def _build_batter_steamer(self, board_proj: dict) -> dict:
        """Extract fusion-compatible steamer dict from board batter projection."""
        pa = board_proj.get("pa", 0) or 0
        return {
            "avg": board_proj.get("avg"),
            "obp": 0.330,  # league-average prior (board omits per-player OBP)
            "slg": board_proj.get("slg"),
            "k_percent": board_proj["k_bat"] / pa if pa > 0 else 0.225,
            "bb_percent": 0.080,  # population prior
            "hr_per_pa": board_proj["hr"] / pa if pa > 0 else 0.035,
            "sb_per_pa": board_proj["nsb"] / pa if pa > 0 else 0.010,
        }

    def _build_pitcher_steamer(self, board_proj: dict) -> dict:
        """Extract fusion-compatible steamer dict from board pitcher projection."""
        return {
            "era": board_proj.get("era"),
            "whip": board_proj.get("whip"),
            "k_percent": 0.22,  # population prior (not in board)
            "bb_percent": 0.07,
            "k_per_nine": board_proj.get("k9"),
            "bb_per_nine": 3.0,  # population prior
        }

    # ------------------------------------------------------------------
    # Statcast dict builders
    # ------------------------------------------------------------------

    def _build_batter_statcast(self, metrics: StatcastBatterMetrics) -> dict:
        """Convert StatcastBatterMetrics ORM row to fusion-compatible dict."""
        obp = None
        if metrics.ops is not None and metrics.slg is not None:
            obp = metrics.ops - metrics.slg
        return {
            "avg": metrics.avg,
            "obp": obp,
            "slg": metrics.slg,
            "xwoba": metrics.xwoba,
            "barrel_pct": (metrics.barrel_percent / 100.0) if metrics.barrel_percent else None,
            "k_percent": None,  # not in StatcastBatterMetrics
            "bb_percent": None,
        }

    def _build_pitcher_statcast(self, metrics: StatcastPitcherMetrics) -> dict:
        """Convert StatcastPitcherMetrics ORM row to fusion-compatible dict."""
        return {
            "era": metrics.era,
            "whip": None,  # not in StatcastPitcherMetrics
            "xera": metrics.xera,
            "k_percent": metrics.k_percent,
            "bb_percent": metrics.bb_percent,
            "k_per_nine": metrics.k_9,
        }

    # ------------------------------------------------------------------
    # Confidence + shrinkage
    # ------------------------------------------------------------------

    def _confidence_score(self, result: FusionResult, sample_size: int) -> float:
        """
        Returns 0.0-1.0 composite confidence scalar.
        Formula: min(1.0, sample_size / 300) * source_multiplier
        """
        sample_factor = min(1.0, sample_size / 300.0)
        if result.source == "fusion":
            source_mult = 1.0
        elif result.source in ("steamer", "statcast_shrunk"):
            source_mult = 0.7
        else:
            source_mult = 0.4
        return sample_factor * source_mult

    def _shrinkage_from_result(self, result: FusionResult, sample_size: int) -> float:
        """
        Degree of regression to mean: 1.0 = full prior (no data), 0.0 = full observation.
        For population_prior: always 1.0. For fusion: sample-dependent.
        """
        if result.source == "population_prior":
            return 1.0
        if result.source == "steamer":
            return 1.0  # prior passthrough unchanged
        # fusion or statcast_shrunk: approximate from sample_size
        # At sample_size=0 -> 1.0 (all prior), at 300 -> 0.5, beyond -> lower
        return max(0.0, 1.0 - min(1.0, sample_size / 300.0) * 0.8)

    # ------------------------------------------------------------------
    # CategoryImpact builder
    # ------------------------------------------------------------------

    def _build_category_impacts(
        self,
        projection_id: int,
        player_type: str,
        values: dict,
        z_pool: dict,
    ) -> list:
        """Build CategoryImpact rows for all relevant categories."""
        # Negative categories: lower is better -- flip direction
        NEGATIVE_CATS = {"ERA", "WHIP"}

        impacts = []
        for category, value in values.items():
            pool_key = category.lower()
            pool = z_pool.get(pool_key, (0.0, 1.0))
            mean, std = pool
            direction = -1 if category in NEGATIVE_CATS else 1
            z = _zscore(value, mean, std, direction)

            # denominator_weight: 1.0 for counting stats, placeholder 1.0 for rate stats
            denominator_weight = 1.0

            impact = CategoryImpact(
                canonical_projection_id=projection_id,
                category=category,
                projected_value=value,
                z_score=z,
                generic_marginal_impact=z,  # placeholder; matchup-specific delta at runtime
                denominator_weight=denominator_weight,
            )
            impacts.append(impact)
        return impacts

    # ------------------------------------------------------------------
    # Z-score pool builder
    # ------------------------------------------------------------------

    def _build_zscore_pools(self, batters: list, pitchers: list) -> dict:
        """
        Build (mean, std) tuples for each fantasy category using board projections.

        Keys: r, hr, rbi, sb, avg, ops (batters) + w, k_pit, sv, era, whip, k9 (pitchers)
        """
        def _collect(players, proj_key):
            return [p.get("proj", {}).get(proj_key) for p in players]

        pool = {}
        # Batter pools (positive direction)
        pool["r"] = _zscore_pool(_collect(batters, "r"))
        pool["hr"] = _zscore_pool(_collect(batters, "hr"))
        pool["rbi"] = _zscore_pool(_collect(batters, "rbi"))
        pool["sb"] = _zscore_pool([p.get("proj", {}).get("nsb") for p in batters])
        pool["avg"] = _zscore_pool(_collect(batters, "avg"))
        pool["ops"] = _zscore_pool(_collect(batters, "ops"))
        pool["obp"] = _zscore_pool([p.get("proj", {}).get("obp") for p in batters])

        # Pitcher pools
        pool["w"] = _zscore_pool(_collect(pitchers, "w"))
        pool["k_pit"] = _zscore_pool(_collect(pitchers, "k_pit"))
        pool["sv"] = _zscore_pool(_collect(pitchers, "sv"))
        pool["era"] = _zscore_pool(_collect(pitchers, "era"))
        pool["whip"] = _zscore_pool(_collect(pitchers, "whip"))
        pool["k9"] = _zscore_pool(_collect(pitchers, "k9"))

        # Map CategoryImpact category strings to pool keys
        return {
            # Batters
            "r": pool["r"],
            "hr": pool["hr"],
            "rbi": pool["rbi"],
            "sb": pool["sb"],
            "avg": pool["avg"],
            "ops": pool["ops"],
            "obp": pool["obp"],
            # Pitchers
            "w": pool["w"],
            "k": pool["k_pit"],
            "sv": pool["sv"],
            "era": pool["era"],
            "whip": pool["whip"],
            "k9": pool["k9"],
        }

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _barrel_decimal(self, metrics: Optional[StatcastBatterMetrics]) -> Optional[float]:
        """Convert barrel_percent (e.g. 12.5) to decimal (0.125)."""
        if metrics is None or metrics.barrel_percent is None:
            return None
        return metrics.barrel_percent / 100.0


# ---------------------------------------------------------------------------
# Top-level convenience entry point
# ---------------------------------------------------------------------------

def assemble_projections(season: int = 2026, projection_date: Optional[date] = None) -> dict:
    """Top-level entry point for daily_ingestion.py job."""
    db = SessionLocal()
    try:
        svc = ProjectionAssemblyService(db, season=season)
        return svc.run(projection_date=projection_date)
    finally:
        db.close()
