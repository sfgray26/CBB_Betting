"""
PredictiveStatsService — FIP/xFIP/SIERA + xwOBA/Hard-Hit%/wRC+ pipeline.

Gated behind feature flag: predictive_stats_v1_enabled (default: disabled).

Usage:
    from backend.services.predictive_stats_service import get_predictive_stats_service
    svc = get_predictive_stats_service()
    if svc.is_enabled():
        stats = svc.get_pitcher_predictive_stats(player_id)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from backend.services.config_service import is_flag_enabled

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

FLAG_NAME = "predictive_stats_v1_enabled"


def _serialize_stats(value: Any) -> Any:
    """Convert dataclass-backed cache objects to plain dicts for API-safe returns."""
    if is_dataclass(value):
        return asdict(value)
    return value


def _estimate_row_count(stats: Any) -> int:
    """Estimate how many player rows were returned for observability purposes."""
    if stats is None:
        return 0
    if isinstance(stats, (list, tuple, set)):
        return len(stats)
    if isinstance(stats, dict):
        if not stats:
            return 0
        nested_values = [
            value
            for value in stats.values()
            if isinstance(value, dict) or is_dataclass(value)
        ]
        return len(stats) if nested_values else 1
    return 1


class PredictiveStatsObservability:
    """Tracks refresh timestamps, row counts, and error states for observability."""

    def __init__(self):
        self.last_pitcher_refresh: Optional[datetime] = None
        self.last_batter_refresh: Optional[datetime] = None
        self.pitcher_row_count: int = 0
        self.batter_row_count: int = 0
        self.last_error: Optional[str] = None
        self.is_stale: bool = True

    def record_pitcher_refresh(self, row_count: int) -> None:
        self.last_pitcher_refresh = datetime.now(_ET)
        self.pitcher_row_count = row_count
        self.is_stale = False
        self.last_error = None
        logger.info("predictive_stats: pitcher refresh complete — %s rows", row_count)

    def record_batter_refresh(self, row_count: int) -> None:
        self.last_batter_refresh = datetime.now(_ET)
        self.batter_row_count = row_count
        self.is_stale = False
        self.last_error = None
        logger.info("predictive_stats: batter refresh complete — %s rows", row_count)

    def record_error(self, error: str) -> None:
        self.last_error = error
        self.is_stale = True
        logger.error("predictive_stats: error — %s", error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag_enabled": is_flag_enabled(FLAG_NAME),
            "last_pitcher_refresh": self.last_pitcher_refresh,
            "last_batter_refresh": self.last_batter_refresh,
            "pitcher_row_count": self.pitcher_row_count,
            "batter_row_count": self.batter_row_count,
            "last_error": self.last_error,
            "is_stale": self.is_stale,
        }


_observability = PredictiveStatsObservability()


class PredictiveStatsService:
    """
    Gates FIP/xFIP/SIERA (FanGraphs/pybaseball) and xwOBA/Hard-Hit%/wRC+ (Savant/pybaseball)
    behind the predictive_stats_v1_enabled feature flag.

    When flag is disabled: returns None / empty results gracefully.
    When flag is enabled: delegates to existing ingestion modules.
    """

    def is_enabled(self) -> bool:
        return is_flag_enabled(FLAG_NAME)

    def get_observability(self) -> dict[str, Any]:
        return _observability.to_dict()

    def get_pitcher_predictive_stats(self, player_id: Optional[int] = None) -> Optional[dict[str, Any]]:
        """
        Return FIP/xFIP/SIERA for pitcher(s).

        Returns None if flag is disabled.
        Delegates to pybaseball_loader for FanGraphs data when an adapter exists.
        """
        if not self.is_enabled():
            logger.debug("predictive_stats: flag disabled, skipping pitcher stats for %s", player_id)
            return None

        try:
            from backend.fantasy_baseball import pybaseball_loader as pybaseball_module

            stats: Any = {}
            loader_cls = getattr(pybaseball_module, "PybaseballLoader", None)
            if loader_cls is not None:
                loader = loader_cls()
                if hasattr(loader, "get_pitcher_fip_stats"):
                    stats = loader.get_pitcher_fip_stats(player_id=player_id) or {}
            elif player_id is None and hasattr(pybaseball_module, "load_pybaseball_pitchers"):
                cached = pybaseball_module.load_pybaseball_pitchers()
                stats = {
                    name: _serialize_stats(value)
                    for name, value in cached.items()
                } if isinstance(cached, dict) else {}

            _observability.record_pitcher_refresh(_estimate_row_count(stats))
            return stats
        except Exception as exc:
            _observability.record_error(f"pitcher stats: {exc}")
            return None

    def get_batter_predictive_stats(self, player_id: Optional[int] = None) -> Optional[dict[str, Any]]:
        """
        Return xwOBA/Hard-Hit%/wRC+ for batter(s).

        Returns None if flag is disabled.
        Delegates to savant_ingestion when an adapter exists, otherwise falls back
        to pybaseball cached batter metrics.
        """
        if not self.is_enabled():
            logger.debug("predictive_stats: flag disabled, skipping batter stats for %s", player_id)
            return None

        try:
            from backend.fantasy_baseball import pybaseball_loader as pybaseball_module
            from backend.fantasy_baseball import savant_ingestion as savant_module

            stats: Any = {}
            svc_cls = getattr(savant_module, "SavantIngestionService", None)
            if svc_cls is not None:
                svc = svc_cls()
                if hasattr(svc, "get_batter_xwoba_stats"):
                    stats = svc.get_batter_xwoba_stats(player_id=player_id) or {}
            elif hasattr(pybaseball_module, "load_pybaseball_batters"):
                cached = pybaseball_module.load_pybaseball_batters()
                if isinstance(cached, dict):
                    if player_id is not None:
                        selected = {
                            name: value
                            for name, value in cached.items()
                            if getattr(value, "player_id", None) == player_id
                        }
                        stats = {
                            name: _serialize_stats(value)
                            for name, value in selected.items()
                        }
                    else:
                        stats = {
                            name: _serialize_stats(value)
                            for name, value in cached.items()
                        }

            _observability.record_batter_refresh(_estimate_row_count(stats))
            return stats
        except Exception as exc:
            _observability.record_error(f"batter stats: {exc}")
            return None

    def get_pitcher_deep_dive(self, player_id: int) -> Optional[dict[str, Any]]:
        """
        Return full pitcher deep-dive (FIP+xFIP+SIERA+Savant metrics).

        Returns None if flag is disabled.
        """
        if not self.is_enabled():
            logger.debug("predictive_stats: flag disabled, skipping pitcher deep dive for %s", player_id)
            return None

        try:
            from backend.fantasy_baseball import pitcher_deep_dive as deep_dive_module

            deep_dive_cls = getattr(deep_dive_module, "PitcherDeepDive", None)
            if deep_dive_cls is not None:
                deep_dive = deep_dive_cls()
                if hasattr(deep_dive, "analyze"):
                    result = deep_dive.analyze(player_id)
                    return _serialize_stats(result)

            return {}
        except Exception as exc:
            _observability.record_error(f"pitcher deep dive: {exc}")
            return None


_service_instance: Optional[PredictiveStatsService] = None


def get_predictive_stats_service() -> PredictiveStatsService:
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictiveStatsService()
    return _service_instance
