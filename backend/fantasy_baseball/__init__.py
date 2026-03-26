"""Fantasy Baseball module — Yahoo Fantasy integration, keeper evaluation, draft assistant."""

# Original exports (preserve backward compatibility)
from backend.fantasy_baseball.yahoo_client import YahooFantasyClient

# Resilience components (new)
from backend.fantasy_baseball.circuit_breaker import CircuitBreaker, CircuitOpenError
from backend.fantasy_baseball.cache_manager import StaleCacheManager, CacheResult, NoDataAvailableError
from backend.fantasy_baseball.position_normalizer import PositionNormalizer, LineupValidationError
from backend.fantasy_baseball.yahoo_client_resilient import ResilientYahooClient, WaiverResponse, LineupResult

# Game-aware lineup validator (new)
from backend.fantasy_baseball.lineup_validator import (
    LineupValidator,
    ScheduleFetcher,
    OptimizedSlot,
    GameStatus,
    PlayerGameInfo,
    LineupValidation,
    LineupSubmission,
    format_lineup_report,
)

# Alias for backward compatibility
YahooClient = YahooFantasyClient

__all__ = [
    # Original (with aliases for backward compatibility)
    "YahooFantasyClient",
    "YahooClient",  # Alias
    # Resilience components
    "CircuitBreaker",
    "CircuitOpenError",
    "StaleCacheManager",
    "CacheResult",
    "NoDataAvailableError",
    "PositionNormalizer",
    "LineupValidationError",
    "ResilientYahooClient",
    "WaiverResponse",
    "LineupResult",
    # Lineup validator (game-aware validation)
    "LineupValidator",
    "ScheduleFetcher",
    "OptimizedSlot",
    "GameStatus",
    "PlayerGameInfo",
    "LineupValidation",
    "LineupSubmission",
    "format_lineup_report",
]
