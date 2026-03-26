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

# Platoon splits fetcher (new)
from backend.fantasy_baseball.platoon_fetcher import (
    PlatoonSplitFetcher,
    PlatoonSplits,
    get_platoon_fetcher,
)

# Category tracker (new)
from backend.fantasy_baseball.category_tracker import (
    CategoryTracker,
    MatchupStatus,
    get_category_tracker,
)

# Smart lineup selector (advanced optimizer)
from backend.fantasy_baseball.smart_lineup_selector import (
    SmartLineupSelector,
    SmartBatterRanking,
    OpposingPitcher,
    CategoryNeed,
    Handedness,
    get_smart_selector,
)

# Elite context (new)
from backend.fantasy_baseball.elite_context import (
    EliteManagerContextBuilder,
    PlayerDecisionContext,
    LineupDecisionReport,
    RiskProfile,
    MatchupStrategy,
    WeatherContext,
    RecentForm,
    LineupSpot,
)

# Pitcher deep dive (new)
from backend.fantasy_baseball.pitcher_deep_dive import (
    PitcherDeepDiveFetcher,
    get_pitcher_fetcher,
)

# Weather (new)
from backend.fantasy_baseball.weather_fetcher import (
    WeatherFetcher,
    GameWeather,
    get_weather_fetcher,
)

# Daily briefing (new)
from backend.fantasy_baseball.daily_briefing import (
    DailyBriefingGenerator,
    DailyBriefing,
    PlayerBriefing,
    CategoryBriefing,
    DecisionAction,
    get_briefing_generator,
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
    # Platoon fetcher
    "PlatoonSplitFetcher",
    "PlatoonSplits",
    "get_platoon_fetcher",
    # Category tracker
    "CategoryTracker",
    "MatchupStatus",
    "get_category_tracker",
    # Smart lineup selector
    "SmartLineupSelector",
    "SmartBatterRanking",
    "OpposingPitcher",
    "CategoryNeed",
    "Handedness",
    "get_smart_selector",
    # Pitcher deep dive
    "PitcherDeepDiveFetcher",
    "get_pitcher_fetcher",
    # Weather
    "WeatherFetcher",
    "GameWeather",
    "get_weather_fetcher",
    # Daily briefing
    "DailyBriefingGenerator",
    "DailyBriefing",
    "PlayerBriefing",
    "CategoryBriefing",
    "DecisionAction",
    "get_briefing_generator",
    # Elite context
    "EliteManagerContextBuilder",
    "PlayerDecisionContext",
    "LineupDecisionReport",
    "RiskProfile",
    "MatchupStrategy",
    "WeatherContext",
    "RecentForm",
    "LineupSpot",
]
