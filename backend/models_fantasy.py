"""
backend/models_fantasy.py -- Fantasy service SQLAlchemy models.

Re-exports fantasy model classes from backend.models during the
strangler-fig migration. After Phase 7 cleanup, class definitions move
here permanently and backend.models is deleted.

DBAlert and DataIngestionLog are re-exported from backend.models during
the transition -- they will be independently redefined in the Fantasy DB
in Phase 7. DO NOT import from backend.models_edge -- ever.
DO NOT import from backend.betting_model -- GUARDIAN FREEZE.
"""
from backend.models import (  # noqa: F401
    FantasyDraftSession,
    FantasyDraftPick,
    FantasyLineup,
    PlayerDailyMetric,
    StatcastPerformance,
    PlayerProjection,
    PatternDetectionAlert,
    UserPreferences,
    ProjectionCacheEntry,
    ProjectionSnapshot,
    PlayerValuationCache,
    # DBAlert and DataIngestionLog shared during transition
    DBAlert,
    DataIngestionLog,
)

__all__ = [
    "FantasyDraftSession",
    "FantasyDraftPick",
    "FantasyLineup",
    "PlayerDailyMetric",
    "StatcastPerformance",
    "PlayerProjection",
    "PatternDetectionAlert",
    "UserPreferences",
    "ProjectionCacheEntry",
    "ProjectionSnapshot",
    "PlayerValuationCache",
    "DBAlert",
    "DataIngestionLog",
]
