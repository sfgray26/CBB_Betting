"""
backend/models_edge.py -- Edge service SQLAlchemy models.

Re-exports betting/analysis model classes from backend.models during the
strangler-fig migration. After Phase 7 cleanup, class definitions move
here permanently and backend.models is deleted.

DO NOT import from backend.models_fantasy -- ever.
DO NOT import from backend.betting_model -- use it directly only in
services that explicitly need it (analysis.py, line_monitor.py, etc.).
"""
from backend.models import (  # noqa: F401
    Base,
    Game,
    Prediction,
    BetLog,
    ModelParameter,
    PerformanceSnapshot,
    DataFetch,
    ClosingLine,
    TeamProfile,
    DBAlert,
    DataIngestionLog,
)

__all__ = [
    "Base",
    "Game",
    "Prediction",
    "BetLog",
    "ModelParameter",
    "PerformanceSnapshot",
    "DataFetch",
    "ClosingLine",
    "TeamProfile",
    "DBAlert",
    "DataIngestionLog",
]
