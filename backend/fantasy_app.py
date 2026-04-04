"""
backend/fantasy_app.py -- Fantasy service FastAPI entry point.

Railway start command: uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT

This app serves ONLY fantasy endpoints:
  - /api/fantasy/* (all fantasy routes)
  - /admin/* (fantasy-specific: ingestion status, job queue, etc.)
  - /health, /

Edge/betting routes are NOT mounted here. Zero knowledge of betting_model,
analysis.py, or any CBB-specific service. GUARDIAN FREEZE enforced by
import structure.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.fantasy import router as fantasy_router
from backend.routers.admin import router as admin_router
from backend.schedulers.fantasy_scheduler import start_fantasy_scheduler, stop_fantasy_scheduler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fantasy service starting -- deployment_role=%s", os.getenv("DEPLOYMENT_ROLE", "fantasy-prod"))
    start_fantasy_scheduler()

    # Startup catch-up: trigger a one-off ensemble refresh if enabled.
    # Note: _startup_catchup is defined inside main.py's lifespan and cannot be
    # directly imported. In Phase 7, the catch-up logic moves here. For now,
    # the DailyIngestionOrchestrator (if enabled) handles freshness via its own scheduler.
    try:
        from backend.utils.deployment import startup_catchup_enabled  # noqa: PLC0415
        catchup_enabled = startup_catchup_enabled()
    except ImportError:
        catchup_enabled = os.getenv("ENABLE_STARTUP_CATCHUP", "true").lower() != "false"

    if catchup_enabled:
        logger.info("Startup catch-up enabled -- DailyIngestionOrchestrator handles freshness")
    else:
        logger.info("Startup catch-up disabled by ENABLE_STARTUP_CATCHUP=false")

    yield
    stop_fantasy_scheduler()
    logger.info("Fantasy service stopped")


app = FastAPI(
    title="Fantasy Baseball API",
    description="MLB fantasy baseball -- lineup optimisation, waiver analysis, projections",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fantasy_router)
app.include_router(admin_router)
