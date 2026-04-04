"""
backend/edge_app.py -- Edge service FastAPI entry point.

Railway start command: uvicorn backend.edge_app:app --host 0.0.0.0 --port $PORT

This app serves ONLY betting/analysis endpoints:
  - /api/predictions/*, /api/performance/*, /api/bets/*, /api/games/*
  - /api/closing-lines/*, /api/tournament/*, /api/feature-flags
  - /admin/* (edge-specific admin operations)
  - /health, /

Fantasy routes are NOT mounted here. If a fantasy client hits this service
by mistake, it receives 404.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.edge import router as edge_router
from backend.routers.admin import router as admin_router
from backend.schedulers.edge_scheduler import start_edge_scheduler, stop_edge_scheduler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Edge service starting -- deployment_role=%s", os.getenv("DEPLOYMENT_ROLE", "primary"))
    start_edge_scheduler()
    yield
    stop_edge_scheduler()
    logger.info("Edge service stopped")


app = FastAPI(
    title="Edge Analytics API",
    description="Sports betting edge analysis engine -- CBB + MLB",
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

app.include_router(edge_router)
app.include_router(admin_router)
