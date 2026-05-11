# README Update Summary — 2026-05-11

## What Changed

Updated `README.md` to reflect the project's current MLB-focused state after pivoting from CBB betting.

## Key Updates

| Section | Before | After |
|---------|--------|-------|
| **Title** | "CBB Edge Analyzer" | "CBB Edge — MLB Fantasy Baseball Platform" |
| **Focus** | CBB betting framework | MLB fantasy baseball (with note that CBB is dormant) |
| **Features** | Odds monitoring, CLV tracking, betting alerts | Lineup optimizer, waiver wire, matchup simulation, budget/FAAB tracking, streaming options |
| **Frontend** | Streamlit (v1) → Next.js (v2) | Next.js 15, React 19, TypeScript, Tailwind CSS |
| **Backend** | FastAPI + SQLAlchemy | FastAPI, SQLAlchemy 2.0, Pydantic v2 |
| **Database** | PostgreSQL (Railway or Supabase) | PostgreSQL (Railway), Alembic migrations |
| **Added** | — | Redis cache, APScheduler with advisory locks, Yahoo OAuth, Statcast/pybaseball, BallDontLie, OpenWeatherMap |
| **Deploy** | Railway.app | Railway.app (unchanged host) |
| **Local dev** | Streamlit dashboard | Next.js dev server on :3000 |
| **Project structure** | CBB-centric (betting_model, odds, ratings) | MLB-centric (fantasy router, war-room pages, ingestion pipelines) |
| **Ops docs** | Monitoring section inline | Link to `docs/incident-response.md` runbook |

## Accuracy Checks

- ✅ `frontend/package.json` confirms Next.js 15, React 19, TypeScript, Tailwind CSS, Recharts
- ✅ `backend/routers/fantasy.py` confirms lineup optimizer, waiver wire, matchup simulation, FAAB budget, streaming endpoints
- ✅ `backend/services/daily_ingestion.py`, `mlb_analysis.py`, `yahoo_ingestion.py` confirm data pipeline
- ✅ `docs/incident-response.md` exists and is referenced
- ✅ `.env.example` confirms Yahoo OAuth, OpenWeatherMap, BallDontLie, Statcast dependencies
- ✅ `requirements.txt` confirms pybaseball, MLB-StatsAPI, requests-oauthlib

## Files Modified

- `README.md` — full rewrite (kept CBB framework as dormant/legacy note)

## Files Created

- `reports/2026-05-11-readme-update.md` — this summary
