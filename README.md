# CBB Edge — MLB Fantasy Baseball Platform

> **Note:** The college basketball (CBB) betting season is closed. The platform has pivoted to **MLB Fantasy Baseball** for the 2026 season. The CBB betting framework remains in the codebase and will reactivate when the season resumes.

---

## 🎯 What It Is

A production-ready fantasy baseball platform that connects to Yahoo Fantasy leagues, ingests real-time MLB data (Statcast, MLB Stats API, BallDontLie, weather), and provides tools for daily lineup management, matchup simulation, and roster optimization.

## 🏆 Key Features

- **Lineup Optimizer** — Auto-generate optimal daily lineups based on matchups, park factors, and player projections
- **Waiver Wire** — Edge-detected waiver recommendations with IL capacity awareness and FAAB budget tracking
- **Matchup Simulation** — Monte Carlo simulation of weekly head-to-head matchups with category-by-category win probability
- **Budget Tracking** — FAAB balance integration and daily exposure caps for paper-traded bets
- **Streaming Options** — Probable pitcher start tracking, two-start detection, and streaming hitter recommendations
- **Real-time Data Ingestion** — Scheduled pipelines for Statcast, Savant, BallDontLie, Yahoo rosters, and weather
- **Performance Dashboard** — Next.js war room with roster lab, matchup preview, and decision explanations
- **Discord Alerts** — Automated notifications for betting opportunities and pipeline issues

## 📊 Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 15, React 19, TypeScript, Tailwind CSS, Radix UI, Recharts |
| **Backend** | FastAPI (Python 3.11+), SQLAlchemy 2.0, Pydantic v2 |
| **Database** | PostgreSQL (Railway), Alembic migrations |
| **Cache** | Redis (Railway) |
| **Scheduler** | APScheduler (asyncio) with advisory-lock job deduplication |
| **Hosting** | Railway.app |
| **Fantasy API** | Yahoo Fantasy Sports API (OAuth 2.0) |
| **MLB Data** | MLB Stats API, Statcast (pybaseball), BallDontLie, OpenWeatherMap |
| **Testing** | pytest, Playwright (e2e) |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+ (for frontend)
- PostgreSQL (local or Railway)
- Yahoo API credentials ([developer.yahoo.com/apps](https://developer.yahoo.com/apps/))

### Backend

```bash
cd cbb-edge
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your DATABASE_URL, YAHOO_CLIENT_ID/SECRET, and API keys

# Initialize database
python scripts/init_db.py

# Run server
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

### Production Deploy

```bash
# Railway CLI
npm install -g @railway/cli
railway login
railway up
```

## 📁 Project Structure

```
cbb-edge/
├── backend/
│   ├── main.py                 # FastAPI app + scheduler
│   ├── models.py              # SQLAlchemy ORM
│   ├── schemas.py              # Pydantic schemas
│   ├── routers/
│   │   └── fantasy.py         # /api/fantasy/* endpoints
│   ├── services/
│   │   ├── daily_ingestion.py # Pipeline orchestrator
│   │   ├── matchup_engine.py  # H2H simulation
│   │   ├── waiver_edge_detector.py
│   │   └── simulation_engine.py
│   └── fantasy_baseball/
│       └── yahoo_client_resilient.py
├── frontend/
│   ├── app/(dashboard)/        # Next.js app router
│   │   ├── war-room/
│   │   │   ├── roster/
│   │   │   ├── streaming/
│   │   │   └── waiver/
│   │   └── today/
│   └── components/
├── docs/
│   └── incident-response.md   # Ops runbook
├── scripts/
│   └── init_db.py
└── reports/                   # Audit & UAT logs
```

## 🔐 Authentication

- **API Key**: `X-API-Key` header for internal endpoints
- **Yahoo OAuth**: Refresh-token flow for fantasy league access
- **Admin Key**: Separate `X-Admin-Key` for diagnostic endpoints

## 📈 Daily Workflow

1. **Morning**: Check the War Room dashboard for today's optimal lineup
2. **Midday**: Review waiver wire recommendations before FAAB runs
3. **Pre-matchup**: Run matchup simulation to preview weekly category battles
4. **Evening**: Verify streaming pitcher starts for tomorrow

## 🔧 Operations

See [`docs/incident-response.md`](docs/incident-response.md) for:
- Database connection troubleshooting
- Scheduler failure recovery
- Yahoo token refresh procedures
- Pipeline starvation diagnosis
- Rollback procedures

## 🧪 Testing

```bash
# Backend
pytest tests/

# Frontend e2e
cd frontend && npm run test:e2e
```

## ⚠️ Disclaimer

This software is for **educational and personal use only**.
- Not financial advice
- No guaranteed profitability
- Gambling involves risk of loss
- Check local laws regarding sports betting
- Never bet more than you can afford to lose

## 📄 License

MIT License — see LICENSE file.

---

**Questions?** Open an issue. **Found a bug?** PRs welcome.
