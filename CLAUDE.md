# CLAUDE.md — CBB Edge Analyzer

## Project Overview

College Basketball (CBB) Edge Analyzer — a production betting framework for NCAA Division 1 basketball. The system fetches real-time odds, integrates multi-source ratings (KenPom, BartTorvik, EvanMiya), runs a **Version 8** Monte Carlo betting model with matchup-specific variance and portfolio-level risk management, and tracks performance via Closing Line Value (CLV). It is **conservative by design**: the model PASSes on 85-95% of games and only recommends bets when the lower-bound edge is positive after uncertainty adjustments.

**Tech stack:** Python 3.11+ · FastAPI · PostgreSQL 15+ · SQLAlchemy 2.0 · Streamlit · APScheduler · NumPy/SciPy

## Repository Structure

```
CBB_Betting/
├── backend/                  # FastAPI application + core logic
│   ├── main.py               # FastAPI app, REST endpoints, APScheduler cron jobs
│   ├── betting_model.py      # Version 8 model: Monte Carlo CI, Shin vig removal,
│   │                         #   matchup-specific SD, dynamic weight re-normalization
│   ├── models.py             # SQLAlchemy ORM (9 tables: games, predictions, bet_logs, etc.)
│   ├── schemas.py            # Pydantic request/response models
│   ├── auth.py               # API key authentication (X-API-Key header)
│   └── services/
│       ├── analysis.py       # Nightly analysis orchestration (V8: injuries, profiles, portfolio)
│       ├── odds.py           # The Odds API client
│       ├── odds_monitor.py   # Event-driven line movement detection (polls every 5 min)
│       ├── ratings.py        # KenPom API + BartTorvik/EvanMiya stubs
│       ├── injuries.py       # Injury scraping (ESPN) + manual overrides + impact tiers
│       ├── portfolio.py      # Portfolio-level Kelly sizing, drawdown circuit breaker
│       ├── matchup_engine.py # Play-by-play matchup geometry + team style profiles
│       ├── performance.py    # ROI, calibration, timeline analytics
│       ├── bet_tracker.py    # Outcome settling, closing line capture
│       ├── clv.py            # Closing Line Value calculation
│       ├── alerts.py         # Alert generation (email/SMS stubs)
│       └── team_mapping.py   # Team name normalization with fuzzy matching
├── dashboard/                # Streamlit multi-page UI
│   ├── app.py                # Main dashboard entry point
│   └── pages/
│       ├── 1_Performance.py
│       ├── 2_CLV_Analysis.py
│       ├── 3_Bet_History.py
│       ├── 4_Calibration.py
│       └── 5_Alerts.py
├── scripts/
│   ├── init_db.py            # Database schema initialization + seeding
│   ├── map_teams.py          # Team name mapping utility
│   └── migrate_v2.py         # Legacy data migration
├── tests/
│   ├── test_betting_model.py # V8 model: Monte Carlo, Shin, matchup SD, weight re-norm
│   ├── test_portfolio.py     # Portfolio manager: exposure caps, drawdown, correlation
│   ├── test_matchup_engine.py# Matchup adjustments: pace, 3PA vs drop, zone, transition
│   ├── test_performance.py   # Performance metrics
│   ├── test_bet_tracker.py   # Bet outcome + CLV integration
│   └── test_alerts.py        # Alert triggering logic
├── .github/workflows/
│   └── deploy.yml            # CI: pytest + coverage → deploy to Railway
├── requirements.txt          # All Python dependencies (pinned versions)
├── railway.json              # Railway.app deployment config
├── .env.example              # Environment variable template
├── setup.py                  # One-command automated setup
└── README.md                 # Primary documentation
```

## Build & Run Commands

### Local Development

```bash
# Create virtual environment and install deps
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env            # Then edit .env with real values

# Initialize database (requires PostgreSQL running)
python scripts/init_db.py

# Run backend API server (with hot-reload)
uvicorn backend.main:app --reload

# Run dashboard (separate terminal)
streamlit run dashboard/app.py
```

### Testing

```bash
# Run full test suite with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=backend

# Run a single test file
pytest tests/test_betting_model.py -v

# Run a single test class or method
pytest tests/test_betting_model.py::TestWeightReNormalization -v
pytest tests/test_betting_model.py::TestMatchupSD::test_style_profiles_affect_sd -v

# Run the new V8 test suites
pytest tests/test_portfolio.py -v
pytest tests/test_matchup_engine.py -v
```

Tests require a PostgreSQL database for the full suite. The betting model, portfolio, and matchup engine tests run without a database.

### Code Quality

```bash
# Formatting
black backend/ tests/ scripts/ dashboard/

# Linting
flake8 backend/ tests/

# Type checking
mypy backend/
```

### Production

```bash
# Start with gunicorn (production)
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT

# Railway deployment happens automatically on push to main (via CI)
```

## Architecture & Key Concepts

### Betting Model (Version 8)

The core model lives in `backend/betting_model.py` (`CBBEdgeModel` class):

- **Dynamic weight re-normalization**: When rating sources (EvanMiya, BartTorvik) are missing, their weight is redistributed proportionally to available sources instead of silently dropping from the margin calculation
- **Shin (1993) vig removal**: Uses the Shin method to extract true probabilities from American odds, accounting for bookmaker favourite-longshot bias (replaces naive proportional method)
- **Matchup-specific SD**: Computes team-pair variance from play-style profiles (pace, 3PA rate, FT rate) instead of a flat `base_sd=11.0`. Falls back to game-total heuristic, then base_sd
- **2-layer Monte Carlo CI**: Simulates both parameter uncertainty and outcome uncertainty to produce confidence intervals on win probabilities
- **Safe Kelly Criterion**: Fractional Kelly sizing with caps (`max_kelly=0.20`, `fractional_kelly_divisor=2.0`)
- **Penalty budget with ceiling**: Penalties applied for data staleness, missing sources, injuries, etc.
- **Conservative threshold**: Only recommends BET when point-estimate edge > 2% AND lower-bound CI edge > 0%

Verdicts: `BET` (rare, 5-15%), `CONSIDER` (marginal edge), `PASS` (majority of games)

### Injury Service (`services/injuries.py`)

Provides real-time roster availability so the model never trades blind into a line that sharp money has already moved:

- **ESPN scraper**: Scrapes injury reports from ESPN college basketball
- **Manual overrides**: API for adding injury entries that take priority over scraped data
- **Impact tiers**: `star` (3.5 pts), `starter` (1.8 pts), `role` (0.7 pts), `bench` (0.2 pts) — can be scaled by usage rate
- **Status weighting**: Out (100%), Doubtful (75%), Questionable (40%), Probable (10%)
- **30-minute cache**: Avoids excessive scraping while staying current

### Portfolio Manager (`services/portfolio.py`)

Prevents the "30 bets on Saturday" problem by managing bankroll exposure:

- **Total exposure cap**: Limits aggregate deployed capital (default 15% of bankroll)
- **Single bet cap**: No single bet exceeds 3% of bankroll
- **Drawdown circuit breaker**: Halts all new bets when drawdown exceeds threshold (default 15%)
- **Conference correlation penalty**: Small scaling penalty for multiple bets in the same conference
- **DB state reconstruction**: Restores pending positions after server restart

### Matchup Engine (`services/matchup_engine.py`)

Computes second-order matchup adjustments that top-level efficiency metrics miss:

- **Pace mismatch**: Increases SD when one team runs and the other grinds
- **3PA vs drop coverage**: Margin boost for shooters attacking drop defence
- **Transition gap**: Margin adjustment for fast-break differential
- **Zone vs 3FG%**: Penalizes zone defence against elite 3-point teams
- **Rebounding differential**: Adjusts for second-chance points and extra possessions
- **Team profile cache**: Loads from BartTorvik or manual config

### Odds Monitor (`services/odds_monitor.py`)

Event-driven line movement detection that complements the nightly batch:

- **5-minute polling**: Configurable interval via `ODDS_MONITOR_INTERVAL_MIN`
- **Movement detection**: Thresholds for spread (1.5pts), total (2.0pts), moneyline (25 units)
- **Golden window**: Lower thresholds when games are < 2 hours from tipoff
- **Callback system**: Fires callbacks for significant movements
- **API quota guard**: Pauses polling when quota drops below reserve
- **Rolling history**: Maintains last 50 snapshots per game

### Database

PostgreSQL with SQLAlchemy ORM. Key tables defined in `backend/models.py`:

| Table | Purpose |
|-------|---------|
| `games` | Game data: teams, date, scores, completion status |
| `predictions` | Model outputs: margins, edges, Kelly fractions, verdicts |
| `bet_logs` | Paper trades and real bets placed |
| `closing_lines` | Captured closing odds for CLV calculation |
| `performance_snapshots` | Daily/weekly aggregate metrics |
| `data_fetches` | API call logging for monitoring |
| `model_parameters` | Versioned model weights and parameters |
| `db_alerts` | System alerts (drawdown, calibration drift) |

Connection setup is in `backend/models.py`: uses `pool_pre_ping=True` for Docker stability, loads `DATABASE_URL` from `.env` via `python-dotenv`.

### Scheduled Jobs

Configured in `backend/main.py` via APScheduler:

| Schedule | Job | Function |
|----------|-----|----------|
| Daily at 3 AM ET | Nightly analysis | `run_nightly_analysis()` — full pipeline with injuries, profiles, portfolio |
| Every 5 minutes | Odds monitor | `_odds_monitor_job()` — detect line movements |
| Every 2 hours | Update outcomes | `update_completed_games()` — settle finished games |
| Every 30 minutes | Capture closing lines | `capture_closing_lines()` — grab lines for CLV |
| Daily at 4 AM ET | Performance snapshot | `_daily_snapshot_job()` — aggregate metrics + alerts |

### Authentication

Simple API key auth via `X-API-Key` header (`backend/auth.py`). Keys are set via environment variables (`API_KEY_USER1` through `API_KEY_USER5`). User1 has admin privileges. A `"dev-key-insecure"` fallback exists only when `ENVIRONMENT=development`.

### API Endpoints

Key endpoints in `backend/main.py`:

- `GET /health` — health check (unauthenticated)
- `GET /api/predictions/today` — today's predictions
- `POST /api/bets/log` — create a bet log entry
- `PUT /api/bets/{id}/outcome` — update bet outcome
- `GET /api/performance/summary` — performance summary stats
- `POST /admin/run-analysis` — manually trigger nightly job (admin only)
- `GET /admin/portfolio/status` — portfolio exposure, drawdown, positions
- `GET /admin/odds-monitor/status` — odds monitor state
- `GET /admin/scheduler/status` — scheduler job listing

Full OpenAPI docs available at `/docs` when the server is running.

## Code Conventions

### Style & Formatting

- **Formatter**: `black` (default settings)
- **Linter**: `flake8`
- **Type checker**: `mypy`
- **Docstrings**: Module-level docstrings on all files; class and function docstrings for public APIs
- **Logging**: Use `logging.getLogger(__name__)` — structured logging via `structlog` is available but standard logging is used throughout

### Patterns

- **Pydantic schemas** (`backend/schemas.py`) for all API request/response validation — never accept raw dicts on endpoints
- **SQLAlchemy ORM** models with `relationship()` for joins — use `get_db()` dependency for session lifecycle
- **Service layer** (`backend/services/`) separates business logic from API routes
- **Singleton pattern** for service instances: `get_injury_service()`, `get_portfolio_manager()`, `get_odds_monitor()`, `get_profile_cache()`
- **Environment variables** for all configuration — loaded via `python-dotenv`, see `.env.example` for full list
- **Dataclasses** for internal data transfer objects (e.g., `GameAnalysis`, `MatchupAdjustment`, `AdjustedSizing`)
- **Type hints** on function signatures throughout the codebase

### Naming

- Files: `snake_case.py`
- Classes: `PascalCase` (e.g., `CBBEdgeModel`, `GameAnalysis`, `PortfolioManager`)
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE` (e.g., `BASE_URL`, `TIER_IMPACT`)
- Test files: `test_<module>.py` with test classes `Test<Feature>`
- Dashboard pages: numbered prefix `N_Name.py` (e.g., `1_Performance.py`)

### Testing Conventions

- Framework: `pytest` with `pytest-asyncio` for async tests
- Test structure: class-based grouping by feature (`TestWeightReNormalization`, `TestPortfolioExposure`)
- Each test method tests one specific behavior
- No fixtures or conftest yet — models are instantiated directly in tests
- Run command noted in test file docstrings: `pytest tests/test_betting_model.py -v`
- Currently 55 tests across 3 core test files (betting model, portfolio, matchup engine)

## Environment Variables

All configuration is via environment variables. See `.env.example` for the complete list. Critical ones:

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `THE_ODDS_API_KEY` | Yes | The Odds API key for live odds |
| `KENPOM_API_KEY` | Yes | KenPom ratings API key |
| `API_KEY_USER1` | Yes | Primary API key (32+ chars, admin) |
| `ENVIRONMENT` | No | `development` or `production` (default: production) |
| `BASE_SD` | No | Model base standard deviation (default: 11.0) |
| `WEIGHT_KENPOM` | No | KenPom weight in composite (default: 0.342) |
| `HOME_ADVANTAGE` | No | Home court advantage in points (default: 3.09) |
| `MAX_KELLY_FRACTION` | No | Kelly fraction cap (default: 0.20) |
| `STARTING_BANKROLL` | No | Starting bankroll in dollars (default: 1000) |
| `MAX_DRAWDOWN_PCT` | No | Auto-pause drawdown threshold (default: 15.0) |
| `ODDS_MONITOR_INTERVAL_MIN` | No | Odds monitor polling interval in minutes (default: 5) |

## CI/CD Pipeline

Defined in `.github/workflows/deploy.yml`:

1. **Test job** (every push/PR to `main`):
   - Ubuntu with PostgreSQL 15 service container
   - Python 3.11
   - Installs deps, initializes test DB, runs `pytest tests/ -v --cov=backend`

2. **Deploy job** (only on push to `main`, after tests pass):
   - Deploys to Railway via `railway up`
   - Uses `RAILWAY_TOKEN` from GitHub Secrets

## Deployment

- **Platform**: Railway.app (configured via `railway.json`)
- **Start command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Health check**: `GET /health`
- **Region**: us-west1
- **Dashboard**: Can be deployed separately on Streamlit Cloud (free tier) or Railway

## Common Development Tasks

### Adding a new API endpoint

1. Define Pydantic schemas in `backend/schemas.py`
2. Add the route in `backend/main.py` with appropriate auth dependency (`verify_api_key` or `verify_admin_api_key`)
3. Implement business logic in the appropriate service file under `backend/services/`
4. Add tests in `tests/`

### Adding a new database table

1. Define the SQLAlchemy model in `backend/models.py`
2. Add relationships to existing models if needed
3. Update `scripts/init_db.py` to create the table
4. Add Pydantic schemas in `backend/schemas.py` if exposed via API

### Modifying the betting model

1. Changes go in `backend/betting_model.py`
2. Update tests in `tests/test_betting_model.py`
3. If changing parameters, update `.env.example` and the `model_parameters` table tracking
4. The model is intentionally conservative — maintain the philosophy of PASSing on most games
5. When adding rating sources, update `self.weights` and the re-normalization logic in `analyze_game()`

### Adding a new matchup factor

1. Add the factor method to `MatchupEngine` in `backend/services/matchup_engine.py`
2. Call it from `analyze_matchup()`
3. If new team data is needed, extend `TeamPlayStyle` dataclass
4. Add tests in `tests/test_matchup_engine.py`

### Adding a dashboard page

1. Create `dashboard/pages/N_PageName.py` (numbered for ordering)
2. Use Streamlit + Plotly/Altair for visualizations
3. Dashboard reads from the same PostgreSQL database via `DATABASE_URL`
