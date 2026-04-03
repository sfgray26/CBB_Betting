# CBB Edge Analyzer

> A quantitative trading-inspired fantasy baseball decision engine

## What This Is

CBB Edge Analyzer treats fantasy baseball like a trading system. It doesn't just tell you who to start - it explains **why**, shows you the **risk-adjusted value**, and lets you decide how much automation you want.

## Quick Start

### Local Development

```bash
# Install dependencies
pnpm install

# Start local infrastructure (Postgres + Redis)
docker-compose up -d

# Run migrations
pnpm db:migrate

# Start all services
pnpm dev

# API is now running at http://localhost:3000
```

### Railway Deployment

```bash
# Login and deploy
railway login
railway link
railway up

# The API and worker will both deploy automatically
```

## How It Works

The system has four layers:

```
┌─────────────────────────────────────┐
│  Data Ingestion                     │
│  MLB Stats API, weather, ballpark   │
├─────────────────────────────────────┤
│  Analytics Engine                   │
│  Monte Carlo sims, factor models    │
├─────────────────────────────────────┤
│  Decision Engine                    │
│  Optimization, risk assessment      │
├─────────────────────────────────────┤
│  Execution                          │
│  Human review or automation         │
└─────────────────────────────────────┘
```

## Core Concepts

### Decision Contracts

The system revolves around immutable contracts:

1. **LineupOptimizationRequest** - "Here's my league, my players, my risk tolerance - give me the optimal lineup"
2. **PlayerValuationReport** - "Here's what this player is worth in this context, with full uncertainty quantified"
3. **ExecutionDecision** - "Here's what you should do, why, and how confident we are"

### Safety-First Automation

Automation isn't binary. The system supports progressive trust:

| Level | Behavior |
|-------|----------|
| 0 (default) | All decisions require human review |
| 1 | Auto-execute only high-confidence, low-risk moves |
| 2 | Auto-execute with daily digest |
| 3 | Full automation with exception alerts |

## Human Interaction

### Via API

```bash
# Get optimal lineup for today
curl -X POST http://localhost:3000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "leagueId": "your-league-id",
    "scoringPeriod": "2025-07-15",
    "riskTolerance": "balanced"
  }'

# Get player valuation
curl http://localhost:3000/players/shohei-ohtani/valuation?date=2025-07-15

# Get pending decisions requiring review
curl http://localhost:3000/decisions/pending
```

### Via CLI (Coming Soon)

```bash
# Interactive lineup optimizer
pnpm cli optimize --league my-league

# Run analytics on a player
pnpm cli analyze "Shohei Ohtani" --verbose
```

## System Architecture

```
Repository Structure:
├── apps/
│   ├── api/              # HTTP API (Railway service)
│   └── worker/           # Background jobs (Railway service)
├── packages/
│   ├── core/             # Domain logic (framework-agnostic)
│   ├── analytics/        # Monte Carlo, projections
│   ├── data/             # Data ingestion adapters
│   └── infrastructure/   # Railway-specific code
```

See [docs/architecture.md](./docs/architecture.md) for the full blueprint.

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# Redis (Railway Upstash or local)
REDIS_URL=redis://localhost:6379

# MLB Stats API
MLB_API_KEY=your_key_here

# Fantasy Platform (for future automation)
YAHOO_CLIENT_ID=...
YAHOO_CLIENT_SECRET=...

# Automation Level (0-3, default 0)
AUTOMATION_LEVEL=0
```

## Development

### Project Structure

- `apps/api` - Fastify HTTP server
- `apps/worker` - Background job processor
- `packages/core` - Domain logic, decision contracts
- `packages/analytics` - Statistical models, simulations
- `packages/data` - Data source adapters
- `packages/infrastructure` - Database, queue, caching

### Adding a New Data Source

1. Create adapter in `packages/data/src/adapters/`
2. Implement the `DataSource` interface
3. Add transformer to normalize to internal schema
4. Register in `apps/worker/src/jobs/data-sync.ts`

### Adding a New Decision Type

1. Define contract in `packages/core/src/decisions/`
2. Create handler in `apps/worker/src/handlers/`
3. Add API route in `apps/api/src/routes/`
4. Update queue processor in `apps/worker/src/worker.ts`

## Deployment

### Railway Services

| Service | Type | Purpose |
|---------|------|---------|
| `api` | Web Service | HTTP API, webhooks |
| `worker` | Worker | Background jobs, cron |
| `redis` | Plugin | Queue, caching |
| `postgres` | Plugin | Data persistence |

### Cron Schedule

The worker runs scheduled jobs:
- Every 15 minutes: Update live projections
- Every hour: Sync player data
- Daily at 6 AM: Generate lineup recommendations
- Daily at 11 PM: End-of-day analytics

## Roadmap

### Phase 1: Foundation (Now)
- [ ] Data ingestion pipeline
- [ ] Basic projections
- [ ] Lineup optimizer
- [ ] Manual execution

### Phase 2: Intelligence
- [ ] Monte Carlo simulations
- [ ] Weather integration
- [ ] Ballpark factors
- [ ] Risk profiles

### Phase 3: Automation
- [ ] Browser automation foundation
- [ ] Progressive automation levels
- [ ] Execution with human override

### Phase 4: Scale
- [ ] Web UI
- [ ] Multi-league support
- [ ] Advanced analytics dashboard

## Why This Architecture

**Contract-first**: Every decision is explicit, auditable, and immutable. No hidden state.

**Framework-agnostic core**: The analytics engine is pure TypeScript. It could run in a browser extension, a Python service, or a CLI tool.

**Railway-native**: Services map cleanly to Railway's model. Background jobs, API, and scheduled tasks each get appropriate resources.

**Safety ladder**: Automation is added progressively. Start with suggestions, end with full auto - same code, different config.

## License

MIT

## Contributing

This is a personal project, but PRs welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md).
