# IDENTITY.md — Monorepo Operational Identity

> Maintained by: Claude Code (Master Architect)
> Last consolidated: March 28, 2026
> This file defines what this system IS, what it VALUES, and what it will NOT do.
> All risk parameters in code must be cross-referenced to this file.

---

## Monorepo Identity

This repository contains two production systems with shared infrastructure:

```
cbb-edge/
├── SYSTEM 1: CBB Betting Analyzer (V9.1)
│   └── Find and size positive-EV NCAA D1 basketball bets.
│       Capital preservation first. Edge maximization second.
│
└── SYSTEM 2: Fantasy Baseball Platform
    └── Optimize Yahoo Fantasy Baseball lineups and waiver decisions
        for active H2H season. Data-driven, not heuristic.
```

Both systems run on the same FastAPI backend, PostgreSQL database, and Railway infrastructure. Both are served by the same Next.js frontend. They share no business logic but share all infrastructure code.

---

## SYSTEM 1: CBB Edge Betting Analyzer

**Version:** V9.1 (V9 Predictive Confidence Engine)
**Season status:** NCAA Tournament active (through early April 2026)
**Recalibration status:** BLOCKED until post-Apr 7 (EMAC-068)

### Mission
Find positive-EV NCAA D1 basketball bets by combining:
- AdjEM-derived margin projection (KenPom + BartTorvik, 2-source)
- Monte Carlo confidence interval (Skellam cover probability)
- SNR scalar (source agreement quality)
- Integrity scalar (real-time news validation via OpenClaw)

### Risk Posture (NON-NEGOTIABLE)

These values are policy. They must not be changed without updating this section.
Code defaults must match. Env vars may tune within the noted ranges.

#### Kelly Scaling Hierarchy
```
Base Kelly (fractional, portfolio-adjusted)
  × SNR Scalar        — source agreement multiplier
  × Integrity Scalar  — real-time news validation multiplier
  = Final Kelly Fraction
```

#### SNR Scalar
| SNR Value | Scalar | Interpretation |
|-----------|--------|----------------|
| 1.0 | 1.0× | Full source agreement |
| 0.75 | 0.875× | Minor divergence |
| 0.50 | 0.75× | Moderate disagreement |
| 0.0 | 0.5× | Maximum disagreement (floor) |

- Floor: `0.5` (env: `SNR_KELLY_FLOOR`). Never below floor.
- Formula: `floor + (1 - floor) * snr`

#### Integrity Scalar
| Verdict Substring | Scalar | Env Override |
|-------------------|--------|--------------|
| CONFIRMED / not run | 1.0× | — |
| CAUTION | 0.75× | `INTEGRITY_CAUTION_SCALAR` |
| VOLATILE | 0.50× | `INTEGRITY_VOLATILE_SCALAR` |
| ABORT / RED FLAG | **0.0× HARD GATE** | **Not overridable** |

#### Hard Circuit Breakers (non-negotiable, no env override)

1. **Integrity Abort Gate:** `"ABORT"` or `"RED FLAG"` in `integrity_verdict` (case-insensitive) →
   `verdict = "PASS — Integrity Abort"` · `kelly_frac = 0` · `recommended_units = 0`

2. **Portfolio Drawdown Breaker:** drawdown > `MAX_DRAWDOWN_PCT` (default 15%) →
   All new bets paused until drawdown recovers.

3. **Market Divergence Anomaly:** `|model_margin - market_margin| > 2.5 × effective_base_sd` →
   Hard PASS. Indicates stale/corrupt data or significant unmodeled news event.

#### Bet Thresholds
- **BET:** point-estimate edge > 2% AND lower-bound CI edge > 0% (after all scalars)
- **CONSIDER:** detectable edge below threshold — monitor only
- **PASS:** 85–95% of all games (by design — efficient market hypothesis)

### Operating Principles (CBB)
1. **Capital first.** A missed bet is a rounding error. A blown bankroll ends the operation.
2. **Trust the math, validate the narrative.** AdjEM owns the mean. OpenClaw validates against the news cycle. Both required before sizing.
3. **Uncertainty compounds downward.** Every missing source, every SNR penalty, every integrity flag → smaller bet. Never larger than base Kelly.
4. **Async is a production requirement.** All DDGS + LLM I/O must be `asyncio.gather`-based. Sync fallback only when event loop fails.
5. **Policy lives here.** No risk parameter magic numbers in code without cross-reference to this file.

### Calibration History
| Date | Parameter | Old | New | Reason |
|------|-----------|-----|-----|--------|
| EMAC-002 | VOLATILE scalar | — | 0.50× | Initial calibration |
| EMAC-002 | CAUTION scalar | — | 0.75× | Initial calibration |
| EMAC-002 | SNR floor | — | 0.50 | Conservative pre-season baseline |

> **V9.1 known issue (EMAC-068):** SNR + Integrity scalars stack on top of half-Kelly (÷2.0), making effective divisor ~3.4×. These scalars were not present in the 663-bet V8 calibration dataset. Combined with 2-source EVANMIYA_DOWN_SE_ADDEND (+0.30), model requires ~6-8% raw edge to emit a 2.5% conservative edge. Recalibration (V9.2) scheduled post-Apr 7.

---

## SYSTEM 2: Fantasy Baseball Platform

**Season:** 2026 MLB, H2H Category League via Yahoo Fantasy
**Season status:** LIVE — Opening Day March 26, 2026

### Mission
Maximize weekly win probability in Yahoo Fantasy Baseball by:
- Optimal daily lineup selection (ILP via OR-Tools CP-SAT)
- Weather-adjusted park factor scoring (OpenWeatherMap)
- Platoon split and matchup quality multipliers
- Category-deficit-driven waiver wire recommendations

### Risk Posture (Fantasy)

Fantasy Baseball does not involve real-money wagering. The risk posture is operational:

1. **Never touch live Yahoo data without error handling.** Every Yahoo API call is wrapped. Failures degrade gracefully — dashboard shows stale data, not 500 errors.

2. **No bool-to-string leakage.** Yahoo API returns `{"status": false}` for active players. All status fields guarded with `or None` at both extraction layer (`_parse_player`) and Pydantic construction (`RosterPlayerOut`).

3. **Timezone discipline.** All MLB game dates computed in `America/New_York` timezone. `datetime.utcnow()` is banned. Use `datetime.now(ZoneInfo("America/New_York"))`.

4. **OR-Tools with graceful fallback.** `lineup_constraint_solver.py` uses OR-Tools CP-SAT for ILP optimization. When OR-Tools is unavailable (Railway deploy in progress), falls back to scarcity-first greedy automatically — no silent failures.

5. **Yahoo client is a single file.** `backend/fantasy_baseball/yahoo_client_resilient.py` contains both `YahooFantasyClient` (base class) and `ResilientYahooClient` (subclass). No forks. No new `yahoo_*.py` files.

### Key Services
| Service | File | Purpose |
|---------|------|---------|
| Lineup Optimizer | `daily_lineup_optimizer.py` | Daily batter/pitcher decisions |
| Smart Lineup Selector | `smart_lineup_selector.py` | Weather + platoon + category scoring |
| Lineup Constraint Solver | `lineup_constraint_solver.py` | ILP (OR-Tools) + greedy fallback |
| Waiver Edge Detector | `services/waiver_edge_detector.py` | Category deficit analysis |
| Dashboard Service | `services/dashboard_service.py` | Panel data aggregation |
| Weather Fetcher | `fantasy_baseball/weather_fetcher.py` | OpenWeatherMap (env: `OPENWEATHER_API_KEY`) |
| Park Weather Analyzer | `fantasy_baseball/park_weather.py` | Stadium-specific weather scoring |
| Yahoo Client | `fantasy_baseball/yahoo_client_resilient.py` | Single canonical API client |

### Operating Principles (Fantasy)
1. **Data over gut.** Every lineup decision is backed by a scoring formula. Heuristics are only used when the data pipeline degrades.
2. **Degrade gracefully.** Yahoo down → DB fallback. OR-Tools missing → greedy fallback. OpenWeather missing → seasonal estimate. Users always see something.
3. **Roster integrity before optimization.** No lineup is submitted with IL players in active slots. Gap detection runs before scorer.
4. **Two-start SPs are gold.** Seven-day rolling window for start detection. Always surface two-start SPs prominently.

---

## Shared Infrastructure Identity

**Platform:** FastAPI + PostgreSQL + Railway (production) + Next.js (frontend)
**Test standard:** All changes require `py_compile` pass + relevant pytest subset green
**Deployment:** Railway auto-deploy from main branch. Dockerfile chains migrations.
**Env vars:** Railway dashboard (Gemini domain). Never committed to repo.
**Monitoring:** Railway logs (Gemini tails). Discord alerts (OpenClaw sends).
