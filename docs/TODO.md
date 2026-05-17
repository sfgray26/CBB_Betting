# Technical Debt & TODO Registry

**Generated:** 2026-05-16  
**Branch:** agent/gemini/todo-documentation-20260516  
**Total TODOs:** 28

---

## Summary by Category

| Category | Count | Priority |
|----------|-------|----------|
| Data Pipeline | 10 | P1 |
| API / Backend | 8 | P1-P2 |
| Features | 7 | P2 |
| Performance | 3 | P3 |

---

## TODO Registry

### Data Pipeline TODOs

| ID | Location | Description | Priority | Est. Effort | Dependencies |
|----|----------|-------------|----------|-------------|--------------|
| DP-1 | `fantasy.py:2074` | Populate `quality_score` from ProbablePitcherSnapshot for pitchers | P1 | 4h | ProbablePitcherSnapshot table |
| DP-2 | `fantasy.py:3277-3280` | Track `fetched_at` timestamps and compute `is_stale` for Yahoo data freshness | P1 | 6h | Yahoo client instrumentation |
| DP-3 | `player_mapper.py:112` | Wire up Yahoo matchup data for game context (opponent, game_time, weather) | P2 | 8h | Yahoo matchup API integration |
| DP-4 | `player_mapper.py:203` | Game context fields PR-5 through PR-12 implementation | P2 | 8h | DP-3 |
| DP-5 | `player_mapper.py:233,236` | Track Yahoo client fetch timestamps for freshness metadata | P2 | 4h | DP-2 |
| DP-6 | `scoreboard_orchestrator.py:394,397` | Track fetched_at and compute is_stale for scoreboard data | P2 | 4h | DP-2 |
| DP-7 | `row_projector.py:251` | Use actual MLB schedule for hitter games remaining | P2 | 6h | MLB schedule API integration |
| DP-8 | `row_projector.py:255` | Use ProbablePitcherSnapshot for accurate SP start predictions | P1 | 4h | ProbablePitcherSnapshot query |
| DP-9 | `row_projector.py:570` | Incorporate off-days from schedule into projections | P3 | 8h | DP-7 |
| DP-10 | `row_projector.py:601` | Implement rotation math + probable pitcher feed integration | P2 | 12h | DP-8, Probable pitcher API |
| DP-11 | `daily_ingestion.py:3701` | Implement direct Yahoo API ownership fetch (PR 4.2.1) | P2 | 6h | Yahoo API client |
| DP-12 | `daily_ingestion.py:3782` | Implement add/drop rate tracking | P3 | 8h | Waiver wire tracking system |

### API / Backend TODOs

| ID | Location | Description | Priority | Est. Effort | Dependencies |
|----|----------|-------------|----------|-------------|--------------|
| API-1 | `fantasy.py:3277-3280` | Freshness tracking for roster endpoint | P1 | 4h | DP-2 |
| API-2 | `scoreboard_orchestrator.py:394-397` | Freshness metadata for matchup scoreboard | P2 | 4h | DP-6 |
| API-3 | `player_mapper.py:233-236` | Freshness tracking for player row mapping | P2 | 4h | DP-5 |
| API-4 | `constraint_helpers.py:338` | Probable pitcher check in constraint validation | P2 | 4h | Probable pitcher data |
| API-5 | `openclaw_telemetry.py:138` | Wire integrity_checks_24h to actual telemetry | P3 | 4h | Telemetry service |
| API-6 | `openclaw_telemetry.py:150` | Wire sharp_signals_24h to sharp_money service | P3 | 4h | Sharp money service |
| API-7 | `fantasy.py:2074` | Pitcher quality_score in waiver recommendations | P1 | 4h | DP-1 |
| API-8 | `decision_tracker.py:220` | Compare user override vs system decision outcomes | P2 | 6h | Decision tracking analytics |

### Features TODOs

| ID | Location | Description | Priority | Est. Effort | Dependencies |
|----|----------|-------------|----------|-------------|--------------|
| FEAT-1 | `elite_context.py:402` | Integrate FanGraphs for deep dive analytics | P2 | 12h | FanGraphs API client |
| FEAT-2 | `elite_context.py:409` | Integrate weather API for game context | P2 | 8h | Weather API integration |
| FEAT-3 | `elite_context.py:416` | Integrate Statcast/pybaseball for advanced metrics | P2 | 10h | Statcast data pipeline |
| FEAT-4 | `elite_context.py:421` | Scrape from MLB.com or fetch from API | P2 | 8h | MLB.com API client |
| FEAT-5 | `elite_context.py:426` | Integrate platoon_fetcher for matchup data | P2 | 6h | Platoon data service |
| FEAT-6 | `daily_briefing.py:547` | Use schedule fetcher for game context | P2 | 4h | Schedule service |
| FEAT-7 | `fantasy.py:3277-3280` | Data freshness indicators in UI | P2 | 6h | DP-2, API-1 |

### Performance TODOs

| ID | Location | Description | Priority | Est. Effort | Dependencies |
|----|----------|-------------|----------|-------------|--------------|
| PERF-1 | `row_projector.py:251` | Optimize games remaining calculation with cached schedule | P3 | 6h | DP-7 |
| PERF-2 | `row_projector.py:570` | Batch off-day calculations for efficiency | P3 | 8h | DP-9 |
| PERF-3 | `scoreboard_orchestrator.py` | Cache Monte Carlo simulation results | P3 | 8h | Caching layer |

---

## Dependencies Graph

```
DP-2 (Freshness Tracking)
├── API-1 (Roster freshness)
├── API-2 (Scoreboard freshness)
├── DP-5 (Player mapper freshness)
├── DP-6 (Scoreboard orchestrator freshness)
└── FEAT-7 (UI freshness indicators)

DP-1 (Quality Score)
├── API-7 (Waiver recommendations)
└── DP-8 (SP start predictions)

DP-3 (Game Context)
├── DP-4 (PR-5 through PR-12)
└── FEAT-2 (Weather)

ProbablePitcherSnapshot
├── DP-1 (Quality score)
├── DP-8 (SP predictions)
├── DP-10 (Rotation math)
└── API-4 (Constraint checks)

MLB Schedule API
├── DP-7 (Hitter games)
├── DP-9 (Off-days)
└── PERF-1 (Optimization)

FanGraphs/Statcast
├── FEAT-1 (FanGraphs)
├── FEAT-3 (Statcast)
└── FEAT-4 (MLB.com)
```

---

## Recommended Implementation Order

### Phase 1: Critical Data Freshness (P1)
1. **DP-2**: Implement Yahoo client timestamp tracking
2. **API-1**: Wire freshness to roster endpoint
3. **DP-1**: Populate pitcher quality_score from ProbablePitcherSnapshot
4. **API-7**: Add quality_score to waiver recommendations

### Phase 2: Game Context & Projections (P2)
5. **DP-8**: Probable pitcher snapshot for SP predictions
6. **DP-7**: MLB schedule integration for hitters
7. **DP-3**: Yahoo matchup data wiring
8. **DP-4**: Complete PR-5 through PR-12 game context

### Phase 3: Advanced Features (P2-P3)
9. **DP-10**: Rotation math and probable pitcher feed
10. **FEAT-2**: Weather API integration
11. **FEAT-3**: Statcast/pybaseball integration
12. **FEAT-1**: FanGraphs deep dive

### Phase 4: Telemetry & Optimization (P3)
13. **API-5**: Integrity checks telemetry
14. **API-6**: Sharp signals telemetry
15. **PERF-1**: Schedule caching optimization
16. **PERF-3**: Monte Carlo result caching

---

## Notes

### Data Freshness Pattern
Multiple TODOs reference `fetched_at`/`is_stale` pattern:
- Standard threshold: 60 minutes for staleness
- Implementation: Track fetch timestamp in Yahoo client, compute staleness on response assembly
- Affected files: `fantasy.py`, `player_mapper.py`, `scoreboard_orchestrator.py`

### Quality Score
- Source: `ProbablePitcherSnapshot.quality_score` column
- Used for: Pitcher streaming recommendations
- Impact: Higher quality scores indicate better streaming targets

### Game Context (PR-5 through PR-12)
Missing fields:
- `opponent_team`: Opposing team abbreviation
- `game_time`: Scheduled start time
- `is_home`: Home/away flag
- `weather`: Game weather conditions
- Source: Yahoo matchup API (not yet integrated)

---

*Last updated: 2026-05-16*
