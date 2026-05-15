# CBB Edge - Project Context for Hermes

## Project Overview
- **Repo**: /mnt/c/Users/sfgra/repos/Fixed/cbb-edge
- **Stack**: Python 3.11, FastAPI, deployed to Railway
- **Branch**: stable/cbb-prod
- **Domain**: Fantasy baseball (MLB) lineup optimization, projections, betting recommendations

## Key Files
- scripts/audit_lite.py — daily health audit
- scripts/todays_bets.py — betting recommendations
- scripts/model_quality_audit.py — model drift check
- scripts/weather_adjust.py — weather-adjusted projections
- scripts/yahoo_lineup_auto.py — automated Yahoo lineup
- scripts/daily_analysis.py — daily performance report

## Critical Bug Report — 5 P1s (Latest Feedback)

### Structural Theme
The codebase has the right ideas but parallel implementations competing for authority — tested solver vs. shipped greedy endpoint, 5-factor matchup model vs. disabled fetch layer, canonical category map vs. 3 page-local remappers.

### Bug 1: Roster Optimize Endpoint Uses Wrong Solver
- **Bug**: Roster optimize endpoint uses a greedy allocator, not the tested scarcity-aware solver
- **Location**: fantasy.py:3447-3477
- **Impact**: Wrong lineup slots despite better code existing

### Bug 2: Inverted Implied Runs Sign
- **Bug**: _implied_runs() sign is inverted for negative home spreads — favored home team gets fewer projected runs
- **Location**: daily_lineup_optimizer.py:420-436
- **Impact**: Pollutes batter and streaming pitcher rankings

### Bug 3: Silent Empty Roster on Missing Count Field
- **Bug**: get_roster() silently returns empty list if Yahoo payload omits count field
- **Location**: yahoo_client_resilient.py:693-701
- **Impact**: Cascading "no roster" failure under Yahoo shape variance

### Bug 4: Disabled Pitcher Handedness Signal
- **Bug**: Pitcher handedness signal (claimed as 35% of matchup score) never activates — hand=None always
- **Location**: matchup_engine.py:227-267
- **Impact**: Matchup scores less discriminating than advertised

### Bug 5: Unsafe Live Projection Fallback
- **Bug**: _get_live_projection(None, "") falls back to ilike("%%") — any DB row matches
- **Location**: projection_assembly_service.py:503-530
- **Impact**: Wrong projection attached to Yahoo-only players

### Recommended Fix Order
1. Unify runtime paths — close the gap between tested solver and shipped endpoint
2. Close parser/math hazards — fix sign inversion and ilike fallback
3. Collapse duplicate contracts — merge competing category maps and remappers
4. Then tune — enable disabled features (handedness signal)

## Goal for Hermes
Review the codebase with these P1s as the lens. Identify:
- Where the parallel implementations diverge
- The safest path to unify them
- Any additional bugs or risks not captured above
- Priority-ranked action plan with estimated effort
