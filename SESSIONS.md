# CBB Edge — Session History

## 2026-02-28 (Saturday)

### 13:00 — Core Portfolio & Mapping Updates (P0)
- **Implemented Global Proportional Scaler (P0)**:
  - Modified `backend/services/analysis.py` to move paper-bet allocation logic outside of the main game-analysis loop.
  - **Removed Sequential Scaling**: The system no longer enforces the daily cap game-by-game, ensuring fair allocation across the entire slate.
  - Updated `_create_paper_bet` to accept `scaled_bet_dollars`, bypassing internal sequential capacity checks when provided.
  - Initialized `current_exposure_acc` before the loop to properly track incremental exposure changes.
- **Resolved North Dakota/Alabama Mapping Collisions**:
  - Added explicit mappings in `backend/services/team_mapping.py` for North Dakota and SWAC teams.
  - Added **Substring Guard** comments to protect against fuzzy-match collisions with flagship programs like "Alabama".
- **Verification**:
  - Confirmed via web search that North Dakota and North Dakota State have distinct AdjEM ratings for the 2026 season (#131 vs #241).
  - Validated `analysis.py` syntax with `py_compile`.
  - Verified portfolio math: All bets are now proportionally scaled to fit within the `MAX_DAILY_EXPOSURE_PCT` limit.

### 16:00 — Phase 1.5 'Fire Fighting' Hand-off
- System stability verified.
- Core portfolio and mapping P0 bugs resolved.
- Ready for Claude hand-off.

### 17:30 — O1-O3 Audit + Edge Calibration Fixes (Claude)
- **O1 Operations Audit**: Triggered live analysis run (134 games, 16 BET, 271s).
  EvanMiya 100% null. 34/134 games missing BartTorvik (mapping gaps). Bet rate 12%.
- **O2 Slate Inspection**: Validated 16 bets. Found North Dakota identity bug (already
  fixed in P1.5), 13.2% edge passed circuit breaker via matchup lift, 5 games with
  novig=50/50 (correct Shin behavior for symmetric markets).
- **O3 Model Quality Audit**: Identified edge inflation (mean 8.4% vs expected 4-5%).
  Three root causes: tight margin_se, model-heavy sigmoid, noisy Markov sims.
- **Implemented 4 calibration fixes** (all tests passing, 358/358):
  1. `BASE_MARGIN_SE` 0.85 → 1.50, `MAX_MARGIN_SE` 1.65 → 2.50
     (betting_model.py + .env.example + 5 tests updated)
  2. EvanMiya 0-team HTTP 200 now counted as failure with auto-drop guard
     (ratings.py — final guard after all strategies exhausted)
  3. Markov n_sims 2000 → 5000 (betting_model.py line ~1185)
  4. Market-blend sigmoid midpoint 6h → 10h (betting_model.py _dynamic_model_weight)
