# K15: Oracle Validation Spec — Consensus Spread Divergence Detection
**Date:** March 22, 2026  
**Status:** Production-Ready Specification  
**Target Implementation:** Post-April 7, 2026  

---

## 1. Executive Summary

The Oracle Validation System detects when our model's projected margin diverges significantly from the market consensus spread. This serves as a data-quality sentinel — flagging potential team-mapping errors, unreported injuries, stale ratings, or line-moving sharp action that our model hasn't captured.

---

## 2. Consensus Spread Formula

### 2.1 Our Model's Projected Margin

```python
# From betting_model.py — analyze_game() method

# Step 1: Collect available rating sources
available_sources = []

# KenPom (required — already checked above)
kenpom_diff = kp_home - kp_away  # AdjEM difference
available_sources.append(('kenpom', kenpom_diff))

# BartTorvik (if available)
if bt_home is not None and bt_away is not None:
    barttorvik_diff = bt_home - bt_away  # AdjEM = AdjOE - AdjDE
    available_sources.append(('barttorvik', barttorvik_diff))

# EvanMiya (if available)
if em_home is not None and em_away is not None:
    evanmiya_diff = em_home - em_away  # BPR difference
    available_sources.append(('evanmiya', evanmiya_diff))

# Step 2: Renormalize weights to available sources
total_weight = sum(weights[src] for src, _ in available_sources)  # weights sum to ~1.0

# Step 3: Compute weighted margin
margin = sum(
    (weights[src] / total_weight) * diff
    for src, diff in available_sources
)

# Step 4: Apply confidence shrinkage for missing sources
n_available = len(available_sources)
n_total = 3  # Expected sources
n_missing = max(0, n_total - n_available)
if n_missing > 0:
    shrinkage = max(0.70, 1.0 - 0.10 * n_missing)
    margin *= shrinkage

# Step 5: Apply adjustments (IN ORDER)
margin += adjusted_hca                    # Pace-adjusted home court advantage
margin += injury_adj                      # Net injury impact (signed)
margin += matchup_margin_adj              # MatchupEngine geometry adjustment
margin += fatigue_margin_adj              # Rest/travel/altitude adjustment

# Step 6: Market blend (if sharp spread available)
if sharp_consensus_spread is not None:
    market_margin = -sharp_consensus_spread  # Convert spread to margin
    model_weight = dynamic_model_weight(hours_to_tipoff, sharp_books_available, injury_adj)
    margin = model_weight * margin + (1 - model_weight) * market_margin

# FINAL: our_projected_margin = margin
```

### 2.2 Market Consensus Spread

```python
# From odds.py — parse_odds_for_game() method

# Sharp consensus (primary)
sharp_consensus_spread = average(spread_home across Pinnacle + Circa)

# Soft-proxy fallback (if no sharp books)
if sharp_books_available == 0:
    if DraftKings and FanDuel agree within 0.5 points:
        sharp_proxy_used = True
        sharp_consensus_spread = (DK_spread + FD_spread) / 2

# Convert to market margin
market_consensus_margin = -sharp_consensus_spread
```

### 2.3 Consensus Spread Definition

```python
# The "Consensus Spread" is the sharp market line BEFORE our model blend
# If no sharp line exists, use best available spread

consensus_spread = (
    odds.get('sharp_consensus_spread') or      # Pinnacle/Circa average
    odds.get('best_spread') or                  # Best available retail
    None
)

consensus_margin = -consensus_spread if consensus_spread else None
```

---

## 3. Oracle Comparison Logic

### 3.1 Core Comparison Formula

```python
def calculate_oracle_divergence(
    our_projected_margin: float,
    consensus_margin: float,
    adjusted_sd: float,
) -> OracleResult:
    """
    Calculate divergence between model and market consensus.
    
    Returns OracleResult with divergence metrics and flag status.
    """
    
    # Absolute divergence in points
    divergence_points = abs(our_projected_margin - consensus_margin)
    
    # Z-score divergence (normalized by game uncertainty)
    # Uses adjusted_sd (already includes penalties, heuristic multipliers, etc.)
    divergence_z = divergence_points / max(adjusted_sd, 1.0)
    
    # Threshold determination (see Section 4)
    threshold_z = get_oracle_threshold(hours_to_tipoff)
    
    # Oracle flag logic
    oracle_flag = divergence_z > threshold_z
    
    return OracleResult(
        divergence_points=divergence_points,
        divergence_z=divergence_z,
        threshold_z=threshold_z,
        oracle_flag=oracle_flag,
        our_margin=our_projected_margin,
        consensus_margin=consensus_margin,
    )
```

### 3.2 Comparison Implementation

```python
# Exact code to insert in analysis.py

# After: analysis = model.analyze_game(...)
# Before: prediction populated and persisted

# --- ORACLE VALIDATION CHECK ---
oracle_result = None
if odds_input.get('sharp_consensus_spread') is not None:
    consensus_margin = -odds_input['sharp_consensus_spread']
    our_margin = analysis.projected_margin
    
    # Use the pre-penalty base SD for cleaner divergence measurement
    # (post-penalty SD can inflate artificially from missing sources)
    base_sd_for_oracle = dynamic_base_sd or model.base_sd
    
    divergence_points = abs(our_margin - consensus_margin)
    divergence_z = divergence_points / max(base_sd_for_oracle, 1.0)
    
    # Dynamic threshold (see Section 4)
    hours_to_tip = hours_to_tipoff or 12.0
    if hours_to_tip >= 24:
        threshold_z = 2.0  # 2.0 sigma = ~95% confidence
    elif hours_to_tip >= 4:
        threshold_z = 2.5  # 2.5 sigma = ~99% confidence
    else:
        threshold_z = 3.0  # 3.0 sigma = ~99.7% confidence (sharp line settling)
    
    oracle_flag = divergence_z > threshold_z
    
    oracle_result = {
        'our_margin': our_margin,
        'consensus_margin': consensus_margin,
        'divergence_points': divergence_points,
        'divergence_z': divergence_z,
        'threshold_z': threshold_z,
        'oracle_flag': oracle_flag,
        'hours_to_tipoff': hours_to_tip,
    }
    
    if oracle_flag:
        logger.warning(
            "ORACLE FLAG [%s @ %s]: divergence=%.2fpts (z=%.2f > threshold=%.2f). "
            "Our margin: %+.1f, Market: %+.1f",
            away_team, home_team,
            divergence_points, divergence_z, threshold_z,
            our_margin, consensus_margin,
        )
        notes.append(
            f"ORACLE FLAG: Model diverges {divergence_points:.1f}pts from market "
            f"({divergence_z:.2f}σ > {threshold_z:.2f}σ threshold)"
        )
```

---

## 4. Recommended Threshold Value & Justification

### 4.1 Threshold Schedule

| Hours to Tipoff | Threshold (z) | Probability | Rationale |
|-----------------|---------------|-------------|-----------|
| ≥ 24 hours | 2.0σ | ~95% | Thin market; model legitimately leads |
| 4–24 hours | 2.5σ | ~99% | Market forming; tighter gate |
| < 4 hours | 3.0σ | ~99.7% | Sharp books settling; extreme divergence = data error |

### 4.2 Justification from Existing Code

The thresholds are derived from existing system patterns:

1. **Z-Divergence Guard** (betting_model.py, line ~715):
   ```python
   _Z_DIVERGENCE_THRESHOLD = 2.5  # Hard PASS at 2.5 sigma
   ```
   This guards against model vs. market margin divergence > 2.5 SD.

2. **Edge Breaker Threshold** (betting_model.py, line ~925):
   ```python
   # >= 24 h: 15% edge threshold
   # 4 <= h < 24: 12% edge threshold  
   # h < 4: exponential decay to 6%
   ```
   The Oracle uses probability (z-score) rather than edge percentage for cleaner statistical interpretation.

3. **Confidence Shrinkage** (betting_model.py, line ~475):
   ```python
   shrinkage = max(0.70, 1.0 - 0.10 * n_missing)
   ```
   Sources missing → higher uncertainty → wider tolerance early.

### 4.3 Threshold Rationale

| Scenario | Expected Behavior |
|----------|-------------------|
| **T-24h, divergence = 2.1σ** | Flagged — early divergence may indicate our injury info isn't priced in yet |
| **T-6h, divergence = 2.6σ** | Flagged — market should be converging; divergence suggests data error |
| **T-1h, divergence = 3.1σ** | Flagged — sharp books are efficient; this is almost certainly a bug |
| **T-1h, divergence = 2.0σ** | NOT flagged — within 3.0σ threshold for near-tipoff |

### 4.4 Environment Variable Override

```python
# Allow operators to adjust without code deploy
ORACLE_THRESHOLD_Z_EARLY = float(os.getenv("ORACLE_THRESHOLD_Z_EARLY", "2.0"))   # >= 24h
ORACLE_THRESHOLD_Z_MID = float(os.getenv("ORACLE_THRESHOLD_Z_MID", "2.5"))       # 4-24h
ORACLE_THRESHOLD_Z_LATE = float(os.getenv("ORACLE_THRESHOLD_Z_LATE", "3.0"))     # < 4h
```

---

## 5. OracleResult Schema

### 5.1 Dataclass Definition

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class OracleResult:
    """
    Complete record of Oracle divergence analysis for a game.
    Persisted to prediction.oracle_result (JSONB) and used for alerting.
    """
    
    # Core identifiers
    game_key: str                              # "AwayTeam@HomeTeam"
    analysis_timestamp: datetime               # When Oracle check ran
    
    # Margin comparison
    our_projected_margin: float                # Our model's final margin
    consensus_margin: float                    # Market consensus margin
    divergence_points: float                   # Absolute difference (pts)
    
    # Statistical divergence
    base_sd_used: float                        # SD used for z-calculation
    divergence_z: float                        # Normalized divergence (sigma)
    threshold_z: float                         # Applied threshold
    
    # Flag and verdict
    oracle_flag: bool                          # True if divergence > threshold
    flag_reason: Optional[str]                 # Human-readable explanation
    
    # Context for debugging
    hours_to_tipoff: Optional[float]           # Time context
    sharp_books_available: int                 # Market data quality
    sources_used: list[str]                    # ["kenpom", "barttorvik", ...]
    
    # Optional: component breakdown
    our_margin_components: Optional[dict]      # {kenpom: X, barttorvik: Y, ...}
    
    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            'game_key': self.game_key,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'our_projected_margin': self.our_projected_margin,
            'consensus_margin': self.consensus_margin,
            'divergence_points': self.divergence_points,
            'base_sd_used': self.base_sd_used,
            'divergence_z': self.divergence_z,
            'threshold_z': self.threshold_z,
            'oracle_flag': self.oracle_flag,
            'flag_reason': self.flag_reason,
            'hours_to_tipoff': self.hours_to_tipoff,
            'sharp_books_available': self.sharp_books_available,
            'sources_used': self.sources_used,
            'our_margin_components': self.our_margin_components,
        }
```

### 5.2 JSON Schema (for API/documentation)

```json
{
  "game_key": "Duke@UNC",
  "analysis_timestamp": "2026-03-22T14:30:00Z",
  "our_projected_margin": 8.5,
  "consensus_margin": 3.2,
  "divergence_points": 5.3,
  "base_sd_used": 11.2,
  "divergence_z": 2.38,
  "threshold_z": 2.5,
  "oracle_flag": false,
  "flag_reason": null,
  "hours_to_tipoff": 18.5,
  "sharp_books_available": 2,
  "sources_used": ["kenpom", "barttorvik", "evanmiya"],
  "our_margin_components": {
    "kenpom_contrib": 3.2,
    "barttorvik_contrib": 2.8,
    "evanmiya_contrib": 1.5,
    "home_advantage": 2.1,
    "injury_adj": -0.8,
    "matchup_adj": 0.4,
    "fatigue_adj": -0.1
  }
}
```

---

## 6. Insertion Point in analysis.py

### 6.1 Exact Location

**File:** `backend/services/analysis.py`  
**Function:** `run_nightly_analysis()`  
**Line:** After line ~1100 (after `analysis = model.analyze_game(...)`), before prediction persistence.

### 6.2 Exact Call Site

```python
# ================================================================
# EXISTING CODE (around line 1100-1120)
# ================================================================

analysis = model.analyze_game(
    game_data=game_input,
    odds=odds_input,
    ratings=ratings_input,
    injuries=game_injuries,
    data_freshness=odds_freshness,
    base_sd_override=dynamic_base_sd,
    home_style=home_style,
    away_style=away_style,
    matchup_margin_adj=matchup_margin_adj,
    hours_to_tipoff=hours_to_tipoff,
    concurrent_exposure=_pass2_concurrent_exposure,
    target_exposure=target_exposure,
    sharp_books_available=odds_input.get("sharp_books_available", 0),
    sharp_proxy_used=bool(game_data.get("sharp_proxy_used", False)),
    integrity_verdict=integrity_verdict,
    fatigue_margin_adj=fatigue_margin_adj,
    fatigue_metadata=fatigue_metadata,
)

# ================================================================
# ORACLE VALIDATION INSERTION (NEW CODE)
# ================================================================

oracle_result = None
oracle_flag = False
if odds_input.get('sharp_consensus_spread') is not None:
    from backend.services.oracle_validator import calculate_oracle_divergence
    
    oracle_result = calculate_oracle_divergence(
        our_margin=analysis.projected_margin,
        consensus_spread=odds_input['sharp_consensus_spread'],
        base_sd=dynamic_base_sd or model.base_sd,
        hours_to_tipoff=hours_to_tipoff,
        game_key=_game_key,
        sharp_books_available=odds_input.get('sharp_books_available', 0),
        sources_used=[src for src, _ in available_sources],  # from betting_model
    )
    oracle_flag = oracle_result.oracle_flag
    
    if oracle_flag:
        logger.warning(
            "ORACLE FLAG [%s]: divergence=%.2fpts (z=%.2f > %.2f)",
            _game_key, oracle_result.divergence_points,
            oracle_result.divergence_z, oracle_result.threshold_z,
        )

# ================================================================
# EXISTING CODE CONTINUES (ReanalysisEngine capture, ~line 1120)
# ================================================================

try:
    _reanalysis_cache[_game_key] = ReanalysisEngine.from_analysis_pass(...)
except Exception as re_exc:
    ...
```

### 6.3 Required Imports

```python
# Add to top of analysis.py
from backend.services.oracle_validator import calculate_oracle_divergence, OracleResult
```

---

## 7. Database Schema Changes

### 7.1 Prediction Table Migration

```sql
-- Migration: Add Oracle flag to predictions table
-- File: migrations/0015_add_oracle_flag.sql

-- Add oracle_flag boolean (nullable, default NULL for historical rows)
ALTER TABLE predictions 
ADD COLUMN oracle_flag BOOLEAN DEFAULT NULL;

-- Add oracle_result JSONB for full context
ALTER TABLE predictions 
ADD COLUMN oracle_result JSONB DEFAULT NULL;

-- Index for efficient querying of flagged predictions
CREATE INDEX idx_predictions_oracle_flag 
ON predictions(oracle_flag) 
WHERE oracle_flag IS NOT NULL;

-- Composite index for admin queries
CREATE INDEX idx_predictions_oracle_date 
ON predictions(prediction_date, oracle_flag) 
WHERE oracle_flag = TRUE;
```

### 7.2 SQLAlchemy Model Update

```python
# backend/models.py — Prediction class

class Prediction(Base):
    # ... existing columns ...
    
    # Oracle validation fields
    oracle_flag = Column(Boolean, nullable=True, default=None, index=True)
    oracle_result = Column(JSONB, nullable=True, default=None)
    
    # ... rest of model ...
```

### 7.3 Migration Script

```python
# scripts/migrate_oracle_fields.py
"""One-time migration to add Oracle fields to predictions table."""

from backend.models import engine, Prediction
from sqlalchemy import Column, Boolean, JSON

def upgrade():
    """Add oracle_flag and oracle_result columns."""
    from sqlalchemy import MetaData, Table
    
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    predictions_table = Table('predictions', metadata, autoload_with=engine)
    
    # Check if columns exist
    existing_cols = {c.name for c in predictions_table.columns}
    
    with engine.connect() as conn:
        if 'oracle_flag' not in existing_cols:
            conn.execute("""
                ALTER TABLE predictions 
                ADD COLUMN oracle_flag BOOLEAN DEFAULT NULL
            """)
            conn.execute("""
                CREATE INDEX idx_predictions_oracle_flag 
                ON predictions(oracle_flag) 
                WHERE oracle_flag IS NOT NULL
            """)
            print("Added oracle_flag column")
        
        if 'oracle_result' not in existing_cols:
            conn.execute("""
                ALTER TABLE predictions 
                ADD COLUMN oracle_result JSONB DEFAULT NULL
            """)
            print("Added oracle_result column")
        
        conn.commit()
    
    print("Migration complete")

if __name__ == "__main__":
    upgrade()
```

---

## 8. Admin Endpoint Specification

### 8.1 Endpoint Definition

```python
# backend/main.py — Admin routes

@app.get("/admin/oracle/flagged", response_model=OracleFlaggedResponse)
async def get_oracle_flagged_predictions(
    date: Optional[date] = Query(None, description="Filter by date (default: today)"),
    min_divergence_z: Optional[float] = Query(None, description="Minimum divergence z-score"),
    include_passed: bool = Query(False, description="Include PASS verdicts or only BET/CONSIDER"),
    db: Session = Depends(get_db),
):
    """
    Retrieve all Oracle-flagged predictions for review.
    
    Returns predictions where model diverged significantly from market consensus.
    Useful for identifying data quality issues or unreported injuries.
    """
    target_date = date or datetime.utcnow().date()
    
    query = db.query(Prediction).join(Game).filter(
        Prediction.prediction_date == target_date,
        Prediction.oracle_flag == True,
    )
    
    if min_divergence_z:
        # Filter by divergence_z in oracle_result JSONB
        query = query.filter(
            Prediction.oracle_result['divergence_z'].astext.cast(float) >= min_divergence_z
        )
    
    if not include_passed:
        query = query.filter(
            ~Prediction.verdict.startswith("PASS")
        )
    
    predictions = query.order_by(
        Prediction.oracle_result['divergence_z'].astext.cast(float).desc()
    ).all()
    
    return OracleFlaggedResponse(
        date=target_date,
        total_flagged=len(predictions),
        predictions=[_format_oracle_prediction(p) for p in predictions],
    )
```

### 8.2 Response Schema

```python
# backend/schemas.py

from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class OraclePredictionDetail(BaseModel):
    """Single Oracle-flagged prediction for admin review."""
    
    prediction_id: int
    game_key: str
    
    # Teams and matchup
    home_team: str
    away_team: str
    
    # Model vs Market
    our_projected_margin: float
    consensus_margin: float
    divergence_points: float
    divergence_z: float
    threshold_z: float
    
    # Verdict info
    model_verdict: str
    model_edge: Optional[float]
    
    # Context
    hours_to_tipoff: Optional[float]
    sharp_books_available: int
    sources_used: List[str]
    
    # Timestamp
    analysis_timestamp: str
    
    # Full breakdown (optional, for deep dives)
    margin_components: Optional[dict] = None

class OracleFlaggedResponse(BaseModel):
    """Response from GET /admin/oracle/flagged."""
    
    date: date
    total_flagged: int
    predictions: List[OraclePredictionDetail]
    
    # Summary statistics
    avg_divergence_z: Optional[float] = None
    max_divergence_z: Optional[float] = None
    
    class Config:
        from_attributes = True
```

### 8.3 Sample Response

```json
{
  "date": "2026-03-22",
  "total_flagged": 3,
  "predictions": [
    {
      "prediction_id": 15432,
      "game_key": "Gonzaga@SaintMarys",
      "home_team": "Saint Mary's",
      "away_team": "Gonzaga",
      "our_projected_margin": -2.5,
      "consensus_margin": 4.8,
      "divergence_points": 7.3,
      "divergence_z": 3.42,
      "threshold_z": 2.5,
      "model_verdict": "Bet 1.00u [T3] Gonzaga (away +2.5) @ -110",
      "model_edge": 0.052,
      "hours_to_tipoff": 6.5,
      "sharp_books_available": 2,
      "sources_used": ["kenpom", "barttorvik"],
      "analysis_timestamp": "2026-03-22T14:30:00Z",
      "margin_components": {
        "kenpom_contrib": 1.2,
        "barttorvik_contrib": 0.8,
        "home_advantage": 2.1,
        "injury_adj": -6.5
      }
    }
  ],
  "avg_divergence_z": 3.15,
  "max_divergence_z": 3.42
}
```

### 8.4 Helper Function

```python
def _format_oracle_prediction(p: Prediction) -> OraclePredictionDetail:
    """Format a Prediction ORM object for the Oracle response."""
    
    oracle_result = p.oracle_result or {}
    game = p.game
    
    return OraclePredictionDetail(
        prediction_id=p.id,
        game_key=f"{game.away_team}@{game.home_team}",
        home_team=game.home_team,
        away_team=game.away_team,
        our_projected_margin=oracle_result.get('our_projected_margin'),
        consensus_margin=oracle_result.get('consensus_margin'),
        divergence_points=oracle_result.get('divergence_points'),
        divergence_z=oracle_result.get('divergence_z'),
        threshold_z=oracle_result.get('threshold_z'),
        model_verdict=p.verdict,
        model_edge=p.edge_conservative,
        hours_to_tipoff=oracle_result.get('hours_to_tipoff'),
        sharp_books_available=oracle_result.get('sharp_books_available', 0),
        sources_used=oracle_result.get('sources_used', []),
        analysis_timestamp=oracle_result.get('analysis_timestamp'),
        margin_components=oracle_result.get('our_margin_components'),
    )
```

---

## 9. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `backend/services/oracle_validator.py` with `calculate_oracle_divergence()`
- [ ] Create `OracleResult` dataclass
- [ ] Run database migration (add `oracle_flag` and `oracle_result` columns)
- [ ] Update `backend/models.py` Prediction class

### Phase 2: Integration
- [ ] Insert Oracle check in `analysis.py` after `model.analyze_game()` call
- [ ] Persist `oracle_flag` and `oracle_result` to Prediction record
- [ ] Add logging for flagged predictions

### Phase 3: Admin API
- [ ] Add Pydantic schemas to `backend/schemas.py`
- [ ] Implement `GET /admin/oracle/flagged` endpoint in `backend/main.py`
- [ ] Add `_format_oracle_prediction()` helper

### Phase 4: Testing & Validation
- [ ] Unit tests for `calculate_oracle_divergence()`
- [ ] Integration test for admin endpoint
- [ ] Backtest on historical predictions to tune thresholds

### Phase 5: Documentation
- [ ] Update API documentation
- [ ] Add Oracle section to operator runbook
- [ ] Document threshold override env vars

---

## 10. Files to Create/Modify

### New Files
```
backend/services/oracle_validator.py      # Core Oracle logic
scripts/migrate_oracle_fields.py          # DB migration
```

### Modified Files
```
backend/models.py                         # Add oracle_flag, oracle_result columns
backend/services/analysis.py              # Insert Oracle check (line ~1100)
backend/schemas.py                        # Add Oracle response schemas
backend/main.py                           # Add admin endpoint
```

---

## 11. Environment Variables

```bash
# Oracle Threshold Configuration (optional overrides)
ORACLE_THRESHOLD_Z_EARLY=2.0     # >= 24 hours to tipoff
ORACLE_THRESHOLD_Z_MID=2.5       # 4-24 hours to tipoff
ORACLE_THRESHOLD_Z_LATE=3.0      # < 4 hours to tipoff

# Enable/disable Oracle (for emergency disable)
ORACLE_ENABLED=true

# Logging verbosity
ORACLE_LOG_LEVEL=WARNING
```

---

**Specification Complete**  
**Ready for Claude Code implementation post-April 7, 2026**
