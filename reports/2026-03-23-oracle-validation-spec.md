# K-15 Oracle Validation System — Production Spec

**Status:** IMPLEMENTED (as of 2026-03-23)  
**Location:** `backend/services/oracle_validator.py`, `backend/models.py`, `backend/services/analysis.py`  
**Ticket:** K-15  

---

## 1. Consensus Spread Formula

### 1.1 Rating Differentials

For each rating source, compute the raw margin differential (home - away):

```python
kenpom_margin = kenpom_home - kenpom_away
barttorvik_margin = barttorvik_home - barttorvik_away
```

### 1.2 Consensus Calculation

The **oracle_spread** is the arithmetic mean of all available rating differentials:

```python
margins: list[float] = []
sources: list[str] = []

if kenpom_home is not None and kenpom_away is not None:
    margins.append(kenpom_home - kenpom_away)
    sources.append("kenpom")

if barttorvik_home is not None and barttorvik_away is not None:
    margins.append(barttorvik_home - barttorvik_away)
    sources.append("barttorvik")

oracle_spread = sum(margins) / len(margins)  # simple average
```

**Key characteristics:**
- Uses only KenPom and BartTorvik (EvanMiya excluded from consensus by design)
- Requires at least one complete rating pair to compute
- No renormalization or weighting — pure arithmetic mean
- Sign convention: positive = home team favored

---

## 2. Oracle Comparison Logic

### 2.1 Divergence Calculation

```python
divergence_points = abs(model_spread - oracle_spread)
divergence_z = divergence_points / ORACLE_SD
```

Where:
- `model_spread`: `analysis.projected_margin` from the betting model
- `ORACLE_SD`: Calibrated standard deviation for rating-system disagreement (default: 4.0)

### 2.2 Time-Weighted Threshold

The flagging threshold tightens as game time approaches:

```python
def _select_threshold(hours_to_tipoff: Optional[float]) -> float:
    if hours_to_tipoff is None or hours_to_tipoff >= 24:
        return ORACLE_THRESHOLD_Z_EARLY  # 2.0
    if hours_to_tipoff >= 4:
        return ORACLE_THRESHOLD_Z_MID    # 2.5
    return ORACLE_THRESHOLD_Z_LATE       # 3.0
```

### 2.3 Flag Condition

```python
flagged = divergence_z >= threshold_z
```

---

## 3. Recommended Threshold Values

| Parameter | Default | Env Override | Justification |
|-----------|---------|--------------|---------------|
| `ORACLE_SD` | 4.0 | `ORACLE_SD` | In CBB, raw AdjEM margins from independent systems typically agree within ±3-5 points per team. A 4-point SD covers ~68% of normal spreads (1 SD). |
| `ORACLE_THRESHOLD_Z_EARLY` | 2.0 | `ORACLE_THRESHOLD_Z_EARLY` | ≥24h before tip: Allow 2σ divergence (~95% confidence interval). Early-day uncertainty is expected; thin markets and lineup uncertainty justify wider tolerance. |
| `ORACLE_THRESHOLD_Z_MID` | 2.5 | `ORACLE_THRESHOLD_Z_MID` | 4-24h: Tighten to 2.5σ. Market forming, sharp lines emerging. Model should converge toward consensus. |
| `ORACLE_THRESHOLD_Z_LATE` | 3.0 | `ORACLE_THRESHOLD_Z_LATE` | <4h: Strict 3σ gate. Near tipoff, any remaining divergence likely indicates missing injury/news or data error, not genuine edge. |

**Rationale for time-weighting:**
- Early: Model has information advantage over stale lines
- Late: Sharp books have captured all public information; divergence is a red flag
- Prevents false flags from legitimate early-day information asymmetry

---

## 4. OracleResult Schema

### 4.1 Dataclass Definition

```python
@dataclass
class OracleResult:
    oracle_spread: float       # Consensus spread (avg of rating differentials)
    model_spread: float        # Our model's projected_margin
    divergence_points: float   # |model_spread - oracle_spread| in raw points
    divergence_z: float        # divergence_points / ORACLE_SD
    threshold_z: float         # Z threshold in effect at this hours_to_tipoff
    flagged: bool              # True when divergence_z >= threshold_z
    sources: list[str]         # Rating systems that contributed (e.g., ["kenpom", "barttorvik"])
```

### 4.2 Serialization

```python
def to_dict(self) -> dict:
    return {
        "oracle_spread": round(self.oracle_spread, 3),
        "model_spread": round(self.model_spread, 3),
        "divergence_points": round(self.divergence_points, 3),
        "divergence_z": round(self.divergence_z, 3),
        "threshold_z": self.threshold_z,
        "flagged": self.flagged,
        "sources": self.sources,
    }
```

### 4.3 Field Types

| Field | Type | Description |
|-------|------|-------------|
| `oracle_spread` | `float` | Mean of KenPom and BartTorvik home-away differentials |
| `model_spread` | `float` | The model's final projected_margin after all adjustments |
| `divergence_points` | `float` | Absolute point difference between model and oracle |
| `divergence_z` | `float` | Normalized divergence (σ units) |
| `threshold_z` | `float` | Time-dependent threshold that was applied |
| `flagged` | `bool` | Whether this prediction exceeded the threshold |
| `sources` | `list[str]` | Which rating systems contributed to consensus |

---

## 5. Insertion Point in analysis.py

### 5.1 Location

**File:** `backend/services/analysis.py`  
**Function:** `run_nightly_analysis()`  
**Call Site:** Lines 1444-1477 (within Pass 2 main loop, after model analysis)

### 5.2 Exact Code Block

```python
# ---- K-15: Oracle Validation --------------------------
# Compare model's projected_margin against KenPom +
# BartTorvik consensus.  Flags irreconcilable divergences
# so analysts can inspect before a bet is placed.
_oracle_result = None
_oracle_flagged = False
try:
    from backend.services.oracle_validator import (
        calculate_oracle_divergence,
    )
    _oracle_result = calculate_oracle_divergence(
        model_spread=analysis.projected_margin,
        kenpom_home=ratings_input.get("kenpom", {}).get("home"),
        kenpom_away=ratings_input.get("kenpom", {}).get("away"),
        barttorvik_home=ratings_input.get("barttorvik", {}).get("home"),
        barttorvik_away=ratings_input.get("barttorvik", {}).get("away"),
        hours_to_tipoff=hours_to_tipoff,
    )
    if _oracle_result is not None:
        _oracle_flagged = _oracle_result.flagged
        if _oracle_flagged:
            logger.warning(
                "Oracle flag: %s @ %s — model=%.1f, consensus=%.1f, z=%.2f (threshold=%.1f)",
                away_team, home_team,
                _oracle_result.model_spread,
                _oracle_result.oracle_spread,
                _oracle_result.divergence_z,
                _oracle_result.threshold_z,
            )
except Exception as _oracle_exc:
    logger.warning(
        "Oracle validation failed for %s @ %s: %s",
        away_team, home_team, _oracle_exc,
    )
```

### 5.3 Context Within Analysis Flow

The oracle check is inserted **after** model analysis completes but **before** sharp money post-processing:

1. Pass 1: Pre-score games by raw edge (lines 873-959)
2. Pass 2: Main loop begins (line 1002)
3. Model inputs built (lines 1026-1096)
4. **Model analysis called** → `model.analyze_game()` (line 1419)
5. **ORACLE VALIDATION** ← **INSERTED HERE** (lines 1444-1477)
6. Sharp money post-processing (lines 1479-1512)
7. Reanalysis engine capture (lines 1514-1534)
8. Prediction persistence (lines 1536-1644)

### 5.4 Persistence

```python
# K-15: Oracle Validation fields (lines 1641-1643)
prediction.oracle_flag = _oracle_flagged if _oracle_result is not None else None
prediction.oracle_result = _oracle_result.to_dict() if _oracle_result is not None else None
```

---

## 6. Prediction Table Column Definition

### 6.1 SQLAlchemy Model

**File:** `backend/models.py` (lines 208-210)

```python
# K-15: Oracle Validation — divergence from rating-system consensus
oracle_flag = Column(Boolean)         # True when z-score ≥ time-weighted threshold
oracle_result = Column(JSON)          # OracleResult.to_dict() snapshot
```

### 6.2 Alembic Migration (if needed)

```python
# Revision for K-15 Oracle Validation
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('predictions', sa.Column('oracle_flag', sa.Boolean(), nullable=True))
    op.add_column('predictions', sa.Column('oracle_result', sa.JSON(), nullable=True))
    
    # Index for fast filtering of flagged predictions
    op.create_index(
        'ix_predictions_oracle_flag',
        'predictions',
        ['oracle_flag'],
        postgresql_where=sa.text('oracle_flag IS TRUE')
    )

def downgrade():
    op.drop_index('ix_predictions_oracle_flag', table_name='predictions')
    op.drop_column('predictions', 'oracle_result')
    op.drop_column('predictions', 'oracle_flag')
```

### 6.3 Column Details

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `oracle_flag` | `BOOLEAN` | Yes | True if flagged, False if checked but not flagged, NULL if check failed/not run |
| `oracle_result` | `JSONB` | Yes | Full `OracleResult.to_dict()` snapshot for debugging |

---

## 7. Admin Endpoint Specification

### 7.1 GET /admin/oracle/flagged

**Purpose:** Retrieve all predictions flagged by the Oracle validation system for analyst review.

#### Request

```http
GET /admin/oracle/flagged?date=2026-03-23&min_z=2.0&verdict_type=bet
```

**Query Parameters:**

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `date` | string (ISO date) | No | Today | Filter by prediction date |
| `min_z` | float | No | 2.0 | Minimum divergence_z to include |
| `verdict_type` | string | No | null | Filter by verdict: "bet", "consider", "pass", or "all" |
| `limit` | int | No | 50 | Max results to return |

#### Response Shape

```json
{
  "status": "ok",
  "date": "2026-03-23",
  "count": 3,
  "thresholds": {
    "oracle_sd": 4.0,
    "early": 2.0,
    "mid": 2.5,
    "late": 3.0
  },
  "flagged_predictions": [
    {
      "prediction_id": 12345,
      "game_id": 67890,
      "game_key": "Duke@North Carolina",
      "game_time": "2026-03-23T19:00:00Z",
      "hours_to_tipoff": 2.5,
      "verdict": "Bet 1.00u [T3] Duke (home -4.5) @ -110",
      "oracle": {
        "oracle_spread": -1.2,
        "model_spread": -4.5,
        "divergence_points": 3.3,
        "divergence_z": 0.825,
        "threshold_z": 3.0,
        "flagged": false,
        "sources": ["kenpom", "barttorvik"]
      },
      "model_outputs": {
        "projected_margin": -4.5,
        "edge_conservative": 0.045,
        "recommended_units": 1.0,
        "pricing_engine": "markov"
      },
      "ratings": {
        "kenpom": { "home": 25.3, "away": 22.1 },
        "barttorvik": { "home": 24.8, "away": 21.5 },
        "evanmiya": { "home": null, "away": null }
      },
      "run_tier": "nightly",
      "created_at": "2026-03-23T06:15:32Z"
    }
  ],
  "summary": {
    "total_analyzed": 45,
    "total_flagged": 3,
    "flag_rate": 0.067,
    "by_tier": {
      "early": 1,
      "mid": 1,
      "late": 1
    }
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "ok" or "error" |
| `date` | string | Date queried (ISO 8601) |
| `count` | int | Number of flagged predictions returned |
| `thresholds` | object | Current threshold configuration |
| `flagged_predictions` | array | Detailed prediction records |
| `flagged_predictions[].oracle` | object | Full OracleResult snapshot |
| `flagged_predictions[].model_outputs` | object | Key model outputs for context |
| `summary` | object | Aggregate statistics |

#### Error Responses

```json
// 400 Bad Request
{
  "status": "error",
  "message": "Invalid date format. Expected: YYYY-MM-DD"
}

// 500 Internal Server Error
{
  "status": "error",
  "message": "Database connection failed"
}
```

### 7.2 Implementation Template

```python
@admin_router.get("/oracle/flagged")
async def get_flagged_predictions(
    date: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD)"),
    min_z: float = Query(2.0, ge=0.0, description="Minimum divergence z-score"),
    verdict_type: Optional[str] = Query(None, regex="^(bet|consider|pass|all)$"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
):
    """
    Retrieve Oracle-flagged predictions for analyst review.
    
    Flags indicate model divergences from rating-system consensus
    that exceed time-weighted thresholds.
    """
    from datetime import date as date_type
    from sqlalchemy import func
    
    # Parse date
    if date:
        try:
            filter_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "Invalid date format. Expected: YYYY-MM-DD")
    else:
        filter_date = datetime.utcnow().date()
    
    # Build query
    query = db.query(Prediction).filter(
        Prediction.prediction_date == filter_date,
        Prediction.oracle_flag == True,
    )
    
    # Apply verdict filter
    if verdict_type and verdict_type != "all":
        if verdict_type == "bet":
            query = query.filter(Prediction.verdict.ilike("Bet%"))
        elif verdict_type == "consider":
            query = query.filter(Prediction.verdict.ilike("CONSIDER%"))
        elif verdict_type == "pass":
            query = query.filter(Prediction.verdict.ilike("PASS%"))
    
    # Apply min_z filter (JSON extraction is DB-specific; PostgreSQL example)
    query = query.filter(
        func.coalesce(
            Prediction.oracle_result["divergence_z"].astext.cast(Float),
            0.0
        ) >= min_z
    )
    
    # Get total counts for summary
    total_analyzed = db.query(Prediction).filter(
        Prediction.prediction_date == filter_date,
        Prediction.oracle_result.isnot(None)
    ).count()
    
    total_flagged = query.count()
    
    # Fetch results with game data joined
    predictions = query.join(Game).order_by(
        Prediction.oracle_result["divergence_z"].desc()
    ).limit(limit).all()
    
    # Build response
    flagged_items = []
    for p in predictions:
        oracle_data = p.oracle_result or {}
        game = p.game
        
        flagged_items.append({
            "prediction_id": p.id,
            "game_id": game.id,
            "game_key": f"{game.away_team}@{game.home_team}",
            "game_time": game.game_date.isoformat() if game.game_date else None,
            "hours_to_tipoff": oracle_data.get("hours_to_tipoff"),
            "verdict": p.verdict,
            "oracle": oracle_data,
            "model_outputs": {
                "projected_margin": p.projected_margin,
                "edge_conservative": p.edge_conservative,
                "recommended_units": p.recommended_units,
                "pricing_engine": p.full_analysis.get("calculations", {}).get("pricing_engine") if p.full_analysis else None,
            },
            "ratings": {
                "kenpom": {"home": p.kenpom_home, "away": p.kenpom_away},
                "barttorvik": {"home": p.barttorvik_home, "away": p.barttorvik_away},
                "evanmiya": {"home": p.evanmiya_home, "away": p.evanmiya_away},
            },
            "run_tier": p.run_tier,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        })
    
    return {
        "status": "ok",
        "date": filter_date.isoformat(),
        "count": len(flagged_items),
        "thresholds": {
            "oracle_sd": float(os.getenv("ORACLE_SD", "4.0")),
            "early": float(os.getenv("ORACLE_THRESHOLD_Z_EARLY", "2.0")),
            "mid": float(os.getenv("ORACLE_THRESHOLD_Z_MID", "2.5")),
            "late": float(os.getenv("ORACLE_THRESHOLD_Z_LATE", "3.0")),
        },
        "flagged_predictions": flagged_items,
        "summary": {
            "total_analyzed": total_analyzed,
            "total_flagged": total_flagged,
            "flag_rate": total_flagged / max(total_analyzed, 1),
        },
    }
```

---

## 8. Environment Configuration Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_SD` | 4.0 | Calibrated SD for rating-system disagreement |
| `ORACLE_THRESHOLD_Z_EARLY` | 2.0 | Z threshold ≥24h before tipoff |
| `ORACLE_THRESHOLD_Z_MID` | 2.5 | Z threshold 4-24h before tipoff |
| `ORACLE_THRESHOLD_Z_LATE` | 3.0 | Z threshold <4h before tipoff |

---

## 9. Operational Notes

### 9.1 When Flags Fire

Oracle flags indicate the model's projected margin diverges significantly from the rating-system consensus. This may indicate:

- **Missing injury/suspension** in our data (most common)
- **Team mapping error** (wrong team being analyzed)
- **Stale rating data** (one source has updated, other hasn't)
- **Legitimate model insight** (matchup engine found something ratings miss)

### 9.2 Recommended Response

| Divergence | Action |
|------------|--------|
| z < 2.0 | No action required |
| 2.0 ≤ z < 2.5 | Logged for review; likely acceptable early in day |
| 2.5 ≤ z < 3.0 | Manual inspection recommended; verify injury data |
| z ≥ 3.0 | **Halt bet placement** until root cause identified |

### 9.3 Integration with Existing Circuit Breakers

The Oracle validation operates **independently** of existing safety mechanisms:

1. **Z-score divergence guard** (betting_model.py lines 1081-1126): PASS if model vs market diverges >2.5σ
2. **Edge circuit breaker** (betting_model.py lines 1651-1776): PASS if edge exceeds dynamic threshold
3. **Oracle validation** (analysis.py lines 1444-1477): FLAG for analyst review (does not auto-PASS)

The Oracle system is purely **observational** — it never blocks a bet automatically, only surfaces warnings for human review.

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-23  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**Reviewed By:** Claude Code (Master Architect)
