# Fatigue Model (V9.1)

**Last Updated:** March 10, 2026  
**Model Version:** v9.1  
**File:** `backend/services/fatigue.py`

---

## Overview

The fatigue model calculates travel and schedule-based adjustments to game predictions. It quantifies the physical toll of travel, rest disparities, and cumulative game load on team performance.

---

## Factors

### 1. Rest Days Penalty

Days since last game impact on performance:

| Rest Days | Penalty | Rationale |
|-----------|---------|-----------|
| 0 (B2B) | 1.8 pts | Severe fatigue, no recovery time |
| 1 | 0.7 pts | Minimal recovery, travel often involved |
| 2 | 0.2 pts | Near-optimal recovery |
| 3+ | 0.0 pts | Full recovery |

**Formula:**
```python
def calculate_rest_penalty(days_since_last_game: int) -> float:
    penalties = {0: 1.8, 1: 0.7, 2: 0.2}
    return penalties.get(days_since_last_game, 0.0)
```

---

### 2. Travel Distance

Miles traveled from previous game venue:

| Distance | Penalty | Rationale |
|----------|---------|-----------|
| 0-500 | 0.0 | Minimal travel impact |
| 500-1000 | 0.3 | Moderate travel |
| 1000-1500 | 0.6 | Significant travel |
| 1500+ | 1.0 | Cross-country, jet lag risk |

**Formula:**
```python
def calculate_travel_penalty(miles: float) -> float:
    if miles <= 500: return 0.0
    if miles <= 1000: return 0.3
    if miles <= 1500: return 0.6
    return 1.0
```

---

### 3. Time Zone Changes

Time zones crossed between venues:

| Zones Crossed | Base Penalty | Eastward Multiplier |
|---------------|--------------|---------------------|
| 0 | 0.0 | - |
| 1 | 0.25 | 1.3x |
| 2 | 0.50 | 1.3x |
| 3 | 0.75 | 1.3x |

**Eastward Penalty:** Flying east (losing hours) is 1.3x harder than westward.

**Examples:**
- LAX → DEN (1 zone west): 0.25 pts
- BOS → LAX (3 zones west): 0.75 pts
- LAX → BOS (3 zones east): 0.75 × 1.3 = 0.975 pts

---

### 4. Altitude Effects

Venues above 3,000 feet impact teams unaccustomed to altitude:

| Venue | Elevation | Penalty for Sea-Level Teams |
|-------|-----------|----------------------------|
| New Mexico | 5,312 ft | 2.0 pts |
| Air Force | 7,258 ft | 3.0 pts |
| Wyoming | 7,220 ft | 3.0 pts |
| Utah State | 4,700 ft | 1.5 pts |
| Colorado | 5,430 ft | 2.0 pts |
| BYU | 4,550 ft | 1.5 pts |
| Nevada | 4,500 ft | 1.5 pts |
| Boise State | 2,700 ft | 0.5 pts |
| Denver | 5,280 ft | 2.0 pts |

**Rule:** Altitude penalty only applies if:
- Team plays at altitude venue AND
- Team is based at sea level (< 1,000 ft home venue)

Teams accustomed to altitude (e.g., New Mexico playing at Colorado) receive no penalty.

---

### 5. Cumulative Load

Games played in recent window:

| Games in 7 Days | Penalty | Games in 14 Days | Penalty |
|-----------------|---------|------------------|---------|
| 0-2 | 0.0 | 0-4 | 0.0 |
| 3 | 0.5 | 5-6 | 0.3 |
| 4 | 1.0 | 7-8 | 0.6 |
| 5+ | 1.5 | 9+ | 1.0 |

**Combined:** Take the higher of the two penalties (not additive).

---

## Integration

### In Betting Model

```python
from backend.services.fatigue import FatigueCalculator

fatigue_calc = FatigueCalculator()

# Inside analyze_game()
fatigue_result = fatigue_calc.calculate(
    team_id=home_team_id,
    game_date=game_date,
    venue=venue,
    previous_venue=last_game_venue,
    schedule=recent_games
)

# Apply adjustment
adjusted_margin = base_margin + fatigue_result.margin_adjustment

# Store metadata for transparency
prediction.fatigue_metadata = fatigue_result.to_dict()
```

### API Response

```json
{
  "prediction": {
    "margin": -2.5,
    "fatigue_adjustment": -1.2,
    "fatigue_breakdown": {
      "rest_days": 0,
      "rest_penalty": 1.8,
      "travel_miles": 1847,
      "travel_penalty": 1.0,
      "time_zones": 2,
      "timezone_penalty": 0.65,
      "altitude_ft": 5256,
      "altitude_penalty": 2.0,
      "games_7_days": 3,
      "games_14_days": 6,
      "load_penalty": 0.5
    }
  }
}
```

---

## Tuning Guidelines

### Adjusting Penalties

Penalty values are in the `PENALTY_CONFIG` dict:

```python
PENALTY_CONFIG = {
    "rest": {0: 1.8, 1: 0.7, 2: 0.2},
    "travel": [(500, 0.0), (1000, 0.3), (1500, 0.6), (float('inf'), 1.0)],
    "timezone": 0.25,
    "timezone_east_multiplier": 1.3,
    "altitude": {
        (3000, 4000): 0.5,
        (4000, 5000): 1.0,
        (5000, 6000): 2.0,
        (6000, float('inf')): 3.0
    }
}
```

### Validation

To validate fatigue model impact:
```bash
python -m pytest tests/test_fatigue.py -v
```

---

## Future Enhancements

- [ ] Player-specific fatigue (minutes load)
- [ ] Historical team fatigue sensitivity
- [ ] Conference tournament travel clustering
- [ ] NBA back-to-back data correlation

---

## References

- Research: "The Effects of Travel on Athletic Performance" (Stanford, 2019)
- NBA B2B studies: 14% win rate decrease for traveling B2B teams
- Altitude physiology: 3-5% VO2 max reduction per 1,000 ft above 5,000 ft
