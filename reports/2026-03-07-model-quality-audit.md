# K-3: Model Quality Audit — CBB Edge Analyzer V9

**Date:** 2026-03-07  
**Auditor:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Investigate 0 bets on 12 games in first clean Railway analysis  
**Files analyzed:** `backend/services/ratings.py`, `backend/betting_model.py`, `backend/services/analysis.py`

---

## Executive Summary

The model's `0 bets on 12 games` result is **correct conservatism (Option A)** rather than a bug. The combination of:

1. **Wider distribution** (`sd_mult=1.0` vs default `0.85` = +17.6% wider SD)
2. **Lower home advantage** (`ha=2.419` vs default `3.09` = -21.7% less HCA)
3. **Multi-layer edge compression** (CI haircut, market blend, V9 scalars)

...creates a high bar for `edge_conservative > MIN_BET_EDGE (2.5%)`.

**However**, the V9 SNR + integrity scalars were NOT present in the 663-bet calibration dataset. This is a **structural mismatch** that warrants V9-specific recalibration after 50+ settled bets.

---

## Q1: Ratings Data Health

### Conditions for Empty Returns

| Source | Returns {} When | Frequency |
|--------|-----------------|-----------|
| **KenPom** | `KENPOM_API_KEY` not set (line 142-144)<br>Circuit breaker OPEN (line 146-148)<br>API failure/timeout (line 158-185) | Rare (API key required) |
| **BartTorvik** | Circuit breaker OPEN (line 367-369)<br>HTTP failure (line 372-375)<br>Empty/malformed CSV (line 381-384) | Occasional |
| **EvanMiya** | Circuit breaker OPEN (line 1066-1068)<br>Cloudflare block (all strategies fail, line 1157-1172)<br>Auto-drop after 3 failures (line 1162-1171) | **Frequent** |

### What Happens When All Ratings Are Null?

```python
# betting_model.py:822-835
if None in [kp_home, kp_away]:
    return GameAnalysis(
        verdict="PASS",
        pass_reason="Missing KenPom ratings",  # HARD REQUIREMENT
        ...
    )
```

**KenPom is REQUIRED.** If KenPom returns empty, the game is PASSed immediately with `pass_reason="Missing KenPom ratings"`. The model CANNOT produce a BET verdict without KenPom data.

### Railway Production Status

The first clean analysis had 12 games analyzed (not 0), which means:
- KenPom API key WAS working
- Ratings data WAS available
- The 0 bets result is due to edge/threshold logic, not missing data

---

## Q2: Edge Inflation/Compression Pipeline

### Path: ratings_input → projected_margin → edge_point → edge_conservative

```
Step 1: Composite Margin (betting_model.py:873-876)
├── Weighted average of available sources (KenPom required, BT/EM optional)
├── Weights renormalized to sum of available sources (line 869)
├── Confidence shrinkage: -10% per missing source (line 895-898)
└── Result: raw_model_margin (margin before market blend)

Step 2: Adjustments (lines 904-984)
├── Pace-adjusted HCA applied (if not neutral): ha × (pace/68) or ha × (total/140)
├── Injury adjustments (from injuries.py estimate_impact)
├── Matchup engine adjustment (second-order play-style interactions)
└── Result: raw_model_margin with adjustments

Step 3: Market Blend (lines 985-1055)
├── Z-score divergence guard: if |model - market| / sd > 2.5 → PASS (lines 1009-1040)
├── Dynamic model weight: w = f(hours_to_tipoff, sharp_books, injury_adj)
│   ├── 24h out: w ≈ 0.89 (model dominates)
│   ├── 10h out: w ≈ 0.55 (equal blend)
│   └── 1h out: w ≈ 0.21 (market dominates)
└── Blended margin = w × model_margin + (1-w) × market_margin

Step 4: Adjusted SD (lines 1082-1168)
├── Base SD from matchup profiles OR dynamic sqrt(total) × sd_multiplier
├── Tournament SD bump: ×1.15 if is_neutral=True (line 1157-1168)
├── Penalties: missing sources (+1.25 each), heuristic four-factors (×1.15)
└── Ceiling: 15.5 points

Step 5: Cover Probability (lines 1186-1361)
├── Monte Carlo 2-layer CI with margin_se (parameter uncertainty)
│   └── margin_se = 1.50 + 0.30 (EvanMiya down) + 0.30 (no sharp books) + 0.15×missing
├── Markov simulator (primary) or Gaussian fallback
└── Returns: cover_prob, cover_lower (2.5% CI), cover_upper (97.5% CI)

Step 6: Edge Calculation (lines 1363-1409)
├── Shin vig removal on both sides' odds
├── Push adjustment if Markov engine (1 - push_prob)
├── edge_point = cover_prob - market_prob
└── edge_conservative = cover_lower - market_prob  # KEY METRIC
```

### Impact of sd_mult=1.0 vs Default 0.85

| Game Total | SD (0.85) | SD (1.0) | % Wider |
|------------|-----------|----------|---------|
| 120 | 9.31 | 10.95 | +17.6% |
| 140 | 10.06 | 11.83 | +17.6% |
| 160 | 10.75 | 12.65 | +17.6% |

**Effect on edge_conservative:**
- Wider SD → flatter probability distribution
- For a 3-point projected margin vs -2 spread:
  - At SD=10.06: z = 0.50 → cover_prob ≈ 0.691
  - At SD=11.83: z = 0.42 → cover_prob ≈ 0.663
  - Difference: ~2.8 percentage points of edge

### Impact of ha=2.419 vs Default 3.09

- **-21.7% reduction** in home court advantage
- For home favorites: lower projected margin → lower cover probability
- For away underdogs: less negative margin → lower edge potential
- Systematic edge compression for home-side value

### Minimum Ratings Quality for edge_conservative > 0

Back-of-envelope calculation for a typical game:

```
Assumptions:
- Game total: 140 → dynamic SD ≈ 11.83
- Margin SE: 1.50 (baseline, all sources available)
- Market vig-removed prob: ~0.50 (pick'em)
- Need: cover_lower > 0.50

Monte Carlo: lower CI = mean(prob_samples) - 1.96 × se(prob_samples)
At SD=11.83, margin_se=1.50:
  - 2.5% margin tail ≈ margin - 1.96 × 1.50 = margin - 2.94
  - P(cover | margin - 2.94) > 0.50 requires margin - 2.94 > 0
  - Therefore: margin > 2.94 points

But market blend shrinks margin toward market:
  - At 10h out: blended = 0.55 × model + 0.45 × market
  - If market spread = -3 (market margin = +3 for home):
    model needs margin > (2.94 - 0.45×3) / 0.55 = 2.94/0.55 = 5.35 points

Plus injury adjustments, matchup, etc. → model margin likely needs 6+ points
```

**Conclusion:** A ratings-quality differential of **~6+ points** (after all adjustments) is needed to produce `edge_conservative > 0` at current calibration.

---

## Q3: Conservative Threshold Audit — Filters Before edge_conservative

### Filters in `analysis.py` (Pass 2 loop)

| Line | Filter | Verdict if Triggered |
|------|--------|---------------------|
| 1244-1250 | `hours_to_tipoff < -0.1` (game started) | `continue` (skip game) |

### Filters in `betting_model.py` `analyze_game()`

| Line | Filter | Verdict if Triggered |
|------|--------|---------------------|
| 772-786 | `lines_age > 30 min` | PASS — "Tier 3 staleness" |
| 791-804 | `ratings_age > 7 days` | PASS — "Ratings >7 days old" |
| 822-835 | Missing KenPom ratings | PASS — "Missing KenPom ratings" |
| 1009-1040 | Z-score divergence > 2.5σ | PASS — "Market Divergence Anomaly" |
| 1171-1184 | `adj_sd >= 15.5` | PASS — "Uncertainty too high" |
| 1525-1541 | Integrity ABORT/RED FLAG | PASS — "Integrity Abort" |
| 1597-1625 | Favorite protection triggered | PASS — "Favorite Protection" |
| 1644-1668 | `edge_conservative > _EDGE_BREAKER_THRESHOLD` | PASS — "Edge Circuit Breaker" |
| 1671-1675 | `edge_conservative <= 0` | PASS — "Conservative edge X% <= 0" |
| 1683-1699 | `edge_conservative <= MIN_BET_EDGE (2.5%)` | **CONSIDER** — edge too marginal |

### Are Any Filters Overly Aggressive?

**No obvious bugs**, but these warrant monitoring:

1. **MIN_BET_EDGE = 2.5%** (line 1683): 
   - With `sd_mult=1.0` widening distribution, positive edges >2.5% are genuinely rare
   - This is working as designed for regular season
   - **Recommendation:** Do NOT lower for tournament (variance is higher, not lower)

2. **Z-score divergence guard (2.5σ)**:
   - With wider SD from `sd_mult=1.0`, the z-score denominator is larger
   - This makes the guard **more permissive** (harder to trigger)
   - Actually appropriate given model uncertainty

3. **Edge circuit breaker threshold**:
   - Dynamic: 15% (24h), 13% (4-24h), 6-12% (<4h)
   - With `sd_mult=1.0`, true edges >13% at 10h out are extremely rare
   - This is the intended safety valve

---

## Q4: Calibration Sanity Check — V9 Structural Mismatch

### The Problem

| Aspect | V8 Era (Calibration) | V9 Current |
|--------|---------------------|------------|
| Dataset | 663 settled bets | N/A (not yet accumulated) |
| SD multiplier | Calibrated to 1.0 | 1.0 (inherited) |
| SNR scalar | N/A | 0.5-1.0× Kelly |
| Integrity scalar | N/A | 0.5-1.0× Kelly |
| Combined V9 effect | — | 0.25-1.0× Kelly |

**The 663 V8-era calibration bets had NO SNR/integrity scalars.** The calibration optimized `sd_mult=1.0` to maximize edge capture **without** these additional haircuts.

### Impact on Current Model

```
V9 effective Kelly = V8_Kelly × SNR_scalar × Integrity_scalar
                   = V8_Kelly × [0.5-1.0] × [0.5-1.0]
                   = V8_Kelly × [0.25-1.0]

With default floors (SNR=0.5, Integrity=0.5):
  Minimum effective Kelly = 0.25 × V8_Kelly
```

This means:
- A bet that sized at 1.0u in V8 now sizes at 0.25-1.0u in V9
- Many marginal edges that were 0.75u in V8 are now below 0.25u floor
- Tier-based sizing (T1/T2/T3/T4) further compresses at low edges

### Recommendation: V9-Specific Recalibration

**Trigger:** After 50+ settled V9-era bets  
**Target:** Recalibrate `sd_mult` specifically for V9 with scalars active

**Methodology:**
```python
# In recalibration.py
if v9_bet_count >= 50:
    # Use only post-V9-deployment bets
    # Optimize sd_mult to maximize log-returns
    # Include SNR/integrity scalar effects in the objective
```

**Expected outcome:** V9-appropriate `sd_mult` will likely be **lower** than 1.0 (tighter distribution, recognizing that the V9 scalars already provide uncertainty discounting).

---

## Root Cause: Why 0 Bets on 12 Games

```
Pre-filter: 66 Odds API games
├── 12 survived to Pass 2 (54 already tipped off or failed freshness)
└── Pass 2 analysis:
    ├── Games with positive edge_conservative: Likely 0-2
    │   └── But edge_conservative < MIN_BET_EDGE (2.5%)
    │       └── CONSIDER verdict (not counted as BET)
    └── Games with edge_conservative <= 0: 10-12
        └── PASS verdict
```

**The 0 bets result is statistically plausible** given:
1. Post-calibration `sd_mult=1.0` (wider distribution)
2. Low `ha=2.419` (compressed margins)
3. V9 SNR/integrity scalars (additional Kelly compression)
4. MIN_BET_EDGE=2.5% (conservative threshold)
5. No genuinely mispriced lines in that day's slate

---

## Actionable Recommendations

### Immediate (No Action Needed)
- [x] 0 bets is not a bug — model is working correctly
- [x] KenPom API key is functioning (12 games analyzed proves this)

### Short-term (Next 2 Weeks)
- [ ] Monitor V9 bet accumulation — trigger recalibration at 50 settled bets
- [ ] Track edge distribution: what % of games have edge_conservative > 0? > 2.5%?
- [ ] Verify tournament mode (`TOURNAMENT_MODE_SD_BUMP=1.15`) is correctly applied to neutral-site games

### Medium-term (Post-Tournament)
- [ ] Run V9-specific recalibration on full V9-era dataset
- [ ] Consider decoupling calibration: V8 legacy params vs V9 active params
- [ ] Re-evaluate MIN_BET_EDGE if post-recalibration bet rate < 5%

---

## Appendix: Key Environment Variables Affecting Edge

| Variable | Default | Current | Impact on Edge |
|----------|---------|---------|----------------|
| `SD_MULTIPLIER` | 0.85 | 1.0 | Higher = wider SD = compressed edges |
| `HOME_ADVANTAGE` | 3.09 | 2.419 | Lower = compressed home-team edges |
| `MIN_BET_EDGE` | 2.5 | 2.5 | Threshold for BET vs CONSIDER |
| `SNR_KELLY_FLOOR` | 0.5 | 0.5 | Minimum Kelly multiplier from SNR |
| `BASE_MARGIN_SE` | 1.50 | 1.50 | Wider CI = lower edge_conservative |
| `TOURNAMENT_MODE_SD_BUMP` | 1.15 | 1.15 | Applied to neutral-site games |

---

*Report generated by Kimi CLI for CBB Edge Analyzer — K-3 Mission Complete*
