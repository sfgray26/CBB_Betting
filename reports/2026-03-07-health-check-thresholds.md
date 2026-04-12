# K-7: A-30 Nightly Health Check Threshold Design

**Date:** 2026-03-07  
**Designer:** Kimi CLI (Deep Intelligence Unit)  
**Mission:** Design thresholds for `_nightly_health_check_job()` in OpenClaw  
**Reference:** HEARTBEAT.md, `backend/services/performance.py`, K-3 Model Quality Audit

---

## Executive Summary

This document specifies production-ready thresholds for the Nightly Health Check loop (O-7). These thresholds balance **early detection** of model degradation against **false positive fatigue**. The design considers:

1. **V9 model characteristics** (sd_mult=1.0, SNR/integrity scalars active)
2. **Tournament timing** (high-variance March environment)
3. **Operational realities** (nightly runs, operator attention constraints)

---

## 1. Available Metrics Inventory

From `performance.py`, the following metrics are available for health checks:

### 1.1 Model Accuracy Metrics (`calculate_model_accuracy`)

| Metric | Description | Data Quality |
|--------|-------------|--------------|
| `mean_mae` | Mean absolute error of margin predictions | ✅ Reliable with 10+ games |
| `median_mae` | Median absolute error (robust to outliers) | ✅ More stable than mean |
| `mae_by_verdict` | MAE split by BET vs PASS verdicts | ✅ Critical for drift detection |
| `mae_by_source` | Per-rating-source MAE (KP, BT, EM) | ⚠️ Requires 7+ snapshots |
| `mean_signed_error` | Bias detection (over/under-predict home) | ✅ Useful for HCA drift |
| `brier_score` | Probability calibration quality | ✅ Good secondary indicator |

### 1.2 Financial Metrics (`calculate_summary_stats`)

| Metric | Description | Alert Suitability |
|--------|-------------|-------------------|
| `current_drawdown` | Peak-to-trough portfolio decline | ✅ PRIMARY circuit breaker input |
| `roi` | Return on invested capital | ⚠️ High variance, weekly better |
| `win_rate` | Proportion of winning bets | ⚠️ Needs 30+ bets for stability |
| `mean_clv` | Average closing line value | ✅ Good model health proxy |

### 1.3 Financial Metrics (`calculate_financial_metrics`)

| Metric | Description | Use Case |
|--------|-------------|----------|
| `sharpe_ratio` | Risk-adjusted return | Monthly review, not nightly |
| `sortino_ratio` | Downside-adjusted return | Monthly review |
| `max_drawdown_pct` | Maximum historical drawdown | ✅ Compare to current drawdown |
| `calmar_ratio` | Return / max drawdown | Trend analysis |

### 1.4 Daily Snapshot Metrics (`generate_daily_snapshot`)

| Metric | Description | Threshold Applicability |
|--------|-------------|-------------------------|
| `pass_rate` | % of games passed (not bet) | ⚠️ Context-dependent |
| `bets_recommended` | Count of BET-tier games | ✅ Min threshold for meaningful stats |
| `calibration_error` | Mean absolute probability calibration error | ✅ < 0.07 = well calibrated |
| `mean_edge` | Average edge of recommended bets | ⚠️ Tournament varies widely |

---

## 2. Threshold Recommendations

### 2.1 MAE (Mean Absolute Error) Thresholds

**Context from K-3 Audit:**
- V9 model uses `sd_mult=1.0` (vs 0.85 default) = 17.6% wider distributions
- Current HA = 2.419 (vs 3.09 default) = 21.7% lower home court advantage
- These compress edges, making accurate margin prediction MORE critical

**Recommended MAE Thresholds:**

| Level | Threshold | Rationale | Action |
|-------|-----------|-----------|--------|
| 🟢 **HEALTHY** | MAE ≤ 9.5 pts | Normal V9 operation | Log INFO, no action |
| 🟡 **WARNING** | 9.5 < MAE ≤ 12.0 | Slight degradation | Log WARNING, note in HANDOFF.md |
| 🟠 **ELEVATED** | 12.0 < MAE ≤ 15.0 | Significant drift | PRIORITY flag, queue recal review |
| 🔴 **CRITICAL** | MAE > 15.0 | Model broken | HALT new bets, escalate to Claude |

**Rationale for 9.5 pt baseline:**
- College basketball has high natural variance (typical game SD ~11-13 pts)
- A model with MAE ~9.5 pts captures ~60-70% of predictable variance
- HEARTBEAT.md originally proposed 3.0 pts — this was unrealistically strict
- 3.0 pts would fire on every single night; 9.5 is achievable but demanding

**Tournament Adjustment:**
- Neutral-site games have ~1.15× higher variance
- MAE threshold relaxes to **≤ 11.0 pts** during tournament (March 18+)

### 2.2 Drawdown Thresholds

From `IDENTITY.md`, the Portfolio Drawdown Breaker is at 15%. The Nightly Health Check should provide **early warning** before this triggers.

**Recommended Drawdown Zones:**

| Zone | Range | Rationale | Action |
|------|-------|-----------|--------|
| 🟢 **GREEN** | 0–8% | Normal variance | Daily summary only |
| 🟡 **YELLOW** | 8–12% | Concerning but within tolerance | Log WARNING, reduce max Kelly by 20% |
| 🟠 **ORANGE** | 12–15% | Approaching circuit breaker | PRIORITY flag, reduce max Kelly by 50% |
| 🔴 **RED** | > 15% | **CIRCUIT BREAKER ACTIVE** | Halt all new bets per `IDENTITY.md` |

**Key distinction:** The 15% hard breaker is enforced by `betting_model.py` (code-level). The Nightly Health Check provides **escalating warnings** as we approach it.

### 2.3 Minimum Predictions for Meaningful Check

**Problem:** On nights with 0–2 BET-tier games, variance dominates signal.

**Recommendation:**

```python
MIN_BETS_FOR_HEALTH_CHECK = 5  # Minimum settled bets in last 7 days
MIN_PREDICTIONS_FOR_MAE = 10   # Minimum predictions with actual_margin
```

**Behavior when below minimum:**
- Skip MAE-based alerts
- Still check drawdown (always valid if portfolio exists)
- Log: "Insufficient sample size for statistical health checks (N=X)"

### 2.4 Secondary Thresholds (Log but don't alert)

These metrics are tracked but don't trigger alerts unless extreme:

| Metric | Log if | Escalate if |
|--------|--------|-------------|
| `win_rate` | < 45% or > 55% (7-day) | < 40% for 14 days |
| `mean_clv` | < 0.5% | < 0% for 7 days |
| `pass_rate` | > 98% (too conservative) or < 70% (too aggressive) | > 99% or < 60% for 5 days |
| `calibration_error` | > 0.05 | > 0.10 for 7 days |

---

## 3. Implementation Specification

### 3.1 Health Check Function Structure

```python
# backend/services/analysis.py — _nightly_health_check_job()

async def _nightly_health_check_job():
    """
    Nightly health check per HEARTBEAT.md spec.
    Runs at 4:30 AM ET daily (30 min after snapshot job).
    """
    db = SessionLocal()
    try:
        alerts = []
        
        # --- 1. Model Accuracy Check ---
        accuracy = calculate_model_accuracy(db, days=7)
        if accuracy.get("count", 0) >= MIN_PREDICTIONS_FOR_MAE:
            mae = accuracy.get("mean_mae")
            if mae:
                if mae > 15.0:
                    alerts.append(("CRITICAL", f"MAE {mae:.1f} > 15.0", "HALT"))
                elif mae > 12.0:
                    alerts.append(("ELEVATED", f"MAE {mae:.1f} > 12.0", "REVIEW"))
                elif mae > 9.5:
                    alerts.append(("WARNING", f"MAE {mae:.1f} > 9.5", "MONITOR"))
        
        # --- 2. Drawdown Check ---
        stats = calculate_summary_stats(db)
        drawdown = stats.get("overall", {}).get("current_drawdown", 0)
        
        if drawdown > 0.15:
            alerts.append(("CRITICAL", f"Drawdown {drawdown:.1%} > 15%", "HALT"))
        elif drawdown > 0.12:
            alerts.append(("ELEVATED", f"Drawdown {drawdown:.1%} > 12%", "REDUCE_50"))
        elif drawdown > 0.08:
            alerts.append(("WARNING", f"Drawdown {drawdown:.1%} > 8%", "REDUCE_20"))
        
        # --- 3. Write to HANDOFF.md if any alerts ---
        if alerts:
            _update_handoff_with_alerts(alerts)
        
        # --- 4. Daily summary to memory ---
        _write_daily_summary(db, alerts)
        
    finally:
        db.close()
```

### 3.2 Alert Severity Mapping

| Severity | Color | HANDOFF.md Action | Discord Notification |
|----------|-------|-------------------|---------------------|
| `CRITICAL` | 🔴 | Add "HEALTH CRITICAL" section | @operator mention |
| `ELEVATED` | 🟠 | Add to "Architect Review Queue" | Channel alert |
| `WARNING` | 🟡 | Log in daily summary only | Silent (logged) |

### 3.3 Consecutive Day Escalation

Per HEARTBEAT.md: "MAE > 3 pts for 7 consecutive days → queue recalibration"

With revised thresholds:

```python
# Track consecutive days at WARNING or above
CONSECUTIVE_MAE_WARNING_THRESHOLD = 5  # days at WARNING or ELEVATED
CONSECUTIVE_MAE_CRITICAL_THRESHOLD = 3  # days at CRITICAL

# Behavior:
# - 5 days at WARNING → escalate to ELEVATED treatment
# - 3 days at CRITICAL → force recalibration regardless of bet count
# - 7 days total with any MAE alerts → queue recalibration review
```

---

## 4. Tournament-Specific Adjustments

**Period:** March 18 – April 7 (First Four through Championship)

| Metric | Regular Season | Tournament |
|--------|---------------|------------|
| MAE WARNING | > 9.5 pts | > 11.0 pts |
| MAE ELEVATED | > 12.0 pts | > 14.0 pts |
| MAE CRITICAL | > 15.0 pts | > 17.0 pts |
| Min bets check | 5 | 3 (smaller sample sizes) |
| Drawdown YELLOW | 8% | 10% (higher baseline variance) |

**Rationale:**
- Neutral sites increase variance (no HCA modeling)
- Single-elimination creates fatter tails
- Sample sizes smaller (fewer games per day in later rounds)

---

## 5. Threshold Validation

### 5.1 Simulation Against Historical Data

If we had 90 days of backtest data, we would verify:

| Check | Expected Frequency | Tolerance |
|-------|-------------------|-----------|
| WARNING fires | ~15–20% of nights | ±5% |
| ELEVATED fires | ~2–5% of nights | ±2% |
| CRITICAL fires | < 1% of nights | ±0.5% |
| Consecutive 5-day WARNING | ~1–2× per season | Acceptable |

### 5.2 Tuning Parameters (Env Vars)

These thresholds should be env-configurable without code changes:

```bash
# MAE thresholds
MAE_WARNING_THRESHOLD=9.5
MAE_ELEVATED_THRESHOLD=12.0
MAE_CRITICAL_THRESHOLD=15.0

# Drawdown thresholds  
DRAWDOWN_YELLOW_PCT=8.0
DRAWDOWN_ORANGE_PCT=12.0
DRAWDOWN_RED_PCT=15.0

# Minimum samples
MIN_BETS_FOR_HEALTH_CHECK=5
MIN_PREDICTIONS_FOR_MAE=10

# Consecutive day triggers
CONSECUTIVE_WARNING_DAYS=5
CONSECUTIVE_CRITICAL_DAYS=3
```

---

## 6. Integration with Existing Systems

### 6.1 HEARTBEAT.md Status Tracker

Update Nightly Health Check row:

```markdown
| Nightly Health Check | READY | — | Thresholds defined (K-7). A-30 implementation pending. |
```

### 6.2 HANDOFF.md Integration

Alerts write to a new "Health Status" section:

```markdown
## Health Status (Auto-Updated by OpenClaw)

| Date | Check | Result | Status |
|------|-------|--------|--------|
| 2026-03-07 | MAE (7d) | 8.2 pts | 🟢 HEALTHY |
| 2026-03-07 | Drawdown | 3.5% | 🟢 GREEN |
| 2026-03-07 | Min bets | 0 | ⚪ INSUFFICIENT DATA |
```

### 6.3 Circuit Breaker Coordination

The Nightly Health Check **does not** directly trigger the portfolio drawdown breaker (that's in `betting_model.py`). Instead:

1. Health check logs WARNING at 8% drawdown
2. Health check logs ELEVATED at 12% drawdown  
3. `betting_model.py` enforces HARD HALT at 15% drawdown
4. If health check detects > 15% but model is still betting → CRITICAL alert

---

## 7. Summary of Recommendations

| Threshold | Current HEARTBEAT.md | K-7 Recommended | Rationale |
|-----------|---------------------|-----------------|-----------|
| MAE WARNING | 3.0 pts | **9.5 pts** | 3.0 was unrealistic; 9.5 is demanding but achievable |
| MAE ELEVATED | — | **12.0 pts** | Significant drift requiring review |
| MAE CRITICAL | — | **15.0 pts** | Model broken, halt betting |
| Drawdown YELLOW | 10% | **8%** | Earlier warning before 15% breaker |
| Drawdown ORANGE | — | **12%** | Reduce sizing by 50% |
| Drawdown RED | 15% | **15%** | Circuit breaker (unchanged per IDENTITY.md) |
| Min bets for check | — | **5 bets / 10 predictions** | Variance dominates below this |
| Consecutive days | 7 | **5 (WARNING), 3 (CRITICAL)** | Faster response to sustained drift |

---

## 8. Next Steps

1. **Claude Code (A-30):** Implement `_nightly_health_check_job()` in `backend/services/analysis.py`
2. **Gemini CLI:** Add env vars to Railway for threshold tuning
3. **OpenClaw:** Wire APScheduler job for 4:30 AM ET daily execution
4. **Post-Implementation:** Monitor for 2 weeks, adjust thresholds if false positive rate > 25%

---

*Report generated by Kimi CLI for CBB Edge Analyzer — K-7 Mission Complete*
