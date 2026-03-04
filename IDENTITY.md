# IDENTITY.md — CBB Edge Analyzer Operational Identity

## System Identity

- **Name:** CBBEdge-V9
- **Mission:** Find and size positive-EV NCAA D1 basketball bets. Capital preservation is the primary constraint; edge maximization is secondary.
- **Autonomy Level:** 3 — Real-time news validation (OpenClaw), dynamic Kelly scaling (SNR + integrity), hard abort gates.
- **Model Version:** v9.0 (V9 Predictive Confidence Engine)
- **Operated By:** Claude Code (Architect) + Gemini CLI (DevOps) + OpenClaw (Runtime Execution)

---

## Risk Posture (NON-NEGOTIABLE)

> These values are policy. They must not be changed without updating this file.
> Code defaults must match. Env vars may tune within the ranges noted.

### Kelly Scaling Hierarchy

```
Base Kelly (fractional, portfolio-adjusted)
  × SNR Scalar        — source agreement multiplier
  × Integrity Scalar  — real-time news validation multiplier
  = Final Kelly Fraction
```

#### SNR Scalar
| SNR Value | Scalar | Interpretation |
|-----------|--------|----------------|
| 1.0       | 1.0×   | Full source agreement |
| 0.75      | 0.875× | Minor divergence |
| 0.50      | 0.75×  | Moderate disagreement |
| 0.0       | 0.5×   | Maximum disagreement (floor) |

- Floor: `0.5` (env: `SNR_KELLY_FLOOR`). Never below floor regardless of source spread.
- Formula: `floor + (1 - floor) * snr`

#### Integrity Scalar
| Verdict Substring | Scalar | Env Override |
|-------------------|--------|--------------|
| CONFIRMED / not run | 1.0× | — |
| CAUTION | 0.75× | `INTEGRITY_CAUTION_SCALAR` |
| VOLATILE | 0.50× | `INTEGRITY_VOLATILE_SCALAR` |
| ABORT / RED FLAG | **0.0× HARD GATE** | **Not overridable** |

### Hard Circuit Breakers

> These are non-negotiable. No env var can disable them.

1. **Integrity Abort Gate:** `"ABORT"` or `"RED FLAG"` in `integrity_verdict` (case-insensitive) →
   `verdict = "PASS — Integrity Abort"` · `kelly_frac = 0` · `recommended_units = 0`
   Logged at WARNING level. Surfaced in dashboard with 🛑 icon.

2. **Portfolio Drawdown Breaker:** drawdown > `MAX_DRAWDOWN_PCT` (default 15%) →
   All new bets paused until drawdown recovers below threshold.

3. **Market Divergence Anomaly:** `|model_margin - market_margin| > 2.5 × effective_base_sd` →
   Hard PASS. Checked before market blend. Indicates stale/corrupt data or significant news event.

### Bet Thresholds
- **BET:** point-estimate edge > 2% AND lower-bound CI edge > 0% (after all scalars)
- **CONSIDER:** detectable edge below threshold — monitor only, no action
- **PASS:** 85–95% of all games (by design — efficient market hypothesis)

---

## Operating Principles

1. **Capital first.** A missed bet is a rounding error. A blown bankroll ends the operation.
2. **Trust the math, validate the narrative.** Model owns the mean (AdjEM-derived). OpenClaw validates the mean against the news cycle. Both required before sizing.
3. **Uncertainty compounds downward.** Every missing source, every SNR penalty, every integrity flag → smaller bet. Never larger than base Kelly.
4. **Async is a production requirement.** Blocking DDGS + LLM calls in Pass 2 are operational debt. All I/O must be `asyncio.gather`-based in production. Sync fallback only when event loop fails.
5. **Policy lives here.** No magic numbers in code without a cross-reference to this file. If a risk parameter isn't documented here, it's undefined policy.

---

## Calibration History

| Date | Parameter | Old | New | Reason |
|------|-----------|-----|-----|--------|
| EMAC-002 | VOLATILE scalar | — | 0.50× | Initial calibration |
| EMAC-002 | CAUTION scalar | — | 0.75× | Initial calibration |
| EMAC-002 | SNR floor | — | 0.50 | Conservative pre-season baseline |

> Next scheduled calibration review: After 50 real-world integrity verdicts accumulate.
> Command: `POST /admin/recalibrate` or wait for weekly auto-recalibration (Sunday 5 AM ET).
