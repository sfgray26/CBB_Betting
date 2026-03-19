# Task: Oracle-Based Model Validation

**STATUS: OPEN** (post-tournament, after V9.2)
**Assigned to:** Kimi CLI (research) + Claude Code (implementation)
**Estimated:** 2 sessions
**Priority:** MEDIUM

## Concept

From Anthropic's C Compiler paper: compare our model against a known-good baseline.
Our oracle: KenPom + BartTorvik consensus. If our spread differs by >5 pts from consensus,
flag for review before the bet is placed.

## Kimi Delegation (Phase 1 — Research)

Kimi reads:
- `backend/services/ratings.py` — how ratings are fetched
- `backend/betting_model.py` — how spread is computed
- `backend/services/analysis.py` — where predictions are stored

Kimi produces:
- `reports/K15_ORACLE_VALIDATION_SPEC.md`
- Defines: oracle consensus formula, threshold (±5 pts), flag workflow

## Claude Implementation (Phase 2 — Code)

After Kimi spec is ready:

1. `backend/services/oracle.py` — `validate_against_consensus(prediction, ratings) -> OracleResult`
2. Wire into `analysis.py` — after each prediction, run oracle check
3. Add `oracle_flag` column to `Prediction` model (boolean)
4. Admin endpoint: `GET /admin/oracle/flagged` — predictions that differed >5 pts from consensus
5. Tests in `tests/test_oracle.py`

## Success Criteria

- [ ] Oracle flags ≥1 prediction in backtesting
- [ ] Admin endpoint returns flagged predictions
- [ ] All tests pass
- [ ] GUARDIAN not violated (check timing against Apr 7)
