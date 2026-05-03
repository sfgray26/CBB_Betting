## 0. Gemini Deployment Bundle — Sessions W+X+Y (BLOCKING — do first)

Deploy commit `00c53ae` and verify statcast signals reach the waiver UI:

```bash
BASE=https://fantasy-app-production-5079.up.railway.app

# 1. Deploy
railway up --detach

# 2. Wait for healthy
curl -s "$BASE/health" | python -m json.tool

# 3. Trigger player_id_mapping sync (V3 — builds BDL->MLBAM bridge)
curl -s -X POST -H "X-API-Key: $API_KEY_USER1" \
  "$BASE/admin/ingestion/run/player_id_mapping" | python -m json.tool
# Expected: "mlbam_found": N, "statcast_patched": M (both > 0)

# 4. Trigger cat_scores backfill (V1 — fixes MCMC win_prob_gain=0.0)
curl -s -X POST -H "X-API-Key: $API_KEY_USER1" \
  "$BASE/admin/ingestion/run/cat_scores_backfill" | python -m json.tool
# Expected: {"status": "success", "records": N} where N > 0

# 5. Run field-coverage diagnostic
curl -s -H "X-API-Key: $API_KEY_USER1" \
  "$BASE/admin/diagnostics/field-coverage" | python -m json.tool
# Expected: scarcity_rank_populated > 0, quality_score_populated > 0

# 6. Check waiver recommendations logs for non-zero win_prob_gain
# POST waiver recommendations and look for [MCMC_DEBUG] lines in Railway logs
# Expected: win_prob_gain != 0.0 for at least some recommendations
# Expected: statcast_loader DB tier log line showing > 0 batters loaded
```

**Report:** deployment success, player_id_mapping mlbam_found count,
cat_scores backfill records count, field-coverage output, sample win_prob_gain values.

---

