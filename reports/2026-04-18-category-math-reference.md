# K4 Category Math Reference Sheet
**Date:** 2026-04-18  
**Author:** Kimi CLI  
**Scope:** One-page math reference for each of the 18 v2 canonical scoring categories.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| `current` | Value already accumulated in the active matchup week (from Yahoo scoreboard) |
| `row` | Rest-of-Week projection (see K2 spec) |
| `proj_final` | `current + row` |
| `opp_proj_final` | Opponent's projected final |
| `margin` | `my_proj_final - opp_proj_final` for higher-is-better; reversed for lower-is-better |
| `flip` | The change needed to reverse who is winning the category |

---

## Batting Categories (9)

### R — Runs
| Item | Value |
|------|-------|
| Yahoo stat_id | 7 |
| Aggregation | Sum |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["R"]` |
| ROW projection | `sum(daily_rate_r × games_remaining)` |
| Proj final | `current_R + row_R` |
| Margin | `proj_final_R - opp_proj_final_R` |
| Delta-to-flip | `opp_proj_final_R - proj_final_R + 1` runs |
| Display string | `"Need +{delta} R"` or `"Lead safe by {abs(delta)} R"` |

### H — Hits
| Item | Value |
|------|-------|
| Yahoo stat_id | 8 |
| Aggregation | Sum |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["H"]` |
| ROW projection | `sum(daily_rate_h × games_remaining)` |
| Proj final | `current_H + row_H` |
| Margin | `proj_final_H - opp_proj_final_H` |
| Delta-to-flip | `opp_proj_final_H - proj_final_H + 1` hits |
| Display string | `"Need +{delta} H"` or `"Lead safe by {abs(delta)} H"` |

### HR_B — Home Runs (Batting)
| Item | Value |
|------|-------|
| Yahoo stat_id | 12 |
| Aggregation | Sum |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["HR"]` |
| ROW projection | `sum(daily_rate_hr × games_remaining)` |
| Proj final | `current_HR + row_HR` |
| Margin | `proj_final_HR - opp_proj_final_HR` |
| Delta-to-flip | `opp_proj_final_HR - proj_final_HR + 1` HR |
| Display string | `"Need +{delta} HR"` or `"Lead safe by {abs(delta)} HR"` |

### RBI — Runs Batted In
| Item | Value |
|------|-------|
| Yahoo stat_id | 13 |
| Aggregation | Sum |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["RBI"]` |
| ROW projection | `sum(daily_rate_rbi × games_remaining)` |
| Proj final | `current_RBI + row_RBI` |
| Margin | `proj_final_RBI - opp_proj_final_RBI` |
| Delta-to-flip | `opp_proj_final_RBI - proj_final_RBI + 1` RBI |
| Display string | `"Need +{delta} RBI"` or `"Lead safe by {abs(delta)} RBI"` |

### K_B — Strikeouts (Batter)
| Item | Value |
|------|-------|
| Yahoo stat_id | 42 |
| Aggregation | Sum |
| Direction | **lower_is_better** |
| Current value | `yahoo_scoreboard["K"]` (batter) |
| ROW projection | `sum(daily_rate_k_b × games_remaining)` |
| Proj final | `current_K_B + row_K_B` |
| Margin | `opp_proj_final_K_B - proj_final_K_B` (note: reversed because lower is better) |
| Delta-to-flip | `proj_final_K_B - opp_proj_final_K_B - 1` fewer K |
| Display string | `"Need {delta} fewer K"` or `"Keep K ≤ {opp_proj_final_K_B - 1}"` |

### TB — Total Bases
| Item | Value |
|------|-------|
| Yahoo stat_id | 4 or 6 (alt) |
| Aggregation | Sum |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["TB"]` |
| ROW projection | `sum(daily_rate_tb × games_remaining)` |
| Proj final | `current_TB + row_TB` |
| Margin | `proj_final_TB - opp_proj_final_TB` |
| Delta-to-flip | `opp_proj_final_TB - proj_final_TB + 1` TB |
| Display string | `"Need +{delta} TB"` or `"Lead safe by {abs(delta)} TB"` |

### AVG — Batting Average
| Item | Value |
|------|-------|
| Yahoo stat_id | 3 |
| Aggregation | weighted_ratio: sum(H) / sum(AB) |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["AVG"]` (Yahoo provides this as a float) |
| Underlying numerators/denominators | Must be fetched or tracked: `current_H`, `current_AB` |
| ROW projection | Team-level: `row_H / row_AB` (component sums, not ratio of ratios) |
| Proj final | `(current_H + row_H) / (current_AB + row_AB)` |
| Margin | `proj_final_AVG - opp_proj_final_AVG` |
| Delta-to-flip | Solve for `delta_H` such that `(current_H + row_H + delta_H) / (current_AB + row_AB + delta_AB) > opp_proj_final_AVG`. Simplification: if `delta_AB ≈ delta_H / current_AVG`, then `delta_H ≈ (opp_proj_final_AVG × (current_AB + row_AB + delta_AB)) - (current_H + row_H)`. For display purposes, approximate: `"Need ~{ceil(delta)} more hits in ~{ceil(delta_ab)} AB"` |
| Display string | `"Keep AVG ≥ {opp_proj_final_AVG:.3f}"` or `"Need ~{delta} more H"` |

### OPS — On-Base Plus Slugging
| Item | Value |
|------|-------|
| Yahoo stat_id | 55 |
| Aggregation | composite: OBP + SLG |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["OPS"]` |
| Underlying components | `current_OBP_num = current_H + current_BB`, `current_OBP_den = current_AB + current_BB`, `current_SLG_num = current_TB` |
| ROW projection | Team-level: `(row_OBP_num / row_OBP_den) + (row_SLG_num / row_SLG_den)` |
| Proj final | `((current_OBP_num + row_OBP_num) / (current_OBP_den + row_OBP_den)) + ((current_SLG_num + row_SLG_num) / (current_SLG_den + row_SLG_den))` |
| Margin | `proj_final_OPS - opp_proj_final_OPS` |
| Delta-to-flip | Complex — approximate with `"Need ~{delta} more total bases"` or `"Keep OPS ≥ {opp_proj_final_OPS:.3f}"` |
| Display string | `"Keep OPS ≥ {opp_proj_final_OPS:.3f}"` or `"Need +{delta} TB-equivalent"` |

### NSB — Net Stolen Bases
| Item | Value |
|------|-------|
| Yahoo stat_id | 60 |
| Aggregation | composite: SB - CS |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["NSB"]` (or derived from SB - CS) |
| ROW projection | `sum(daily_rate_nsb × games_remaining)` |
| Proj final | `current_NSB + row_NSB` |
| Margin | `proj_final_NSB - opp_proj_final_NSB` |
| Delta-to-flip | `opp_proj_final_NSB - proj_final_NSB + 1` NSB |
| Display string | `"Need +{delta} NSB"` or `"Lead safe by {abs(delta)} NSB"` |

---

## Pitching Categories (9)

### W — Wins
| Item | Value |
|------|-------|
| Yahoo stat_id | 23 |
| Aggregation | Sum |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["W"]` |
| ROW projection | `sum(remaining_start_prob × win_prob_per_start)` |
| Proj final | `current_W + row_W` |
| Margin | `proj_final_W - opp_proj_final_W` |
| Delta-to-flip | `opp_proj_final_W - proj_final_W + 1` wins |
| Display string | `"Need +{delta} W"` or `"Lead safe by {abs(delta)} W"` |

### L — Losses
| Item | Value |
|------|-------|
| Yahoo stat_id | 24 |
| Aggregation | Sum |
| Direction | **lower_is_better** |
| Current value | `yahoo_scoreboard["L"]` |
| ROW projection | `sum(remaining_start_prob × loss_prob_per_start)` |
| Proj final | `current_L + row_L` |
| Margin | `opp_proj_final_L - proj_final_L` (reversed) |
| Delta-to-flip | `proj_final_L - opp_proj_final_L - 1` fewer losses |
| Display string | `"Keep L ≤ {opp_proj_final_L - 1}"` or `"Need {delta} fewer L"` |

### HR_P — Home Runs Allowed
| Item | Value |
|------|-------|
| Yahoo stat_id | 35 |
| Aggregation | Sum |
| Direction | **lower_is_better** |
| Current value | `yahoo_scoreboard["HRA"]` |
| ROW projection | `sum(daily_rate_hr_p × projected_remaining_ip / 9)` |
| Proj final | `current_HR_P + row_HR_P` |
| Margin | `opp_proj_final_HR_P - proj_final_HR_P` (reversed) |
| Delta-to-flip | `proj_final_HR_P - opp_proj_final_HR_P - 1` fewer HRA |
| Display string | `"Keep HRA ≤ {opp_proj_final_HR_P - 1}"` or `"Need {delta} fewer HRA"` |

### K_P — Strikeouts (Pitcher)
| Item | Value |
|------|-------|
| Yahoo stat_id | 28 |
| Aggregation | Sum |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["K"]` (pitching) |
| ROW projection | `sum(daily_rate_k_p × games_remaining)` |
| Proj final | `current_K_P + row_K_P` |
| Margin | `proj_final_K_P - opp_proj_final_K_P` |
| Delta-to-flip | `opp_proj_final_K_P - proj_final_K_P + 1` K |
| Display string | `"Need +{delta} K"` or `"Lead safe by {abs(delta)} K"` |

### ERA — Earned Run Average
| Item | Value |
|------|-------|
| Yahoo stat_id | 26 |
| Aggregation | weighted_ratio: `(ER × 27) / IP_outs` |
| Direction | **lower_is_better** |
| Current value | `yahoo_scoreboard["ERA"]` |
| Underlying numerators/denominators | `current_ER`, `current_IP` |
| ROW projection | Team-level: `9 × row_ER / row_IP` |
| Proj final | `9 × (current_ER + row_ER) / (current_IP + row_IP)` |
| Margin | `opp_proj_final_ERA - proj_final_ERA` (reversed) |
| Delta-to-flip | Solve for `delta_ER` such that `9 × (current_ER + row_ER + delta_ER) / (current_IP + row_IP) < opp_proj_final_ERA`. Rearranged: `delta_ER < (opp_proj_final_ERA × (current_IP + row_IP) / 9) - (current_ER + row_ER)`. Since `delta_ER` must be **earned runs allowed**, display as: `"Allow ≤{floor(delta)} more ER in remaining {row_IP:.1f} IP"` |
| Display string | `"Keep ERA < {opp_proj_final_ERA:.2f}"` or `"Allow ≤{delta} more ER"` |

### WHIP — Walks + Hits per IP
| Item | Value |
|------|-------|
| Yahoo stat_id | 27 |
| Aggregation | weighted_ratio: `(H_allowed + BB_allowed) / IP` |
| Direction | **lower_is_better** |
| Current value | `yahoo_scoreboard["WHIP"]` |
| Underlying numerators/denominators | `current_H_allowed`, `current_BB_allowed`, `current_IP` |
| ROW projection | Team-level: `(row_H_allowed + row_BB_allowed) / row_IP` |
| Proj final | `(current_H_allowed + row_H_allowed + current_BB_allowed + row_BB_allowed) / (current_IP + row_IP)` |
| Margin | `opp_proj_final_WHIP - proj_final_WHIP` (reversed) |
| Delta-to-flip | Solve for `delta_baserunners` such that `(current_num + row_num + delta) / (current_IP + row_IP) < opp_proj_final_WHIP`. `delta = floor(opp_proj_final_WHIP × (current_IP + row_IP) - (current_num + row_num))`. Display: `"Allow ≤{delta} more H+BB"` |
| Display string | `"Keep WHIP < {opp_proj_final_WHIP:.3f}"` or `"Allow ≤{delta} more H+BB"` |

### K_9 — Strikeouts per Nine
| Item | Value |
|------|-------|
| Yahoo stat_id | 57 |
| Aggregation | weighted_ratio: `(K_P × 27) / IP_outs` |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["K/9"]` |
| Underlying numerators/denominators | `current_K_P`, `current_IP` |
| ROW projection | Team-level: `27 × row_K_P / row_IP_outs` |
| Proj final | `27 × (current_K_P + row_K_P) / (current_IP_outs + row_IP_outs)` |
| Margin | `proj_final_K_9 - opp_proj_final_K_9` |
| Delta-to-flip | Solve for `delta_K` such that `27 × (current_K_P + row_K_P + delta_K) / (current_IP_outs + row_IP_outs) > opp_proj_final_K_9`. `delta_K = ceil((opp_proj_final_K_9 × (current_IP_outs + row_IP_outs) / 27) - (current_K_P + row_K_P)) + 1`. Display: `"Need +{delta} K"` |
| Display string | `"Need +{delta} K to pass"` or `"Lead safe — keep K/9 ≥ {opp_proj_final_K_9:.2f}"` |

### QS — Quality Starts
| Item | Value |
|------|-------|
| Yahoo stat_id | 29 |
| Aggregation | Sum |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["QS"]` |
| ROW projection | `sum(remaining_starts × qs_rate)` |
| Proj final | `current_QS + row_QS` |
| Margin | `proj_final_QS - opp_proj_final_QS` |
| Delta-to-flip | `opp_proj_final_QS - proj_final_QS + 1` QS |
| Display string | `"Need +{delta} QS"` or `"Lead safe by {abs(delta)} QS"` |

### NSV — Net Saves
| Item | Value |
|------|-------|
| Yahoo stat_id | 83 |
| Aggregation | composite: SV - BS |
| Direction | higher_is_better |
| Current value | `yahoo_scoreboard["NSV"]` (or derived) |
| ROW projection | `sum(remaining_appearances × (sv_rate - bs_rate))` |
| Proj final | `current_NSV + row_NSV` |
| Margin | `proj_final_NSV - opp_proj_final_NSV` |
| Delta-to-flip | `opp_proj_final_NSV - proj_final_NSV + 1` NSV |
| Display string | `"Need +{delta} NSV"` or `"Lead safe by {abs(delta)} NSV"` |

---

## Margin Sign Convention (Critical for Phase 3 Implementation)

```python
def compute_margin(my_proj_final: float, opp_proj_final: float, is_lower_better: bool) -> float:
    if is_lower_better:
        # For ERA: lower is better, so if my ERA is 3.50 and opp is 4.00, I'm winning
        # Margin should be positive when I'm winning
        return opp_proj_final - my_proj_final
    else:
        return my_proj_final - opp_proj_final
```

**Rule:** `margin > 0` always means "I am winning this category." This is the convention used in `MatchupScoreboardRow.current_margin` (`contracts.py:280`).

---

## Delta-to-Flip: Unified Formula

### Counting Stats (higher_is_better)
```python
delta = opp_proj_final - my_proj_final + 1
```
If `delta <= 0`, the category is already won.

### Counting Stats (lower_is_better)
```python
delta = my_proj_final - opp_proj_final - 1
```
If `delta <= 0`, the category is already won.

### Ratio Stats (higher_is_better) — e.g., AVG, OPS, K/9
```python
# Solve for the increment needed to my numerator such that my ratio exceeds opponent's
# Approximation: assume denominator stays constant (new AB/IP are known)
delta_numerator = ceil(opp_proj_final * my_denominator - my_numerator) + 1
```

### Ratio Stats (lower_is_better) — e.g., ERA, WHIP
```python
# Solve for the maximum additional numerator allowed such that my ratio stays below opponent's
delta_numerator = floor(opp_proj_final * my_denominator - my_numerator)
```

---

## Recommended Next Actions (for Claude)

1. **Implement `compute_margin()` and `compute_delta_to_flip()`** as pure functions in a new module (e.g., `backend/fantasy_baseball/category_math.py`).
2. **Add unit tests** for ratio-stat delta-to-flip with known values (e.g., ERA 3.50 vs 4.00 with 20 IP remaining).
3. **Wire into scoreboard endpoint** so `MatchupScoreboardRow.current_margin`, `projected_margin`, and `delta_to_flip` are populated.
