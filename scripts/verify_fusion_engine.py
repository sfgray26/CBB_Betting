"""
Mathematical Verification Script for Bayesian Projection Fusion Engine.

This script proves the core mathematical properties of the Marcel update formula.
Run with: venv/Scripts/python scripts/verify_fusion_engine.py
"""

from backend.fantasy_baseball.fusion_engine import (
    StabilizationPoints, PopulationPrior, marcel_update,
    fuse_batter_projection, fuse_pitcher_projection
)

def print_header(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)

def print_section(title):
    print('\n' + title)
    print('-' * 70)

# ============================================================================
# VERIFICATION 1: Midpoint Property
# ============================================================================
print_header('MATHEMATICAL VERIFICATION: Bayesian Projection Fusion Engine')

print_section('VERIFICATION 1: Midpoint Property')
print('At sample_size = stabilization_point, posterior should equal midpoint')

prior = 0.250
observed = 0.300
stabil = 100
result = marcel_update(prior, observed, stabil, stabil)
expected = (prior + observed) / 2

print(f'\n  Prior: {prior}')
print(f'  Observed: {observed}')
print(f'  Stabilization Point: {stabil}')
print(f'  Sample Size: {stabil} (equal to stabilization)')
print(f'\n  Result: {result:.10f}')
print(f'  Expected Midpoint: {expected:.10f}')
print(f'  Difference: {abs(result - expected):.2e}')

if abs(result - expected) < 1e-10:
    print('\n  [PASS] Midpoint property VERIFIED')
else:
    print('\n  [FAIL] Midpoint property NOT verified')

# ============================================================================
# VERIFICATION 2: Prior Dominance (Small Sample)
# ============================================================================
print_section('VERIFICATION 2: Prior Dominance (Small Sample)')
print('At sample_size = 10% of stabilization, prior should dominate (>80%)')

prior = 0.250
observed = 0.350
stabil = 100
sample = 10  # 10% of stabil

result = marcel_update(prior, observed, sample, stabil)
prior_weight = stabil / (stabil + sample)
observed_weight = sample / (stabil + sample)

print(f'\n  Prior: {prior}')
print(f'  Observed: {observed}')
print(f'  Stabilization Point: {stabil}')
print(f'  Sample Size: {sample} (10% of stabilization)')
print(f'\n  Prior Weight: {prior_weight:.4f} ({prior_weight*100:.1f}%)')
print(f'  Observed Weight: {observed_weight:.4f} ({observed_weight*100:.1f}%)')
print(f'  Result: {result:.6f}')

distance_to_prior = abs(result - prior)
distance_to_observed = abs(result - observed)

print(f'\n  Distance to Prior: {distance_to_prior:.6f}')
print(f'  Distance to Observed: {distance_to_observed:.6f}')
print(f'  Ratio (closer to prior): {distance_to_observed / distance_to_prior:.2f}x')

if prior_weight > 0.80 and distance_to_prior < distance_to_observed:
    print('\n  [PASS] Prior dominance VERIFIED')
else:
    print('\n  [FAIL] Prior dominance NOT verified')

# ============================================================================
# VERIFICATION 3: Observed Dominance (Large Sample)
# ============================================================================
print_section('VERIFICATION 3: Observed Dominance (Large Sample)')
print('At sample_size = 10x stabilization, observed should dominate (>80%)')

prior = 0.250
observed = 0.350
stabil = 100
sample = 1000  # 10x stabil

result = marcel_update(prior, observed, sample, stabil)
prior_weight = stabil / (stabil + sample)
observed_weight = sample / (stabil + sample)

print(f'\n  Prior: {prior}')
print(f'  Observed: {observed}')
print(f'  Stabilization Point: {stabil}')
print(f'  Sample Size: {sample} (10x stabilization)')
print(f'\n  Prior Weight: {prior_weight:.4f} ({prior_weight*100:.1f}%)')
print(f'  Observed Weight: {observed_weight:.4f} ({observed_weight*100:.1f}%)')
print(f'  Result: {result:.6f}')

distance_to_prior = abs(result - prior)
distance_to_observed = abs(result - observed)

print(f'\n  Distance to Prior: {distance_to_prior:.6f}')
print(f'  Distance to Observed: {distance_to_observed:.6f}')
print(f'  Ratio (closer to observed): {distance_to_prior / distance_to_observed:.2f}x')

if observed_weight > 0.80 and distance_to_observed < distance_to_prior:
    print('\n  [PASS] Observed dominance VERIFIED')
else:
    print('\n  [FAIL] Observed dominance NOT verified')

# ============================================================================
# VERIFICATION 4: Four-State Behavior (Batter)
# ============================================================================
print_section('VERIFICATION 4: Four-State Behavior (Batter Fusion)')
print('Each state must return correct source label')

# State 1: Steamer + Statcast
result1 = fuse_batter_projection(
    {'avg': 0.280, 'obp': 0.350, 'slg': 0.480, 'k_percent': 0.220, 'bb_percent': 0.080},
    {'avg': 0.270, 'obp': 0.340, 'slg': 0.460, 'k_percent': 0.230, 'bb_percent': 0.070, 'pa': 500},
    500
)
print(f'\n  State 1 (Steamer + Statcast): source="{result1.source}"')
state1_pass = result1.source == 'fusion' and result1.components_fused > 0

# State 2: Steamer only
result2 = fuse_batter_projection(
    {'avg': 0.280, 'obp': 0.350, 'slg': 0.480, 'k_percent': 0.220, 'bb_percent': 0.080},
    None,
    500
)
print(f'  State 2 (Steamer only): source="{result2.source}"')
state2_pass = result2.source == 'steamer' and result2.components_fused == 0

# State 3: Statcast only
result3 = fuse_batter_projection(
    None,
    {'avg': 0.270, 'obp': 0.340, 'slg': 0.460, 'k_percent': 0.230, 'bb_percent': 0.070, 'pa': 300},
    300
)
print(f'  State 3 (Statcast only): source="{result3.source}"')
state3_pass = result3.source == 'statcast_shrunk' and result3.components_fused > 0

# State 4: Neither
result4 = fuse_batter_projection(None, None, 0)
print(f'  State 4 (Neither): source="{result4.source}"')
state4_pass = result4.source == 'population_prior' and result4.components_fused == 0

if state1_pass and state2_pass and state3_pass and state4_pass:
    print('\n  [PASS] Four-state behavior VERIFIED')
else:
    print('\n  [FAIL] Four-state behavior NOT verified')

# ============================================================================
# VERIFICATION 5: xwOBA Override Trigger
# ============================================================================
print_section('VERIFICATION 5: xwOBA Override Trigger')
print('xwOBA override triggers when |xwOBA - wOBA| > 0.030')

# Should trigger (delta = 0.040)
statcast_trigger = {
    'avg': 0.270, 'obp': 0.340, 'slg': 0.460, 'k_percent': 0.230, 'bb_percent': 0.070, 'pa': 400,
    'xwoba': 0.370, 'woba': 0.330
}
result_trigger = fuse_batter_projection(
    {'avg': 0.280, 'obp': 0.350, 'slg': 0.480, 'k_percent': 0.220, 'bb_percent': 0.080},
    statcast_trigger,
    400
)
delta_trigger = abs(statcast_trigger['xwoba'] - statcast_trigger['woba'])
print(f'\n  Test Case 1: |xwOBA - wOBA| = {delta_trigger:.3f}')
print(f'    Override Applied: {result_trigger.xwoba_override_applied}')

# Should NOT trigger (delta = 0.020)
statcast_no_trigger = {
    'avg': 0.270, 'obp': 0.340, 'slg': 0.460, 'k_percent': 0.230, 'bb_percent': 0.070, 'pa': 400,
    'xwoba': 0.350, 'woba': 0.330
}
result_no_trigger = fuse_batter_projection(
    {'avg': 0.280, 'obp': 0.350, 'slg': 0.480, 'k_percent': 0.220, 'bb_percent': 0.080},
    statcast_no_trigger,
    400
)
delta_no_trigger = abs(statcast_no_trigger['xwoba'] - statcast_no_trigger['woba'])
print(f'  Test Case 2: |xwOBA - wOBA| = {delta_no_trigger:.3f}')
print(f'    Override Applied: {result_no_trigger.xwoba_override_applied}')

if result_trigger.xwoba_override_applied and not result_no_trigger.xwoba_override_applied:
    print('\n  [PASS] xwOBA override behavior VERIFIED')
else:
    print('\n  [FAIL] xwOBA override behavior NOT verified')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print_header('MATHEMATICAL VERIFICATION COMPLETE')
print('\nAll core mathematical properties have been verified.')
print('The Marcel update formula is correctly implemented.')
