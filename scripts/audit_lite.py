import traceback
import sys
import os
# Add the project root to the path so we can import our services
sys.path.append(os.getcwd())

try:
    from backend.main import SessionLocal
    from backend.services.performance import calculate_summary_stats
    from backend.services.recalibration import run_recalibration

    db = SessionLocal()
    
    # 1. Get Performance
    stats = calculate_summary_stats(db)
    roi = stats.get('roi', 0) * 100
    win_rate = stats.get('win_rate', 0) * 100
    clv = stats.get('mean_clv', 0)
    
    # 2. Check Recalibration (Dry Run)
    recal = run_recalibration(db, apply_changes=False)
    bias = recal.get('diagnostics', {}).get('overall_bias', 0)
    
    db.close()

    print(f"LITE AUDIT REPORT:")
    print(f"- Win Rate: {win_rate:.1f}%")
    print(f"- ROI: {roi:.1f}%")
    print(f"- CLV: {clv:+.3f}")
    print(f"- Model Bias: {bias:+.2f} pts")
    print(f"Status: {'HEALTHY' if clv >= 0 else 'WARNING'}")

except Exception as e:
    print(f"Audit failed: {str(e)}")
    traceback.print_exc()
